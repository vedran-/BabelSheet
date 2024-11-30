from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator
from ..utils.llm_handler import LLMHandler
import pandas as pd
import json
from ..utils.qa_handler import QAHandler
from ..term_base.term_base_handler import TermBaseHandler
from ..sheets.sheets_handler import SheetsHandler, CellData
from ..utils.ui_manager import UIManager
import asyncio
import logging

logger = logging.getLogger(__name__)
ui = UIManager()

class TranslationManager:
    def __init__(self, config: Dict, sheets_handler: SheetsHandler, term_base_handler: TermBaseHandler):
        """Initialize Translation Manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        llm_config = config.get('llm', {})
        
        self.sheets_handler = sheets_handler
        self.term_base_handler = term_base_handler
        
        # Initialize LLM Handler with correct parameters from config
        self.llm_handler = LLMHandler(
            api_key=llm_config.get('api_key'),
            base_url=llm_config.get('api_url', "https://api.openai.com/v1"),
            model=llm_config.get('model', 'o1-mini'),
            temperature=llm_config.get('temperature', 0.3),
            config=llm_config
        )
        
        # Initialize QA Handler
        self.qa_handler = QAHandler(
            max_length=config.get('max_length'),
            llm_handler=self.llm_handler,
            non_translatable_patterns=config.get('qa', {}).get('non_translatable_patterns', [])
        )
        
        # Get batch configuration
        self.batch_size = llm_config.get('batch_size', 10)
        self.batch_delay = llm_config.get('batch_delay', 1)
        self.max_retries = llm_config.get('max_retries', 3)
        self.retry_delay = llm_config.get('retry_delay', 1)  # seconds

    async def ensure_sheet_translations(self, sheet_name: str, source_lang: str, 
                                        target_langs: List[str], use_term_base: bool) -> None:
        """Ensure all terms in the term base have translations for target languages.
        This should be called before starting translation of other sheets.
        
        Args:
            sheet_name: Name of the sheet to ensure translations for
            source_lang: Source language code
            target_langs: List of target language codes
        """
        logger.debug(f"Ensuring translations for sheet '{sheet_name}' from '{source_lang}' to {target_langs}")
        
        # Get the sheet data
        df = self.sheets_handler.get_sheet_data(sheet_name)
        # Detect missing translations
        missing_translations = self._detect_missing_translations(df, source_lang, target_langs)
        if len(missing_translations) > 0:
            logger.info(f"[{sheet_name}] Found missing translations for {len(missing_translations)} languages: " + 
                       ", ".join(f"{lang} ({len(items)} items)" for lang, items in missing_translations.items()))

        if missing_translations and len(missing_translations) > 0:
            # Process all translations and update the DataFrame
            await self._process_missing_translations(df, missing_translations, use_term_base)
            
            logger.debug(f"Updated sheet '{sheet_name}' with new translations")
        else:
            logger.debug(f"No missing translations found for sheet '{sheet_name}'")

    def _detect_missing_translations(self, df: pd.DataFrame, source_lang: str, target_langs: List[str]) -> Dict[str, List[int]]:
        """Detect missing translations in the DataFrame.
        
        Args:
            df: DataFrame containing translations
            source_lang: Source language code
            target_langs: List of target language codes
            
        Returns:
            Dictionary mapping language codes to missing translations
        """
        missing_translations = {}
        source_lang_idx = self.sheets_handler.get_column_indexes(df, [source_lang])[0]
        target_langs_idx = self.sheets_handler.get_column_indexes(df, target_langs, create_if_missing=True)
        rows_count = df.shape[0]

        # Check each target language
        for i in range(len(target_langs)):
            lang = target_langs[i]
            lang_idx = target_langs_idx[i]
            lang_missing = []
            
            logger.debug(f"Checking {rows_count} rows for language {lang}")

            # Check each row
            for row_idx in range(rows_count):
                if row_idx == 0:
                    continue
                
                source_cell = df.iloc[row_idx][source_lang_idx]
                # Skip if source text is empty
                if source_cell.is_empty():
                    logger.debug(f"Skipping row {row_idx} for language {lang} because source text is empty")
                    continue

                target_cell = df.iloc[row_idx][lang_idx]
                logger.debug(f"Checking row {row_idx} for language {lang}: {source_cell.value} -> {target_cell}")
                # Check if translation is missing or empty
                if target_cell is None or target_cell.is_empty():
                    if pd.isna(target_cell):
                        target_cell = CellData(value=None)
                        df.loc[row_idx, lang_idx] = target_cell

                    context = self.sheets_handler.get_row_context(df, row_idx)

                    lang_missing.append(
                    {
                        'row_idx': row_idx,
                        'col_idx': lang_idx,
                        'source_text': source_cell.value,
                        'target_cell': target_cell,
                        'context': context
                    })

            if len(lang_missing) > 0:
                missing_translations[lang] = lang_missing
        
        return missing_translations

    async def _process_missing_translations(self, df: pd.DataFrame, missing_translations: Dict[str, List[Dict[str, Any]]],
                                            use_term_base: bool) -> None:
        sheet_name = df.attrs['sheet_name']
        skipped_items = []
        
        total_items = sum(len(items) for items in missing_translations.values())
        processed_items = 0
        
        ui.start()
        ui.info(f"Starting batch translation for {len(missing_translations)} languages ({total_items} items total)")
        
        try:
            while len(missing_translations) > 0:
                lang, missing_items = next(iter(missing_translations.items()))
                batch = missing_items[:self.batch_size]
                
                # Add pending translations to UI
                for item in batch:
                    ui.add_translation_entry(item['source_text'], lang)
                
                # Perform translation
                contexts = [self.sheets_handler.get_row_context(df, item['row_idx']) for item in batch]
                
                batch_translations = await self._perform_translation(
                    source_texts=[item['source_text'] for item in batch],
                    source_lang=self.config['languages']['source'],
                    target_lang=lang,
                    contexts=contexts,
                    issues=[item.get('last_issues', []) for item in batch],
                    use_term_base=use_term_base
                )
                
                for idx, (missing_item, (translation, issues)) in enumerate(zip(batch, batch_translations)):
                    processed_items += 1
                    
                    if issues:
                        ui.warning(f"Translation '{translation}' has issues for {lang}: {issues}")
                        all_issues = missing_item.get('last_issues', []) + [{
                            'translation': translation,
                            'issues': issues,
                        }]
                        missing_item['last_issues'] = all_issues
                        
                        if len(all_issues) >= self.max_retries:
                            ui.critical(f"Max retries reached for {lang} item {missing_item['source_text']}. Giving up.")
                            skipped_items.append(missing_item)
                            missing_items.remove(missing_item)
                        
                        ui.complete_translation(missing_item['source_text'], lang, translation, str(issues))
                    else:
                        self.sheets_handler.modify_cell_data(
                            sheet_name=sheet_name,
                            row=missing_item['row_idx'],
                            col=missing_item['col_idx'],
                            value=translation
                        )
                        missing_items.remove(missing_item)
                        ui.complete_translation(missing_item['source_text'], lang, translation)
                
                if len(missing_items) == 0:
                    missing_translations.pop(lang)
                    ui.info(f"Completed all translations for {lang}")
                
                # Start new batch
                ui.start_new_batch()
            
            if skipped_items:
                ui.warning(f"Skipped {len(skipped_items)} items due to max retries")
        finally:
            ui.stop()

    def _prepare_texts_with_contexts(self, source_texts: List[str], 
                                     contexts: List[Dict[str, Any]], 
                                     issues: List[Dict[str, Any]],
                                     term_base: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """Prepare texts with their contexts for translation.
        
        Args:
            source_texts: List of texts to translate
            contexts: List of context strings for each text
            issues: List of issues for each text
            term_base: Optional term base dictionary
            
        Returns:
            Combined text string with contexts and term base
        """

        def escape(text: str) -> str:
            return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def escape_xml(key: str, item: Any) -> str:
            escaped_key = escape(key)
            escaped_item = escape(str(item))
            return f"<{escaped_key}>{escaped_item}</{escaped_key}>"

        texts_with_contexts = []
        for i, (text, context, issues) in enumerate(zip(source_texts, contexts, issues)):
            exc = []
            # Add context
            for key, item in context.items():
                exc.append(escape_xml(key, item))
            # Add issues
            for issue in issues:
                exc.append(f"<FAILED_TRANSLATION>'{escape(issue['translation'])}' error: {escape(issue['issues'])}</FAILED_TRANSLATION>")
            expanded_context = "".join(exc)
            texts_with_contexts.append(f"<text id='{i+1}'>{text}</text>\n<context id='{i+1}'>{expanded_context}</context>")
        
        combined_texts = "\n\n".join(texts_with_contexts)

        if term_base:
            term_base_xml = ["<term_base>"]
            for term, data in term_base.items():
                term_base_xml.append(f"<term><source>{escape(term)}</source><translation>{escape(data["translation"])}</translation><context>{escape(data["context"])}</context></term>")
            term_base_xml.append("</term_base>")
            combined_texts += "\n\nTerm Base References:\n" + "\n".join(term_base_xml)
            
        return combined_texts

    def _create_translation_prompt(self, combined_texts: str, target_lang: str, source_lang: str) -> str:
        """Create the translation prompt with all necessary instructions.
        
        Args:
            combined_texts: Prepared texts with contexts
            target_lang: Target language code
            source_lang: Source language code
            
        Returns:
            Complete prompt string
        """
        return f"""You are a world-class expert in translating to {target_lang}, 
specialized for casual mobile games. Translate the following texts professionally:

{combined_texts}

Rules:
- Use provided term base for consistency
- Don't translate special terms which match the following patterns: {str(self.config['qa']['non_translatable_patterns'])}
- Keep appropriate format (uppercase/lowercase)
- Replace newlines with \\n
- Keep translations lighthearted and fun
- Keep translations concise to fit UI elements
- Localize all output text, except special terms between markup characters
- Review previous failed translations and avoid making the same mistakes

Additionally:
- Identify any important unique terms, like character or item names, in the source text that should be added to the term base.
- Only suggest game-specific terms or terms requiring consistent translation
- Don't suggest terms which are common language words or phrases - only suggest terms which are unique to the game.
- For each suggested term, provide:
  * The term in the source language
  * A suggested translation
  * A brief comment explaining its usage/context in the source language ({source_lang})
- Don't suggest terms that are already in the term base
- Don't suggest special terms that match the non-translatable patterns

Translate each text maintaining all rules. Return translations and term suggestions in a structured JSON format."""

    def _get_translation_schema(self) -> Dict[str, Any]:
        """Get the schema for translation response validation.
        
        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "translation": {
                                "type": "string",
                                "description": "The translated text"
                            },
                            "notes": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Any translator notes or warnings"
                            }
                        },
                        "required": ["translation"]
                    }
                },
                "term_suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_term": {
                                "type": "string",
                                "description": "The term in the source language"
                            },
                            "suggested_translation": {
                                "type": "string",
                                "description": "Suggested translation for the term"
                            },
                            "comment": {
                                "type": "string",
                                "description": f"Brief explanation of term usage/context"
                            }
                        },
                        "required": ["source_term", "suggested_translation", "comment"]
                    },
                    "description": "Suggested terms (names) to add to the term base"
                }
            },
            "required": ["translations"]
        }

    async def _get_llm_translations(self, source_texts: List[str], source_lang: str,
                                  target_lang: str, contexts: List[str], issues: List[Dict[str, Any]],
                                  term_base: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Get translations from LLM.
        
        Args:
            source_texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: List of context strings for each text
            issues: List of issues for each text
            term_base: Optional term base dictionary
            
        Returns:
            Dictionary containing translations and term suggestions
        """
        # Prepare texts and create prompt
        combined_texts = self._prepare_texts_with_contexts(source_texts, contexts, issues, term_base)
        prompt = self._create_translation_prompt(combined_texts, target_lang, source_lang)
        
        # Get translation schema and generate completion
        translation_schema = self._get_translation_schema()
        response = await self.llm_handler.generate_completion(
            messages=[
                {"role": "system", "content": f"You are a world-class expert in translating to {target_lang}."},
                {"role": "user", "content": prompt}
            ],
            json_schema=translation_schema
        )
        
        # Process response
        return self.llm_handler.extract_structured_response(response)

    async def _handle_term_suggestions(self, term_suggestions: List[Dict[str, str]], target_lang: str) -> None:
        """Handle term suggestions by adding them to the term base.
        
        Args:
            term_suggestions: List of term suggestions
            target_lang: Target language code
        """
        if self.term_base_handler and term_suggestions:
            for term in term_suggestions:
                self.term_base_handler.add_new_term(
                    term=term["source_term"],
                    comment=term["comment"],
                    translations={target_lang: term["suggested_translation"]}
                )
            self.logger.info(f"[TERM_BASE] Added {len(term_suggestions)} new terms to the term base")
            self.sheets_handler.save_changes()

    async def _validate_translations(self, source_texts: List[str], translations: List[Dict[str, Any]], 
                                   contexts: List[str], term_base: Optional[Dict[str, Dict[str, Any]]],
                                   previous_issues: List[Dict[str, Any]],
                                   target_lang: str) -> List[Tuple[str, List[str]]]:
        """Validate translations and combine with translator notes.
        
        Args:
            source_texts: Original texts
            translations: Translation results
            term_base: Optional term base
            target_lang: Target language code
            
        Returns:
            List of tuples containing (translated_text, issues)
        """
        results = []
        syntax_validation_tasks = []
        
        # First validate syntax for all translations
        for i, source_text in enumerate(source_texts):
            translated_text = translations[i]["translation"]
            context = contexts[i]
            task = self.qa_handler.validate_translation_syntax(
                source_text=source_text,
                translated_text=translated_text,
                context=context,
                term_base=term_base,
                target_lang=target_lang
            )
            syntax_validation_tasks.append(task)
            
        # Wait for all syntax validations to complete
        syntax_results = await asyncio.gather(*syntax_validation_tasks)
        
        # Prepare items for LLM validation
        llm_validation_items = []
        llm_validation_indexes = []
        
        for i, (translation_dict, syntax_issues) in enumerate(zip(translations, syntax_results)):
            translated_text = translation_dict["translation"]
            translator_notes = translation_dict.get("notes", [])
            
            # If there are no syntax issues, add to LLM validation batch
            if len(syntax_issues) == 0:
                llm_validation_items.append({
                    'source_text': source_texts[i],
                    'translated_text': translated_text,
                    'context': contexts[i],
                    'previous_issues': previous_issues[i]
                })
                llm_validation_indexes.append(i)
            
            # Store initial results with syntax issues
            results.append((translated_text, translator_notes + syntax_issues))
        
        # Perform batch LLM validation if needed
        if llm_validation_items:
            llm_results = await self.qa_handler.validate_with_llm_batch(llm_validation_items, target_lang)
            
            # Update results with LLM validation issues
            for batch_idx, result_idx in enumerate(llm_validation_indexes):
                translated_text, current_issues = results[result_idx]
                llm_issues = llm_results[batch_idx]
                results[result_idx] = (translated_text, current_issues + llm_issues)
        
        return results

    async def _perform_translation(self, source_texts: List[str], source_lang: str, 
                            target_lang: str, contexts: List[str], issues: List[Dict[str, Any]],
                            use_term_base: bool
                            ) -> List[Tuple[str, List[str]]]:
        """Internal method to perform batch translation and update cells.
        
        Args:
            source_texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: List of context strings for each text
            issues: List of issues for each text
            term_base: Optional term base dictionary
            
        Returns:
            List of tuples containing (translated_text, issues) for each input text
        """

        term_base = self.term_base_handler.get_terms_for_language(target_lang) if use_term_base else None

        # Get translations from LLM
        result = await self._get_llm_translations(
            source_texts=source_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            contexts=contexts,
            issues=issues,
            term_base=term_base
        )
        
        translations = result["translations"]
        term_suggestions = result.get("term_suggestions", [])
        
        # Handle term suggestions
        await self._handle_term_suggestions(term_suggestions, target_lang)
        
        # Validate translations
        return await self._validate_translations(
            source_texts=source_texts,
            translations=translations,
            contexts=contexts,
            term_base=term_base,
            previous_issues=issues,
            target_lang=target_lang)
