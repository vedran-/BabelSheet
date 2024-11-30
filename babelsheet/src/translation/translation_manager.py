from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator
from ..utils.llm_handler import LLMHandler
import pandas as pd
import json
from ..utils.qa_handler import QAHandler
from ..term_base.term_base_handler import TermBaseHandler
from ..sheets.sheets_handler import SheetsHandler, CellData
from ..utils.ui_manager import create_ui_manager
import asyncio
import logging
from datetime import datetime
import os
import pathlib

logger = logging.getLogger(__name__)

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
        
        # Initialize UI Manager
        self.ui = create_ui_manager(config)
        
        # Initialize statistics
        self.stats = {
            'successful_translations': 0,
            'failed_translations': 0,
            'failed_items': []  # List to store details of failed translations
        }
        
        # Setup translation logging
        output_config = config.get('output', {})
        self.output_dir = pathlib.Path(output_config.get('dir', 'translation_logs'))
        self._setup_translation_logging()

    def _setup_translation_logging(self):
        """Setup the translation logging directory and files."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for both files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup failed translations file
        self.failed_translations_file = self.output_dir / f"failed_translations_{timestamp}.txt"
        self._initialize_translations_file(self.failed_translations_file, "Failed")
        
        # Setup successful translations file
        self.successful_translations_file = self.output_dir / f"successful_translations_{timestamp}.txt"
        self._initialize_translations_file(self.successful_translations_file, "Successful")
        
        logger.info(f"Failed translations will be logged to: {self.failed_translations_file}")
        logger.info(f"Successful translations will be logged to: {self.successful_translations_file}")

    def _initialize_translations_file(self, file_path: pathlib.Path, log_type: str):
        """Create and initialize a translations log file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{log_type} Translations Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def _log_translation_header(self, f):
        """Write common header information to the log file."""
        f.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")

    def _log_failed_translation(self, failed_item: Dict[str, Any]):
        """Log a failed translation to the output file."""
        try:
            with open(self.failed_translations_file, 'a', encoding='utf-8') as f:
                self._log_translation_header(f)
                f.write(f"Sheet: {failed_item['sheet_name']}\n")
                f.write(f"Language: {failed_item['lang']}\n")
                f.write(f"Source Text: {failed_item['source_text']}\n")
                f.write(f"Last Translation Attempt: {failed_item['last_translation']}\n")
                f.write("Issues:\n")
                for issue in failed_item['issues']:
                    f.write(f"  - {issue}\n")
                f.write("\n" + "=" * 80 + "\n\n")
        except Exception as e:
            logger.error(f"Failed to log failed translation: {e}")
            # Continue execution even if logging fails

    def _log_successful_translation(self, sheet_name: str, lang: str, source_text: str, translation: str):
        """Log a successful translation to the output file."""
        try:
            with open(self.successful_translations_file, 'a', encoding='utf-8') as f:
                self._log_translation_header(f)
                f.write(f"Sheet: {sheet_name}\n")
                f.write(f"Language: {lang}\n")
                f.write(f"Source Text: {source_text}\n")
                f.write(f"Translation: {translation}\n")
                f.write("\n" + "=" * 80 + "\n\n")
        except Exception as e:
            logger.error(f"Failed to log successful translation: {e}")
            # Continue execution even if logging fails

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
        
        #if len(missing_translations) > 0:
        #    logger.info(f"[{sheet_name}] Found missing translations for {len(missing_translations)} languages: " + 
        #               ", ".join(f"{lang} ({len(items)} items)" for lang, items in missing_translations.items()))

        if missing_translations and len(missing_translations) > 0:
            # Process all translations and update the DataFrame
            await self._process_missing_translations(df, missing_translations, use_term_base)
            
            logger.debug(f"Updated sheet '{sheet_name}' with new translations")
            
            # Display statistics after processing each sheet
            self._display_statistics(sheet_name)
        else:
            logger.debug(f"No missing translations found for sheet '{sheet_name}'")

    def _display_statistics(self, sheet_name: Optional[str] = None) -> None:
        """Display translation statistics.
        
        Args:
            sheet_name: Optional sheet name for context
        """
        total = self.stats['successful_translations'] + self.stats['failed_translations']
        if total == 0:
            return
            
        # Calculate success rate
        success_rate = (self.stats['successful_translations'] / total) * 100
        
        # Create header
        header = "Translation Statistics"
        if sheet_name:
            header += f" for {sheet_name}"
            
        # Display summary
        self.ui.info("=" * 80)
        self.ui.info(header)
        self.ui.info("-" * 80)
        self.ui.info(f"Total translations attempted: {total}")
        self.ui.info(f"Successful translations: {self.stats['successful_translations']} ({success_rate:.1f}%)")
        self.ui.info(f"Failed translations: {self.stats['failed_translations']} ({100-success_rate:.1f}%)")
        
        # Display failed items if any
        if self.stats['failed_items']:
            self.ui.info("\nFailed Translations:")
            self.ui.info("-" * 80)
            for item in self.stats['failed_items']:
                self.ui.info(f"Sheet: {item['sheet_name']}")
                self.ui.info(f"Language: {item['lang']}")
                self.ui.info(f"Source text: {item['source_text']}")
                self.ui.info(f"Last attempt: {item['last_translation']}")
                self.ui.info(f"Issues:")
                for issue in item['issues']:
                    self.ui.info(f"  - {issue}")
                self.ui.info("-" * 40)
        self.ui.info("=" * 80)

    def _detect_missing_translations(self, df: pd.DataFrame, source_lang: str, target_langs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
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
                source_cell = df.iloc[row_idx][source_lang_idx]
                # Skip if source text is empty
                if source_cell is None or source_cell.is_empty():
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

                    lang_missing.append({
                        'sheet_name': df.attrs['sheet_name'],
                        'row_idx': row_idx,
                        'col_idx': lang_idx,
                        'source_text': source_cell.value,
                        'target_cell': target_cell,
                        'context': context
                    })

            if len(lang_missing) > 0:
                if lang not in missing_translations:
                    missing_translations[lang] = []
                missing_translations[lang].extend(lang_missing)
        
        return missing_translations

    async def collect_all_missing_translations(self, source_lang: str, target_langs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Collect all missing translations from all sheets, organized by language.
        
        Args:
            source_lang: Source language code
            target_langs: List of target language codes
            
        Returns:
            Dictionary mapping language codes to lists of missing translations across all sheets
        """
        all_missing_translations = {}
        
        # Get all sheet names
        sheet_names = self.sheets_handler.get_sheet_names()
        total_sheets = len(sheet_names)
        
        self.ui.info(f"Collecting missing translations from {total_sheets} sheets...")
        
        # Process each sheet
        for i, sheet_name in enumerate(sheet_names, 1):
            # Skip term base sheet as it's handled separately
            if self.term_base_handler and sheet_name == self.term_base_handler.sheet_name:
                continue
                
            self.ui.info(f"Analyzing sheet {i}/{total_sheets}: {sheet_name}")
            
            # Get the sheet data
            df = self.sheets_handler.get_sheet_data(sheet_name)
            df.attrs['sheet_name'] = sheet_name  # Store sheet name in DataFrame attributes
            
            # Detect missing translations for this sheet
            sheet_missing = self._detect_missing_translations(df, source_lang, target_langs)
            
            # Report findings for this sheet
            if sheet_missing:
                sheet_total = sum(len(items) for items in sheet_missing.values())
                self.ui.info(f"Found {sheet_total} missing translations in {sheet_name}:")
                for lang, items in sheet_missing.items():
                    self.ui.info(f"  - {lang}: {len(items)} items")
            
            # Merge with all missing translations
            for lang, items in sheet_missing.items():
                if lang not in all_missing_translations:
                    all_missing_translations[lang] = []
                all_missing_translations[lang].extend(items)
        
        # Log final summary
        if all_missing_translations:
            total_missing = sum(len(items) for items in all_missing_translations.values())
            self.ui.info("\nCollection complete. Summary of all missing translations:")
            for lang, items in all_missing_translations.items():
                self.ui.info(f"  - {lang}: {len(items)} items")
            self.ui.info(f"Total missing translations: {total_missing}")
        else:
            self.ui.info("\nNo missing translations found in any sheet.")
        
        return all_missing_translations

    async def translate_all_sheets(self, source_lang: str, target_langs: List[str], use_term_base: bool = True) -> None:
        """Translate all missing translations across all sheets, organized by language.
        
        Args:
            source_lang: Source language code
            target_langs: List of target language codes
            use_term_base: Whether to use the term base for translations
        """
        
        # Now collect and process all other sheets by language
        all_missing_translations = await self.collect_all_missing_translations(source_lang, target_langs)
        
        if not all_missing_translations:
            logger.info("No missing translations found in any sheet")
            return
        
        # Process all missing translations
        await self._process_missing_translations(None, all_missing_translations, use_term_base=True)
        
        # Display final statistics
        self._display_statistics()
        
        # Save all changes
        self.sheets_handler.save_changes()

    async def _process_missing_translations(self, df: Optional[pd.DataFrame], missing_translations: Dict[str, List[Dict[str, Any]]], use_term_base: bool) -> None:
        """Process missing translations, either for a single sheet or across all sheets.
        
        Args:
            df: DataFrame containing translations (None if processing across all sheets)
            missing_translations: Dictionary mapping language codes to missing translations
            use_term_base: Whether to use the term base for translations
        """
        skipped_items = []
        
        total_items = sum(len(items) for items in missing_translations.values())
        processed_items = 0
        source_lang = self.config['languages']['source']
        
        self.ui.start()
        self.ui.info(f"Starting batch translation for {len(missing_translations)} languages ({total_items} items total)")

        last_lang = None
        async def check_language_change(lang: str) -> None:
            nonlocal last_lang
            if last_lang == lang:
                return
            
            last_lang = lang
            self.ui.info(f"Processing language {lang} ({len(missing_translations[lang])} items)")

            # Ensure term base translations are up to date if we have a term base
            if self.term_base_handler and use_term_base:
                logger.debug("Ensuring term base translations are up to date...")
                self.ui.debug("Processing term base sheet first...")
                await self.ensure_sheet_translations(
                    sheet_name=self.term_base_handler.sheet_name,
                    source_lang=source_lang,
                    target_langs=[lang],
                    use_term_base=False  # Don't use term base for translating itself
                )
                self.ui.debug("Term base processing complete.")

        try:
            while len(missing_translations) > 0:
                lang, missing_items = next(iter(missing_translations.items()))
                await check_language_change(lang)
                
                batch = missing_items[:self.batch_size]
                
                # Add pending translations to UI
                for item in batch:
                    #retry_count = len(item.get('last_issues', []))
                    #if retry_count > 0:
                    #    # Add exponential backoff delay for retries
                    #    delay = self.retry_delay * (2 ** (retry_count - 1))
                    #    self.ui.info(f"Waiting {delay}s before retry #{retry_count + 1} for '{item['source_text']}'")
                    #    await asyncio.sleep(delay)
                    self.ui.add_translation_entry(item['source_text'], lang)
                
                # Perform translation
                contexts = [item['context'] for item in batch]
                
                batch_translations = await self._perform_translation(
                    source_texts=[item['source_text'] for item in batch],
                    source_lang=source_lang,
                    target_lang=lang,
                    contexts=contexts,
                    issues=[item.get('last_issues', []) for item in batch],
                    use_term_base=use_term_base
                )
                
                for idx, (missing_item, (translation, issues)) in enumerate(zip(batch, batch_translations)):
                    processed_items += 1
                    sheet_name = missing_item['sheet_name']
                    
                    if issues:
                        self.ui.warning(f"Translation '{translation}' has issues for {lang}: {issues}")
                        all_issues = missing_item.get('last_issues', []) + [{
                            'translation': translation,
                            'issues': issues,
                        }]
                        missing_item['last_issues'] = all_issues
                        
                        if len(all_issues) >= self.max_retries:
                            self.ui.critical(f"Max retries reached for {lang} item {missing_item['source_text']}. Giving up.")
                            skipped_items.append(missing_item)
                            missing_items.remove(missing_item)
                            
                            # Track failed translation
                            self.stats['failed_translations'] += 1
                            failed_item = {
                                'sheet_name': sheet_name,
                                'lang': lang,
                                'source_text': missing_item['source_text'],
                                'last_translation': translation,
                                'issues': [str(issue['issues']) for issue in all_issues]
                            }
                            self.stats['failed_items'].append(failed_item)
                            self._log_failed_translation(failed_item)
                            
                            self.ui.complete_translation(missing_item['source_text'], lang, translation, str(issues))
                        else:
                            # Keep the item in the list for retry, but mark it as in progress
                            self.ui.complete_translation(missing_item['source_text'], lang, translation, str(issues))
                            self.ui.add_translation_entry(missing_item['source_text'], lang, "⏳")
                    else:
                        # Only handle as successful if there are no issues
                        self.sheets_handler.modify_cell_data(
                            sheet_name=sheet_name,
                            row=missing_item['row_idx'],
                            col=missing_item['col_idx'],
                            value=translation
                        )
                        missing_items.remove(missing_item)
                        self.ui.complete_translation(missing_item['source_text'], lang, translation)
                        
                        # Track successful translation and log it
                        self.stats['successful_translations'] += 1
                        self._log_successful_translation(sheet_name, lang, missing_item['source_text'], translation)
                
                if len(missing_items) == 0:
                    missing_translations.pop(lang)
                    self.ui.info(f"Completed all translations for {lang}")
                
                # Start new batch
                self.ui.start_new_batch()
                self.sheets_handler.save_changes()
            
            if skipped_items:
                self.ui.warning(f"Skipped {len(skipped_items)} items due to max retries")
        finally:
            self.ui.stop()

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
- Keep translations lighthearted and fun, but precise
- Keep translations concise to fit UI elements
- Localize all output text, except special terms between markup characters
- It is ok to be polarizing, don't be neutral - but avoid offensive language
- Review previous failed translations and avoid making the same mistakes

Additionally, identify any important unique terms, like character or item names, in the source text that should be added to the term base.
Term Base Rules:
- Only suggest game-specific terms or terms requiring consistent translation, like character or item names
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
            self.logger.info(f"[TERM_BASE] Added {len(term_suggestions)} new terms to the term base: {', '.join(term['source_term'] for term in term_suggestions)}")
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
            results.append((translated_text, syntax_issues))
        
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
