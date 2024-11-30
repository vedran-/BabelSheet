from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator
from ..utils.llm_handler import LLMHandler
import pandas as pd
import json
from ..utils.qa_handler import QAHandler
from ..term_base.term_base_handler import TermBaseHandler
from ..sheets.sheets_handler import SheetsHandler, CellData
import asyncio
import logging

logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, config: Dict, sheets_handler: SheetsHandler, term_base_handler: TermBaseHandler):
        """Initialize Translation Manager.
        
        Args:
            config: Configuration dictionary containing all necessary settings
            sheets_handler: Pre-initialized SheetsHandler
            term_base_handler: Pre-initialized TermBaseHandler
        """
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
        missing_translations = self.detect_missing_translations(df, source_lang, target_langs)
        if len(missing_translations) > 0:
            logger.info(f"[{sheet_name}] Found missing translations for {len(missing_translations)} languages: {list(missing_translations.keys())}")

        if missing_translations and len(missing_translations) > 0:
            # Process all translations and update the DataFrame
            await self._process_missing_translations(missing_translations, use_term_base)
            
            logger.debug(f"Updated sheet '{sheet_name}' with new translations")
        else:
            logger.debug(f"No missing translations found for sheet '{sheet_name}'")

    def detect_missing_translations(self, df: pd.DataFrame, source_lang: str, target_langs: List[str]) -> Dict[str, List[int]]:
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

    async def _process_missing_translations(self, missing_translations: Dict[str, List[Dict[str, Any]]],
                                            use_term_base: bool) -> None:
        # First ensure term base is complete for all target languages
        #if use_term_base:
        #    await self.ensure_term_base_translations(target_langs)
        
        # Process each language group
        for lang, missing_items in missing_translations.items():
            texts = []
            contexts = []
            cells = []
            
            # Extract texts, contexts, and row indices
            for item in missing_items:
                texts.append(item['source_text'])
                contexts.append(item['context'])
                cells.append(item['row_idx'])
            
            # Get term base for this language
            term_base = self.term_base_handler.get_terms_for_language(lang) if use_term_base else None
            
            # Process the batch
            await self._batch_translate(
                texts=texts,
                target_lang=lang,
                contexts=contexts,
                cells=cells,
                term_base=term_base
            )

    async def _batch_translate(self, texts: List[str], target_lang: str,
                            contexts: List[str],
                            cells: List[Any],
                            term_base: Dict[str, Dict[str, Any]] = None
                            ):
        """Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            target_lang: Target language code
            contexts: List of context strings for each text
            cells: List of cell indices for DataFrame updates
            term_base: Term base dictionary
            
        Yields:
            List of dictionaries containing translations and metadata for each batch
        """
        if len(texts) != len(contexts) or len(texts) != len(cells):
            raise ValueError("Number of texts, contexts and cells must match")
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_contexts = contexts[i:i + self.batch_size]
            batch_cells = cells[i:i + self.batch_size]
            
            logger.info(f"Processing batch {i//self.batch_size + 1} of {(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            await self._perform_translation_with_retry(batch_texts, batch_contexts, batch_cells, target_lang, term_base)
            
            # Apply batch delay if this is not the last batch
            if i + self.batch_size < len(texts):
                logger.debug(f"Waiting {self.batch_delay} seconds before processing next batch...")
                await asyncio.sleep(self.batch_delay)

    async def _perform_translation_with_retry(self, batch_texts: List[str], batch_contexts: List[str], 
                                              batch_cells: List[Any], target_lang: str,
                                              term_base: Dict[str, Dict[str, Any]] = None):
        """Perform translation for a batch of texts with retry logic."""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Process the entire batch at once
                batch_translations = await self._perform_translation(
                    source_texts=batch_texts,
                    source_lang=self.config['languages']['source'],
                    target_lang=target_lang,
                    contexts=batch_contexts,
                    cells=batch_cells,
                    term_base=term_base
                )

                # Process each translation in the batch
                batch_results = []
                for j, (text, context, (translation, issues)) in enumerate(zip(batch_texts, batch_contexts, batch_translations)):
                    result = {
                        'source_text': text,
                        'translated_text': translation,
                        'context': context,
                        'issues': issues,
                        'batch_number': j//self.batch_size + 1
                    }
            
                    batch_results.append(result)

            except (
                TimeoutError,  # Network timeouts
                ConnectionError,  # Connection issues
                json.JSONDecodeError,  # Response parsing errors
                asyncio.TimeoutError,  # Async timeouts
                #ValueError,
            ) as e:
                last_error = e
                retries += 1
                if retries == self.max_retries:
                    raise
                logger.warning(f"Translation attempt {retries} failed, retrying in {self.retry_delay} seconds. Error: {e.__class__.__name__}: {str(e)}")
                logger.warning(f"Translation attempt {retries} failed, retrying in {self.retry_delay} seconds. Error: {e.__class__.__name__}: {str(e)} in {e.__traceback__.tb_frame.f_code.co_filename.split('/')[-1]}:{e.__traceback__.tb_lineno}")
                await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff

    def _prepare_texts_with_contexts(self, source_texts: List[str], 
                                     contexts: List[Dict[str, Any]], 
                                     term_base: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """Prepare texts with their contexts for translation.
        
        Args:
            source_texts: List of texts to translate
            contexts: List of context strings for each text
            term_base: Optional term base dictionary
            
        Returns:
            Combined text string with contexts and term base
        """
        texts_with_contexts = []
        for i, (text, context) in enumerate(zip(source_texts, contexts)):
            exc = []
            for key, item in context.items():
                escaped_key = key.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                escaped_item = str(item).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                exc.append(f"<{escaped_key}>{escaped_item}</{escaped_key}>")
            expanded_context = "".join(exc)
            texts_with_contexts.append(f"<text id='{i+1}'>{text}</text>\n<context id='{i+1}'>{expanded_context}</context>")
        
        combined_texts = "\n\n".join(texts_with_contexts)

        if term_base:
            term_base_xml = ["<term_base>"]
            for term, data in term_base.items():
                escaped_term = term.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                escaped_translation = data["translation"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                escaped_context = data["context"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                term_base_xml.append(f"<term><source>{escaped_term}</source><translation>{escaped_translation}</translation><context>{escaped_context}</context></term>")
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
            self.logger.info(f"Added {len(term_suggestions)} new terms to the term base")
            self.sheets_handler.save_changes()

    async def _validate_translations(self, source_texts: List[str], translations: List[Dict[str, Any]], 
                                   term_base: Optional[Dict[str, Dict[str, Any]]], target_lang: str) -> List[Tuple[str, List[str]]]:
        """Validate translations and combine with translator notes.
        
        Args:
            source_texts: Original texts
            translations: Translation results
            term_base: Optional term base
            target_lang: Target language code
            
        Returns:
            List of tuples containing (translated_text, issues)
        """
        validation_tasks = []
        for i, source_text in enumerate(source_texts):
            translated_text = translations[i]["translation"]
            task = self.qa_handler.validate_translation(
                source_text=source_text,
                translated_text=translated_text,
                term_base=term_base,
                skip_llm_on_issues=False,
                target_lang=target_lang
            )
            validation_tasks.append(task)
        
        validation_results = await asyncio.gather(*validation_tasks)
        
        results = []
        for i, (translation_result, validation_issues) in enumerate(zip(translations, validation_results)):
            translated_text = translation_result["translation"]
            translator_notes = translation_result.get("notes", [])
            results.append((translated_text, translator_notes + validation_issues))
        
        return results

    async def _perform_translation(self, source_texts: List[str], source_lang: str, 
                                 target_lang: str, contexts: List[str],
                                 cells: List[Any],
                                 term_base: Dict[str, Dict[str, Any]] = None
                                 ) -> List[Tuple[str, List[str]]]:
        """Internal method to perform batch translation.
        
        Args:
            source_texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: List of context strings for each text
            term_base: Optional term base dictionary
            
        Returns:
            List of tuples containing (translated_text, issues) for each input text
        """
        # Prepare texts and create prompt
        combined_texts = self._prepare_texts_with_contexts(source_texts, contexts, term_base)
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
        result = self.llm_handler.extract_structured_response(response)
        translations = result["translations"]
        term_suggestions = result.get("term_suggestions", [])
        
        # Handle term suggestions
        await self._handle_term_suggestions(term_suggestions, target_lang)
        
        # Validate translations
        return await self._validate_translations(source_texts, translations, term_base, target_lang)
