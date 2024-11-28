from typing import Optional, List, Dict, Any, Tuple
from ..utils.llm_handler import LLMHandler
import pandas as pd
import json
from ..utils.qa_handler import QAHandler
import asyncio
import logging

logger = logging.getLogger(__name__)

class TranslationManager:
    def __init__(self, config: Dict):
        self.config = config
        llm_config = config.get('llm', {})
        
        # Initialize LLM Handler with correct parameters from config
        self.llm_handler = LLMHandler(
            api_key=llm_config.get('api_key'),
            base_url=llm_config.get('api_url', "https://api.openai.com/v1"),
            model=llm_config.get('model', 'gpt-4'),
            temperature=llm_config.get('temperature', 0.3)
        )
        
        # Initialize QA Handler with LLM Handler
        self.qa_handler = QAHandler(
            max_length=config.get('max_length'),
            llm_handler=self.llm_handler
        )
        
        # Get batch configuration
        self.batch_size = llm_config.get('batch_size', 50)
        self.batch_delay = llm_config.get('batch_delay', 1)
        self.max_retries = llm_config.get('max_retries', 3)
        self.retry_delay = llm_config.get('retry_delay', 1)  # seconds
        
    def detect_missing_translations(self, df: pd.DataFrame, 
                                  source_lang: str = 'en',
                                  target_langs: List[str] = None) -> Dict[str, List[int]]:
        """Detect missing translations in the dataframe."""
        missing_translations = {}
        
        if target_langs is None:
            # Get all language columns except source
            target_langs = [col for col in df.columns if col != source_lang]
        
        # Ensure source_lang exists
        if source_lang not in df.columns:
            raise ValueError(f"Source language column '{source_lang}' not found in dataframe")
        
        for lang in target_langs:
            if lang in df.columns:
                # Find rows where target language is empty but source has content
                missing_mask = df[lang].isna() & df[source_lang].notna()
                missing_translations[lang] = df[missing_mask].index.tolist()
            else:
                # If column doesn't exist, all rows with source content need translation
                source_content_mask = df[source_lang].notna()
                missing_translations[lang] = df[source_content_mask].index.tolist()
                # Add the missing column to the dataframe
                df[lang] = pd.NA
                
        return missing_translations

    def create_translation_prompt(self, text: str, context: str, 
                                term_base: Dict[str, str],
                                target_lang: str) -> str:
        """Create a context-aware translation prompt."""
        prompt = f"""You are a world-class expert in translating to {target_lang}, 
specialized for casual mobile games. Translate the following text professionally:

Text: {text}

Context: {context}

Term Base References:
{json.dumps(term_base, indent=2)}

Rules:
- Use provided term base for consistency
- Don't translate text between markup characters [] and {{}}
- Keep appropriate format (uppercase/lowercase)
- Replace newlines with \\n
- Keep translations lighthearted and fun
- Keep translations concise to fit UI elements
- Localize all output text

Translate the text maintaining all rules."""
        return prompt

    async def translate_text(self, source_texts: List[str], source_lang: str, target_lang: str, 
                           contexts: List[str] = None, term_base: Dict[str, str] = None) -> List[Tuple[str, List[str]]]:
        """Translate a batch of texts and validate the translations.
        
        Args:
            source_texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: Optional list of context strings for each text
            term_base: Optional term base dictionary
            
        Returns:
            List of tuples containing (translated_text, issues) for each input text
        """
        if contexts is None:
            contexts = [""] * len(source_texts)
            
        if len(contexts) != len(source_texts):
            raise ValueError("Number of contexts must match number of source texts")
            
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                results = await self._perform_translation(
                    source_texts, source_lang, target_lang, contexts, term_base
                )
                return results
            except Exception as e:
                last_error = e
                retries += 1
                if retries == self.max_retries:
                    logger.error(f"Failed to translate after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Translation attempt {retries} failed, retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff

    async def _perform_translation(self, source_texts: List[str], source_lang: str, 
                                 target_lang: str, contexts: List[str],
                                 term_base: Dict[str, str] = None) -> List[Tuple[str, List[str]]]:
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
        # Create a combined prompt for all texts
        texts_with_contexts = []
        for i, (text, context) in enumerate(zip(source_texts, contexts)):
            texts_with_contexts.append(f"Text {i+1}: {text}\nContext {i+1}: {context}")
        
        combined_texts = "\n\n".join(texts_with_contexts)
        
        prompt = f"""You are a world-class expert in translating to {target_lang}, 
specialized for casual mobile games. Translate the following texts professionally:

{combined_texts}

Term Base References:
{json.dumps(term_base, indent=2)}

Rules:
- Use provided term base for consistency
- Don't translate text between markup characters [] and {{}}
- Keep appropriate format (uppercase/lowercase)
- Replace newlines with \\n
- Keep translations lighthearted and fun
- Keep translations concise to fit UI elements
- Localize all output text

Translate each text maintaining all rules. Return translations in a structured format."""

        translation_schema = {
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
                }
            },
            "required": ["translations"]
        }

        response = await self.llm_handler.generate_completion(
            messages=[
                {"role": "system", "content": f"You are a professional translator for {target_lang}."},
                {"role": "user", "content": prompt}
            ],
            json_schema=translation_schema
        )
        
        result = self.llm_handler.extract_structured_response(response)
        translations = result["translations"]
        
        # Validate all translations in parallel
        validation_tasks = []
        for i, source_text in enumerate(source_texts):
            translated_text = translations[i]["translation"]
            task = self.qa_handler.validate_translation(
                source_text=source_text,
                translated_text=translated_text,
                term_base=term_base,
                skip_llm_on_issues=False
            )
            validation_tasks.append(task)
        
        validation_results = await asyncio.gather(*validation_tasks)
        
        # Combine translations with their validation results
        results = []
        for i, (translation_result, validation_issues) in enumerate(zip(translations, validation_results)):
            translated_text = translation_result["translation"]
            translator_notes = translation_result.get("notes", [])
            results.append((translated_text, translator_notes + validation_issues))
        
        return results

    async def batch_translate(self, texts: List[str], target_lang: str,
                            contexts: List[str] = None,
                            term_base: Dict[str, str] = None,
                            df: Optional[pd.DataFrame] = None,
                            row_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Translate a batch of texts with configurable batch size and delay.
        
        Args:
            texts: List of texts to translate
            target_lang: Target language code
            contexts: Optional list of context strings for each text
            term_base: Optional term base dictionary
            df: Optional DataFrame to update with translations
            row_indices: Optional list of row indices for DataFrame updates
            
        Yields:
            List of dictionaries containing translations and metadata for each batch
        """
        if contexts is None:
            contexts = [""] * len(texts)
        
        if len(texts) != len(contexts):
            raise ValueError("Number of texts and contexts must match")
        
        # If DataFrame is provided, ensure the target language column exists
        if df is not None:
            if target_lang not in df.columns:
                df[target_lang] = pd.NA
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_contexts = contexts[i:i + self.batch_size]
            batch_indices = row_indices[i:i + self.batch_size] if row_indices else None
            
            logger.info(f"Processing batch {i//self.batch_size + 1} of {(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            # Process the entire batch at once
            batch_translations = await self.translate_text(
                source_texts=batch_texts,
                source_lang=self.config['languages']['source'],
                target_lang=target_lang,
                contexts=batch_contexts,
                term_base=term_base
            )
            
            # Process results and update DataFrame
            batch_results = []
            for j, (text, context, (translation, issues)) in enumerate(zip(batch_texts, batch_contexts, batch_translations)):
                # Update DataFrame with just the translated text if row index is provided
                if df is not None and batch_indices is not None:
                    df.at[batch_indices[j], target_lang] = translation
                
                batch_results.append({
                    'source_text': text,
                    'translated_text': translation,
                    'context': context,
                    'issues': issues,
                    'target_lang': target_lang,
                    'needs_update': True,  # Signal that this batch needs to be written to sheets
                    'batch_number': i//self.batch_size + 1
                })
            
            # Signal that this batch is complete and should be written to sheets
            yield batch_results
            
            # Apply batch delay if this is not the last batch
            if i + self.batch_size < len(texts):
                logger.debug(f"Waiting {self.batch_delay} seconds before processing next batch...")
                await asyncio.sleep(self.batch_delay)

    async def extract_terms(self, text: str, context: str, target_lang: str) -> Dict[str, Dict[str, str]]:
        """Extract potential terms from translated text."""
        prompt = f"""Analyze this text and its translation for any important terms that should be added to the term base.
Only extract terms that are specific to games or require consistent translation.

Original Text: {text}
Context: {context}

Rules for term extraction:
- Only extract terms that need consistent translation
- Focus on game-specific terminology
- Include proper nouns that need consistent translation
- Don't include common words or phrases
- Don't include terms that are part of markup [xxx] or {{xxx}}"""

        terms_schema = {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "object",
                    "patternProperties": {
                        "^.*$": {
                            "type": "object",
                            "properties": {
                                "comment": {"type": "string"}
                            },
                            "required": ["comment"]
                        }
                    }
                }
            },
            "required": ["terms"]
        }

        try:
            response = await self.llm_handler.generate_completion(
                messages=[
                    {"role": "system", "content": "You are a terminology extraction expert."},
                    {"role": "user", "content": prompt}
                ],
                json_schema=terms_schema
            )
            
            result = self.llm_handler.extract_structured_response(response)
            return result["terms"]
            
        except Exception as e:
            print(f"Term extraction failed: {e}")
            return {}

    async def process_translations(self, translations: List[Dict], df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Process a batch of translations with validation."""
        results = []
        
        # Group translations by target language for efficient batch processing
        by_language = {}
        for item in translations:
            lang = item['target_lang']
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(item)
        
        # Process each language group
        for lang, items in by_language.items():
            texts = []
            contexts = []
            row_indices = []
            
            # Extract texts, contexts, and row indices
            for item in items:
                texts.append(item['source_text'])
                contexts.append(item.get('context', ''))
                if 'row_idx' in item:
                    row_indices.append(item['row_idx'])
            
            # Get term base for this language if available
            term_base = {}  # You might want to implement term base lookup here
            
            # Process the batch
            batch_results = await self.batch_translate(
                texts=texts,
                target_lang=lang,
                contexts=contexts,
                term_base=term_base,
                df=df,
                row_indices=row_indices if row_indices else None
            )
            
            results.extend(batch_results)
        
        return results