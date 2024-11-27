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
        
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
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

    async def translate_text(self, source_text: str, source_lang: str, target_lang: str, 
                            context: str = "", term_base: Dict[str, str] = None,
                            df: Optional[pd.DataFrame] = None,
                            row_idx: Optional[int] = None) -> Tuple[str, List[str]]:
        """Translate text and validate the translation."""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                translated_text, issues = await self._perform_translation(
                    source_text, source_lang, target_lang, context, term_base
                )
                
                # Update DataFrame if provided
                if df is not None and row_idx is not None:
                    if target_lang not in df.columns:
                        df[target_lang] = pd.NA
                    df.at[row_idx, target_lang] = translated_text
                
                return translated_text, issues
            except Exception as e:
                last_error = e
                retries += 1
                if retries == self.max_retries:
                    logger.error(f"Failed to translate after {self.max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Translation attempt {retries} failed, retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff

    async def _perform_translation(self, source_text: str, source_lang: str, 
                                 target_lang: str, context: str,
                                 term_base: Dict[str, str] = None) -> Tuple[str, List[str]]:
        """Internal method to perform the actual translation."""
        prompt = self.create_translation_prompt(source_text, context, term_base or {}, target_lang)
        
        translation_schema = {
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

        response = await self.llm_handler.generate_completion(
            messages=[
                {"role": "system", "content": f"You are a professional translator for {target_lang}."},
                {"role": "user", "content": prompt}
            ],
            json_schema=translation_schema
        )
        
        result = self.llm_handler.extract_structured_response(response)
        translated_text = result["translation"]
        translator_notes = result.get("notes", [])
        
        # Validate the translation
        issues = await self.qa_handler.validate_translation(
            source_text=source_text,
            translated_text=translated_text,
            term_base=term_base,
            skip_llm_on_issues=False
        )
        
        return translated_text, translator_notes + issues

    async def batch_translate(self, texts: List[str], target_lang: str,
                            contexts: List[str] = None,
                            term_base: Dict[str, str] = None,
                            df: Optional[pd.DataFrame] = None) -> List[str]:
        """Translate a batch of texts."""
        if contexts is None:
            contexts = [""] * len(texts)
        
        if len(texts) != len(contexts):
            raise ValueError("Number of texts and contexts must match")
        
        # If DataFrame is provided, ensure the target language column exists
        if df is not None:
            if target_lang not in df.columns:
                df[target_lang] = pd.NA
                # Get indices of rows that need translation
                source_lang = self.config['languages']['source']
                indices_to_translate = df[df[source_lang].notna()].index.tolist()
                # Update texts and contexts lists to include all rows needing translation
                texts.extend([df.iloc[idx][source_lang] for idx in indices_to_translate])
                contexts.extend([""] * len(indices_to_translate))
        
        translations = []
        for text, context in zip(texts, contexts):
            translation, _ = await self.translate_text(
                source_text=text,
                source_lang=self.config['languages']['source'],
                target_lang=target_lang,
                context=context,
                term_base=term_base,
                df=df
            )
            translations.append(translation)
            
        return translations

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
        
        # First detect any missing translations if DataFrame is provided
        if df is not None:
            # Get all target languages from both existing translations and DataFrame columns
            target_langs = set()
            # Add languages from existing translations
            target_langs.update(item['target_lang'] for item in translations)
            # Add all non-source language columns from DataFrame
            source_lang = self.config['languages']['source']
            target_langs.update(col for col in df.columns if col != source_lang)
            
            missing_translations = self.detect_missing_translations(
                df=df,
                source_lang=source_lang,
                target_langs=list(target_langs)  # Convert set back to list
            )
            
            # Add any missing translations to the batch
            for target_lang, missing_indices in missing_translations.items():
                for idx in missing_indices:
                    if df.iloc[idx][source_lang]:  # If source text exists
                        translations.append({
                            'source_text': df.iloc[idx][source_lang],
                            'source_lang': source_lang,
                            'target_lang': target_lang,
                            'row_idx': idx  # Add row index for DataFrame updates
                        })
        
        # Process all translations
        for item in translations:
            translated_text, issues = await self.translate_text(
                item['source_text'],
                item['source_lang'],
                item['target_lang'],
                df=df,
                row_idx=item.get('row_idx')  # Pass row index if available
            )
            
            results.append({
                'source_text': item['source_text'],
                'translated_text': translated_text,
                'source_lang': item['source_lang'],
                'target_lang': item['target_lang'],
                'validation_issues': issues,
                'passed_validation': len(issues) == 0
            })
            
        return results