from typing import Optional, List, Dict, Any, Tuple
from ..utils.llm_handler import LLMHandler
import pandas as pd
import json
from ..utils.qa_handler import QAHandler

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
        
    def detect_missing_translations(self, df: pd.DataFrame, 
                                  source_lang: str = 'en',
                                  target_langs: List[str] = None) -> Dict[str, List[int]]:
        """Detect missing translations in the dataframe."""
        missing_translations = {}
        
        if target_langs is None:
            # Get all language columns except source
            target_langs = [col for col in df.columns if col != source_lang]
            
        for lang in target_langs:
            if lang in df.columns:
                # Find rows where target language is empty but source has content
                missing_mask = df[lang].isna() & df[source_lang].notna()
                missing_translations[lang] = df[missing_mask].index.tolist()
                
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

    async def translate_text(self, source_text: str, source_lang: str, target_lang: str, context: str = "", term_base: Dict[str, str] = None) -> Tuple[str, List[str]]:
        """Translate text and validate the translation."""
        # Create translation prompt with context
        prompt = self.create_translation_prompt(source_text, context, term_base or {}, target_lang)
        
        # Use generate_completion instead of generate_text
        response = await self.llm_handler.generate_completion([
            {"role": "system", "content": f"You are a professional translator for {target_lang}."},
            {"role": "user", "content": prompt}
        ])
        
        translated_text = self.llm_handler.extract_completion_text(response)
        
        # Validate the translation
        issues = await self.qa_handler.validate_translation(
            source_text=source_text,
            translated_text=translated_text,
            term_base=term_base,
            skip_llm_on_issues=False
        )
        
        return translated_text, issues

    async def batch_translate(self, texts: List[str], target_lang: str,
                            contexts: List[str] = None,
                            term_base: Dict[str, str] = None) -> List[str]:
        """Translate a batch of texts."""
        if contexts is None:
            contexts = [""] * len(texts)
        
        if len(texts) != len(contexts):
            raise ValueError("Number of texts and contexts must match")
            
        translations = []
        for text, context in zip(texts, contexts):
            # Updated to match new translate_text signature
            translation, _ = await self.translate_text(
                source_text=text,
                source_lang=self.config['languages']['source'],
                target_lang=target_lang,
                context=context,
                term_base=term_base
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
- Don't include terms that are part of markup [xxx] or {{xxx}}

Return the response in this exact format:
{{"terms": {{"term1": {{"comment": "context for term1"}}, "term2": {{"comment": "context for term2"}}, ...}}}}

If no terms found, return {{"terms": {{}}}}"""

        try:
            response = await self.llm_handler.generate_completion([
                {"role": "system", "content": "You are a terminology extraction expert."},
                {"role": "user", "content": prompt}
            ])
            
            extracted = json.loads(self.llm_handler.extract_completion_text(response))
            return extracted.get('terms', {})
            
        except Exception as e:
            print(f"Term extraction failed: {e}")
            return {}

    async def process_translations(self, translations: List[Dict]) -> List[Dict]:
        """Process a batch of translations with validation."""
        results = []
        for item in translations:
            translated_text, issues = await self.translate_text(
                item['source_text'],
                item['source_lang'],
                item['target_lang']
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