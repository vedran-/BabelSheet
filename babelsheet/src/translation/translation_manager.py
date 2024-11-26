from typing import Optional, List, Dict, Any
from ..utils.llm_handler import LLMHandler
import pandas as pd
import json
from ..utils.qa_handler import QAHandler

class TranslationManager:
    def __init__(self, 
                 api_key: str, 
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4", 
                 temperature: float = 0.3,
                 max_length: Optional[int] = None):
        """Initialize the Translation Manager."""
        # Initialize LLM handler
        self.llm_handler = LLMHandler(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature
        )
            
        # Initialize QA handler
        self.qa_handler = QAHandler(max_length=max_length)
        
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

    async def translate_text(self, text: str, target_lang: str, 
                           context: str = "", term_base: Dict[str, str] = None,
                           term_base_handler = None) -> str:
        """Translate text and validate the translation."""
        translation = await super().translate_text(text, target_lang, context, term_base)
        
        # Validate translation
        issues = self.qa_handler.validate_translation(text, translation, term_base)
        
        if issues:
            # Log issues
            print(f"QA issues found in translation:")
            for issue in issues:
                print(f"- {issue}")
            
            # Try one more time with issues in the prompt
            prompt = self.create_translation_prompt(text, context, term_base, target_lang)
            prompt += "\n\nPrevious translation had these issues:\n"
            prompt += "\n".join(f"- {issue}" for issue in issues)
            prompt += "\nPlease provide a corrected translation."
            
            response = await self.llm_handler.generate_completion([
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ])
            
            translation = self.llm_handler.extract_completion_text(response)
            
            # Validate again
            issues = self.qa_handler.validate_translation(text, translation, term_base)
            if issues:
                print("Warning: Translation still has QA issues:")
                for issue in issues:
                    print(f"- {issue}")
        
        return translation

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
            translation = await self.translate_text(text, target_lang, context, term_base)
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