from typing import Dict, List, Any, Optional
from ..utils.qa_handler import QAHandler
from .translation_dictionary import TranslationDictionary

class TranslationPrompts:
    def __init__(self, config: Dict, qa_handler: QAHandler, translation_dictionary: TranslationDictionary):
        """Initialize Translation Prompts."""
        self.config = config
        self.qa_handler = qa_handler
        self.translation_dictionary = translation_dictionary
        self.use_override = config.get('qa', {}).get('use_override', False)

    def escape(self, text: str) -> str:
        """Escape special characters in text for XML-like format."""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def prepare_texts_with_contexts(self, source_texts: List[str], contexts: List[Dict[str, Any]], 
                                  issues: List[Dict[str, Any]],
                                  target_lang: str) -> str:
        """Prepare texts with their contexts for translation.
        
        Args:
            source_texts: List of texts to translate
            contexts: List of context dictionaries for each text
            issues: List of issues for each text
            term_base: Optional term base dictionary
            
        Returns:
            Combined texts with contexts as a string
        """
        texts_with_contexts = []
        
        for i, (text, context_dict, issue_list) in enumerate(zip(source_texts, contexts, issues)):
            # Extract non-translatable terms for this text
            text_terms = self.qa_handler.extract_non_translatable_terms(text)

            # Prepare context
            exc = []
            for key, value in context_dict.items():
                if value:  # Only add non-empty context
                    exc.append(f"{self.escape(key)}: {self.escape(str(value))}")
            
            relevant_translations = self.translation_dictionary.get_relevant_translations(text, target_lang)
            if len(relevant_translations) > 0:
                exc.append(f"  RELEVANT_TERM_BASE:\n    - {'\n   - '.join(f'{rt['term']}: {rt['translation']}' for rt in relevant_translations)}")
            
            # Add non-translatable terms to context if any exist
            if text_terms and len(text_terms) > 0:
                terms_str = ", ".join(f"`{term}`" for term in text_terms)
                exc.append(f"  NON_TRANSLATABLE_TERMS, which must be preserved exactly as is: {terms_str}")
            
            # Add issues
            for issue in issue_list:
                exc.append(f"  FAILED_TRANSLATION: `{self.escape(issue['translation'])}` failed because: {self.escape(issue['issues'])}")
            
            expanded_context = "\n".join(exc)
            texts_with_contexts.append(f"<text id='{i+1}'>{text}</text>\n<context id='{i+1}'>\n{expanded_context}\n</context>")
        
        return f"# Texts to Translate ({len(source_texts)} texts)\n\n" + "\n\n".join(texts_with_contexts)


    def create_translation_prompt(self, combined_texts: str, target_lang: str, source_lang: str, 
                                term_base: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """Create the translation prompt with all necessary instructions.
        
        Args:
            combined_texts: Prepared texts with contexts
            target_lang: Target language code
            source_lang: Source language code
            term_base: Optional term base dictionary
            
        Returns:
            Complete prompt string
        """
        prompt = f"""You are a world-class expert in translating to {target_lang}, 
specialized for casual mobile games. Your task is to provide accurate and culturally appropriate translations while learning from any previous translation attempts.

"""
        # Add term base at the beginning if available
        # OBSOLETE - now we use whole document as a term base, as shorter texts are translated first.
        if False and term_base and len(term_base) > 0:
            prompt += "Term Base References (use these for translation consistency):\n"
            for term, data in term_base.items():
                prompt += f"- {term}: {data['translation']} (Context: {data['context']})\n"
            prompt += "\n"

        override_instructions = ''
        if self.use_override:
            override_instructions = f"""5. Override validation issues - IMPORTANT:
   - By default, the 'override' field should be empty
   - For each text that has FAILED_TRANSLATION in its context, if you believe the previous translation was actually correct despite validation issues:
     * Provide that previously failed but now corrected translation again
     * In the 'override' field, explain in detail why the validation issues were false positives, in source language ({source_lang}). This will tell the app to use this translation despite previous issues.
   - Only fill in override when you are 100% certain the previous translation was completely correct and was rejected due to issues that can be safely ignored.
"""

        prompt += f"""
# Translation Rules:
- Use provided term base for consistency. For names, try to use the translation from the term base. For other terms and phrases, try to use your best judgement what would be the most appropriate translation of the source text, provided wider context.
- Preserve all non-translatable terms exactly as specified in each text's context. Those are special terms which match the following patterns: {str(self.config['qa']['non_translatable_patterns'])}
- Keep appropriate format (uppercase/lowercase)
- Replace newlines with \\n, and quotes with \\" or \\'
- Keep translations lighthearted and fun, but precise
- Keep translations concise to fit UI elements
- Localize all output text, except special terms between markup characters
- It is ok to be polarizing, don't be neutral - but avoid offensive language
- If translation spans multiple lines, try to keep the same line breaks as the source text, and also try to make each row equal in length if possible

# Critical Instructions for Previously Failed Translations:
When you see FAILED_TRANSLATION tags in the context:
1. These represent previous translation attempts that were rejected
2. Study each failed translation and its error message carefully
3. Identify specific issues that caused the rejection
4. Ensure your new translation:
   - Addresses all previous error points
   - Maintains the original meaning
   - Avoids similar mistakes
   - Improves upon the previous attempts
{override_instructions}

# Term Base Management:
Identify any important unique terms in the source text that should be added to the term base:
- Only suggest game-specific names requiring consistent translation
- Don't suggest common language words/phrases
- Don't suggest terms already in the term base
- Don't suggest special terms matching non-translatable patterns
For each suggested term, provide:
  * The term in the source language
  * A suggested translation
  * A brief comment explaining its usage/context in the source language ({source_lang})

  
{combined_texts}
  

Return translations and term suggestions in a structured JSON format."""
        return prompt

    def get_translation_schema(self) -> Dict[str, Any]:
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
                            "text_id": {
                                "type": "integer",
                                "description": "The ID of the text being translated (as shown in the prompt)"
                            },
                            "translation": {
                                "type": "string",
                                "description": "The translated text"
                            },
                            **({} if not self.use_override else {
                                "override": {
                                    "type": "string", 
                                    "description": "Optional reason for overriding validation issues, or empty string. Only provide this when 100% certain that the translation is correct despite validation issues."
                                }
                            })
                        },
                        "required": ["text_id", "translation"] + (["override"] if self.use_override else [])
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
                                "description": "Brief explanation of term usage/context"
                            }
                        },
                        "required": ["source_term", "suggested_translation", "comment"]
                    },
                    "description": "Suggested terms (names) to add to the term base"
                }
            },
            "required": ["translations"]
        }