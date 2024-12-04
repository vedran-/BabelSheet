from typing import Dict, List, Any, Optional
import json

class TranslationPrompts:
    def __init__(self, config: Dict):
        """Initialize Translation Prompts."""
        self.config = config

    def escape(self, text: str) -> str:
        """Escape special characters in text for XML-like format."""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def prepare_texts_with_contexts(self, source_texts: List[str], contexts: List[Dict[str, Any]], 
                                  issues: List[Dict[str, Any]], qa_handler,
                                  term_base: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """Prepare texts with their contexts for translation.
        
        Args:
            source_texts: List of texts to translate
            contexts: List of context dictionaries for each text
            issues: List of issues for each text
            qa_handler: QA Handler instance for extracting non-translatable terms
            term_base: Optional term base dictionary
            
        Returns:
            Combined texts with contexts as a string
        """
        texts_with_contexts = []
        
        for i, (text, context_dict, issue_list) in enumerate(zip(source_texts, contexts, issues)):
            # Extract non-translatable terms for this text
            text_terms = qa_handler._extract_non_translatable_terms(text)
            
            # Prepare context
            exc = []
            for key, value in context_dict.items():
                if value:  # Only add non-empty context
                    exc.append(f"{self.escape(key)}: {self.escape(str(value))}")
            
            # Add non-translatable terms to context if any exist
            if text_terms and len(text_terms) > 0:
                terms_str = ", ".join(f"'{term}'" for term in text_terms)
                exc.append(f"<non_translatable_terms>The following terms must be preserved exactly as is: {terms_str}</non_translatable_terms>")
            
            # Add issues
            for issue in issue_list:
                exc.append(f"FAILED_TRANSLATION: '{self.escape(issue['translation'])}' failed because: {self.escape(issue['issues'])}")
            
            expanded_context = "\n".join(exc)
            texts_with_contexts.append(f"<text id='{i+1}'>{text}</text>\n<context id='{i+1}'>{expanded_context}</context>")
        
        return "\n\n".join(texts_with_contexts)

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
        if term_base and len(term_base) > 0:
            prompt += "Term Base References (use these for translation consistency):\n"
            for term, data in term_base.items():
                prompt += f"- {term}: {data['translation']} (Context: {data['context']})\n"
            prompt += "\n"

        prompt += f"""Texts to Translate:
{combined_texts}

Translation Rules:
- Use provided term base for consistency
- Preserve all non-translatable terms exactly as specified in each text's context
- Don't translate special terms which match the following patterns: {str(self.config['qa']['non_translatable_patterns'])}
- Keep appropriate format (uppercase/lowercase)
- Replace newlines with \\n, and quotes with \\" or \\'
- Keep translations lighthearted and fun, but precise
- Keep translations concise to fit UI elements
- Localize all output text, except special terms between markup characters
- It is ok to be polarizing, don't be neutral - but avoid offensive language

Critical Instructions for Previously Failed Translations:
When you see FAILED_TRANSLATION tags in the context:
1. These represent previous translation attempts that were rejected
2. Study each failed translation and its error message carefully
3. Identify specific issues that caused the rejection
4. Ensure your new translation:
   - Addresses all previous error points
   - Maintains the original meaning
   - Avoids similar mistakes
   - Improves upon the previous attempts
5. Double-check your translation against all identified issues before submitting

Term Base Management:
Identify any important unique terms in the source text that should be added to the term base:
- Only suggest game-specific names requiring consistent translation
- Don't suggest common language words/phrases
- Don't suggest terms already in the term base
- Don't suggest special terms matching non-translatable patterns
For each suggested term, provide:
  * The term in the source language
  * A suggested translation
  * A brief comment explaining its usage/context in the source language ({source_lang})

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
                            }
                        },
                        "required": ["text_id", "translation"]
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