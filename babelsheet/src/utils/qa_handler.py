import re
from typing import Dict, List, Tuple, Optional
from .llm_handler import LLMHandler

class QAHandler:
    def __init__(self, max_length: Optional[int] = None, llm_handler: Optional[LLMHandler] = None):
        """Initialize QA Handler.
        
        Args:
            max_length: Maximum allowed length for translations (if None, no limit)
            llm_handler: LLMHandler instance for AI-powered validation
        """
        self.max_length = max_length
        self.llm_handler = llm_handler
        
    async def validate_translation(self, 
                                   source_text: str, 
                                   translated_text: str,
                                   term_base: Dict[str, str] = None,
                                   skip_llm_on_issues: bool = True) -> List[str]:
        """Validate a translation and return list of issues found.
        
        Args:
            source_text: Original text
            translated_text: Translated text to validate
            term_base: Optional terminology base for validation
            skip_llm_on_issues: If True, skips LLM validation when other issues are found
        """
        issues = []
        
        # Basic format validation
        format_issues = self._validate_format(source_text, translated_text)
        issues.extend(format_issues)
        
        # Markup preservation check
        markup_issues = self._validate_markup(source_text, translated_text)
        issues.extend(markup_issues)
        
        # Character limit check
        if self.max_length and len(translated_text) > self.max_length:
            issues.append(f"Translation exceeds maximum length of {self.max_length} characters")
            
        # Term base compliance check
        if term_base:
            term_issues = self._validate_term_base(translated_text, term_base)
            issues.extend(term_issues)
            
        # Only proceed with LLM validation if conditions are met
        if self.llm_handler and (not skip_llm_on_issues or not issues):
            llm_issues = await self._validate_with_llm(source_text, translated_text)
            issues.extend(llm_issues)
            
        return issues
    
    def _validate_format(self, source: str, translation: str) -> List[str]:
        """Check format consistency between source and translation."""
        issues = []
        
        # Check case consistency for all-caps words
        source_caps = re.findall(r'\b[A-Z]{2,}\b', source)
        for word in source_caps:
            if word in source and not any(word in t for t in re.findall(r'\b[A-Z]{2,}\b', translation)):
                issues.append(f"Capitalization mismatch for term: {word}")
        
        # Check newline preservation
        if source.count('\\n') != translation.count('\\n'):
            issues.append("Newline count mismatch between source and translation")
            
        # Check ending punctuation
        source_end = re.search(r'[.!?:,]$', source)
        trans_end = re.search(r'[.!?:,]$', translation)
        if bool(source_end) != bool(trans_end):
            issues.append("Ending punctuation mismatch")
            
        return issues
    
    def _validate_markup(self, source: str, translation: str) -> List[str]:
        """Check if markup is preserved correctly."""
        issues = []
        
        # Check square brackets
        source_brackets = re.findall(r'\[.*?\]', source)
        trans_brackets = re.findall(r'\[.*?\]', translation)
        if len(source_brackets) != len(trans_brackets):
            issues.append("Square bracket markup count mismatch")
        else:
            for s, t in zip(source_brackets, trans_brackets):
                if s != t:
                    issues.append(f"Square bracket content modified: {s} -> {t}")
        
        # Check curly braces
        source_braces = re.findall(r'\{.*?\}', source)
        trans_braces = re.findall(r'\{.*?\}', translation)
        if len(source_braces) != len(trans_braces):
            issues.append("Curly brace markup count mismatch")
        else:
            for s, t in zip(source_braces, trans_braces):
                if s != t:
                    issues.append(f"Curly brace content modified: {s} -> {t}")
                    
        return issues
    
    def _validate_term_base(self, translation: str, term_base: Dict[str, str]) -> List[str]:
        """Check if translation uses terms from term base correctly."""
        issues = []
        
        for source_term, expected_translation in term_base.items():
            # Skip empty translations in term base
            if not expected_translation:
                continue
                
            # Check if source term's translation is used consistently
            if source_term.lower() in translation.lower() and expected_translation.lower() not in translation.lower():
                issues.append(f"Term base mismatch: '{source_term}' should be translated as '{expected_translation}'")
                
        return issues 
    
    async def _validate_with_llm(self, source_text: str, translated_text: str) -> List[str]:
        """Use LLM to validate translation quality."""
        prompt = (
            f"You are a professional translation validator. Please review this translation:\n\n"
            f"Source text: {source_text}\n"
            f"Translated text: {translated_text}\n\n"
            f"Evaluate the following aspects:\n"
            f"1. Semantic accuracy (does it convey the same meaning?)\n"
            f"2. Cultural appropriateness\n"
            f"3. Natural flow and readability\n"
            f"4. Consistency in tone and style\n\n"
            f"Respond in the following format:\n"
            f"If the translation is perfect: 'TRANSLATION_OK'\n"
            f"If there are issues, list each issue on a new line starting with '- '\n"
            f"Be concise but specific in describing issues."
        )
        
        try:
            response = await self.llm_handler.generate_completion([
                {"role": "system", "content": "You are a professional translation validator."},
                {"role": "user", "content": prompt}
            ])
            
            completion_text = self.llm_handler.extract_completion_text(response)
            
            if not completion_text or completion_text.isspace():
                return ["LLM validation failed: Empty response"]
                
            if "TRANSLATION_OK" in completion_text:
                return []
                
            # Extract issues (lines starting with '- ')
            issues = [
                line[2:].strip() 
                for line in completion_text.split('\n') 
                if line.strip().startswith('- ')
            ]
            
            return [f"LLM found issue: {issue}" for issue in issues] if issues else ["LLM validation failed: Unexpected response format"]
            
        except Exception as e:
            return [f"LLM validation failed: {str(e)}"]