import re
from typing import Dict, List, Tuple, Optional, Pattern
from .llm_handler import LLMHandler
import logging

class QAHandler:
    def __init__(self, max_length: Optional[int] = None, llm_handler: Optional[LLMHandler] = None, non_translatable_patterns: Optional[List[Dict[str, str]]] = None):
        """Initialize QA Handler.
        
        Args:
            max_length: Maximum allowed length for translations (if None, no limit)
            llm_handler: LLMHandler instance for AI-powered validation
            non_translatable_patterns: List of pattern dicts with 'start' and 'end' keys
        """
        self.max_length = max_length
        self.llm_handler = llm_handler
        self.logger = logging.getLogger(__name__)
        
        # Validate and compile patterns if provided
        if non_translatable_patterns:
            try:
                self.patterns = self._compile_patterns(non_translatable_patterns)
                self.logger.debug(f"Compiled {len(self.patterns)} non-translatable patterns")
            except Exception as e:
                self.logger.warning(f"Failed to compile patterns: {e}")
                self.patterns = None
        else:
            self.patterns = None
            self.logger.debug("No non-translatable patterns configured")
        
    async def validate_translation(self, source_text: str, translated_text: str,
                                 term_base: Optional[Dict[str, str]] = None,
                                 skip_llm_on_issues: bool = False) -> List[str]:
        """Validate translation quality."""
        try:
            self.logger.debug(f"Validating translation:\nSource: {source_text}\nTranslated: {translated_text}")
            
            issues = []
            
            # Check for non-translatable terms only if patterns are configured
            if self.patterns:
                source_terms = self._extract_non_translatable_terms(source_text)
                for term in source_terms:
                    if term not in translated_text:
                        issues.append(f"Non-translatable term '{term}' is missing in translation")
            
            # Basic format validation
            format_issues = self._validate_format(source_text, translated_text)
            if format_issues:
                self.logger.warning(f"Found format issues: {format_issues}")
                issues.extend(format_issues)
            
            # Markup preservation check
            markup_issues = self._validate_markup(source_text, translated_text)
            if markup_issues:
                self.logger.warning(f"Found markup issues: {markup_issues}")
                issues.extend(markup_issues)
            
            # Character limit check
            if self.max_length and len(translated_text) > self.max_length:
                issue = f"Translation exceeds maximum length of {self.max_length} characters"
                self.logger.warning(issue)
                issues.append(issue)
            
            # Validate term base usage if provided
            if term_base:
                tb_issues = self._validate_term_base(translated_text, term_base)
                if tb_issues:
                    self.logger.warning(f"Found term base issues: {tb_issues}")
                    issues.extend(tb_issues)
            
            # If we found issues and skip_llm_on_issues is True, return early
            if issues and skip_llm_on_issues:
                self.logger.info("Skipping LLM validation due to existing issues")
                return issues
            
            # Perform LLM-based validation
            if self.llm_handler:  # Only if LLM handler is configured
                try:
                    llm_issues = await self._validate_with_llm(source_text, translated_text)
                    if llm_issues:
                        self.logger.warning(f"Found LLM validation issues: {llm_issues}")
                        issues.extend(llm_issues)
                except Exception as e:
                    self.logger.error(f"LLM validation failed: {str(e)}", exc_info=True)
                    issues.append(f"LLM validation error: {str(e)}")
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Translation validation failed: {str(e)}", exc_info=True)
            raise ValueError(f"Translation validation failed: {str(e)}")
    
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
        
        # First check non-translatable terms if patterns exist
        if self.patterns:
            source_terms = self._extract_non_translatable_terms(source)
            for term in source_terms:
                if term not in translation:
                    issues.append(f"Non-translatable term '{term}' must appear exactly as in source")
                elif translation.count(term) != source.count(term):
                    issues.append(f"Non-translatable term '{term}' appears {translation.count(term)} times in translation but {source.count(term)} times in source")
        
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
            f"4. Consistency in tone and style"
        )

        validation_schema = {
            "type": "object",
            "properties": {
                "is_valid": {
                    "type": "boolean",
                    "description": "Whether the translation is valid"
                },
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of identified issues"
                }
            },
            "required": ["is_valid", "issues"]
        }
        
        try:
            response = await self.llm_handler.generate_completion(
                messages=[
                    {"role": "system", "content": "You are a professional translation validator."},
                    {"role": "user", "content": prompt}
                ],
                json_schema=validation_schema
            )
            
            result = self.llm_handler.extract_structured_response(response)
            
            if result["is_valid"]:
                return []
            
            return [f"LLM found issue: {issue}" for issue in result["issues"]]
            
        except Exception as e:
            return [f"LLM validation failed: {str(e)}"]
    
    def _compile_patterns(self, patterns: List[Dict[str, str]]) -> List[Pattern]:
        """Compile regex patterns for non-translatable terms."""
        if not patterns:
            self.logger.warning("Empty patterns list provided")
            return []
            
        compiled_patterns = []
        for i, pattern in enumerate(patterns):
            try:
                if 'start' not in pattern or 'end' not in pattern:
                    raise ValueError(f"Pattern {i} missing 'start' or 'end' key")
                    
                start = pattern['start'].strip()
                end = pattern['end'].strip()
                
                if not start or not end:
                    raise ValueError(f"Pattern {i} has empty 'start' or 'end' value")
                
                # Escape special regex characters in start/end markers
                start = re.escape(start)
                end = re.escape(end)
                
                # Create pattern that matches the entire term including markers
                regex = f"({start}[^{end}]+?{end})"  # Capture the entire term
                compiled = re.compile(regex)
                
                # Test the pattern
                test_str = f"{pattern['start']}TEST{pattern['end']}"
                matches = compiled.findall(test_str)
                if not matches or matches[0] != test_str:  # Verify exact match
                    raise ValueError(f"Pattern {i} failed validation test")
                    
                compiled_patterns.append(compiled)
                self.logger.debug(f"Successfully compiled pattern: {regex}")
                
            except Exception as e:
                self.logger.error(f"Failed to compile pattern {i}: {str(e)}")
                continue
                
        return compiled_patterns
    
    def _extract_non_translatable_terms(self, text: str) -> List[str]:
        """Extract all non-translatable terms from text using configured patterns.
        
        Returns exact matches including the pattern characters. For example, if the pattern
        is defined as start='{[' and end=']}', and the text contains '{[NUMBER]}', the entire
        string '{[NUMBER]}' will be returned as a non-translatable term.
        """
        if not self.patterns:
            return []
            
        terms = []
        for pattern in self.patterns:
            matches = pattern.findall(text)
            # findall returns tuples if we have groups, so we need to get first element
            terms.extend(m[0] if isinstance(m, tuple) else m for m in matches)
        return terms