import re
from typing import Dict, List, Tuple, Optional

class QAHandler:
    def __init__(self, max_length: Optional[int] = None):
        """Initialize QA Handler.
        
        Args:
            max_length: Maximum allowed length for translations (if None, no limit)
        """
        self.max_length = max_length
        
    def validate_translation(self, 
                           source_text: str, 
                           translated_text: str,
                           term_base: Dict[str, str] = None) -> List[str]:
        """Validate a translation and return list of issues found."""
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