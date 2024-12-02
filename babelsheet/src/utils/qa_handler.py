import re
from typing import Dict, List, Tuple, Optional, Pattern, Any
from .llm_handler import LLMHandler
import logging
import json

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
        
    async def validate_translation_syntax(self, source_text: str, translated_text: str,
                                 context: str, term_base: Optional[Dict[str, Dict[str, Any]]] = None,
                                 target_lang: str = None) -> List[str]:
        """Validate translation quality.
        
        Args:
            source_text: Original text
            translated_text: Translated text to validate
            context: Context of the translation
            term_base: Optional term base dictionary
            target_lang: Target language code for term base validation
        """
        issues = []
        
        try:
            self.logger.debug(f"Validating translation: '{source_text}' => '{translated_text}' ({target_lang})")
            
            # Check for non-translatable terms only if patterns are configured
            if self.patterns:
                source_terms = self._extract_non_translatable_terms(source_text)
                for term in source_terms:
                    if term not in translated_text:
                        issues.append(f"Non-translatable term '{term}' is missing in translation")
            
            # Basic format validation
            format_issues = self._validate_format(source_text, translated_text)
            if format_issues:
                self.logger.debug(f"Found format issues: {format_issues}")
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
            if term_base and target_lang:
                self.logger.debug(f"Starting term base validation for language: {target_lang}")
                tb_issues = self._validate_term_base(translated_text, term_base, target_lang)
                if tb_issues:
                    self.logger.warning(f"Found term base issues: {tb_issues}")
                    issues.extend(tb_issues)
            elif term_base:
                self.logger.warning("Term base provided but target language missing - skipping term base validation")
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Translation validation failed: {str(e)}", exc_info=True)
            raise ValueError(f"Translation validation failed: {str(e)}")
    
    def _validate_format(self, source: str, translation: str) -> List[str]:
        """Check format consistency between source and translation."""
        issues = []

        def get_caps_flags(text: str) -> tuple[bool, bool]:
            """
            Returns two flags for a text:
            1. has_all_caps_word: True if text contains any word in ALL CAPS
            2. is_all_words_caps: True if all words are in ALL CAPS
            """
            def clean_word(word: str) -> str:
                """Extract only letter characters from a word."""
                return ''.join(c for c in word if c.isalpha())
            
            # Normalize newlines first
            text = text.replace('\\n', ' ')
            
            # Split by any whitespace
            words = [w for w in text.split() if w]
            
            # Extract only letter characters and keep words that have at least one letter
            valid_words = []
            for word in words:
                cleaned = clean_word(word)
                if cleaned:  # Include if it has at least one letter
                    valid_words.append(cleaned)
            
            if not valid_words:
                return False, False
                
            # Check if any word is all caps
            has_all_caps = any(w.upper() == w for w in valid_words)
            
            # Check if all words are caps
            all_words_caps = all(w.upper() == w for w in valid_words)
            
            return has_all_caps, all_words_caps

        # Get capitalization flags for both texts
        source_has_caps, source_all_caps = get_caps_flags(source)
        trans_has_caps, trans_all_caps = get_caps_flags(translation)

        # Compare flags
        if source_has_caps != trans_has_caps:
            issues.append(
                f"Capitalization mismatch: source {'has' if source_has_caps else 'does not have'} "
                f"ALL CAPS words, but translation {'has' if trans_has_caps else 'does not have'} them. "
                f"Source: '{source}', Translation: '{translation}'"
            )
        elif source_all_caps != trans_all_caps:
            issues.append(
                f"Capitalization mismatch: {'all' if source_all_caps else 'not all'} words in source are "
                f"ALL CAPS, but {'all' if trans_all_caps else 'not all'} words in translation are. "
                f"Source: '{source}', Translation: '{translation}'"
            )
        
        def count_newlines(text: str) -> int:
            """Count newlines in text, handling both \\n and \n"""
            return text.count('\\n') + text.count('\n')
        
        # Check newline preservation using normalized counts, allowing 1 line difference
        source_newlines = count_newlines(source)
        trans_newlines = count_newlines(translation)
        if abs(source_newlines - trans_newlines) > 1:  # Allow difference of 1
            issues.append(f"Newline count mismatch between source ({source}) and translation ({translation})")
            
        # Check ending punctuation (after trimming)
        source_end = re.search(r'[.!?:,]$', source.strip())
        trans_end = re.search(r'[.!?:,]$', translation.strip())
        if bool(source_end) != bool(trans_end):
            issues.append(f"Ending punctuation does not match between source ({source.strip()}) and translation ({translation.strip()})")
            
        return issues
    
    def _validate_markup(self, source: str, translation: str) -> List[str]:
        """Check if markup is preserved correctly."""
        issues = []
        
        # First check non-translatable terms if patterns exist
        if self.patterns:
            source_terms = self._extract_non_translatable_terms(source)
            for term in source_terms:
                if term not in translation:
                    issues.append(f"Non-translatable term '{term}' must appear exactly as in source ({source})")
                elif translation.count(term) != source.count(term):
                    issues.append(f"Non-translatable term '{term}' appears {translation.count(term)} times in translation ({translation}) but {source.count(term)} times in source ({source})")
        
        # Check square brackets
        source_brackets = re.findall(r'\[.*?\]', source)
        trans_brackets = re.findall(r'\[.*?\]', translation)
        if len(source_brackets) != len(trans_brackets):
            issues.append(f"Square bracket markup count mismatch between source ({source}) and translation ({translation})")
        else:
            for s, t in zip(source_brackets, trans_brackets):
                if s != t:
                    issues.append(f"Square bracket content modified: {s} -> {t} between source ({source}) and translation ({translation})")
        
        # Check curly braces
        source_braces = re.findall(r'\{.*?\}', source)
        trans_braces = re.findall(r'\{.*?\}', translation)
        if len(source_braces) != len(trans_braces):
            issues.append(f"Curly brace markup count mismatch between source ({source}) and translation ({translation})")
        else:
            for s, t in zip(source_braces, trans_braces):
                if s != t:
                    issues.append(f"Curly brace content modified: {s} -> {t} between source ({source}) and translation ({translation})")
                    
        return issues
    
    def _validate_term_base(self, translation: str, term_base: Dict[str, Dict[str, Any]], target_lang: str) -> List[str]:
        """Check if translation uses terms from term base correctly.
        
        Args:
            translation: The translated text to validate
            term_base: Term base dictionary with structure {term: {'translations': Dict[str, str], 'comment': str}}
            target_lang: Target language code
        """
        issues = []
        self.logger.debug(f"Validating translation against term base for language: {target_lang}")
        
        for source_term, term_data in term_base.items():
            expected_translation = term_data.get('translation', '')
            comment = term_data.get('comment', '')
            
            if not expected_translation:
                self.logger.warning(f"Skipping term '{source_term}' - no translation for {target_lang}")
                continue
            
            # Check if source term appears in translation
            source_term_lower = source_term.lower()
            translation_lower = expected_translation.lower()
            
            # Use word boundary check to avoid partial matches
            if f" {source_term_lower} " in f" {translation_lower} ":
                expected_lower = expected_translation.lower()
                
                # Check if expected translation is used
                if expected_lower not in translation_lower:
                    issue = f"Term base mismatch: '{source_term}' should be translated as '{expected_translation}'"
                    if comment:
                        issue += f" (Note: {comment})"
                    self.logger.warning(f"Found term base issue: {issue}")
                    issues.append(issue)
                else:
                    self.logger.debug(f"Term '{source_term}' correctly translated as '{expected_translation}'")
                
        if not issues:
            self.logger.debug("No term base issues found")
        return issues
    
    
    async def validate_with_llm_batch(self, items: List[Dict[str, str]], target_lang: str, term_base: Optional[Dict[str, Dict[str, Any]]] = None) -> List[List[str]]:
        """Use LLM to validate multiple translations at once.
        
        Args:
            items: List of dictionaries containing 'source_text', 'translated_text', and 'context'
            target_lang: Target language code
            term_base: Optional term base dictionary
            
        Returns:
            List of lists containing validation issues for each translation
        """
        combined_prompt = (
            f"You are a professional translation validator for {target_lang} language. "
            f"Your task is to meticulously evaluate each translation for accuracy, consistency, and adherence to provided guidelines.\n\n"
        )
        
        # Add term base at the beginning if available
        if term_base:
            combined_prompt += (
                "Term Base Guidelines:\n"
                "- Verify that all term base translations are used consistently\n"
                "- Check that game-specific terms match their approved translations\n"
                "- Ensure special terms are preserved exactly as specified\n"
                "- Flag any deviations from term base translations\n\n"
                "Term Base Entries:\n"
            )
            for term, data in term_base.items():
                combined_prompt += f"- {term}: {data['translation']} (Context: {data['context']})\n"
            combined_prompt += "\n"

        for i, item in enumerate(items, 1):
            combined_prompt += (
                f"Translation #{i}:\n"
                f"Source text: {item['source_text']}\n"
                f"Translated text: {item['translated_text']}\n"
                f"Context: {item['context']}\n"
            )

            if item['previous_issues'] and len(item['previous_issues']) > 0:
                combined_prompt += f"Previous issues: {item['previous_issues']}\n"

            combined_prompt += "\n"

        combined_prompt += (
            f"For each translation, evaluate:\n"
            f"1. Semantic accuracy (does it convey the same meaning?)\n"
            f"2. Regional and linguistic appropriateness (suitable for target language region, while maintaining the original tone even if provocative)\n"
            f"3. Natural flow and readability\n"
            f"4. Consistency in tone and style\n"
            f"5. Correct usage of term base translations\n"
            f"6. Preservation of special terms and markup\n"
            f"7. Resolution of previous issues (if any)\n\n"
            f"Pay special attention to:\n"
            f"- Consistent use of approved terminology\n"
            f"- Proper handling of game-specific terms\n"
            f"- Whether previous translation issues have been properly addressed\n"
            f"- Regional language conventions while preserving original intent and tone\n"
            f"- Cultural nuances specific to the target language region"
            f"- Keeping with syntax of the source text (e.g. punctuation, capitalization)"
        )

        validation_schema = {
            "type": "object",
            "properties": {
                "validations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "translation_number": {"type": "integer"},
                            "is_valid": {"type": "boolean"},
                            "issues": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["translation_number", "is_valid", "issues"]
                    }
                }
            },
            "required": ["validations"]
        }
        
        response = await self.llm_handler.generate_completion(
            messages=[
                {"role": "system", "content": f"You are a professional translation validator for {target_lang} language."},
                {"role": "user", "content": combined_prompt}
            ],
            json_schema=validation_schema
        )

        # Extract content from LiteLLM response
        content = response.choices[0].message.content.strip()
        
        # Extract JSON block if present
        json_block_start_idx = content.find("```json")
        if json_block_start_idx != -1:
            json_block_end_idx = content.rfind("```")
            if json_block_end_idx != -1:
                content = content[json_block_start_idx + len("```json"):json_block_end_idx]
        
        content = content.strip()
        
        # Parse JSON response
        try:
            result = json.loads(content)
            validations = result.get("validations", [])
            
            # Convert to list of issue lists, maintaining original order
            all_issues = []
            for item in validations:
                if item["is_valid"]:
                    all_issues.append([])
                else:
                    all_issues.append([f"LLM issue: {issue}" for issue in item["issues"]])
            
            return all_issues

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return [[f"LLM validation failed: Could not parse JSON response"]] * len(items)
        
        except AttributeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return [[f"LLM validation failed: Invalid JSON response"]] * len(items)

    async def validate_with_llm(self, source_text: str, translated_text: str, context: str, issues: List[str], target_lang: str) -> List[str]:
        """Use LLM to validate translation quality."""
        items = [{
            'source_text': source_text,
            'translated_text': translated_text,
            'context': context,
            'issues': issues
        }]
        
        results = await self.validate_with_llm_batch(items, target_lang)
        return results[0] if results else []
            
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
                regex = f"{start}.*?{end}"  # Simplified pattern to match exact terms
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
        # TODO - fix full pattern matching
        if not self.patterns:
            return []
            
        terms = []
        for pattern in self.patterns:
            matches = pattern.findall(text)
            # findall returns tuples if we have groups, so we need to get first element
            terms.extend(m[0] if isinstance(m, tuple) else m for m in matches)
        return terms