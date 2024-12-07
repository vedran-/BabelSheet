import re
from typing import Dict, List, Tuple, Optional, Pattern, Any
from .llm_handler import LLMHandler
from .ui_manager import UIManager
from ..translation.translation_dictionary import TranslationDictionary
import logging
import json
import asyncio
import spacy

class QAHandler:
    def __init__(self,
                 config: Dict[str, Any],
                 llm_handler: Optional[LLMHandler] = None, 
                 ui: Optional[UIManager] = None,
                 translation_dictionary: Optional[TranslationDictionary] = None,
                 non_translatable_patterns: Optional[List[Dict[str, str]]] = None):
        """Initialize QA Handler.
        
        Args:
            max_length: Maximum allowed length for translations (if None, no limit)
            llm_handler: LLMHandler instance for AI-powered validation
            non_translatable_patterns: List of pattern dicts with 'start' and 'end' keys
            translation_dictionary: TranslationDictionary instance
        """
        self.config = config
        self.max_length = self.config.get('qa', {}).get('max_length', 1000)
        self.llm_handler = llm_handler
        self.logger = logging.getLogger(__name__)
        self.ui = ui
        self.translation_dictionary = translation_dictionary

        # Validate and compile patterns if provided
        self.non_translatable_patterns = non_translatable_patterns
        if non_translatable_patterns:
            try:
                self.patterns = self._compile_patterns(non_translatable_patterns)
                self.ui.info(f"Compiled {len(self.patterns)} non-translatable patterns")
            except Exception as e:
                self.ui.critical(f"Failed to compile patterns: {e}")
                self.patterns = None
                raise Exception(f"Failed to compile patterns: {e}")
        else:
            self.patterns = None
            self.ui.info("No non-translatable patterns configured")

        try:
            self.nlp = spacy.load("xx_sent_ud_sm")
        except OSError:
            spacy_model_name = "xx_sent_ud_sm"
            self.ui.info(f"Downloading required spacy language model <b><font color='yellow'>{spacy_model_name}</font></b>...")
            spacy.cli.download(spacy_model_name)
            self.nlp = spacy.load(spacy_model_name)
            self.ui.info(f"Downloaded spacy language model <b><font color='yellow'>{spacy_model_name}</font></b>!")
        
    async def validate_translation_syntax(self, source_text: str, translated_text: str,
                                 context: str, term_base: Optional[Dict[str, Dict[str, Any]]] = None,
                                 target_lang: str = None, override: Optional[str] = None) -> List[str]:
        """Validate translation quality.
        
        Args:
            source_text: Original text
            translated_text: Translated text to validate
            context: Context of the translation
            term_base: Optional term base dictionary
            target_lang: Target language code for term base validation
            override: Optional override reason
        """

        if override:
            self.ui.info(f"Validation override used for text <b>'{source_text}'</b>. Reason: {override}")
            return []

        issues = []
        
        try:
            #self.logger.debug(f"Validating translation: '{source_text}' => '{translated_text}' ({target_lang})")
            
            # Check for non-translatable terms only if patterns are configured
            if self.patterns:
                source_terms = self.extract_non_translatable_terms(source_text)
                for term in source_terms:
                    if term not in translated_text:
                        issues.append(f"Non-translatable term '{term}' is missing in translation")
            
            # Basic format validation
            format_issues = self._validate_format(source_text, translated_text, target_lang)
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
                #self.logger.debug(f"Starting term base validation for language: {target_lang}")
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

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text using spaCy."""
        doc = self.nlp(text.replace('\\n', '\n')
                       .replace('\\t', '\t')
                       .replace('\\"', '"')
                       .replace("\\'", "'"))
        return [token.text for token in doc if not token.is_space]
    def _analyze_capitalization(self, text: str ) -> Dict[str, List[str]]:
        n = {}
        n['words'] = self._extract_words(text)
        n['all_caps_words'] = [word for word in n['words'] if len(word) > 1 and word.isupper()]
        n['non_all_caps_words'] = [word for word in n['words'] if len(word) > 1 and not word.isupper()]
        n['has_caps'] = len(n['all_caps_words']) > 0
        # Check if more than 70% of the words are in ALL CAPS
        n['has_all_caps'] = len(n['all_caps_words']) / len(n['words']) > 0.7
        return n

    @staticmethod
    def _get_punctuation_sets() -> Dict[str, str]:
        """Get punctuation marks organized by script type."""
        return {
            'latin': ".!?:,",
            'cjk': "。！？：，｡︕？︓，",
            'rtl': "؟؛،۔",  # RTL punctuation
            'ethiopic': "።፧፨፠"  # Ethiopic script punctuation
        }
    @staticmethod
    def _get_punctuation_type_sets() -> Dict[str, str]:
        """Get punctuation marks organized by semantic type."""
        return {
            'question': "?？؟፧",  # Latin, CJK, Arabic, Ethiopic
            'exclamation': "!！︕፨",
            'period': ".。｡።۔",
        }
    def _get_ending_punctuation(self, text: str) -> str:
        """Get the last character if it's a punctuation mark.
        
        Args:
            text: The text to check for ending punctuation
            
        Returns:
            The ending punctuation character if found, empty string otherwise
        """
        if not text.strip():
            return ""
        last_char = text.strip()[-1]
        
        # Combine all punctuation marks
        all_puncts = ''.join(self._get_punctuation_sets().values())
        
        if last_char in all_puncts:
            return last_char
        return ""
    def _get_punctuation_type(self, char: str) -> str:
        """Determine the semantic type of a punctuation mark.
        
        Args:
            char: The punctuation character to check
            
        Returns:
            The semantic type of the punctuation ('question', 'exclamation', 'period', or 'other')
        """
        punct_types = self._get_punctuation_type_sets()
        
        for punct_type, chars in punct_types.items():
            if char in chars:
                return punct_type
        return "other"
    def _validate_ending_punctuation(self, source: str, translation: str) -> List[str]:
        """Validate ending punctuation between source and translation.
        
        Args:
            source: The source text
            translation: The translated text
            
        Returns:
            List of issues found with punctuation, empty if no issues
        """
        issues = []
        source_end = self._get_ending_punctuation(source)
        trans_end = self._get_ending_punctuation(translation)
        
        if bool(source_end) != bool(trans_end):
            # One has punctuation, the other doesn't
            issues.append(f"Ending punctuation does not match between source text and translation")
        elif source_end and trans_end:
            # Both have punctuation - check if they're semantically equivalent
            source_type = self._get_punctuation_type(source_end)
            trans_type = self._get_punctuation_type(trans_end)
            
            if source_type != trans_type:
                issues.append(f"Ending punctuation type mismatch: source uses {source_end} ({source_type}) but translation uses {trans_end} ({trans_type})")
        
        return issues

    def _validate_format(self, source: str, translation: str, target_lang: str) -> List[str]:
        """Check format consistency between source and translation."""
        issues = []

        # Check capitalization
        source_analysis = self._analyze_capitalization(source)
        translation_analysis = self._analyze_capitalization(translation)
        if source_analysis['has_all_caps'] and not translation_analysis['has_all_caps']:
            issues.append(f"Capitalization mismatch: Words in the translation should be in ALL CAPS, as they are in the source text.")
        elif not source_analysis['has_all_caps'] and translation_analysis['has_all_caps']:
            issues.append(f"Capitalization mismatch: Translation has too many words in ALL CAPS, compared to the source text. Please match the source text's capitalization per word in the translation.")
        elif source_analysis['has_caps'] and not translation_analysis['has_caps'] \
            and max(len(word) for word in source_analysis['all_caps_words']) > 2:   # We can ignore words that are 2 characters or less, like I, WC, etc.
            issues.append(f"Capitalization mismatch: Some words in the translation should be in ALL CAPS, as they are in the source text. Please add ALL CAPS to the words in the translation that are in ALL CAPS in the source text.")
        elif not source_analysis['has_caps'] and translation_analysis['has_caps'] \
            and max(len(word) for word in translation_analysis['all_caps_words']) > 2: # We can ignore words that are 2 characters or less, like I, WC, etc.
            issues.append(f"Capitalization mismatch: Translation has too many words in ALL CAPS, compared to the source text. Please remove ALL CAPS from the words in the translation that are not in ALL CAPS in the source text.")
        
        # Check newline preservation using normalized counts, allowing 1 line difference
        def count_newlines(text: str) -> int:
            """Count newlines in text, handling both \\n and \n"""
            return text.count('\\n') + text.count('\n')
        source_newlines = count_newlines(source)
        trans_newlines = count_newlines(translation)
        if abs(source_newlines - trans_newlines) > 1:  # Allow difference of 1
            issues.append(f"Newline count mismatch between source text ({source_newlines} rows) and translation ({trans_newlines} rows). ")
            
        # Check ending punctuation
        punctuation_issues = self._validate_ending_punctuation(source, translation)
        issues.extend(punctuation_issues)
            
        return issues
    
    def _validate_markup(self, source: str, translation: str) -> List[str]:
        """Check if markup is preserved correctly."""
        issues = []
        
        # First check non-translatable terms if patterns exist
        if self.patterns:
            source_terms = self.extract_non_translatable_terms(source)
            for term in source_terms:
                if term not in translation:
                    issues.append(f"Non-translatable term `{term}` must appear exactly as in source text")
                elif translation.count(term) != source.count(term):
                    issues.append(f"Non-translatable term `{term}` appears {translation.count(term)} times in translation, but {source.count(term)} times in source text.")
        
        # Check square brackets
        source_brackets = re.findall(r'\[.*?\]', source)
        trans_brackets = re.findall(r'\[.*?\]', translation)
        if len(source_brackets) != len(trans_brackets):
            issues.append(f"Square bracket markup count mismatch between source text and translation")
        else:
            for s, t in zip(source_brackets, trans_brackets):
                if s != t:
                    issues.append(f"Square bracket content modified: `{s}` -> `{t}` between source text and translation")
        
        # Check curly braces
        source_braces = re.findall(r'\{.*?\}', source)
        trans_braces = re.findall(r'\{.*?\}', translation)
        if len(source_braces) != len(trans_braces):
            issues.append(f"Curly brace markup count mismatch between source text and translation")
        else:
            for s, t in zip(source_braces, trans_braces):
                if s != t:
                    issues.append(f"Curly brace content modified: `{s}` -> `{t}` between source text and translation")
        
        # Check < and >
        source_angle_brackets = re.findall(r'<.*?>', source)
        trans_angle_brackets = re.findall(r'<.*?>', translation)
        if len(source_angle_brackets) != len(trans_angle_brackets):
            issues.append(f"Angle bracket markup count mismatch between source ({source}) and translation ({translation})")
        else:
            for s, t in zip(source_angle_brackets, trans_angle_brackets):
                if s != t:
                    issues.append(f"Angle bracket content modified: `{s}` -> `{t}` between source text and translation")

        return issues
    
    def _validate_term_base(self, translation: str, term_base: Dict[str, Dict[str, Any]], target_lang: str) -> List[str]:
        """Check if translation uses terms from term base correctly.
        
        Args:
            translation: The translated text to validate
            term_base: Term base dictionary with structure {term: {'translations': Dict[str, str], 'comment': str}}
            target_lang: Target language code
        """
        issues = []
        #self.logger.debug(f"Validating translation against term base for language: {target_lang}")
        
        for source_term, term_data in term_base.items():
            expected_translation = term_data.get('translation', '')
            comment = term_data.get('comment', '')
            
            if not expected_translation:
                self.logger.warning(f"Skipping term `{source_term}` - no translation for {target_lang}")
                continue
            
            # Check if source term appears in translation
            source_term_lower = source_term.lower()
            translation_lower = expected_translation.lower()
            
            # Use word boundary check to avoid partial matches
            if f" {source_term_lower} " in f" {translation_lower} ":
                expected_lower = expected_translation.lower()
                
                # Check if expected translation is used
                if expected_lower not in translation_lower:
                    issue = f"Term base mismatch: `{source_term}` should be translated as `{expected_translation}`"
                    if comment:
                        issue += f" (Note: {comment})"
                    self.logger.warning(f"Found term base issue: {issue}")
                    issues.append(issue)
                else:
                    #self.logger.debug(f"Term '{source_term}' correctly translated as '{expected_translation}'")
                    pass
                
        #if not issues:
        #    self.logger.debug("No term base issues found")
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
        if False and term_base:
            combined_prompt += (
                "# Term Base Guidelines:\n"
                "- Verify that all term base translations are used consistently\n"
                "- Check that game-specific terms and names match their approved translations\n"
                "- Ensure special terms are preserved exactly as specified\n"
                "- Flag any deviations from term base translations\n"
            )
            
            if self.non_translatable_patterns and len(self.non_translatable_patterns) > 0:
                combined_prompt += (
                    "- Exception to this rule are non-translatable terms, which must be preserved exactly as is\n"  
                    f"- Non-translatable terms will match the following patterns: {str(self.non_translatable_patterns)}\n"
                )

            if False:
                combined_prompt += "\nTerm Base Entries:\n"
                for term, data in term_base.items():
                    combined_prompt += f"- {term}: {data['translation']} (Context: {data['context']})\n"
                combined_prompt += "\n"

        combined_prompt += (
            f"\n# Essential Requirements:\n"
            f"1. Grammatical and semantic accuracy:\n"
            f"   - Translation is grammatically correct in target language\n"
            f"   - Conveys identical meaning to source text\n"
            f"   - Key messages and nuances are preserved\n"
            f"   - Maintains tone while being accurate\n\n"
            f"2. Term base compliance:\n"
            f"   - Names MUST match RELEVANT_TRANSLATIONS exactly\n"
            f"   - Other terms should follow RELEVANT_TRANSLATIONS unless doing so would:\n"
            f"      * Result in grammatical errors\n"
            f"      * Create awkward or unnatural phrasing\n"
            f"      * Significantly impact readability\n"
            f"   - Deviations from term base are acceptable when they improve:\n"
            f"      * Grammatical correctness\n"
            f"      * Natural flow\n"
            f"      * Clarity of meaning\n"
            f"   - Any deviation should preserve the core meaning of the term\n\n"
            f"3. Technical formatting:\n"
            f"   - Non-translatable terms preserved exactly as in source\n"
            f"   - Special terms between markup characters left untranslated\n"
            f"   - Capitalization matches source text (e.g., UPPERCASE words stay UPPERCASE)\n"
            f"   - Line breaks positioned more or less the same as source text\n"
            f"   - Multi-line translations maintain equal line lengths where possible\n\n"
            f"4. Style requirements:\n"
            f"   - Maintains lighthearted and fun tone\n"
            f"   - Takes clear positions rather than being neutral\n"
            f"   - Avoids offensive language\n"
            f"   - Fits UI space constraints\n"
            f"   - Preserves source text's level of formality\n\n"
            f"5. Regional appropriateness:\n"
            f"   - Uses correct regional language variants\n"
            f"   - Respects cultural nuances of target region\n"
            f"   - Properly localized except for protected terms\n\n"
            f"# For Failed Translations:\n"
            f"When FAILED_TRANSLATION tags are present:\n"
            f"- Each previous error must be explicitly addressed\n"
            f"- New translation must avoid all previously identified issues\n"
            f"- Solutions should improve upon rejected versions while maintaining accuracy\n\n"
            f"# Final Checks:\n"
            f"- Translation reads naturally in target language\n"
            f"- All technical requirements are met\n"
            f"- Previous issues are resolved\n"
            f"- Deviations from term base (if any) are justified\n"
            f"- Formatting matches source exactly\n\n"
        )

        if self.config['llm']['additional_llm_context']:
            combined_prompt += f"\n# Wider context:\n{self.config['llm']['additional_llm_context']}\n\n"

        combined_prompt += f"\n# Translations to Validate ({len(items)} texts):\n"

        for i, item in enumerate(items, 1):
            combined_prompt += (
                f"\n## Translation #{i} ##\n"
                f"Source text: {item['source_text'].replace('\n', '\\n')}\n"
                f"Translated text to validate: {item['translated_text'].replace('\n', '\\n')}\n"
                f"Context: {item['context']}\n"
            )

            relevant_translations = self.translation_dictionary.get_relevant_translations(item['source_text'], target_lang)
            if len(relevant_translations) > 0:
                combined_prompt += f"RELEVANT_TRANSLATIONS:\n    - {'\n   - '.join(f'{rt['term'].replace('\n', '\\n')}: {rt['translation'].replace('\n', '\\n')}' for rt in relevant_translations)}\n"

            if item['previous_issues'] and len(item['previous_issues']) > 0:
                combined_prompt += f"Previously rejected translations:\n    - {'\n   - '.join(f'`{issue['translation'].replace('\n', '\\n')}` failed because: {issue['issues']}' for issue in item['previous_issues'])}\n"

            combined_prompt += "\n"

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
        
        try:
            quotes_warning = "When providing feedback, never use double quotes - instead, use ` (backtick). For example, write `word` not ''word'' or ""word"". This ensures the JSON response remains valid."
            response = await self.llm_handler.generate_completion(
                messages=[
                    {"role": "system", "content": (
                        f"You are a professional translation validator for {target_lang} language. {quotes_warning}"
                    )},
                    {"role": "user", "content": combined_prompt + f"\n\n{quotes_warning}"}
                ],
                json_schema=validation_schema
            )
        except Exception as e:
            self.logger.error(f"LLM API call failed: {str(e)}, sleeping for 3 minutes before retrying...")
            await asyncio.sleep(180)
            return await self.validate_with_llm_batch(items, target_lang, term_base)

        # Extract content from LiteLLM response
        content = response.choices[0].message.content.strip()
        
        # Extract JSON block if present
        json_block_start_idx = content.find("```json")
        if json_block_start_idx != -1:
            json_block_end_idx = content.rfind("```")
            if json_block_end_idx != -1:
                content = content[json_block_start_idx + len("```json"):json_block_end_idx]
        
        content = content.strip()
        
        # Fix common escaping issues that might break JSON
        content = content.replace(r"\'", "'")  # Replace escaped single quotes
        content = content.replace(r'\"', '"')  # Replace escaped double quotes if any
        
        # Parse JSON response
        try:
            # Try to fix any remaining JSON issues
            try:
                result = json.loads(content)
            except json.JSONDecodeError as first_error:
                # If initial parse fails, try with ast.literal_eval as it's more forgiving
                import ast
                try:
                    # Convert single quotes to double quotes for JSON compatibility
                    content = content.replace("'", '"')
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Log the original content for debugging
                    self.logger.error(f"Failed to parse JSON content: {content}")
                    self.logger.error(f"Original error: {first_error}")
                    raise
            
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
            self.logger.error(f"Problematic content: {content}")
            return [[f"LLM technical issue: Could not parse JSON response. Did you use double quotes by mistake?"]] * len(items)
        
        except Exception as e:
            self.logger.error(f"Unexpected error while parsing LLM response: {e}")
            self.logger.error(f"Problematic content: {content}")
            return [[f"LLM technical issue: {str(e)}"]] * len(items)

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
    
    def extract_non_translatable_terms(self, text: str) -> List[str]:
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
    
