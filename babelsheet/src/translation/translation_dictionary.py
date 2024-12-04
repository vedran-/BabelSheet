from typing import List, Dict, Optional, Any

class TranslationDictionary:
    def __init__(self, ctx):
        """Initialize Translation Dictionary."""
        self.ctx = ctx
        self.sheets_handler = ctx.sheets_handler
        self.ui = ctx.ui
        self.dictionary: Dict[str, Dict[str, str]] = {}
    
    def _remove_substrings(self, text: str, substrings: List[str]) -> str:
        """Remove all substrings from text."""
        for substring in substrings:
            text = text.replace(substring, "")
        return text

    def initialize_from_sheets(self) -> None:
        """Load all translations into the dictionary from the sheets."""
        self.ui.info("Initializing Translation Dictionary...")
        for sheet_name in self.sheets_handler.get_sheet_names():
            df = self.sheets_handler.get_sheet_data(sheet_name)
            source_idx = self.sheets_handler.get_column_indexes(df, [self.ctx.source_lang])[0]
            target_idxs = self.sheets_handler.get_column_indexes(df, self.ctx.target_langs)
            if source_idx is None or len(target_idxs) == 0:
                self.ui.debug(f"No source or target columns found in sheet '{sheet_name}'")
                continue

            for index in range(1, len(df)):
                source_text = self.sheets_handler.get_cell_value(df, index, source_idx)
                if source_text is None:
                    continue

                for target_idx, target_lang in zip(target_idxs, self.ctx.target_langs):
                    target_text = self.sheets_handler.get_cell_value(df, index, target_idx)
                    if target_text is None:
                        continue
                    self.add_translation(source_text, target_lang, target_text)

    def add_translation(self, source_text: str, target_lang: str, translation: str) -> None:
        """Add a new translation to the dictionary."""
        if source_text is None or translation is None:
            return
        
        source_text = str(source_text).strip()
        translation = str(translation).strip()
        if source_text == "" or translation == "":
            return

        if target_lang not in self.dictionary:
            self.dictionary[target_lang] = {}
        
        if source_text in self.dictionary[target_lang]:
            if translation != self.dictionary[target_lang][source_text]:
                self.ui.critical(f"Translation for {source_text} in {target_lang} already exists but with different translation: '{self.dictionary[target_lang][source_text]}' -> '{translation}'")

        self.dictionary[target_lang][source_text] = translation

        # Also add to dictionary same source/target text but without variables
        variables = self.ctx.qa_handler.extract_non_translatable_terms(source_text)
        if len(variables) > 0:
            trimmed_source_text = self._remove_substrings(source_text, variables).strip()
            trimmed_translation = self._remove_substrings(translation, variables).strip()
            if trimmed_source_text != "" and trimmed_translation != "":
                self.add_translation(trimmed_source_text, target_lang, trimmed_translation)

    def get_translation(self, source_text: str, target_lang: str) -> Optional[str]:
        """Get translation for a specific text in target language."""
        if target_lang not in self.dictionary:
            return None
        if source_text not in self.dictionary[target_lang]:
            return None
        return self.dictionary[target_lang][source_text.strip()]

    def get_relevant_translations(self, source_text: str, target_lang: str) -> List[str]:
        """Get all relevant translations for a specific text in target language."""

        # First, remove variables from the source text
        variables = self.ctx.qa_handler.extract_non_translatable_terms(source_text)
        if len(variables) > 0:
            source_text = self._remove_substrings(source_text, variables).strip()

        haystack = source_text.strip().lower()
        dict = self.dictionary.get(target_lang, {})
        if len(dict) == 0:
            return []

        relevant_translations = []
        for source_term, translation_term in dict.items():
            if source_term.lower() in haystack:
                relevant_translations.append({'term': source_term, 'translation': translation_term})
        return relevant_translations
