from typing import Dict, Optional
import pandas as pd
from ..sheets.sheets_handler import GoogleSheetsHandler

class TermBaseHandler:
    def __init__(self, sheets_handler: GoogleSheetsHandler, sheet_name: str,
                 term_column: str = "EN TERM",
                 comment_column: str = "COMMENT",
                 translation_prefix: str = "TRANSLATION_"):
        """Initialize the Term Base Handler.
        
        Args:
            sheets_handler: Initialized GoogleSheetsHandler
            sheet_name: Name of the sheet containing term base
            term_column: Name of the column containing English terms
            comment_column: Name of the column containing comments
            translation_prefix: Prefix for translation columns
        """
        self.sheets_handler = sheets_handler
        self.sheet_name = sheet_name
        self.term_column = term_column
        self.comment_column = comment_column
        self.translation_prefix = translation_prefix
        self.term_base: Dict[str, Dict[str, str]] = {}
        self.load_term_base()
        
    def load_term_base(self) -> None:
        """Load the term base from the Google Sheet."""
        try:
            df = self.sheets_handler.read_sheet(self.sheet_name)
            
            # Get all translation columns (starting with translation_prefix)
            translation_columns = [col for col in df.columns 
                                if col.startswith(self.translation_prefix)]
            
            # Process each row
            for _, row in df.iterrows():
                term = row[self.term_column]
                if pd.isna(term):
                    continue
                    
                comment = row.get(self.comment_column, '')
                
                # Get translations for all languages
                translations = {
                    col[len(self.translation_prefix):].lower(): row[col]
                    for col in translation_columns
                    if pd.notna(row[col]) and row[col] != ''
                }
                
                self.term_base[term] = {
                    'comment': comment,
                    'translations': translations
                }
                
        except Exception as e:
            print(f"Error loading term base: {e}")
            self.term_base = {}
            
    def get_terms_for_language(self, lang: str) -> Dict[str, str]:
        """Get all terms for a specific language."""
        terms = {}
        for term, data in self.term_base.items():
            if lang in data['translations']:
                terms[term] = data['translations'][lang]
        return terms
    
    def add_term(self, term: str, comment: str = "", 
                 translations: Dict[str, str] = None) -> None:
        """Add a new term to the term base."""
        if translations is None:
            translations = {}
            
        # Update local cache
        if term in self.term_base:
            self.term_base[term]['comment'] = comment
            self.term_base[term]['translations'].update(translations)
        else:
            self.term_base[term] = {
                'comment': comment,
                'translations': translations
            }
            
        # Update sheet
        self._save_term_base()
        
    def update_translations(self, term: str, 
                          translations: Dict[str, str]) -> None:
        """Update translations for an existing term."""
        if term not in self.term_base:
            raise KeyError(f"Term not found: {term}")
            
        self.term_base[term]['translations'].update(translations)
        self._save_term_base()
        
    def _save_term_base(self) -> None:
        """Save the term base to the Google Sheet."""
        # Get all languages
        languages = set()
        for term_data in self.term_base.values():
            languages.update(term_data['translations'].keys())
            
        # Prepare data for DataFrame
        data = []
        for term, term_data in self.term_base.items():
            row = {
                self.term_column: term,
                self.comment_column: term_data['comment']
            }
            
            # Add translations
            for lang in languages:
                col = f"{self.translation_prefix}{lang.upper()}"
                row[col] = term_data['translations'].get(lang, '')
                
            data.append(row)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Update sheet
        # Note: This needs to be implemented in GoogleSheetsHandler
        self.sheets_handler.update_sheet(self.sheet_name, df) 