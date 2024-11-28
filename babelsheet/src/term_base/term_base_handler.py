from typing import Dict, Optional, Any
import pandas as pd
from ..sheets.sheets_handler import GoogleSheetsHandler
import logging

logger = logging.getLogger(__name__)

class TermBaseHandler:
    def __init__(self, sheets_handler: GoogleSheetsHandler, sheet_name: str):
        """Initialize Term Base Handler.
        
        Args:
            sheets_handler: Google Sheets handler instance
            sheet_name: Name of the sheet containing the term base
        """
        self.sheets_handler = sheets_handler
        self.sheet_name = sheet_name
        self.term_column = "EN TERM"
        self.comment_column = "COMMENT"
        self.term_base: Dict[str, Dict[str, Any]] = {}  # {term: {'translations': Dict[str, str], 'comment': str}}
        self.logger = logging.getLogger(__name__)
        
        # Only try to load term base if sheets handler is ready
        if self.sheets_handler.is_ready():
            try:
                self.load_term_base()
            except Exception as e:
                self.logger.error(f"Failed to load term base during initialization: {e}")
                self.logger.info("Term base will be empty until next successful load")
        else:
            self.logger.info("Sheets handler not ready - term base will be loaded when spreadsheet ID is set")
        
    def load_term_base(self) -> None:
        """Load the term base from the Google Sheet.
        
        The term base is stored in a sheet with the following columns:
        - EN TERM: Source term in English
        - COMMENT: Optional comment about the term
        - Language codes: Direct language codes (e.g., 'es' for Spanish)
        
        Raises:
            ValueError: If sheet access fails or required columns are missing
        """
        try:
            # Verify sheets handler is properly configured
            if not self.sheets_handler.is_ready():
                raise ValueError("Google Sheets handler is not properly configured")
            
            self.logger.info(f"Loading term base from sheet: {self.sheet_name}")
            df = self.sheets_handler.read_sheet(self.sheet_name)
            
            # Verify required columns exist
            if self.term_column not in df.columns:
                raise ValueError(f"Required column '{self.term_column}' not found in term base sheet")
            
            # Get all translation columns (excluding term and comment columns)
            translation_columns = [col for col in df.columns 
                                if col not in [self.term_column, self.comment_column]]
            self.logger.debug(f"Found translation columns: {translation_columns}")
            
            # Clear existing term base before loading
            self.term_base.clear()
            
            # Process each row
            for _, row in df.iterrows():
                term = row[self.term_column]
                if pd.isna(term):
                    continue
                    
                comment = row.get(self.comment_column, '')
                
                # Get translations for all languages
                translations = {
                    col.lower(): row[col]
                    for col in translation_columns
                    if pd.notna(row[col]) and row[col] != ''
                }
                
                self.term_base[term] = {
                    'comment': comment,
                    'translations': translations
                }
                self.logger.debug(f"Loaded term: {term} with translations for {list(translations.keys())}")
                
            self.logger.info(f"Successfully loaded {len(self.term_base)} terms")
                
        except Exception as e:
            self.logger.error(f"Error loading term base: {e}", exc_info=True)
            raise
    
    def _save_term_base(self) -> None:
        """Save the term base to the Google Sheet."""
        try:
            logger.info(f"Saving term base to sheet: {self.sheet_name}")
            
            # Get all languages
            languages = set()
            for term_data in self.term_base.values():
                languages.update(term_data['translations'].keys())
            logger.debug(f"Found languages: {languages}")
            
            # Prepare data for DataFrame
            data = []
            for term, term_data in self.term_base.items():
                row = {
                    self.term_column: term,
                    self.comment_column: term_data['comment']
                }
                
                # Add translations
                for lang in languages:
                    col = lang.lower()  # Use language code directly
                    row[col] = term_data['translations'].get(lang, '')
                    
                data.append(row)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Update sheet
            self.sheets_handler.update_sheet(self.sheet_name, df)
            logger.info(f"Successfully saved {len(data)} terms to term base")
            
        except Exception as e:
            logger.error(f"Error saving term base: {e}", exc_info=True)
            raise
    
    def get_terms_for_language(self, lang: str) -> Dict[str, Dict[str, Any]]:
        """Get all terms for a specific language.
        
        Args:
            lang: Language code
            
        Returns:
            Dictionary mapping terms to their data (translations and comments)
        """
        terms = {}
        for term, data in self.term_base.items():
            if lang in data['translations']:
                terms[term] = {
                    'translations': {lang: data['translations'][lang]},
                    'comment': data['comment']
                }
        return terms
    
    def add_term(self, term: str, comment: str = "", translations: Dict[str, str] = None) -> None:
        """Add a new term to the term base or update translations for an existing term.
        
        Args:
            term: The term to add or update
            comment: Comment about the term (only used for new terms)
            translations: Dictionary mapping language codes to translations
        """
        if translations is None:
            translations = {}
            
        # Update local cache
        if term in self.term_base:
            # For existing terms, only update translations
            self.term_base[term]['translations'].update(translations)
        else:
            # For new terms, set both comment and translations
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
        