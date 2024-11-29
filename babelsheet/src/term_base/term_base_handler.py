from typing import Dict, Optional, Any
import pandas as pd
from ..sheets.sheets_handler import SheetsHandler
import logging

logger = logging.getLogger(__name__)

class TermBaseHandler:
    def __init__(self, ctx):
        """Initialize Term Base Handler.
        
        Args:
            sheets_handler: Google Sheets handler instance
            ctx: Context object
        """
        self.ctx = ctx
        self.sheets_handler = ctx.sheets_handler
        self.sheet_name = ctx.config['term_base']['sheet_name']
        self.term_column_name = ctx.source_lang
        self.logger = logging.getLogger(__name__)
        self.load_term_base()
        
    def load_term_base(self) -> None:
        """Load the term base from the Google Sheet.
        
        The term base is stored in a sheet with the following columns:
        - EN TERM: Source term in English
        - COMMENT: Optional comment about the term
        - Language codes: Direct language codes (e.g., 'es' for Spanish)
        
        Raises:
            ValueError: If sheet access fails or required columns are missing
        """
        self.logger.info(f"Loading term base from sheet: {self.sheet_name}")
        self.sheet_data = self.sheets_handler.get_sheet_data(self.sheet_name)

        # Find the index of the term column
        self.term_column_index = self.sheets_handler.get_column_indexes(self.sheet_data, [self.term_column_name])[0]
        if self.term_column_index == -1:
            raise ValueError(f"Required column '{self.term_column_name}' not found in term base sheet")

        # Find the indexes of the context columns
        self.logger.info(f"Successfully loaded {len(self.sheet_data.iloc[:, self.term_column_index])} terms")

    def get_terms_for_language(self, lang: str) -> Dict[str, Dict[str, Any]]:
        """Get all terms for a specific language.
        
        Args:
            lang: Language code
            
        Returns:
            Dictionary mapping terms to their data (translations and comments)
        """

        lang_context_column_index = self.sheets_handler.get_column_indexes(self.sheet_data, [lang])[0]

        terms = {}
        for i, row in self.sheet_data.iterrows():
            if i == 0:
                continue

            term = row[self.term_column_index].value

            translation = row[lang_context_column_index].value
            if pd.isna(translation):
                self.logger.critical(f"Language `{lang}`: term not found: `{term}`")
                continue

            term_combined_context = ' '.join([row[idx].value for idx in self.sheet_data.attrs['context_column_indexes']])

            terms[term] = { 
                'translation': translation,
                'context': term_combined_context
            }

        return terms
