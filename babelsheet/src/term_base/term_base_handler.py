from typing import Dict, Optional, Any
import pandas as pd
from ..sheets.sheets_handler import SheetsHandler, CellData
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
        self.logger.debug(f"Loading term base from sheet: {self.sheet_name}")
        self.sheet_data = self.sheets_handler.get_sheet_data(self.sheet_name)

        # Find the index of the term column
        self.term_column_index = self.sheets_handler.get_column_indexes(self.sheet_data, [self.term_column_name])[0]
        if self.term_column_index == -1:
            raise ValueError(f"Required column '{self.term_column_name}' not found in term base sheet")

        # Find the indexes of the context columns
        self.logger.info(f"Successfully loaded Term Base from sheet '{self.sheet_name}' ({len(self.sheet_data.iloc[:, self.term_column_index])} terms)")

    def get_terms_for_language(self, lang: str) -> Dict[str, Dict[str, Any]]:
        """Get all terms for a specific language."""
        
        lang_column_index = self.sheets_handler.get_column_indexes(self.sheet_data, [lang])[0]
        if lang_column_index == -1:
            raise KeyError(f"Language column '{lang}' not found in term base sheet")

        terms = {}
        for i, row in self.sheet_data.iterrows():
            if i == 0:
                continue

            term = row[self.term_column_index].value

            translation = row[lang_column_index].value
            if pd.isna(translation):
                self.logger.critical(f"Language `{lang}`: term not found: `{term}`")
                continue

            term_combined_context = ' '.join([row[idx].value for idx in self.sheet_data.attrs['context_column_indexes']])

            terms[term] = { 
                'translation': translation,
                'context': term_combined_context
            }

        return terms

    def add_new_term(self, term: str, comment: str, translations: Dict[str, str]) -> None:
        """Add a new term to the term base"""

        terms = self.sheet_data.iloc[1:, self.term_column_index].values
        term_idx = next((i for i, t in enumerate(terms) if term in t.value), None)

        def _set_translations(term_idx: int, translations: Dict[str, str]):
            for lang, translation in translations.items():
                column_idx = self.sheets_handler.get_column_indexes(self.sheet_data, [lang], create_if_missing=True)[0]
                current_cell = self.sheet_data.iloc[term_idx + 1, column_idx]
                if current_cell is not None and not current_cell.is_empty():
                    self.logger.critical(f"Translation for language `{lang}` already exists for term `{term}` - ignoring")
                    continue

                self.sheets_handler.modify_cell_data(self.sheet_name, term_idx + 1, column_idx, translation)

        if term_idx is not None:
            self.logger.warn(f"Term `{term}` already exists in term base")
            _set_translations(term_idx, translations)
            return

        # Initialize row with empty CellData for all columns
        row = [CellData(None) for _ in range(len(self.sheet_data.columns))]
        
        # Set term
        row[self.term_column_index] = CellData(term, is_synced=False)

        # Set comment
        comment_column_idx = self.sheet_data.attrs['context_column_indexes'][0]
        row[comment_column_idx] = CellData(comment, is_synced=False)

        # Add new row and get its index
        new_row_idx = len(self.sheet_data) - 1  # Get index before adding
        self.sheets_handler.add_new_row(self.sheet_data, row)

        # Set translations with correct index
        _set_translations(new_row_idx, translations)
