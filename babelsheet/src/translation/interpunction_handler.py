import re
from typing import List
import pandas as pd
from ..sheets.sheets_handler import SheetsHandler
from ..utils.ui.base_ui_manager import UIManager

class InterpunctionHandler:
    def __init__(self, sheets_handler: SheetsHandler, ui: UIManager):
        """Initialize Interpunction Handler."""
        self.sheets_handler = sheets_handler
        self.ui = ui

    async def check_interpunction_spacing(self, sheet_name: str, target_langs: List[str]) -> int:
        """Check and fix spaces before interpunction marks in the specified languages.
        
        Args:
            sheet_name: Name of the sheet to check
            target_langs: List of target language codes to check
            ui: UI manager instance for displaying progress
            
        Returns:
            Number of fixes made
        """
        # Get the sheet data
        df = self.sheets_handler.get_sheet_data(sheet_name)
        
        # Get column indexes for target languages
        target_langs_idx = self.sheets_handler.get_column_indexes(df, target_langs)
        column_names = self.sheets_handler.get_column_names(df)
        rows_count = df.shape[0]

        fixes_made = 0
        
        # Update UI
        self.ui.info(f"<b>Checking interpunction spacing in `<font color='cyan'>{sheet_name}</font>`...</b>")
        
        # Check each target language
        for i, lang_idx in enumerate(target_langs_idx):
            lang = column_names[lang_idx]
            
            # Check each row
            for row_idx in range(rows_count):
                cell = self.sheets_handler.get_cell_value(df, row_idx, lang_idx)
                
                # Skip if cell is empty
                if cell is None or pd.isna(cell) or (hasattr(cell, 'is_empty') and cell.is_empty()):
                    continue
                
                text = cell.value if hasattr(cell, 'value') else str(cell)
                
                # Fix spacing before interpunction marks
                fixed_text = InterpunctionHandler.fix_interpunction_spacing(text)

                # If text was changed, update the cell
                if fixed_text != text:
                    self.sheets_handler.modify_cell_data(sheet_name, row_idx, lang_idx, fixed_text)
                    self.ui.info(f"    🔧 [{lang}] Fixed spacing: '{fixed_text}'")
                    fixes_made += 1
        
        # Update sheet if any fixes were made
        if fixes_made > 0:
            self.sheets_handler.save_changes()
            self.ui.info(f"<b>Made <font color='yellow'>{fixes_made}</font> spacing fixes in `<font color='cyan'>{sheet_name}</font>`</b>")
        else:
            self.ui.info(f"<b>No spacing issues found in `<font color='cyan'>{sheet_name}</font>`</b>")
            
        return fixes_made

    @staticmethod
    def fix_interpunction_spacing(text: str) -> str:
        """Fix spaces before interpunction marks in the given text.
        
        Args:
            text: Text to fix
            
        Returns:
            Fixed text with proper spacing before interpunction marks
        """
        # Non-breaking space character
        nbsp = '\u00A0'
        
        # Replace regular space before interpunction marks with non-breaking space
        patterns = [
            (r' !', f'{nbsp}!'),
            (r' \?', f'{nbsp}?'),
            (r' :', f'{nbsp}:'),
            (r' ;', f'{nbsp};'),
            #(r' »', f'{nbsp}»'),
            #(r'« ', f'«{nbsp}'),
            (r' %', f'{nbsp}%'),
            (r' ,', f'{nbsp},')
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        return result 