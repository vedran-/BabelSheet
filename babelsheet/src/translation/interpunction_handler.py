import pandas as pd
import pathlib
import re
import os
from datetime import datetime
from typing import List
from ..sheets.sheets_handler import SheetsHandler
from ..utils.ui.base_ui_manager import UIManager

class InterpunctionHandler:
    def __init__(self, sheets_handler: SheetsHandler, ui: UIManager, log_output_dir: pathlib.Path):
        """Initialize Interpunction Handler."""
        self.sheets_handler = sheets_handler
        self.ui = ui
        self.log_file_name = os.path.join(log_output_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_interpunction_handler.log")

    async def check_interpunction_spacing(self, sheet_name: str, target_langs: List[str], nbsp_mode: str = 'unicode') -> int:
        """Check and fix spaces before interpunction marks in the specified languages.
        
        Args:
            sheet_name: Name of the sheet to check
            target_langs: List of target language codes to check
            mode: Mode of operation ('unicode', 'html', 'nobr')
            
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

        nbsp_mode = self.sheets_handler.ctx.config.get('translation', {}).get('space_to_nbsp_mode', 'nobr')

        # Update UI
        self.ui.info(f"<b>Checking interpunction spacing in `<font color='cyan'>{sheet_name}</font>` using mode: {nbsp_mode}...</b>")

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
                fixed_text = self.fix_interpunction_spacing(text, nbsp_mode)

                # If text was changed, update the cell
                if fixed_text != text:
                    self.sheets_handler.modify_cell_data(sheet_name, row_idx, lang_idx, fixed_text)
                    self.ui.info(f"    🔧 [{lang}] `{text}` ➜ `{fixed_text}`")
                    # Log the change
                    with open(self.log_file_name, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"[{sheet_name}:{lang}:{row_idx}] `{text}` ➜ `{fixed_text}`\n")
                    fixes_made += 1

        # Update sheet if any fixes were made
        if fixes_made > 0:
            self.sheets_handler.save_changes()
            self.ui.info(f"<b>Made <font color='yellow'>{fixes_made}</font> spacing fixes in `<font color='cyan'>{sheet_name}</font>`</b>")
        else:
            self.ui.info(f"<b>No spacing issues found in `<font color='cyan'>{sheet_name}</font>`</b>")

        return fixes_made

    @staticmethod
    def fix_interpunction_spacing(text: str, nbsp_mode: str = 'unicode') -> str:
        """Fix spaces before interpunction marks in the given text.
        
        Args:
            text: Text to fix
            nbsp_mode: Mode of operation ('unicode', 'html', 'nobr', 'space')
                - 'unicode': Replace space with unicode non-breaking space
                - 'html': Replace space with <nbsp> HTML tag
                - 'nobr': Wrap space and interpunction with <nobr> tag
                - 'space': Use regular space (no transformation)
            
        Returns:
            Fixed text with proper spacing before interpunction marks
            
        Raises:
            ValueError: If an unsupported mode is provided
        """
        VALID_MODES = {'unicode', 'html', 'nobr', 'space'}
        if nbsp_mode not in VALID_MODES:
            raise ValueError(f"Mode must be one of {VALID_MODES}, got: {nbsp_mode}")
            
        if not isinstance(text, str):
            text = str(text)
            
        # First, normalize existing non-breaking spaces to regular spaces
        text = text.replace('\u00A0', ' ')
        text = re.sub(r'<nbsp>', ' ', text)
        text = re.sub(r'<nobr>(.*?)</nobr>', r'\1', text)

        # If space mode, return text as is (with normalized spaces)
        if nbsp_mode == 'space':
            return text

        # Define interpunction marks
        interpunction = r'[!?:;%,]'
        
        if nbsp_mode == 'unicode':
            # Replace space before interpunction with unicode non-breaking space
            return re.sub(rf'\s+({interpunction})', f'\u00A0\\1', text)
            
        elif nbsp_mode == 'html':
            # Replace space before interpunction with <nbsp> tag
            return re.sub(rf'\s+({interpunction})', r'<nbsp>\1', text)
            
        else:  # mode == 'nobr'
            # Wrap space and interpunction with <nobr> tag
            return re.sub(rf'\s+({interpunction})', r'<nobr> \1</nobr>', text)