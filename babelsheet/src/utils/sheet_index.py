import re

class SheetIndex:
    """Helper class to handle sheet indexing consistently."""
    
    @staticmethod
    def to_column_letter(n: int) -> str:
        """Convert column number to letter (1 = A, 27 = AA)."""
        result = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            result = chr(65 + remainder) + result
        return result
    
    @staticmethod
    def from_column_letter(col: str) -> int:
        """Convert column letter to number (A = 1, AA = 27)."""
        result = 0
        for char in col:
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result
    
    @staticmethod
    def to_sheet_row(df_index: int) -> int:
        """Convert DataFrame index to sheet row number."""
        return df_index + 2  # +2 for 1-based index and header
    
    @staticmethod
    def to_df_index(sheet_row: int) -> int:
        """Convert sheet row number to DataFrame index."""
        return sheet_row - 2  # -2 for 1-based index and header
    
    @staticmethod
    def get_cell_reference(col_idx: int, row_idx: int) -> str:
        """Get A1 notation for cell reference."""
        return f"{SheetIndex.to_column_letter(col_idx + 1)}{SheetIndex.to_sheet_row(row_idx)}"
    
    @staticmethod
    def parse_cell_reference(cell_ref: str) -> tuple[str, int]:
        """Parse A1 notation into column letter and row number."""
        match = re.match(r'([A-Z]+)(\d+)', cell_ref)
        if not match:
            raise ValueError(f"Invalid cell reference: {cell_ref}")
        col, row = match.groups()
        return col, int(row) 