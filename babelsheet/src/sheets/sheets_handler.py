from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import logging
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from ..utils.sheet_index import SheetIndex
from ..utils.auth import get_credentials

logger = logging.getLogger(__name__)

@dataclass
class CellData:
    value: Any
    is_synced: bool = True
    
    def __str__(self) -> str:
        return str(self.value)

class SheetData:
    def __init__(self, name: str, data: Dict[Tuple[int, int], CellData]):
        self.name = name
        self.data = data  # (row, col) -> CellData
        self._max_row = max([pos[0] for pos in data.keys()]) if data else 0
        self._max_col = max([pos[1] for pos in data.keys()]) if data else 0
        self.column_names = [data.get((0, col), CellData('')).value for col in range(self._max_col + 1)]
    
    def get_cell(self, row: int, col: int) -> Optional[CellData]:
        return self.data.get((row, col))
    
    def set_cell(self, row: int, col: int, value: Any):
        cell = CellData(value, is_synced=False)
        self.data[(row, col)] = cell
        self._max_row = max(self._max_row, row)
        self._max_col = max(self._max_col, col)
    
    def get_unsynced_cells(self) -> Dict[str, Any]:
        """Returns a dict of A1 notation -> value for unsynced cells"""
        updates = {}
        for (row, col), cell in self.data.items():
            if not cell.is_synced:
                a1_notation = f"{SheetIndex.to_column_letter(col+1)}{row+1}"
                updates[a1_notation] = cell.value
        return updates
    
    def add_column(self, col_idx: int, column_title: str):
        """Add a new column to the sheet"""
        self._max_col += 1
        self.data[(0, col_idx)] = CellData(column_title)
    
    def mark_synced(self, cells: List[str]):
        """Mark cells as synced using A1 notation"""
        for cell in cells:
            col, row = SheetIndex.parse_cell_reference(cell)
            col_idx = SheetIndex.from_column_letter(col) - 1
            row_idx = SheetIndex.to_df_index(row)
            if (row_idx, col_idx) in self.data:
                self.data[(row_idx, col_idx)].is_synced = True

    @classmethod
    def from_dataframe(cls, name: str, df: pd.DataFrame) -> 'SheetData':
        """Create SheetData from DataFrame"""
        data = {}

        # Store all data as before
        for row in range(len(df.index)):
            for col in range(len(df.columns)):
                value = df.iloc[row, col]
                if pd.notna(value):  # Only store non-empty cells
                    data[(row, col)] = CellData(value)
        return cls(name, data)

class SheetsHandler:
    def __init__(self, credentials_path: str = None, credentials: Optional[Credentials] = None):
        self.service = None
        self.current_spreadsheet_id = None
        self._sheets: Dict[str, SheetData] = {}  # sheet_name -> SheetData
        self.credentials_path = credentials_path
        self.credentials = credentials
        self._initialize_service()

    def _initialize_service(self) -> None:
        """Initialize the Google Sheets service."""
        try:
            if self.credentials is None:
                self.credentials = get_credentials()
            
            self.service = build('sheets', 'v4', credentials=self.credentials)
            logger.debug("Successfully initialized Google Sheets service")
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets service: {str(e)}")
            raise
    
    def load_spreadsheet(self, spreadsheet_id: str):
        """Load entire spreadsheet into memory"""
        self.current_spreadsheet_id = spreadsheet_id
        logger.info(f"Loading entire spreadsheet {spreadsheet_id}")
        
        try:
            # Get all sheet names
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id
            ).execute()
            
            sheets = spreadsheet.get('sheets', [])
            for sheet in sheets:
                sheet_name = sheet['properties']['title']
                logger.debug(f"Loading sheet: {sheet_name}")
                
                # Load sheet data
                result = self.service.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=sheet_name
                ).execute()
                
                values = result.get('values', [])
                df = pd.DataFrame(values)
                #logger.debug(f"Loaded sheet {sheet_name}:\n{df}")
                
                # Convert to our internal format
                self._sheets[sheet_name] = SheetData.from_dataframe(sheet_name, df)
                
            logger.info(f"Loaded {len(self._sheets)} sheets into memory: {self.get_sheet_names()}")

        except Exception as e:
            logger.error(f"Error loading spreadsheet: {str(e)}")
            raise
    
    def save_changes(self):
        """Save all unsynced changes to Google Sheets"""
        if not self.current_spreadsheet_id:
            raise ValueError("No spreadsheet loaded")
            
        logger.info("Saving unsynced changes to Google Sheets")
        
        for sheet_name, sheet_data in self._sheets.items():
            updates = sheet_data.get_unsynced_cells()
            if not updates:
                logger.debug(f"No changes to sync for sheet: {sheet_name}")
                continue
                
            logger.info(f"Syncing {len(updates)} changes in sheet: {sheet_name}")
            
            # Convert to batch request format
            batch_data = [
                {
                    'range': f'{sheet_name}!{cell_ref}',
                    'values': [[value]]
                }
                for cell_ref, value in updates.items()
            ]
            
            body = {
                'valueInputOption': 'RAW',
                'data': batch_data
            }
            
            try:
                result = self.service.spreadsheets().values().batchUpdate(
                    spreadsheetId=self.current_spreadsheet_id,
                    body=body
                ).execute()
                
                if 'totalUpdatedCells' in result:
                    # Mark cells as synced
                    sheet_data.mark_synced(updates.keys())
                    logger.info(f"Successfully synced {result['totalUpdatedCells']} cells in {sheet_name}")
                else:
                    logger.warning(f"Update completed but no cell count returned for {sheet_name}")
                    
            except Exception as e:
                logger.error(f"Error syncing changes for sheet {sheet_name}: {str(e)}")
                raise
    
    def get_sheet_names(self) -> List[str]:
        """Get all sheet names"""
        return list(self._sheets.keys())

    def get_sheet_data(self, sheet_name: str) -> SheetData:
        if sheet_name not in self._sheets:
            raise ValueError(f"Sheet {sheet_name} not loaded")
        return self._sheets[sheet_name]
    
    def set_cell_value(self, sheet_name: str, cell_ref: str, value: Any):
        """Set a cell value in memory"""
        if sheet_name not in self._sheets:
            raise ValueError(f"Sheet {sheet_name} not loaded")
            
        col, row = SheetIndex.parse_cell_reference(cell_ref)
        col_idx = SheetIndex.from_column_letter(col) - 1
        row_idx = SheetIndex.to_df_index(row)
        
        self._sheets[sheet_name].set_cell(row_idx, col_idx, value)
        
    def get_cell_value(self, sheet_name: str, cell_ref: str) -> Any:
        """Get a cell value from memory"""
        if sheet_name not in self._sheets:
            raise ValueError(f"Sheet {sheet_name} not loaded")
            
        col, row = SheetIndex.parse_cell_reference(cell_ref)
        col_idx = SheetIndex.from_column_letter(col) - 1
        row_idx = SheetIndex.to_df_index(row)
        
        cell = self._sheets[sheet_name].get_cell(row_idx, col_idx)
        return cell.value if cell else None
    
    def add_new_column(self, sheet_name: str, column_letter: str, column_title: str):
        """Add a new column to the sheet"""
        col_idx = SheetIndex.from_column_letter(column_letter) - 1
        self._sheets[sheet_name].add_column(col_idx, column_title)

    def get_column_indexes(self, sheet_data: SheetData, column_names: List[str]) -> List[int]:
        """Get the indexes of the context columns"""
        return [i for i, col in enumerate(sheet_data.column_names) 
            if any(col_name.lower() == col.lower() for col_name in column_names)]
