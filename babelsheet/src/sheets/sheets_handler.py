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

class SheetsHandler:
    def __init__(self, ctx, credentials: Credentials = None):
        self.ctx = ctx
        self.service = None
        self._sheets: Dict[str, pd.DataFrame] = {}  # sheet name -> sheet data
        self.credentials = credentials
        self._initialize_service()
        self.load_spreadsheet(ctx.spreadsheet_id)

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

                if sheet_name in self._sheets:
                    logger.error(f"Sheet {sheet_name} already loaded, skipping")
                    continue
                
                # Load sheet data
                result = self.service.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=sheet_name
                ).execute()
                
                values = [[CellData(value) for value in row] for row in result.get('values', [])]
                df = pd.DataFrame(values)

                self._sheets[sheet_name] = df
                
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
            updates = self.get_unsynced_cells(sheet_data)
            if not updates:
                logger.debug(f"No changes to sync for sheet: {sheet_name}")
                continue
                
            logger.info(f"Syncing {len(updates)} changes in sheet: {sheet_name}")
            
            # Convert to batch request format
            batch_data = [
                {
                    'range': f'{sheet_name}!{cell_ref}',
                    'values': [[value.value]]
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
                    for cell_ref, cell in updates.items():
                        cell.is_synced = True

                    logger.info(f"Successfully synced {result['totalUpdatedCells']} cells in {sheet_name}")
                else:
                    logger.critical(f"Update completed but no cell count returned for {sheet_name}")
                    
            except Exception as e:
                logger.error(f"Error syncing changes for sheet {sheet_name}: {str(e)}")
                raise
    
    def get_sheet_names(self) -> List[str]:
        """Get all sheet names"""
        return list(self._sheets.keys())

    def get_sheet_data(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name not in self._sheets:
            raise ValueError(f"Sheet {sheet_name} not loaded")
        return self._sheets[sheet_name]
    

    def modify_cell_data(self, sheet_name: str, row: int, col: int, value: Any):
        """Modify a cell data in memory"""
        if sheet_name not in self._sheets:
            raise ValueError(f"Sheet {sheet_name} not loaded")

        cell = self._sheets[sheet_name].iloc[row, col]
        if cell is None:
            cell = CellData(value)
            self._sheets[sheet_name].iloc[row, col] = cell
        else:
            cell.value = value
        cell.is_synced = False

    def get_column_indexes(self, sheet_data: pd.DataFrame, column_names: List[str]) -> List[int]:
        """Get the indexes of the context columns"""

        column_indexes = []
        for colIdx in sheet_data.columns:
            column_name = sheet_data.iloc[0, colIdx].value
            if any(col_name.lower() == column_name.lower() for col_name in column_names):
                column_indexes.append(colIdx)

        return column_indexes

    def get_unsynced_cells(self, sheet_data: pd.DataFrame) -> Dict[str, Any]:
        """Get all unsynced cells"""
        updates = {}
        for rowIdx, row in sheet_data.iterrows():
            for colIdx, cell in enumerate(row):
                if cell and not cell.is_synced:
                    updates[f'{SheetIndex.to_column_letter(colIdx + 1)}{rowIdx + 2}'] = cell

        return updates


    #########################################################
    ########## Cell value operations #######################
    #########################################################
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

