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
    
    def is_empty(self) -> bool:
        return self is None or self.value is None or str(self.value).strip() == ''

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

                df.attrs['context_column_indexes'] = self.get_column_indexes(df, self.ctx.config['context_columns']['patterns'])

                self._sheets[sheet_name] = df
                
            logger.debug(f"Loaded {len(self._sheets)} sheets into memory: {self.get_sheet_names()}")

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

    def get_unsynced_cells(self, sheet_data: pd.DataFrame) -> Dict[str, Any]:
        """Get all unsynced cells"""
        updates = {}
        for rowIdx, row in sheet_data.iterrows():
            for colIdx, cell in enumerate(row):
                if cell and not cell.is_synced:
                    updates[f'{SheetIndex.to_column_letter(colIdx + 1)}{rowIdx + 1}'] = cell

        return updates

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

    def add_new_column(self, sheet_data: pd.DataFrame, column_title: str) -> int:
        """Add a new column to end of the sheet"""
        empty_column = [CellData(None) for _ in range(len(sheet_data))]
        idx = len(sheet_data.columns)
        sheet_data.insert(idx, idx, empty_column)
        sheet_data.iloc[0, idx] = CellData(column_title, is_synced=False)
        return idx
    
    def add_new_row(self, sheet_data: pd.DataFrame, row_values: List[Any]) -> int:
        """Add a new row to end of the sheet"""
        idx = len(sheet_data.index)
        sheet_data.loc[idx] = row_values
        return idx


    def get_sheet_names(self) -> List[str]:
        """Get all sheet names"""
        return list(self._sheets.keys())

    def get_sheet_data(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name not in self._sheets:
            raise ValueError(f"Sheet {sheet_name} not loaded")
        return self._sheets[sheet_name]

    def get_column_names(self, sheet_data: pd.DataFrame, lower_case: bool = False) -> List[str]:
        """Get all column names from 1st row"""
        return [cell.value.lower() if lower_case else cell.value for cell in sheet_data.iloc[0]]        

    def get_column_indexes(self, sheet_data: pd.DataFrame, column_names: List[str], create_if_missing: bool = False) -> List[int]:
        """Get the indexes of the context columns"""
        sheet_column_names = self.get_column_names(sheet_data, lower_case=True)
        column_indexes = []
        
        for col_name in column_names:
            try:
                col_idx = sheet_column_names.index(col_name.lower())
            except ValueError:
                if create_if_missing:
                    col_idx = self.add_new_column(sheet_data, col_name)
                else:
                    continue
            column_indexes.append(col_idx)

        return column_indexes

    def get_row_context(self, sheet_data: pd.DataFrame, row_idx: int) -> List[str]:
        """Get the context of a row"""
        column_names = self.get_column_names(sheet_data)
        return [{column_names[col_idx]: sheet_data.iloc[row_idx][col_idx].value} for col_idx in sheet_data.attrs['context_column_indexes']]
