from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from typing import Optional, Dict, Any, List
import pandas as pd

class GoogleSheetsHandler:
    def __init__(self, credentials: Credentials):
        """Initialize the Google Sheets handler."""
        self.service = build('sheets', 'v4', credentials=credentials)
        self.current_spreadsheet_id: Optional[str] = None

    def set_spreadsheet(self, spreadsheet_id: str) -> None:
        """Set the current spreadsheet ID."""
        self.current_spreadsheet_id = spreadsheet_id

    def get_all_sheets(self) -> List[str]:
        """Get all sheet names from the current spreadsheet."""
        if not self.current_spreadsheet_id:
            raise ValueError("Spreadsheet ID not set")

        result = self.service.spreadsheets().get(
            spreadsheetId=self.current_spreadsheet_id
        ).execute()

        return [sheet['properties']['title'] for sheet in result.get('sheets', [])]

    def read_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Read a sheet and return it as a pandas DataFrame."""
        if not self.current_spreadsheet_id:
            raise ValueError("Spreadsheet ID not set")

        result = self.service.spreadsheets().values().get(
            spreadsheetId=self.current_spreadsheet_id,
            range=sheet_name
        ).execute()

        values = result.get('values', [])
        if not values:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(values[1:], columns=values[0])
        return df

    def update_sheet(self, sheet_name: str, df: pd.DataFrame) -> None:
        """Update an entire sheet with new data."""
        if not self.current_spreadsheet_id:
            raise ValueError("Spreadsheet ID not set")

        # Convert DataFrame to list of lists
        values = [df.columns.tolist()]  # Header row
        values.extend(df.values.tolist())  # Data rows

        # Clear existing content
        self.service.spreadsheets().values().clear(
            spreadsheetId=self.current_spreadsheet_id,
            range=sheet_name
        ).execute()

        # Update with new content
        body = {
            'values': values
        }

        self.service.spreadsheets().values().update(
            spreadsheetId=self.current_spreadsheet_id,
            range=sheet_name,
            valueInputOption='RAW',
            body=body
        ).execute()

    def write_translations(self, sheet_name: str, updates: Dict[str, Any]) -> None:
        """Write translations back to the sheet."""
        if not self.current_spreadsheet_id:
            raise ValueError("Spreadsheet ID not set")

        # Prepare the batch update
        batch_data = []
        for cell_range, value in updates.items():
            batch_data.append({
                'range': f'{sheet_name}!{cell_range}',
                'values': [[value]]
            })

        body = {
            'valueInputOption': 'USER_ENTERED',
            'data': batch_data
        }

        self.service.spreadsheets().values().batchUpdate(
            spreadsheetId=self.current_spreadsheet_id,
            body=body
        ).execute() 