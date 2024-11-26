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

    def get_context_from_row(self, row: pd.Series, context_patterns: List[str], ignore_case: bool = True) -> str:
        """Extract context from all matching columns in a row."""
        contexts = []
        
        for column in row.index:
            # Skip empty values
            if pd.isna(row[column]):
                continue
            
            # Check if column matches any context pattern
            column_name = column.lower() if ignore_case else column
            if any(pattern.lower() in column_name if ignore_case else pattern in column_name 
                   for pattern in context_patterns):
                contexts.append(f"{column}: {row[column]}")
        
        return "\n".join(contexts)

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

    def ensure_language_columns(self, sheet_name: str, languages: List[str], force: bool = False, dry_run: bool = False) -> None:
        """Ensure all required language columns exist in the sheet."""
        df = self.read_sheet(sheet_name)
        existing_columns = df.columns.tolist()
        
        # Find missing language columns
        missing_langs = [lang for lang in languages if lang not in existing_columns]
        
        if missing_langs:
            # Show warning
            print(f"\nWARNING: The following language columns are missing in sheet '{sheet_name}':")
            for lang in missing_langs:
                print(f"  - {lang}")
            
            if not force and not dry_run:
                response = input("\nWould you like to add these columns? [Y/n]: ")
                if response.lower() not in ['', 'y', 'yes']:
                    raise ValueError(f"Cannot proceed without required language columns: {', '.join(missing_langs)}")
            
            if dry_run:
                print("\nWould add the following columns:")
                for lang in missing_langs:
                    print(f"  - {lang}")
                return
            
            # Add new columns to the DataFrame
            for lang in missing_langs:
                df[lang] = ''
                print(f"Added column: {lang}")
            
            # Update the sheet with new columns
            self.update_sheet(sheet_name, df)
            print(f"\nSuccessfully added {len(missing_langs)} new language column(s)")