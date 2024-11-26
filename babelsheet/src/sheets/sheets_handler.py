from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from typing import Optional, Dict, Any, List
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)

class GoogleSheetsHandler:
    def __init__(self, credentials: Credentials):
        """Initialize the Google Sheets handler."""
        self.service = build('sheets', 'v4', credentials=credentials)
        self.current_spreadsheet_id: Optional[str] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._sheet_cache: Dict[str, pd.DataFrame] = {}  # Add cache for DataFrames

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

    def _dump_data(self, message: str, data: Any) -> None:
        """Helper method to dump data in a readable format."""
        if hasattr(logging, 'TRACE'):  # Only dump if TRACE level is enabled
            print(f"Dumping data for {message}")
            print(data)
            if isinstance(data, pd.DataFrame):
                formatted = (
                    f"\nDataFrame Shape: {data.shape}\n"
                    f"Columns: {data.columns.tolist()}\n"
                    f"Data (row by row):\n"
                )
                # Add each row with its index
                for idx, row in data.iterrows():
                    formatted += f"\nRow {idx + 2}: {{\n"  # +2 because idx starts at 0 and we have a header row
                    for col in data.columns:
                        formatted += f"  {col}: {row[col]}\n"
                    formatted += "}\n"
            elif isinstance(data, list) and data and isinstance(data[0], list):
                # This is for raw sheet data
                formatted = f"\nRaw Sheet Data ({len(data)} rows):\n"
                if data:
                    formatted += f"\nHeaders: {data[0]}\n"
                    for idx, row in enumerate(data[1:], start=2):  # Start at 2 to match sheet row numbers
                        formatted += f"\nRow {idx}: {{\n"
                        for col_idx, value in enumerate(row):
                            col_name = data[0][col_idx] if col_idx < len(data[0]) else f"Column {col_idx}"
                            formatted += f"  {col_name}: {value}\n"
                        formatted += "}\n"
            else:
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
            self.logger.trace(f"{message}:\n{formatted}")

    def _get_sheet_data(self, sheet_name: str) -> pd.DataFrame:
        """Get sheet data, using cache if available."""
        if sheet_name in self._sheet_cache:
            self.logger.debug(f"Using cached data for sheet: {sheet_name}")
            return self._sheet_cache[sheet_name]

        df = self.read_sheet(sheet_name)
        self._sheet_cache[sheet_name] = df
        return df

    def clear_cache(self):
        """Clear the sheet cache."""
        self._sheet_cache.clear()

    def read_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Read a sheet and return it as a pandas DataFrame."""
        self.logger.debug(f"Reading sheet: {sheet_name}")
        if not self.current_spreadsheet_id:
            self.logger.error("Spreadsheet ID not set")
            raise ValueError("Spreadsheet ID not set")

        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.current_spreadsheet_id,
                range=sheet_name
            ).execute()
            
            values = result.get('values', [])
            self.logger.debug(f"Retrieved {len(values)} rows from sheet")
            self._dump_data("Raw sheet data", values)
            
            if not values:
                self.logger.warning(f"No data found in sheet: {sheet_name}")
                return pd.DataFrame()

            # Add validation for mismatched columns
            header = values[0]
            data = values[1:]
            
            # Pad rows that have fewer columns than the header
            max_cols = len(header)
            padded_data = []
            for row in data:
                # Explicitly convert empty or missing values to empty string
                padded_row = []
                for i in range(max_cols):
                    if i < len(row):
                        padded_row.append(row[i] if row[i] != '' else None)
                    else:
                        padded_row.append(None)
                padded_data.append(padded_row)
            
            df = pd.DataFrame(padded_data, columns=header)
            self.logger.debug(f"Created DataFrame with shape: {df.shape}")
            self._dump_data("Processed DataFrame", df)
            
            # Convert empty strings to None
            df = df.replace(r'^\s*$', None, regex=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading sheet: {str(e)}")
            raise

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

    def process_sheet(self, sheet_name: str, target_langs: List[str]) -> Dict[str, List[int]]:
        """Process sheet and return missing translations for each language."""
        self.logger.debug(f"Processing sheet: {sheet_name}")
        self.logger.debug(f"Target languages: {target_langs}")
        
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        
        # Normalize target languages
        target_langs = [lang.lower() for lang in target_langs]
        self.logger.debug(f"Normalized target languages: {target_langs}")
        
        # Use cached data if available
        df = self._get_sheet_data(sheet_name)
        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        self.logger.debug(f"Processing sheet with {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Initialize result dictionary
        missing_translations = {lang: [] for lang in target_langs}
        
        # First check if language columns exist
        for lang in target_langs:
            matching_cols = [col for col in df.columns if 
                            col == lang or 
                            col.startswith(f'translation_{lang}') or 
                            col.endswith(f'_{lang}')]
            
            self.logger.debug(f"Found matching columns for {lang}: {matching_cols}")
            
            if not matching_cols:
                # If no column exists for this language, mark all rows as missing
                self.logger.debug(f"No column found for language {lang}")
                # Add all rows (adding 2 to account for 0-based index and header row)
                missing_translations[lang] = list(range(2, len(df) + 2))
                continue
            
            # If column exists, check for missing translations
            col = matching_cols[0]  # Use the first matching column
            for idx, row in df.iterrows():
                value = row[col]
                self.logger.debug(f"Checking value for {lang} at row {idx + 2}: '{value}'")
                
                # Check if translation is missing (None, NaN, empty string, or whitespace)
                if pd.isna(value) or str(value).strip() == '':
                    missing_translations[lang].append(idx + 2)
                    self.logger.debug(f"Missing translation in {sheet_name} for language {lang} at row {idx + 2}")
        
        self.logger.debug(f"Missing translations: {missing_translations}")
        return missing_translations

    def ensure_language_columns(self, sheet_name: str, languages: List[str], force: bool = False, dry_run: bool = False) -> None:
        """Ensure all required language columns exist in the sheet."""
        # Use cached data if available
        df = self._get_sheet_data(sheet_name)
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
            # Update cache with new columns
            self._sheet_cache[sheet_name] = df
            print(f"\nSuccessfully added {len(missing_langs)} new language column(s)")

    def get_sheet_as_dataframe(self, sheet_name: str) -> pd.DataFrame:
        """Read sheet data and return as pandas DataFrame."""
        # Use cached version if available
        return self._get_sheet_data(sheet_name)
        
    def update_sheet_from_dataframe(self, sheet_name: str, df: pd.DataFrame) -> None:
        """Update sheet with data from DataFrame."""
        if not self.current_spreadsheet_id:
            raise ValueError("Spreadsheet ID not set")
            
        # Convert DataFrame to values list
        headers = df.columns.tolist()
        values = [headers] + df.fillna('').values.tolist()
        
        body = {
            'values': values
        }
        
        range_name = f"{sheet_name}"
        self.service.spreadsheets().values().update(
            spreadsheetId=self.current_spreadsheet_id,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute()