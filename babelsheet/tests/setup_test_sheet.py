from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from babelsheet.src.utils.auth import GoogleAuth
from babelsheet.src.sheets.sheets_handler import GoogleSheetsHandler

def create_test_sheet():
    """Create and populate test Google Sheet."""
    # Initialize auth
    auth = GoogleAuth(
        credentials_file="config/credentials.json",
        token_file="config/token.pickle",
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    
    credentials = auth.authenticate()
    
    # Create new spreadsheet
    service = build('sheets', 'v4', credentials=credentials)
    spreadsheet = {
        'properties': {
            'title': 'BabelSheet Test Sheet'
        },
        'sheets': [
            {'properties': {'title': 'Sheet1'}},
            {'properties': {'title': 'Sheet2'}},
            {'properties': {'title': 'Sheet3'}},
            {'properties': {'title': '[E] Term Base'}}
        ]
    }
    
    spreadsheet = service.spreadsheets().create(body=spreadsheet).execute()
    spreadsheet_id = spreadsheet['spreadsheetId']
    print(f"Created test sheet with ID: {spreadsheet_id}")
    
    # Initialize sheet handler
    handler = GoogleSheetsHandler(credentials)
    handler.set_spreadsheet(spreadsheet_id)
    
    # Prepare test data
    sheet1_data = pd.DataFrame({
        'key': ['key1', 'key2', 'key3', 'key4', 'key5', 'key6'  ],
        'en': ['Hello', 'Start Game', 'Continue', 'Defeat the boss!', 'More coins, more problems!', 'Slow Joe was here!'],
        'es': ['', '', '', '', '', ''],
        'fr': ['', '', '', '', '', ''],
        'de': ['', '', '', '', '', ''],
        'context': ['', 'Button text', 'Menu option', 'Game text', 'Dialogue', 'Comment']
    })
    
    sheet2_data = pd.DataFrame({
        'key': ['key1', 'key2', 'key3', 'key4'],
        'en': ['{[COINS]} collected', '[PLAYER] wins!', 'Level {[NUMBER]}', 'You have <NUMBER> coins'],
        'es': ['', '', '', ''],
        'fr': ['', '', '', ''],
        'de': ['', '', '', ''],
        'context': ['HUD text', 'Victory screen', 'Level indicator', 'Amount of coins the player has']
    })
    
    sheet3_data = pd.DataFrame({
        'key': ['key1', 'key2', 'key3'],
        'en': ['Awesome!', 'Defeat the boss', 'Epic reward!'],
        'es': ['', '', ''],
        'fr': ['', '', ''],
        'de': ['', '', ''],
        'context': ['Casual exclamation', 'Tutorial text', 'Reward popup']
    })
    
    term_base_data = pd.DataFrame({
        'EN TERM': ['coins', 'boss'],
        'COMMENT': ['Game currency', 'Enemy type'],
        'es': ['monedas', 'jefe'],
        'fr': ['pièces', 'patron'],
        'de': ['Münzen', 'Boss']
    })
    
    # Update sheets
    handler.update_sheet('Sheet1', sheet1_data)
    handler.update_sheet('Sheet2', sheet2_data)
    handler.update_sheet('Sheet3', sheet3_data)
    handler.update_sheet('[E] Term Base', term_base_data)
    
    print("Test data populated successfully")
    return spreadsheet_id

if __name__ == "__main__":
    spreadsheet_id = create_test_sheet()
    print("\nTest sheet setup complete!")
    print(f"Use this spreadsheet ID for testing: {spreadsheet_id}")
    print(f"Spreadsheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")