from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
from typing import Optional

class GoogleAuth:
    def __init__(self, credentials_file: str, token_file: str, scopes: list[str]):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = scopes
        self.credentials: Optional[Credentials] = None

    def authenticate(self) -> Credentials:
        """Handles the complete authentication flow."""
        self.credentials = self._load_credentials()
        
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self._refresh_credentials()
            else:
                self._new_authentication()
            
            self._save_credentials()
        
        return self.credentials

    def _load_credentials(self) -> Optional[Credentials]:
        """Load credentials from token file if it exists."""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'rb') as token:
                    return pickle.load(token)
            except Exception as e:
                print(f"Error loading credentials: {e}")
                if os.path.exists(self.token_file):
                    os.remove(self.token_file)
        return None

    def _refresh_credentials(self) -> None:
        """Refresh expired credentials."""
        try:
            self.credentials.refresh(Request())
        except Exception as e:
            print(f"Error refreshing credentials: {e}")
            self._new_authentication()

    def _new_authentication(self) -> None:
        """Perform new authentication flow."""
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file, self.scopes)
            self.credentials = flow.run_local_server(port=0)
        except Exception as e:
            raise Exception(f"Authentication failed: {e}")

    def _save_credentials(self) -> None:
        """Save credentials to token file."""
        try:
            with open(self.token_file, 'wb') as token:
                pickle.dump(self.credentials, token)
        except Exception as e:
            print(f"Error saving credentials: {e}") 