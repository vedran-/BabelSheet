# BabelSheet

BabelSheet is an automated translation tool for Google Sheets that uses AI to provide high-quality translations while maintaining consistency through a term base.

## Features

- Direct integration with Google Sheets
- AI-powered translations with context awareness
- Automated term base management
- Quality assurance checks
- Support for custom LLM endpoints
- Google Translate fallback
- Batch processing capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/babelsheet.git
cd babelsheet
```

2. Create and activate a virtual environment:

On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

On Unix/MacOS:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

1. Create a Google Cloud project and enable the Google Sheets API:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing one
   - Enable Google Sheets API and Google Drive API
   - Create OAuth 2.0 credentials
   - Download the credentials file

2. Set up the configuration:
```bash
# Create config directory
mkdir -p config

# Copy and rename credentials file
cp path/to/downloaded/credentials.json config/credentials.json

# Copy and customize config file
cp config/config.yaml.example config/config.yaml
```

3. Edit config/config.yaml to match your needs:
```yaml
google_sheets:
  credentials_file: "credentials.json"
  token_file: "token.pickle"
  scopes:
    - "https://www.googleapis.com/auth/spreadsheets"
    - "https://www.googleapis.com/auth/drive.readonly"

llm:
  provider: "openai"
  api_key: ""  # Set via LLM_API_KEY environment variable
  model: "gpt-4"
  temperature: 0.3
  api_url: "https://api.openai.com/v1"  # Optional: Change for custom LLM endpoint

term_base:
  sheet_name: "term_base"  # Name of the sheet containing term base
  columns:
    term: "EN TERM"
    comment: "COMMENT"
    translation_prefix: "TRANSLATION_"
```

4. Set your LLM API key:

On Windows (PowerShell):
```powershell
$env:LLM_API_KEY="your-api-key"
```

On Unix/MacOS:
```bash
export LLM_API_KEY=your-api-key
```

## First Run

On first run, the tool will:
1. Open your browser for Google authentication
2. Save the authentication token for future use
3. Create necessary files and directories

```bash
python -m babelsheet init
```

## Usage

1. Prepare your Google Sheet:
   - Create sheets for translation
   - Add a 'term_base' sheet with columns: EN TERM, COMMENT, TRANSLATION_XX
   - Make sure you have 'key', 'en', and target language columns

2. Translate missing texts:
```bash
# Basic translation
python -m babelsheet translate --sheet-id="your-sheet-id" --target-langs="es,fr,de"

# With options
python -m babelsheet translate \
    --sheet-id="your-sheet-id" \
    --target-langs="es,fr,de" \
    --force  # Add missing columns without confirmation
    --dry-run  # Show what would be done without making changes

# Preview changes
python -m babelsheet translate --sheet-id="your-sheet-id" --target-langs="es" --dry-run
```

### CLI Options
- `--sheet-id`: Google Sheet ID to process
- `--target-langs`: Comma-separated list of target languages
- `--force`: Add missing language columns without confirmation
- `--dry-run`: Show what would be done without making changes

## Sheet Structure

### Translation Sheets
- Each sheet should have these columns:
  - A key column (usually 'key' or 'id')
  - Source language column (e.g., 'en' for English)
  - Target language columns (e.g., 'es' for Spanish)
  - Optional context column

### Term Base Sheet
- Must contain these columns:
  - EN TERM: The English term
  - COMMENT: Context or usage notes
  - TRANSLATION_XX: Translation columns (e.g., TRANSLATION_ES for Spanish)

## Quality Assurance

The tool automatically checks:
- Format consistency
- Markup preservation ([],{})
- Term base compliance
- Character limits
- Newline handling

## Error Handling

- Authentication errors will prompt for re-authentication
- Translation failures will fall back to Google Translate if enabled
- QA issues will trigger a retry with specific feedback
- All errors are logged for review

## Troubleshooting

Common issues:

1. Module not found errors:
   - Make sure you've installed the package with `pip install -e .`
   - Verify you're in the correct directory

2. Authentication errors:
   - Check that credentials.json is in the config directory
   - Delete token.pickle to force re-authentication
   - Verify API access in Google Cloud Console

3. LLM API errors:
   - Verify LLM_API_KEY environment variable is set
   - Check API endpoint URL in config.yaml
   - Verify API key permissions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Development

### Running Tests

1. Unit Tests:
```bash
pytest babelsheet/tests/
```

2. Integration Tests:
To run integration tests, you need:
- A test Google Sheet with appropriate permissions
- Valid credentials.json
- LLM API key

```bash
# Set up test environment
export TEST_SHEET_ID="your-test-sheet-id"
export LLM_API_KEY="your-api-key"

# Run integration tests
pytest babelsheet/tests/ --integration
```

Note: Integration tests will modify the test sheet. Use a dedicated test sheet, not a production one. 