# BabelSheet

BabelSheet is an automated translation tool for Google Sheets that uses AI to provide high-quality translations while maintaining consistency through a term base.

## Features

- Seamless Google Sheets Integration
  - Direct read/write access to your spreadsheets
  - Automated column management
  - Real-time updates
- Advanced Translation Capabilities
  - AI-powered translations with context awareness
  - Support for cloud LLMs (OpenAI, Anthropic Claude)
  - Optional local LLM support
- Quality Control
  - Automated term base management
  - Quality assurance checks
  - Parallel validation processing
  - Strict non-translatable terms validation
- Efficient Processing
  - Batch processing with configurable size
  - Immediate batch updates
  - Error resilience

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

## Sheet Structure

### Translation Sheets
- Each sheet should have these columns:
  - A key column (usually 'key' or 'id')
  - Source language column (e.g., 'en' for English)
  - Target language columns (e.g., 'es' for Spanish)
  - Optional context columns (matched by patterns in config)

### Term Base Sheet
- Must contain these columns:
  - EN TERM: The English term
  - COMMENT: Context or usage notes
  - TRANSLATION_XX: Translation columns (e.g., TRANSLATION_ES for Spanish)

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
  api_key: ""  # Set via LLM_API_KEY environment variable
  model: "o1-mini"  # The LLM model to use
  temperature: 0.3  # Controls randomness in responses
  api_url: "https://api.openai.com/v1"  # LLM API endpoint
  max_retries: 3    # Maximum retry attempts
  retry_delay: 1    # Initial delay between retries
  batch_size: 50    # Number of rows to process in a batch
  batch_delay: 1    # Delay between batches

qa:
  max_length: 1000  # Maximum length for QA validation
  non_translatable_patterns:
    - start: "{["   # Matches {[TERM]}
      end: "]}"
    - start: "<"    # Matches <TERM>
      end: ">"

term_base:
  sheet_name: "term_base"  # Name of the sheet containing term base
  columns:
    term: "EN TERM"
    comment: "COMMENT"
    translation_prefix: "TRANSLATION_"
```

Place the downloaded credentials.json file in the config directory.

## Usage

1. LLM Compatibility
2. Batch Processing
3. Quality Assurance
4. Error Handling
5. Troubleshooting
6. Development
7. Contributing
8. License

## LLM Compatibility

BabelSheet supports any LLM that provides an OpenAI-compatible API endpoint. Here are some examples:

### Local LLMs

NOTE: Local LLMs are not recommended for production use, as they are not as good as top-tier cloud LLMs, like OpenAI, Anthropic, or Google.

1. **LM Studio**:
```yaml
llm:
  api_key: "not-needed"  # Can be any string
  model: "local-model"   # Will be ignored, model is selected in LM Studio
  api_url: "http://localhost:1234/v1"  # Default LM Studio port
```

2. **Ollama**:
```yaml
llm:
  api_key: "not-needed"
  model: "mistral"  # Or any other model you have pulled
  api_url: "http://localhost:11434/v1"  # Ollama with OpenAI compatibility
```

### Cloud Services

1. **Anthropic Claude**:
```yaml
llm:
  api_key: "your-anthropic-key"
  model: "claude-3-5-sonnet-20241022"  # or claude-3-5-sonnet-20240620
  api_url: "https://api.anthropic.com/v1"
```

2. **OpenAI**:
```yaml
llm:
  api_key: "your-openai-key"
  model: "o1-mini"  # or gpt-4o, gpt-4o-mini, o1-preview
  api_url: "https://api.openai.com/v1"
```

3. **Azure OpenAI**:
```yaml
llm:
  api_key: "your-azure-key"
  model: "your-deployment-name"
  api_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment-name"
```


4. **Set your LLM API key**:

On Windows (PowerShell):
```powershell
$env:LLM_API_KEY="your-api-key"
```

On Unix/MacOS:
```bash
export LLM_API_KEY=your-api-key
```

## Usage

1. Prepare your Google Sheet:
   - Create sheets for translation
   - Add a 'term_base' sheet with columns: EN TERM, COMMENT, TRANSLATION_XX
   - Make sure you have 'key', 'en', and target language columns

2. Initialize the tool:
```bash
python -m babelsheet init
```

3. Translate missing texts:
```bash
# Basic translation
python -m babelsheet translate --sheet-id="your-sheet-id" --target-langs="es,fr,de"

# With batch processing options
python -m babelsheet translate \
    --sheet-id="your-sheet-id" \
    --target-langs="es,fr,de" \
    --force  # Add missing columns without confirmation

# Preview changes
python -m babelsheet translate --sheet-id="your-sheet-id" --target-langs="es" --dry-run

# Translate with specific batch size
# Edit config.yaml to set batch_size: 3
python -m babelsheet translate --sheet-id="your-sheet-id" --target-langs="es,fr,de"
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
  - Optional context columns (matched by patterns in config)

### Term Base Sheet
- Must contain these columns:
  - EN TERM: The English term
  - COMMENT: Context or usage notes
  - TRANSLATION_XX: Translation columns (e.g., TRANSLATION_ES for Spanish)

## Batch Processing

The application supports batch processing of translations to optimize API usage and performance. For best translation quality, we recommend:
- Batch size: 5-20 rows (default: 10)
- Configure in config.yaml:
```yaml
llm:
  batch_size: 10    # Recommended range: 5-20
  batch_delay: 1    # Delay between batches in seconds
```


2. **Batch Flow**:
   - Texts are grouped into batches
   - Each batch is translated together
   - Translations are validated in parallel
   - Results are written to Google Sheets immediately
   - Process continues with next batch

3. **Benefits**:
   - Efficient resource usage
   - Immediate feedback
   - Progress visibility
   - Error resilience

## Quality Assurance

The tool automatically checks:
- Format consistency
- Markup preservation ([],{})
- Term base compliance
- Character limits
- Newline handling
- Cultural appropriateness
- Translation accuracy

## Error Handling

- Authentication errors will prompt for re-authentication
- Translation failures are retried with exponential backoff
- Batch failures don't affect other batches
- QA issues trigger retranslation
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
   - Check if local LLM server is running
   - Verify model name matches your setup

4. Batch processing issues:
   - Check batch_size in config.yaml
   - Verify network stability
   - Check LLM server capacity
   - Monitor memory usage

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

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

