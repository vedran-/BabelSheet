# Automated Localization System Architecture

## 1. Overview
The application will be a Python app, named BabelSheet, done with domain-driven design (DDD) principles, that handles the entire localization workflow:
- Google Sheets integration (read/write)
- Missing translations detection
- AI-powered translation with context
- Batch processing
- Term base management
- Quality assurance checks

## 2. Core Components

### 2.1 Google Sheets Handler
```python
class GoogleSheetsHandler:
    def __init__(self, credentials_file: str, spreadsheet_id: str)
    def authenticate()
    def get_all_sheets() -> List[str]
    def read_sheet(sheet_name: str) -> pd.DataFrame
    def write_translations(sheet_name: str, updates: Dict[str, Any])
    def batch_write_translations(updates: Dict[str, Dict[str, Any]])
```

### 2.2 Translation Manager
```python
class TranslationManager:
    def __init__(self, openai_api_key: str, term_base_file: str)
    def detect_missing_translations(df: pd.DataFrame) -> Dict[str, List[int]]
    def create_translation_prompt(text: str, context: str, term_base: Dict) -> str
    def translate_text(text: str, target_lang: str, context: str) -> str
    def batch_translate(texts: List[str], target_lang: str, contexts: List[str]) -> List[str]
```

### 2.3 Term Base Handler
```python
class TermBaseHandler:
    def __init__(self, term_base_file: str)
    def load_term_base() -> Dict[str, Dict[str, str]]
    def get_terms_for_language(lang: str) -> Dict[str, str]
    def update_term_base(new_terms: Dict[str, Dict[str, str]])
```

## 3. Workflow

1. **Initialization**
   - Load configuration (API keys, file paths, sheet IDs)
   - Initialize Google Sheets connection
   - Load term base
   - Set up OpenAI client

2. **Analysis Phase**
   - Fetch all sheets
   - For each sheet:
     - Read all content
     - Identify missing translations
     - Group by language for batch processing
   - Generate translation report

3. **Translation Phase**
   - For each language:
     - Create batches of manageable size
     - Generate context-aware prompts
     - Call OpenAI API with appropriate context
     - Validate translations against term base
     - Queue successful translations

4. **Update Phase**
   - Group translations by sheet
   - Batch update Google Sheets
   - Generate completion report

## 4. Configuration Structure
```yaml
google_sheets:
  credentials_file: "path/to/credentials.json"
  spreadsheet_id: "your-sheet-id"
  batch_size: 50

llm_provider:
  type: "openai"  # or "anthropic" or "custom"
  api_key: "your-api-key"
  api_url: "https://api.openai.com/v1"  # customizable endpoint
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 500

languages:
  source: "en"  # source language
  target: ["es", "ru", "fr", "de"]  # list of target languages to translate into
  
sheet_structure:
  key_column: "key"  # column containing the unique identifier
  text_column: "english"  # column containing the source text
  context_columns:  # columns containing contextual information
    - name: "description"
      type: "context"
    - name: "comment"
      type: "context"
    - name: "trigger"
      type: "technical"  # technical columns won't be translated but preserved

term_base:
  file_path: "path/to/term_base.csv"
  update_frequency: "daily"
  structure:
    term_column: "EN TERM"
    comment_column: "COMMENT"
    translation_prefix: "TRANSLATION_"  # e.g., "TRANSLATION_ES" for Spanish

prompts:
  translation: |
    You are a world-class expert in translating {source_lang} to {target_lang}, 
    specialized for casual mobile games. You will be doing professional-grade 
    translations of in-game texts.
    
    Rules:
    - Use provided term base for consistency
    - Don't translate text between markup characters [] and {}
    - Keep appropriate format (uppercase/lowercase)
    - Replace newlines with \\n
    - Keep translations lighthearted and fun
    - Keep translations concise to fit UI elements
    - Localize all output text
    
  term_base_extraction: |
    Please extract which terms would be worth having in the Term Base for future use.
    Only extract new terms not already present in the term base.

logging:
  level: "INFO"
  file: "translation_log.txt"
  report_dir: "reports"
```

## 5. Error Handling and Recovery
- Rate limiting handling
- API error recovery
- Session persistence
- Checkpoint system for long-running processes
- Validation of translations before updating sheets

## 6. Quality Assurance
- Term base compliance checking
- Format string validation
- Character limit verification
- Special character handling
- HTML/XML tag preservation

## 7. Future Enhancements
- Translation memory system
- Multiple API provider support
- Machine translation confidence scoring
- Interactive review process
- Automated term base suggestions

## 8. Command Line Interface
```bash
python BabelSheet.py --mode=full  # Complete analysis and translation
python BabelSheet.py --mode=analyze  # Only detect missing translations
python BabelSheet.py --mode=translate --sheet="Sheet1"  # Translate specific sheet
python BabelSheet.py --mode=update --batch="batch_001.json"  # Update from saved batch
```