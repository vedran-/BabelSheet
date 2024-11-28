# BabelSheet Code Structure and Architecture

## Domain-Driven Design (DDD) Architecture

BabelSheet follows Domain-Driven Design principles, organizing code around business domains and their logic. The project is structured into distinct bounded contexts, each handling specific business capabilities.

## Project Structure

```
babelsheet/
├── src/                      # Source code
│   ├── translation/          # Translation domain
│   │   └── translation_manager.py
│   ├── sheets/              # Google Sheets domain
│   │   └── sheets_handler.py
│   ├── term_base/           # Term Base domain
│   │   └── term_base_handler.py
│   ├── utils/               # Utility modules
│   │   ├── auth.py
│   │   ├── llm_handler.py
│   │   └── qa_handler.py
│   └── cli/                 # CLI interface
│       └── main.py
├── config/                  # Configuration
│   ├── config.yaml         # Active configuration
│   └── config.yaml.example # Example configuration template
├── docs/                    # Documentation
└── tests/                   # Test suite
```

## Domain Services

### Translation Domain (`src/translation/`)

The core domain handling translation logic and orchestration.

#### TranslationManager

Main service for managing translations, implementing batch processing and validation.

```python
class TranslationManager:
    def __init__(self, config: Dict)
    
    # Core translation methods
    async def translate_text(self, source_texts: List[str], source_lang: str, target_lang: str, 
                           contexts: List[str], term_base: Dict[str, str]) -> List[Tuple[str, List[str]]]
    
    async def batch_translate(self, texts: List[str], target_lang: str,
                            contexts: List[str], term_base: Dict[str, str],
                            df: Optional[pd.DataFrame],
                            row_indices: Optional[List[int]]) -> AsyncGenerator[List[Dict[str, Any]], None]
    
    # Helper methods
    def detect_missing_translations(self, df: pd.DataFrame, 
                                  source_lang: str,
                                  target_langs: List[str]) -> Dict[str, List[int]]
    
    async def _perform_translation(self, source_texts: List[str], source_lang: str,
                                 target_lang: str, contexts: List[str],
                                 term_base: Dict[str, str]) -> List[Tuple[str, List[str]]]
    
    async def extract_terms(self, text: str, context: str, target_lang: str) -> Dict[str, Dict[str, str]]
```

Key Features:
- True batch processing with configurable batch size
- Immediate batch updates to Google Sheets
- Context-aware translations
- Term base integration
- Quality assurance validation
- Parallel validation processing
- Configurable retry mechanism
- Batch-level error handling

### Google Sheets Domain (`src/sheets/`)

Handles all interactions with Google Sheets API.

#### GoogleSheetsHandler

```python
class GoogleSheetsHandler:
    def __init__(self, credentials: Credentials)
    
    # Sheet operations
    def set_spreadsheet(self, spreadsheet_id: str)
    def get_all_sheets(self) -> List[str]
    def read_sheet(self, sheet_name: str) -> pd.DataFrame
    def update_sheet(self, sheet_name: str, df: pd.DataFrame)
    def write_translations(self, sheet_name: str, updates: Dict[str, Any])
    
    # Context handling
    def get_context_from_row(self, row: pd.Series, context_patterns: List[str],
                           ignore_case: bool = True) -> str
    
    # Sheet management
    def ensure_language_columns(self, sheet_name: str, langs: List[str],
                              force: bool = False) -> bool
    
    # Cache management
    def clear_cache(self)
    def _get_sheet_data(self, sheet_name: str) -> pd.DataFrame
```

### Term Base Domain (`src/term_base/`)

Manages terminology consistency across translations.

#### TermBaseHandler

```python
class TermBaseHandler:
    def __init__(self, sheets_handler: GoogleSheetsHandler, sheet_name: str)
    
    def load_term_base(self) -> Dict[str, Dict[str, str]]
    def update_term_base(self, new_terms: Dict[str, Dict[str, str]])
```

### Utility Services (`src/utils/`)

#### LLMHandler

Manages interactions with Language Model APIs.

```python
class LLMHandler:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float)
    
    async def generate_completion(self, messages: List[Dict], json_schema: Dict) -> str
    def extract_structured_response(self, response: str) -> Dict
```

#### QAHandler

Handles translation quality assurance.

```python
class QAHandler:
    def __init__(self, max_length: int, llm_handler: LLMHandler)
    
    async def validate_translation(self, source_text: str, translated_text: str,
                                 term_base: Dict[str, str],
                                 skip_llm_on_issues: bool = False) -> List[str]
```

## Configuration

### config.yaml Structure

```yaml
google_sheets:
  credentials_file: str       # Path to Google API credentials
  token_file: str            # Path to token storage
  scopes: List[str]          # Required API scopes
  spreadsheet_id: Optional[str] # Can be provided via CLI

llm:
  api_key: str               # Set via LLM_API_KEY environment variable
  model: str                 # e.g., "gpt-4", "claude-3-sonnet"
  temperature: float         # Controls randomness (0.0-1.0)
  api_url: str              # LLM API endpoint
  max_retries: int          # Maximum retry attempts
  retry_delay: int          # Delay between retries (seconds)
  batch_size: int           # Rows per batch (default: 50)
  batch_delay: int          # Delay between batches (seconds)

qa:
  max_length: int           # Maximum length for QA validation
  non_translatable_patterns:
    - start: "{["   # Matches {[TERM]}
      end: "]}"
    - start: "<"    # Matches <TERM>
      end: ">"

context_columns:
  patterns: List[str]       # Column name patterns for context
  ignore_case: bool         # Case-insensitive pattern matching

term_base:
  sheet_name: str           # Term base sheet name
  columns:
    term: str              # Column for source terms
    comment: str           # Column for term comments
    translation_prefix: str # Prefix for translation columns

languages:
  source: str              # Source language code
  target: List[str]        # Default target languages
```

## Domain Events and Workflows

1. Translation Workflow:
   - Detect missing translations
   - Group by language
   - Process in configurable batches
   - Validate translations in parallel
   - Immediate batch updates to Google Sheets
   - Retry failed translations
   - Handle batch-level errors

2. Term Base Management:
   - Load existing terms
   - Extract new terms
   - Update term base
   - Apply terms in translations

3. Quality Assurance:
   - Validate translations
   - Check term base compliance
   - Verify cultural appropriateness
   - Ensure format preservation

## Design Patterns Used

1. **Repository Pattern**: Used in GoogleSheetsHandler for data access abstraction
2. **Factory Pattern**: Used in configuration and handler initialization
3. **Strategy Pattern**: Used in translation and validation processes
4. **Observer Pattern**: Used in logging and event handling
5. **Command Pattern**: Used in CLI implementation
6. **Generator Pattern**: Used in batch processing for memory efficiency

## Bounded Contexts

1. **Translation Context**:
   - Core domain
   - Handles translation logic and orchestration
   - Manages batch processing
   - Interfaces with LLM services

2. **Sheet Management Context**:
   - Supporting domain
   - Manages Google Sheets interactions
   - Handles data persistence
   - Manages batch updates

3. **Term Base Context**:
   - Supporting domain
   - Manages terminology consistency
   - Provides translation references

4. **Quality Assurance Context**:
   - Supporting domain
   - Validates translations
   - Ensures quality standards
   - Handles parallel validation

## Extension Points

1. **New Translation Services**:
   - Implement new LLM handlers
   - Add new translation strategies
   - Customize batch processing

2. **Additional Validation Rules**:
   - Extend QAHandler
   - Add new validation strategies
   - Implement custom validation flows

3. **Term Base Enhancements**:
   - Add term extraction methods
   - Implement term suggestions
   - Add term validation rules

4. **Sheet Providers**:
   - Support additional spreadsheet services
   - Implement new data sources
   - Add custom update strategies

## Best Practices for Development

1. **Adding New Features**:
   - Identify the appropriate domain
   - Follow DDD principles
   - Maintain bounded contexts
   - Add appropriate tests
   - Consider batch processing implications

2. **Configuration Changes**:
   - Update config.yaml schema
   - Update config.yaml.example
   - Update validation logic
   - Document new settings

3. **Error Handling**:
   - Use domain-specific exceptions
   - Implement proper logging
   - Maintain transaction boundaries
   - Handle batch-level errors

4. **Testing**:
   - Write unit tests for domain logic
   - Add integration tests for workflows
   - Test batch processing scenarios
   - Test error recovery

### Quality Assurance Context

Quality assurance includes several validation checks:
- Non-translatable terms preservation
- Term base compliance
- Cultural appropriateness
- Format preservation

#### Non-Translatable Terms

The system supports configurable patterns for identifying terms that should not be translated:
```yaml
qa:
  non_translatable_patterns:
    - start: "{["   # Matches {[TERM]}
      end: "]}"
    - start: "<"    # Matches <TERM>
      end: ">"
```

These patterns are used to:
1. Extract terms from source text
2. Verify their presence in translated text
3. Ensure exact preservation of terms