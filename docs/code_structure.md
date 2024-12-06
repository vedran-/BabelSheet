# BabelSheet Code Structure and Architecture

## Domain-Driven Design (DDD) Architecture

BabelSheet follows Domain-Driven Design principles, organizing code around business domains and their logic. The project is structured into distinct bounded contexts, each handling specific business capabilities.

## Project Structure

```
babelsheet/
├── src/                      # Source code
│   ├── translation/          # Translation domain
│   │   ├── translation_manager.py
│   │   ├── translation_prompts.py
│   │   └── translation_dictionary.py
│   ├── sheets/              # Google Sheets domain
│   │   └── sheets_handler.py
│   ├── term_base/           # Term Base domain
│   │   └── term_base_handler.py
│   ├── utils/               # Utility modules
│   │   ├── auth.py         # Authentication handling
│   │   ├── llm_handler.py  # LLM integration
│   │   ├── qa_handler.py   # Quality assurance
│   │   ├── sheet_index.py  # Sheet indexing
│   │   ├── ui_manager.py   # UI abstraction
│   │   └─�� ui/             # UI implementations
│   │       ├── base_ui_manager.py
│   │       └── graphical_ui_manager.py
│   └── cli/                 # CLI interface
│       └── main.py
├── config/                  # Configuration
│   ├── config.yaml         # Active configuration
│   └── config.yaml.example # Example configuration
├── docs/                    # Documentation
├── translation_logs/        # Translation logs
└── tests/                   # Test suite
```

## Domain Services

### Translation Domain (`src/translation/`)

The core domain handling translation logic and orchestration.

#### TranslationManager

Main service orchestrating the translation process:
- Batch processing
- Translation workflow management
- Error handling and retries
- Term base integration
- Quality assurance coordination

```python
class TranslationManager:
    async def collect_all_missing_translations(self, source_lang: str, target_langs: List[str])
    async def _process_missing_translations(self, df: pd.DataFrame, missing_translations: Dict)
    async def _perform_translation(self, source_texts: List[str], source_lang: str, target_lang: str)
    async def _validate_translations(self, source_texts: List[str], translations: List[Dict])
```

#### TranslationPrompts

Manages LLM prompts and response schemas:
- Translation instructions
- Term base integration
- Non-translatable terms handling
- Override instructions
- Response schema definitions

#### TranslationDictionary

Handles translation memory and caching:
- Stores successful translations
- Prevents duplicate translations
- Provides translation suggestions
- Maintains consistency

### Google Sheets Domain (`src/sheets/`)

Handles all interactions with Google Sheets API.

#### SheetsHandler

```python
class SheetsHandler:
    def modify_cell_data(self, sheet_name: str, row: int, col: int, value: str)
    def get_sheet_data(self, sheet_name: str) -> pd.DataFrame
    def get_sheet_names(self) -> List[str]
    def ensure_sheet_exists(self, sheet_name: str) -> bool
```

### Term Base Domain (`src/term_base/`)

Manages terminology consistency across translations.

#### TermBaseHandler

```python
class TermBaseHandler:
    def get_term_base(self, target_lang: str) -> Dict[str, Dict[str, Any]]
    def add_term(self, source_term: str, target_lang: str, translation: str, comment: str)
    def update_term_base(self, terms: Dict[str, Dict[str, Any]])
```

### Utility Services (`src/utils/`)

#### LLMHandler

Manages interactions with Language Model APIs:
- Multiple provider support (OpenAI, Anthropic, etc.)
- Response parsing and validation
- Error handling and retries
- Rate limiting

```python
class LLMHandler:
    async def get_completion(self, messages: List[Dict], json_schema: Dict) -> Dict
    def extract_structured_response(self, response: str) -> Dict
```

#### QAHandler

Comprehensive translation quality assurance:
- Format validation
- Term base compliance
- Non-translatable terms verification
- Cultural appropriateness
- Parallel validation processing

```python
class QAHandler:
    async def validate_with_llm_batch(self, items: List[Dict], target_lang: str)
    def validate_syntax(self, source_text: str, translated_text: str) -> List[str]
```

#### UI System

Modern terminal UI system with multiple implementations:

##### UI Factory

```python
class UIFactory:
    @staticmethod
    def create_ui_manager(config: Dict, llm_handler: LLMHandler) -> BaseUIManager
```

##### BaseUIManager
- Core UI functionality
- Progress tracking
- Status updates
- Error reporting
- Abstract base class for UI implementations

##### GraphicalUIManager
- Rich terminal UI
- Real-time updates
- Color-coded status
- Interactive elements
- Translation progress
- Statistics display
- Table-based layout
- Live updates

##### ConsoleUIManager
- Simple console output
- Basic progress indicators
- Error reporting
- Non-interactive mode
- Minimal dependencies

## Configuration System

### config.yaml Structure

```yaml
google_sheets:
  credentials_file: str
  token_file: str
  spreadsheet_id: str
  scopes: List[str]

languages:
  source: str
  target: List[str]

llm:
  api_key: str
  model: str
  temperature: float
  max_retries: int
  batch_size: int
  batch_delay: int

qa:
  max_length: int
  non_translatable_patterns:
    - start: str
      end: str

term_base:
  sheet_name: str
  columns:
    term: str
    comment: str
    translation_prefix: str
  add_terms_to_term_base: bool

output:
  dir: str

ui:
  type: str  # graphical or simple
```

## Core Workflows

### Translation Process

1. **Initialization**
   - Load configuration
   - Initialize services
   - Connect to Google Sheets
   - Load term base
   - Setup UI system
   - Initialize logging

2. **Analysis**
   - Detect missing translations
   - Group by language
   - Sort by text length
   - Prepare batches
   - Calculate statistics

3. **Translation**
   - Process in batches
   - Apply term base
   - Handle retries
   - Update sheets
   - Log progress
   - Update UI
   - Cache translations

4. **Validation**
   - Syntax checking
   - Term verification
   - Format validation
   - Cultural checks
   - Parallel processing
   - Override handling

5. **Completion**
   - Update term base
   - Generate statistics
   - Create logs
   - Report results
   - Clean up resources

### Error Handling

1. **Translation Errors**
   - Automatic retries
   - Detailed logging
   - User notifications
   - Progress preservation

2. **API Errors**
   - Rate limiting
   - Connection retries
   - Fallback options
   - Error reporting

3. **Validation Errors**
   - Issue categorization
   - Retry suggestions
   - Override options
   - Progress tracking

### Logging System

1. **Translation Logs**
   - Success/failure tracking
   - Detailed error messages
   - Translation attempts
   - Validation issues
   - Term base updates

2. **Performance Logs**
   - Batch processing times
   - API response times
   - Memory usage
   - Cache statistics

3. **Debug Logs**
   - API interactions
   - Sheet operations
   - Term base updates
   - Validation details

## Extension Points

1. **LLM Providers**
   - Add new providers
   - Custom configurations
   - Response handling
   - Error management

2. **UI Systems**
   - New UI implementations
   - Custom displays
   - Progress tracking
   - User interaction

3. **Validation Rules**
   - Custom validators
   - New quality checks
   - Language-specific rules
   - Format verification

4. **Term Base Features**
   - Term extraction
   - Suggestion systems
   - Context handling
   - Override management

5. **Logging Extensions**
   - Custom log formats
   - Additional metrics
   - External logging systems
   - Performance monitoring

## Development Guidelines

1. **Adding Features**
   - Follow DDD principles
   - Maintain bounded contexts
   - Add comprehensive tests
   - Update documentation

2. **Code Style**
   - Type hints
   - Async/await
   - Error handling
   - Logging

3. **Testing**
   - Unit tests
   - Integration tests
   - Mock external services
   - Test configurations

4. **Documentation**
   - Code comments
   - Type hints
   - README updates
   - API documentation

5. **Performance Optimization**
   - Batch size tuning
   - Cache management
   - Memory optimization
   - API rate limiting

6. **Logging Best Practices**
   - Use appropriate log levels
   - Include context
   - Structure log messages
   - Monitor performance metrics