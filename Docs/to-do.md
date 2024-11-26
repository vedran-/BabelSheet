# BabelSheet Implementation To-Do List

## Phase 1: Core Foundation (Priority: Highest)
### Project Setup
- [x] Create project structure
- [x] Set up virtual environment
- [x] Create requirements.txt with initial dependencies
- [x] Set up Google Sheets authentication
  - [x] Implement OAuth2 flow
  - [x] Add token persistence
  - [x] Add token refresh handling

### Basic Components
- [x] Implement GoogleSheetsHandler
  - [x] Implement sheet reading
  - [x] Implement sheet writing
  - [x] Add basic error handling
  - [x] Add sheet update functionality
- [x] Implement TranslationManager
  - [x] Add LLM integration with custom endpoint support
  - [x] Add Google Translate fallback
  - [x] Implement basic prompt handling
- [x] Create simple CLI interface
  - [x] Add init command
  - [x] Add translate command

## Phase 2: Essential Functionality (Priority: High)
### Translation Workflow
- [x] Implement missing translations detection
- [x] Create translation prompt
  - [x] Include term base references
  - [x] Add context from comments
  - [x] Preserve markup ([],{})
  - [x] Handle newlines (\\n)
- [x] Add batch translation
  - [x] Basic progress tracking
  - [x] Simple resume capability

### Term Base
- [x] Implement term base as sheet
- [x] Add term base reading from sheet
- [x] Add term base sheet updates
- [x] Implement automated term extraction
  - [x] Extract terms from new translations
  - [x] Update term base automatically
  - [x] Ensure term base consistency

## Phase 3: Quality Assurance (Priority: Medium)
- [ ] Basic format validation
- [ ] Term base compliance check
- [ ] Character limit validation
- [ ] Markup preservation check

## Documentation
- [ ] Basic usage guide
- [ ] Translation guidelines

## Progress Tracking
- Total Tasks: 26
- Completed: 22
- In Progress: 0
- Remaining: 4