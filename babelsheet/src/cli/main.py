import sys
import asyncio
import click
import yaml
import os
from typing import Dict, Any, List
from ..utils.auth import get_credentials
from ..sheets.sheets_handler import GoogleSheetsHandler
from ..translation.translation_manager import TranslationManager
from ..term_base.term_base_handler import TermBaseHandler
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_config(config):
    """Validate the configuration."""
    # Define expected types and requirements for config values
    validation_schema = {
        'google_sheets': {
            'credentials_file': str,
            'token_file': str,
            'scopes': list,
            'spreadsheet_id': (str, type(None))  # Optional field
        },
        'languages': {
            'source': str,
            'target': list
        },
        'term_base': {
            'sheet_name': str
        },
        'context_columns': {
            'patterns': list,
            'ignore_case': bool
        },
        'llm': {
            'api_url': str,
            'temperature': (int, float),
            'batch_size': (int, type(None)),  # Optional
            'batch_delay': (int, float, type(None)),  # Optional
            'max_retries': (int, type(None)),  # Optional
            'retry_delay': (int, float, type(None))  # Optional
        },
        'qa': {
            'non_translatable_patterns': list,
            'max_length': (int, type(None))  # Optional
        }
    }

    def validate_dict(config_section, schema, path=""):
        for key, expected in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in config_section:
                if current_path == "google_sheets.spreadsheet_id":
                    config_section[key] = None  # Set default to None for spreadsheet_id
                else:
                    raise ValueError(f"Missing required config value: {current_path}")
            
            if isinstance(expected, dict):
                if not isinstance(config_section[key], dict):
                    raise TypeError(f"Invalid type for {current_path}. Expected dict")
                validate_dict(config_section[key], expected, current_path)
            else:
                if config_section[key] is not None:  # Skip type validation for None values
                    if not isinstance(config_section[key], expected if not isinstance(expected, tuple) else expected):
                        raise TypeError(
                            f"Invalid type for {current_path}. "
                            f"Expected {expected}, got {type(config_section[key])}"
                        )

    validate_dict(config, validation_schema)

def setup_logging(verbose_level: int):
    """Set up logging with different verbosity levels.
    
    Level 0: INFO (default)
    Level 1: DEBUG
    Level 2: DEBUG + Data dumps
    """
    level = logging.INFO
    if verbose_level >= 1:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set TRACE level for data dumps
    if verbose_level >= 2:
        logging.TRACE = 5  # Custom level below DEBUG
        logging.addLevelName(logging.TRACE, 'TRACE')
        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.TRACE):
                self._log(logging.TRACE, message, args, **kwargs)
        logging.Logger.trace = trace

def async_command(f):
    """Decorator to run async click commands."""
    def wrapper(*args, **kwargs):
        try:
            return asyncio.run(f(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error during async execution: {str(e)}")
            sys.exit(1)
    return wrapper

@click.group()
@click.option('--config', default='config/config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """BabelSheet - Automated translation tool for Google Sheets."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)

@cli.command(name='translate')
@click.option(
    '--target-langs',
    '-t',
    required=True,
    help='Comma-separated list of target languages (e.g., "fr,es,de")'
)
@click.option(
    '--sheet-id',
    '-s',
    required=False,  # Made optional since it could be in config
    help='Google Sheet ID to process'
)
@click.option(
    '--verbose',
    '-v',
    count=True,
    help='Increase verbosity (use -v for info, -vv for debug, -vvv for trace)'
)
@click.option(
    '--force',
    '-f',
    is_flag=True,
    help='Force processing without confirmation prompts'
)
@click.pass_context
def translate_command(ctx, target_langs, sheet_id, verbose, force):
    """Translate missing entries in the specified Google Sheet."""
    # Update config with CLI sheet_id if provided
    if sheet_id:
        ctx.obj['config']['google_sheets']['spreadsheet_id'] = sheet_id
    
    # Validate config after potential sheet_id update
    validate_config(ctx.obj['config'])
    
    # Ensure we have a sheet_id from somewhere
    if not ctx.obj['config']['google_sheets']['spreadsheet_id']:
        raise click.UsageError("Spreadsheet ID must be provided either in config or via --sheet-id option")
    
    return async_command(translate)(ctx, target_langs, verbose, force)

async def translate(ctx, target_langs, verbose, force):
    """Translate missing entries in the specified Google Sheet."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Split target languages if provided as comma-separated string
        langs = [lang.strip() for lang in target_langs.split(',')]
        
        # Initialize handlers
        creds = get_credentials()
        sheets_handler = GoogleSheetsHandler(creds)
        
        # Get spreadsheet ID from config (already validated)
        spreadsheet_id = ctx.obj['config']['google_sheets']['spreadsheet_id']
        sheets_handler.set_spreadsheet(spreadsheet_id)
        
        # Initialize term base handler with sheet name from config
        try:
            term_base_handler = TermBaseHandler(
                sheets_handler=sheets_handler,
                sheet_name=ctx.obj['config']['term_base']['sheet_name']
            )
            logger.info(f"Successfully initialized term base handler with sheet: {ctx.obj['config']['term_base']['sheet_name']}")
        except Exception as e:
            logger.warning(f"Failed to initialize term base handler: {e}")
            logger.info("Proceeding without term base")
            term_base_handler = None
        
        # Initialize TranslationManager with the config and handlers
        translation_manager = TranslationManager(
            config=ctx.obj['config'],
            sheets_handler=sheets_handler,
            term_base_handler=term_base_handler
        )
        
        # First, ensure term base translations are up to date if we have a term base
        if term_base_handler:
            logger.info("Ensuring term base translations are up to date...")
            try:
                await translation_manager.ensure_term_base_translations(langs)
            except Exception as e:
                logger.error(f"Failed to update term base translations: {e}")
                logger.info("Proceeding with translation without term base updates")
        
        # Process each sheet
        sheet_names = sheets_handler.get_all_sheets()
        for sheet_name in sheet_names:
            logger.info(f"\nProcessing sheet: {sheet_name}")
            
            # Skip term base sheet if it exists
            if term_base_handler and sheet_name == ctx.obj['config']['term_base']['sheet_name']:
                continue
            
            # Get initial sheet data
            df = sheets_handler._get_sheet_data(sheet_name)
            
            # First ensure all required language columns exist
            columns_added = False
            try:
                columns_added = sheets_handler.ensure_language_columns(sheet_name, langs, force=force)
            except ValueError as e:
                logger.error(f"Error: {str(e)}")
                continue
                
            # If columns were added, refresh the sheet data to get the new structure
            if columns_added:
                # Clear the cache to force a fresh fetch
                sheets_handler._sheet_cache.pop(sheet_name, None)
                # Get fresh data with new columns
                df = sheets_handler.read_sheet(sheet_name)
            
            # Detect missing translations
            missing = translation_manager.detect_missing_translations(
                df=df,
                source_lang=ctx.obj['config']['languages']['source'],
                target_langs=langs
            )
            
            # Process translations for each language
            for lang, missing_indices in missing.items():
                if not missing_indices:
                    logger.info(f"No missing translations for {lang}")
                    continue
                
                logger.info(f"Translating {len(missing_indices)} entries to {lang}")
                
                # Get texts and contexts for translation
                texts = df.loc[missing_indices, ctx.obj['config']['languages']['source']].tolist()
                contexts = []
                
                # Get contexts from configured context columns if they exist
                for idx in missing_indices:
                    context_parts = []
                    for pattern in ctx.obj['config']['context_columns']['patterns']:
                        matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
                        for col in matching_cols:
                            if pd.notna(df.loc[idx, col]):
                                context_parts.append(str(df.loc[idx, col]))
                    contexts.append(" | ".join(context_parts))
                
                # Get term base for this language
                term_base = term_base_handler.get_terms_for_language(lang)
                
                # Translate and process each batch
                async for batch_results in translation_manager.batch_translate(
                    texts=texts,
                    target_lang=lang,
                    contexts=contexts,
                    term_base=term_base,
                    df=df,
                    row_indices=missing_indices
                ):
                    # Log any translation issues
                    for result in batch_results:
                        if result.get('issues'):
                            logger.debug(f"Translation issues for '{result['source_text']}' -> '{result['translated_text']}':")
                            for issue in result['issues']:
                                logger.debug(f"  - {issue}")
                    
                    # Print token usage statistics at the end
                    translation_manager.llm_handler.print_token_usage()

                    # Update the sheet with the translated data for this batch
                    logger.info(f"Updating sheet with batch {batch_results[0]['batch_number']} translations...")
                    sheets_handler.update_sheet(sheet_name, df)
                    logger.info(f"Batch {batch_results[0]['batch_number']} completed and saved")
                
                logger.info(f"Completed all translations for {lang}")
            
            logger.info(f"Completed processing sheet: {sheet_name}")
        
        # Print token usage statistics at the end
        translation_manager.llm_handler.print_token_usage()
        
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        raise e

@cli.command()
@click.pass_context
def init(ctx):
    """Initialize configuration and authentication."""
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    
    # Create default config if it doesn't exist
    if not os.path.exists('config/config.yaml'):
        with open('config/config.yaml', 'w') as f:
            yaml.dump(ctx.obj['config'], f)
        click.echo("Created default configuration file")
    
    click.echo("Initializing Google authentication...")
    auth = GoogleAuth(
        credentials_file=ctx.obj['config']['google_sheets']['credentials_file'],
        token_file=ctx.obj['config']['google_sheets']['token_file'],
        scopes=ctx.obj['config']['google_sheets']['scopes']
    )
    auth.authenticate()
    click.echo("Authentication successful")

def main():
    """Entry point for the CLI."""
    cli(obj={})

if __name__ == '__main__':
    main() 