import sys
import asyncio
import click
import yaml
import os
from typing import Dict, Any, List
from ..utils.auth import get_credentials
from ..sheets.sheets_handler import SheetsHandler
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
            #'batch_delay': (int, float, type(None)),  # Optional
            'max_retries': (int, type(None)),  # Optional
            'retry_delay': (int, float, type(None))  # Optional
        },
        'qa': {
            'non_translatable_patterns': list,
            'max_length': (int, type(None))  # Optional
        },
        'ui': {
            'simple_output': (bool, type(None))  # Optional
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
        format='[%(levelname)s @ %(filename)s:%(lineno)d] %(message)s'
        #format='[%(asctime)s %(levelname)s @ %(pathname)s:%(lineno)d] %(message)s'
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
        return asyncio.run(f(*args, **kwargs))
        """
        try:
            return asyncio.run(f(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error during async execution: {str(e)}")
            sys.exit(1)
        """
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
    '--simple-output',
    is_flag=True,
    help='Use simple console output instead of fancy UI'
)
@click.pass_context
def translate_command(ctx, target_langs, sheet_id, verbose, simple_output):
    """Translate missing entries in the specified Google Sheet."""
    setup_logging(verbose)

    # Update config with CLI sheet_id if provided
    if sheet_id:
        ctx.obj['config']['google_sheets']['spreadsheet_id'] = sheet_id
        
    # Update config with simple output option if provided
    if simple_output:
        if 'ui' not in ctx.obj['config']:
            ctx.obj['config']['ui'] = {}
        ctx.obj['config']['ui']['simple_output'] = True
    
    # Validate config after potential updates
    validate_config(ctx.obj['config'])
    
    # Ensure we have a sheet_id from somewhere
    if not ctx.obj['config']['google_sheets']['spreadsheet_id']:
        raise click.UsageError("Spreadsheet ID must be provided either in config or via --sheet-id option")
    
    return async_command(translate)(ctx, target_langs, verbose)

async def translate(ctx, target_langs, verbose):
    """Translate missing entries in the specified Google Sheet."""
    logger = logging.getLogger(__name__)
    
    # Split target languages if provided as comma-separated string
    ctx.config = ctx.obj['config']
    ctx.source_lang = ctx.config['languages']['source']
    ctx.target_langs = [lang.strip() for lang in target_langs.split(',')]
    
    # Initialize SheetsHandler
    creds = get_credentials()
    ctx.spreadsheet_id = ctx.config['google_sheets']['spreadsheet_id']
    ctx.sheets_handler = SheetsHandler(ctx, creds)

    # Initialize TermBaseHandler
    ctx.term_base_handler = TermBaseHandler(ctx)
    logger.debug(f"Successfully initialized Term Base handler with sheet: {ctx.term_base_handler.sheet_name}")

    """
    #terms = ctx.term_base_handler.get_terms_for_language(ctx.target_langs[0])
    #logger.info(terms)

    print('----------------')
    print(ctx.sheets_handler.get_sheet_data('Sheet1'))
    ctx.sheets_handler.modify_cell_data('Sheet1', 1, 3, 'test')
    print('----------------')
    print(ctx.sheets_handler.get_sheet_data('Sheet1'))
    ctx.sheets_handler.modify_cell_data('Sheet1', 2, 3, 'test2')
    print('----------------')
    print(ctx.sheets_handler.get_sheet_data('Sheet1'))
    print('----------------')
    ctx.sheets_handler.save_changes()
    sys.exit()
    """

    # Initialize TranslationManager with the config and handlers
    translation_manager = TranslationManager(
        config=ctx.config,
        sheets_handler=ctx.sheets_handler,
        term_base_handler=ctx.term_base_handler
    )

    # First, ensure term base translations are up to date if we have a term base
    if ctx.term_base_handler:
        logger.debug("Ensuring term base translations are up to date...")
        await translation_manager.ensure_sheet_translations(ctx.term_base_handler.sheet_name, 
            ctx.source_lang, ctx.target_langs, use_term_base=False)
        translation_manager.llm_handler.print_token_usage()


    # Process each sheet
    sheet_names = ctx.sheets_handler.get_sheet_names()
    for sheet_name in sheet_names:
        logger.debug(f"Processing sheet: {sheet_name}")
        
        # Skip term base sheet if it exists
        if ctx.term_base_handler and sheet_name == ctx.config['term_base']['sheet_name']:
            continue
        
        # Translate missing entries in the sheet
        await translation_manager.ensure_sheet_translations(sheet_name, 
            ctx.source_lang, ctx.target_langs, use_term_base=True)

        logger.debug(f"Completed processing sheet: {sheet_name}")
        translation_manager.llm_handler.print_token_usage()
        

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
    try:
        creds = get_credentials()
        click.echo("Authentication successful")
    except Exception as e:
        click.echo(f"Authentication failed: {str(e)}")
        sys.exit(1)

def main():
    """Entry point for the CLI."""
    cli(obj={})

if __name__ == '__main__':
    main() 