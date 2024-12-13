import sys
import asyncio
import click
import yaml
import os
import threading
import queue
import traceback
from typing import Dict, Any, List, Tuple
from datetime import datetime
from ..utils.auth import get_credentials
from ..utils.llm_handler import LLMHandler
from ..utils.qa_handler import QAHandler
from ..sheets.sheets_handler import SheetsHandler
from ..translation.translation_manager import TranslationManager
from ..translation.translation_dictionary import TranslationDictionary
from ..translation.interpunction_handler import InterpunctionHandler
from ..term_base.term_base_handler import TermBaseHandler
import logging
import pandas as pd
from ..utils.ui_manager import create_ui_manager

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
            'model': str,
            'temperature': (int, float),
            'batch_size': (int, type(None)),  # Optional
            #'batch_delay': (int, float, type(None)),  # Optional
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

@click.group()
@click.option('--config', default='config/config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """BabelSheet - Automated translation tool for Google Sheets."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)

def _initialize_context(ctx: click.Context, target_langs: str, sheet_id: str, verbose: int, 
                       require_source_lang: bool = True) -> Tuple[threading.Thread, queue.Queue]:
    """Initialize common context for commands.
    
    Args:
        ctx: Click context
        target_langs: Comma-separated list of target languages
        sheet_id: Google Sheet ID
        verbose: Verbosity level
        require_source_lang: Whether source language is required (default: True)
        
    Returns:
        Tuple of (thread, error_queue)
    """
    setup_logging(verbose)

    # Start timing
    ctx.start_time = datetime.now()

    # Update config with CLI sheet_id if provided
    if sheet_id:
        ctx.obj['config']['google_sheets']['spreadsheet_id'] = sheet_id
        
    # Validate config after potential updates
    validate_config(ctx.obj['config'])
    
    # Ensure we have a sheet_id from somewhere
    if not ctx.obj['config']['google_sheets']['spreadsheet_id']:
        raise click.UsageError("Spreadsheet ID must be provided either in config or via --sheet-id option")

    # Split target languages if provided as comma-separated string
    ctx.config = ctx.obj['config']
    ctx.target_langs = [lang.strip() for lang in target_langs.split(',')]
    
    # Set source language if required
    if require_source_lang:
        ctx.source_lang = ctx.config['languages']['source']
    
    # Initialize LLM Handler
    llm_config = ctx.config.get('llm', {})
    ctx.llm_handler = LLMHandler(
        ctx=ctx,
        api_key=llm_config.get('api_key'),
        model=llm_config.get('model', 'anthropic/claude-3-5-sonnet'),
        temperature=llm_config.get('temperature', 0.3),
    )

    # Set UI type to graphical
    if 'ui' not in ctx.config:
        ctx.config['ui'] = {}
    ctx.config['ui']['type'] = 'graphical'

    ctx.ui = create_ui_manager(ctx, ctx.llm_handler)

    return queue.Queue()

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
    '--dry-run',
    '-d',
    is_flag=True,
    default=False,
    help='Dry run the translation process without saving changes to Google Sheets'
)
@click.pass_context
def translate_command(ctx, target_langs, sheet_id, verbose, dry_run):
    """Translate missing entries in the specified Google Sheet."""
    error_queue = _initialize_context(ctx, target_langs, sheet_id, verbose)
    ctx.dry_run = dry_run

    # Create and start translation thread
    def run_translate():
        try:
            asyncio.run(translate(ctx, ctx.target_langs, verbose))
        except Exception as e:
            error_queue.put((e, traceback.format_exc()))
            logger.error(f"Translation thread error: {str(e)}")
            logger.error(traceback.format_exc())

    translation_thread = threading.Thread(target=run_translate)
    translation_thread.daemon = True
    translation_thread.start()

    # Set thread info in UI manager for monitoring
    ctx.ui.set_thread_info(translation_thread, error_queue)

    # Start UI in main thread
    try:
        ctx.ui.start()
    except KeyboardInterrupt:
        logger.info("Translation interrupted by user")
        ctx.ui.stop()
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Main thread error: {str(e)}")
        logger.critical(traceback.format_exc())
        ctx.ui.stop()
        sys.exit(1)

async def translate(ctx, target_langs, verbose):
    """Translate missing entries in the specified Google Sheet."""
    logger = logging.getLogger(__name__)

    try:
        # Initialize SheetsHandler
        creds = get_credentials()
        ctx.spreadsheet_id = ctx.config['google_sheets']['spreadsheet_id']
        ctx.sheets_handler = SheetsHandler(ctx, creds)
        ctx.sheets_handler.initialize()

        # Initialize Translation Dictionary
        ctx.translation_dictionary = TranslationDictionary(ctx)

        # Initialize QA Handler
        ctx.qa_handler = QAHandler(
            ctx=ctx,
            llm_handler=ctx.llm_handler, ui=ctx.ui,
            translation_dictionary=ctx.translation_dictionary,
            non_translatable_patterns=ctx.config.get('qa', {}).get('non_translatable_patterns', [])
        )

        ctx.translation_dictionary.initialize_from_sheets()

        # Initialize TermBaseHandler
        ctx.term_base_handler = TermBaseHandler(ctx)
        logger.debug(f"Successfully initialized Term Base handler with sheet: {ctx.term_base_handler.sheet_name}")

        # Initialize TranslationManager with the config and handlers
        translation_manager = TranslationManager(
            ctx=ctx,
            sheets_handler=ctx.sheets_handler,
            qa_handler=ctx.qa_handler,
            term_base_handler=ctx.term_base_handler,
            llm_handler=ctx.llm_handler,
            ui=ctx.ui,
            translation_dictionary=ctx.translation_dictionary
        )

        # Translate all sheets
        await translation_manager.translate_all_sheets(
            source_lang=ctx.source_lang,
            target_langs=ctx.target_langs,
            use_term_base=True
        )

        logger.debug("Translation completed")
        ctx.ui.print_overall_stats()

    except Exception as e:
        logger.error(f"Error in translate function: {str(e)}")
        logger.error(traceback.format_exc())
        raise  # Re-raise the exception to be caught by the thread handler

@cli.command(name='check-spacing')
@click.option(
    '--target-langs',
    '-t',
    required=True,
    help='Comma-separated list of target languages to check (e.g., "fr,es,de")'
)
@click.option(
    '--sheet-id',
    '-s',
    required=False,
    help='Google Sheet ID to process'
)
@click.option(
    '--verbose',
    '-v',
    count=True,
    help='Increase verbosity (use -v for info, -vv for debug, -vvv for trace)'
)
@click.option(
    '--dry-run',
    '-d',
    is_flag=True,
    default=False,
    help='Dry run the check spacing process without saving changes to Google Sheets'
)
@click.pass_context
def check_spacing_command(ctx, target_langs, sheet_id, verbose, dry_run):
    """Check and fix spaces before interpunction marks in the specified languages."""
    error_queue = _initialize_context(ctx, target_langs, sheet_id, verbose, require_source_lang=False)

    # Create and start check spacing thread
    def run_check_spacing():
        try:
            asyncio.run(check_spacing(ctx, ctx.target_langs, verbose))
        except Exception as e:
            error_queue.put((e, traceback.format_exc()))
            logger.error(f"Check spacing thread error: {str(e)}")
            logger.error(traceback.format_exc())

    ctx.dry_run = dry_run
    check_spacing_thread = threading.Thread(target=run_check_spacing)
    check_spacing_thread.daemon = True
    check_spacing_thread.start()

    # Set thread info in UI manager for monitoring
    ctx.ui.set_thread_info(check_spacing_thread, error_queue)

    # Start UI in main thread
    try:
        ctx.ui.start()
    except KeyboardInterrupt:
        logger.info("Check spacing interrupted by user")
        ctx.ui.stop()
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Main thread error: {str(e)}")
        logger.critical(traceback.format_exc())
        ctx.ui.stop()
        sys.exit(1)

async def check_spacing(ctx, target_langs, verbose):
    """Check and fix spaces before interpunction marks in the specified languages."""
    logger = logging.getLogger(__name__)

    try:
        # Initialize SheetsHandler
        creds = get_credentials()
        ctx.spreadsheet_id = ctx.config['google_sheets']['spreadsheet_id']
        ctx.sheets_handler = SheetsHandler(ctx, creds)
        ctx.sheets_handler.initialize()

        # Initialize Interpunction Handler
        interpunction_handler = InterpunctionHandler(ctx.sheets_handler, ctx.ui)

        # Get all sheet names
        sheet_names = ctx.sheets_handler.get_sheet_names()

        # Track total fixes
        total_fixes = 0

        # Process each sheet
        for sheet_name in sheet_names:
            fixes = await interpunction_handler.check_interpunction_spacing(sheet_name, target_langs)
            total_fixes += fixes

        logger.info(f"Completed checking interpunction spacing. Total fixes made: {total_fixes}")
        ctx.ui.info(f"<b>Completed checking interpunction spacing. <font color='green'>Total texts fixed:</font> <font color='yellow'>{total_fixes}</font></b>")

    except Exception as e:
        logger.error(f"Error during check spacing: {str(e)}")
        logger.error(traceback.format_exc())
        ctx.ui.error(f"<b><font color='red'>Error during check spacing: {str(e)}</font></b>")
        raise

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