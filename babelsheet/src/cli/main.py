import asyncio
import click
import yaml
import os
from typing import Dict, Any
from ..utils.auth import GoogleAuth
from ..sheets.sheets_handler import GoogleSheetsHandler
from ..translation.translation_manager import TranslationManager
from ..term_base.term_base_handler import TermBaseHandler

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.group()
@click.option('--config', default='config/config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """BabelSheet - Automated translation tool for Google Sheets."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)

@cli.command()
@click.option('--sheet-id', help='Google Sheet ID to process')
@click.option('--target-langs', help='Comma-separated list of target languages')
@click.pass_context
async def translate(ctx, sheet_id: str, target_langs: str):
    """Translate missing texts in the specified Google Sheet."""
    config = ctx.obj['config']
    
    # Initialize components
    auth = GoogleAuth(
        credentials_file=config['google_sheets']['credentials_file'],
        token_file=config['google_sheets']['token_file'],
        scopes=config['google_sheets']['scopes']
    )
    
    credentials = auth.authenticate()
    
    sheets_handler = GoogleSheetsHandler(credentials)
    sheets_handler.set_spreadsheet(sheet_id or config['google_sheets']['spreadsheet_id'])
    
    # Initialize term base with sheet handler
    term_base = TermBaseHandler(
        sheets_handler=sheets_handler,
        sheet_name=config['term_base']['sheet_name'],
        term_column=config['term_base']['columns']['term'],
        comment_column=config['term_base']['columns']['comment'],
        translation_prefix=config['term_base']['columns']['translation_prefix']
    )
    
    translation_manager = TranslationManager(
        api_key=os.getenv('LLM_API_KEY'),
        base_url=config['llm'].get('api_url', 'https://api.openai.com/v1'),
        model=config['llm']['model'],
        temperature=config['llm']['temperature']
    )
    
    # Get target languages
    target_language_list = (target_langs.split(',') if target_langs 
                          else config['languages']['target'])
    
    # Process each sheet
    for sheet_name in sheets_handler.get_all_sheets():
        if sheet_name == config['term_base']['sheet_name']:
            continue  # Skip the term base sheet itself
            
        click.echo(f"\nProcessing sheet: {sheet_name}")
        
        # Read sheet data
        df = sheets_handler.read_sheet(sheet_name)
        
        # Detect missing translations
        missing = translation_manager.detect_missing_translations(
            df, 
            source_lang=config['languages']['source'],
            target_langs=target_language_list
        )
        
        # Process each language
        for lang in target_language_list:
            if lang not in missing or not missing[lang]:
                click.echo(f"No missing translations for {lang}")
                continue
                
            click.echo(f"\nTranslating {len(missing[lang])} texts to {lang}...")
            
            # Get term base for this language
            terms = term_base.get_terms_for_language(lang)
            
            # Process each missing translation
            updates = {}
            for idx in missing[lang]:
                source_text = df.iloc[idx][config['languages']['source']]
                context = df.iloc[idx].get('context', '')
                
                try:
                    translation = await translation_manager.translate_text(
                        text=source_text,
                        target_lang=lang,
                        context=context,
                        term_base=terms,
                        term_base_handler=term_base  # Pass term_base_handler for automatic updates
                    )
                    
                    # Prepare update
                    cell_range = f"{chr(65 + df.columns.get_loc(lang))}{idx + 2}"
                    updates[cell_range] = translation
                    
                    click.echo(f"Translated [{idx + 2}]: {source_text[:30]}... → {translation[:30]}...")
                    
                except Exception as e:
                    click.echo(f"Error translating row {idx + 2}: {e}")
            
            # Batch update the sheet
            if updates:
                sheets_handler.write_translations(sheet_name, updates)
                click.echo(f"Updated {len(updates)} translations in {lang}")

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