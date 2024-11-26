import pytest
from click.testing import CliRunner
from ..src.cli.main import cli
import os

@pytest.fixture
def runner():
    return CliRunner()

def test_translate_dry_run(runner):
    """Test translate command with --dry-run option."""
    result = runner.invoke(cli, [
        'translate',
        '--sheet-id', 'test-sheet-id',
        '--target-langs', 'es,fr',
        '--dry-run'
    ])
    assert result.exit_code == 0
    assert 'DRY RUN MODE' in result.output

def test_translate_force(runner):
    """Test translate command with --force option."""
    result = runner.invoke(cli, [
        'translate',
        '--sheet-id', 'test-sheet-id',
        '--target-langs', 'es,fr',
        '--force'
    ])
    assert result.exit_code == 0

def test_translate_missing_columns(runner):
    """Test translate command with missing language columns."""
    result = runner.invoke(cli, [
        'translate',
        '--sheet-id', 'test-sheet-id',
        '--target-langs', 'es,fr'
    ], input='y\n')  # Simulate user confirming column addition
    assert result.exit_code == 0 