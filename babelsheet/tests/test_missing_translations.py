import pandas as pd
import pytest
from babelsheet.src.sheets.sheets_handler import SheetsHandler, CellData
from babelsheet.src.translation.translation_manager import (
    detect_missing_translations,
    scan_missing_translations,
    format_missing_by_source,
)


class _FakeCtx:
    def __init__(self):
        self.config = {
            'context_columns': {'patterns': [], 'ignore_case': True},
        }
        self.ui = None


def _make_sheet_data():
    rows = [
        [CellData('en'), CellData('es'), CellData('de')],
        [CellData('HOLD TO SHOOT'), CellData(None), CellData(None)],
        [CellData('PLAY AGAIN'), CellData('JUGAR DE NUEVO'), CellData(None)],
        [CellData(None), CellData(None), CellData(None)],
    ]
    df = pd.DataFrame(rows)
    df.attrs['context_column_indexes'] = []
    df.attrs['sheet_name'] = 'UI'
    return df


@pytest.fixture
def sheets_handler():
    ctx = _FakeCtx()
    handler = SheetsHandler(ctx)
    handler._sheets = {'UI': _make_sheet_data()}
    return handler


def test_format_missing_by_source():
    all_missing = {
        'es': [
            {'source_text': 'HOLD TO SHOOT', 'sheet_name': 'UI'},
        ],
        'de': [
            {'source_text': 'HOLD TO SHOOT', 'sheet_name': 'UI'},
            {'source_text': 'PLAY AGAIN', 'sheet_name': 'UI'},
        ],
    }

    lines = format_missing_by_source(all_missing)

    assert lines == [
        '`HOLD TO SHOOT` [UI]: missing de, es',
        '`PLAY AGAIN` [UI]: missing de',
    ]


def test_detect_missing_translations(sheets_handler):
    df = sheets_handler.get_sheet_data('UI')

    missing = detect_missing_translations(
        sheets_handler, df, 'en', ['es', 'de'], create_if_missing=False
    )

    assert set(missing.keys()) == {'es', 'de'}
    assert {item['source_text'] for item in missing['es']} == {'HOLD TO SHOOT'}
    assert {item['source_text'] for item in missing['de']} == {
        'HOLD TO SHOOT', 'PLAY AGAIN'
    }


def test_detect_missing_translations_skips_absent_column(sheets_handler):
    rows = [
        [CellData('en'), CellData('es')],
        [CellData('HOLD TO SHOOT'), CellData(None)],
    ]
    df = pd.DataFrame(rows)
    df.attrs['context_column_indexes'] = []
    df.attrs['sheet_name'] = 'Partial'
    sheets_handler._sheets['Partial'] = df

    missing = detect_missing_translations(
        sheets_handler, df, 'en', ['es', 'de'], create_if_missing=False
    )

    assert list(missing.keys()) == ['es']
    assert missing['es'][0]['source_text'] == 'HOLD TO SHOOT'


def test_detect_missing_translations_skips_source_lang_column(sheets_handler):
    rows = [
        [CellData('en'), CellData('en')],
        [CellData('HOLD TO SHOOT'), CellData(None)],
    ]
    df = pd.DataFrame(rows)
    df.attrs['context_column_indexes'] = []
    df.attrs['sheet_name'] = 'SameCol'
    sheets_handler._sheets['SameCol'] = df

    missing = detect_missing_translations(
        sheets_handler, df, 'en', ['en'], create_if_missing=False
    )

    assert missing == {}


def test_scan_missing_translations_merges_sheets(sheets_handler):
    rows = [
        [CellData('en'), CellData('es')],
        [CellData('HOLD TO SHOOT'), CellData(None)],
    ]
    df = pd.DataFrame(rows)
    df.attrs['context_column_indexes'] = []
    df.attrs['sheet_name'] = 'Other'
    sheets_handler._sheets['Other'] = df

    all_missing = scan_missing_translations(
        sheets_handler, 'en', ['es'], create_if_missing=False
    )

    source_texts = {item['source_text'] for item in all_missing['es']}
    assert source_texts == {'HOLD TO SHOOT'}


def test_format_missing_by_source_merges_langs_for_same_text():
    all_missing = {
        'es': [{'source_text': 'HOLD TO SHOOT', 'sheet_name': 'UI'}],
        'de': [{'source_text': 'HOLD TO SHOOT', 'sheet_name': 'UI'}],
    }

    assert format_missing_by_source(all_missing) == [
        '`HOLD TO SHOOT` [UI]: missing de, es',
    ]


def test_format_missing_by_source_separate_lines_per_sheet():
    all_missing = {
        'es': [
            {'source_text': 'HOLD TO SHOOT', 'sheet_name': 'UI'},
            {'source_text': 'HOLD TO SHOOT', 'sheet_name': 'Tutorial'},
        ],
    }

    assert format_missing_by_source(all_missing) == [
        '`HOLD TO SHOOT` [Tutorial]: missing es',
        '`HOLD TO SHOOT` [UI]: missing es',
    ]


def test_scan_missing_translations_skips_term_base_sheet(sheets_handler):
    all_missing = scan_missing_translations(
        sheets_handler,
        'en',
        ['es'],
        term_base_sheet_name='UI',
        create_if_missing=False,
    )

    assert all_missing == {}
