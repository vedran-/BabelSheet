from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QTableWidget, QTableWidgetItem, QTextEdit, QLabel,
                          QGridLayout, QHeaderView)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from rich.text import Text
from ..llm_handler import LLMHandler
import threading
import logging
import traceback
import queue

logger = logging.getLogger(__name__)

class StatusIcons:
    WAITING = " "  # Stopwatch - shows waiting state clearly
    PREPARING = "⌛"  # Hammer and wrench - shows setup/preparation
    TRANSLATING = "📖"  # Arrows in circle - indicates ongoing process
    VALIDATING = "🔍"  # Magnifying glass - indicates validation
    SUCCESS = "✅"  # Direct hit - shows perfect completion
    FAILED = "❌"  # Broken heart - more friendly than prohibition signs
    INFO = "💡"  # Light bulb - represents information/knowledge
    WARNING = "⚠️"  # Police car light - grabs attention for warnings


class UISignals(QObject):
    """Signals for thread-safe UI updates."""
    debug_signal = pyqtSignal(str)
    info_signal = pyqtSignal(str)
    warning_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    critical_signal = pyqtSignal(str)
    set_translation_list_signal = pyqtSignal(list) # new_items
    update_translation_item_signal = pyqtSignal(object) # item

    on_translation_started_signal = pyqtSignal(object)  # item
    on_translation_ended_signal = pyqtSignal(object)

    start_new_batch_signal = pyqtSignal()
    add_term_base_signal = pyqtSignal(str, str, str, str)
    check_thread_signal = pyqtSignal()  # New signal for checking thread status

class GraphicalUIManager:
    def __init__(self, max_history: int = 100, status_lines: int = 10, llm_handler: LLMHandler = None):
        """Initialize Graphical UI Manager."""
        self.llm_handler = llm_handler
        self.app = QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("BabelSheet Translator")
        self.window.resize(1200, 800)
        
        # Create signals object
        self.signals = UISignals()
        
        # Connect signals to slots with QueuedConnection to ensure thread safety
        self.signals.debug_signal.connect(self._debug, Qt.ConnectionType.QueuedConnection)
        self.signals.info_signal.connect(self._info, Qt.ConnectionType.QueuedConnection)
        self.signals.warning_signal.connect(self._warning, Qt.ConnectionType.QueuedConnection)
        self.signals.error_signal.connect(self._error, Qt.ConnectionType.QueuedConnection)
        self.signals.critical_signal.connect(self._critical, Qt.ConnectionType.QueuedConnection)
        self.signals.set_translation_list_signal.connect(self._set_translation_list, Qt.ConnectionType.QueuedConnection)
        self.signals.update_translation_item_signal.connect(self._update_translation_item, Qt.ConnectionType.QueuedConnection)

        self.signals.on_translation_started_signal.connect(self._on_translation_started, Qt.ConnectionType.QueuedConnection)
        self.signals.on_translation_ended_signal.connect(self._on_translation_ended, Qt.ConnectionType.QueuedConnection)
        self.signals.start_new_batch_signal.connect(self._start_new_batch, Qt.ConnectionType.QueuedConnection)
        self.signals.check_thread_signal.connect(self._check_thread_status, Qt.ConnectionType.QueuedConnection)
        
        # Set dark fusion theme
        self.app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.app.setPalette(palette)
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.window.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create UI components
        self._setup_stats_panel()
        self._setup_translation_table()
        self._setup_status_panel()
        
        # Initialize data structures
        self.translation_entries = []
        self.status_messages = deque(maxlen=status_lines)

        # Statistics
        self.overall_stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0
        }
        
        # Setup thread check timer
        self.thread_check_timer = QTimer()
        self.thread_check_timer.timeout.connect(lambda: self.signals.check_thread_signal.emit())
        self.thread_check_timer.start(50)  # Check every 50ms

        self.thread_update_stats_timer = QTimer()
        self.thread_update_stats_timer.timeout.connect(lambda: self._ui_update_statistics())
        self.thread_update_stats_timer.start(1000)  # Update every second
        
        # Thread management
        self.should_stop = False
        self.error_queue = None
        self.translation_thread = None

    def _setup_stats_panel(self):
        """Setup the statistics panel."""
        self.stats_widget = QWidget()
        self.stats_layout = QGridLayout(self.stats_widget)
        
        self.total_label = QLabel("Total Attempts: 0")
        self.llm_stats_label = QLabel("LLM Stats: 0 tokens, $0.00")
        self.stats_layout.addWidget(QLabel("📊 Statistics"), 0, 0, 1, 1)
        self.stats_layout.addWidget(self.total_label, 0, 1)
        self.stats_layout.addWidget(self.llm_stats_label, 0, 2)
        
        self.layout.addWidget(self.stats_widget)

    def _setup_translation_table(self):
        """Setup the translation progress table."""
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Time", "Source Text", "Language", "Status", "Translation"])
        
        # Enable word wrap for the table
        self.table.setWordWrap(True)
        
        # Set column stretching
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        
        # Set default row height
        self.table.verticalHeader().setDefaultSectionSize(60)
        
        # Enable automatic row height adjustment
        # TODO: Perhaps disable this
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        self.layout.addWidget(self.table)

    def _setup_status_panel(self):
        """Setup the status messages panel."""
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.layout.addWidget(self.status_text)

    def _find_item_index(self, item) -> int:
        """Find the index of an item in the translation entries."""
        for i, entry in enumerate(self.translation_entries):
            if entry['source_text'] == item['source_text'] and entry['lang'] == item['lang']:
                return i
        return -1

    def _create_table_item(self, text: str, multiline: bool = False) -> QTableWidgetItem:
        """Create a table item with proper formatting."""
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make item read-only
        if multiline:
            # Enable text wrapping for multiline items
            item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        return item

    def _get_status_color(self, status: str) -> QColor:
        """Get the color for a status."""
        #if entry_type == "term_base":
        #    return QColor(0, 0, 255, 50)  # Blue with alpha
        if status.startswith(StatusIcons.SUCCESS):
            return QColor(0, 255, 0, 50)  # Green with alpha
        elif status.startswith(StatusIcons.FAILED):
            return QColor(255, 0, 0, 50)  # Red with alpha
        elif status.startswith(StatusIcons.VALIDATING) \
            or status.startswith(StatusIcons.TRANSLATING) \
            or status.startswith(StatusIcons.PREPARING):
            return QColor(255, 255, 0, 50)  # Yellow with alpha
        else:
            return QColor(192, 192, 192, 50)  # Gray with alpha

    def _ui_update_statistics(self):
        """Update the statistics display."""
        total = self.overall_stats['total_attempts']
        if total > 0:
            success_rate = (self.overall_stats['successful'] / total) * 100
            fail_rate = (self.overall_stats['failed'] / total) * 100
        else:
            success_rate = fail_rate = 0.0
        self.total_label.setText(f"Total Attempts: {total}/{len(self.translation_entries)}, {StatusIcons.SUCCESS} {self.overall_stats['successful']} ({success_rate:.1f}%), {StatusIcons.FAILED} {self.overall_stats['failed']} ({fail_rate:.1f}%)")

        llm_stats = self.llm_handler.get_usage_stats()
        prompt_tokens = llm_stats.get("prompt_tokens", 0)
        completion_tokens = llm_stats.get("completion_tokens", 0)
        total_tokens = llm_stats.get("total_tokens", 0)
        total_cost = llm_stats.get("total_cost", 0)
        self.llm_stats_label.setText(f"LLM tokens: {total_tokens} ({prompt_tokens} prompt, {completion_tokens} completion), cost: ${total_cost:.6f}")

    def _ui_update_console(self):
        """Update the status messages display."""
        # Update status panel
        status_text = ""
        for msg in self.status_messages:
            status_text += f"{msg}\n"
        self.status_text.setText(status_text)

    def _ui_update_translation_table(self):
        """Update the translation table."""
        # Update translation table only if there are changes
        all_entries = self.translation_entries
        
        # Temporarily disable auto-resizing
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        
        # Set row count if needed
        if self.table.rowCount() != len(all_entries):
            self.table.setRowCount(len(all_entries))
        
        # Update all rows
        for i, entry in enumerate(reversed(all_entries)):
            self._ui_update_table_row(i, entry)
        
        # Re-enable and perform one-time resize
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def _ui_update_table_row(self, row_idx: int, entry: dict):
        """Update a single table row."""
        time_item = self._create_table_item(entry["time"])
        source_item = self._create_table_item(entry["source_text"], multiline=True)
        lang_item = self._create_table_item(entry["lang"])
        status_item = self._create_table_item(entry["status"])
        
        translation = entry.get("translation", "")

        if entry.get("last_issues"):
            translation += "\n" + "=" * 40
            translation += "\nPrevious Attempts:"
            for attempt in entry["last_issues"]:
                translation += f"\n▶ Translation: '{attempt['translation']}'"
                if attempt['issues']:
                    translation += "\n * " + "\n *".join(attempt['issues'])

        translation_item = self._create_table_item(translation, multiline=True)
        
        # Set colors based on status
        color = self._get_status_color(entry["status"])
        for item in [time_item, source_item, lang_item, status_item, translation_item]:
            item.setBackground(color)
            
        self.table.setItem(row_idx, 0, time_item)
        self.table.setItem(row_idx, 1, source_item)
        self.table.setItem(row_idx, 2, lang_item)
        self.table.setItem(row_idx, 3, status_item)
        self.table.setItem(row_idx, 4, translation_item)
        
        # Adjust row height if needed
        self.table.resizeRowToContents(row_idx)

    def _ui_repaint_all(self):
        """Update the display with current data."""
        self._ui_update_statistics()
        self._ui_update_translation_table()
        self._ui_update_console()


    def _check_thread_status(self):
        """Check translation thread status and handle errors."""
        if self.error_queue is None:
            return
            
        try:
            error, tb = self.error_queue.get_nowait()
            logger.critical("Error in translation thread:")
            logger.critical(tb)
            self.stop()
        except queue.Empty:
            pass

    def start(self):
        """Start the graphical interface."""
        try:
            self.window.show()
            self.app.exec()
        except Exception as e:
            logger.error(f"Error in GraphicalUIManager: {str(e)}")
            logger.error(traceback.format_exc())
            self.stop()
            raise
        
    def stop(self):
        """Stop the graphical interface."""
        try:
            self.should_stop = True
            if self.app:
                self.app.quit()
        except Exception as e:
            logger.error(f"Error stopping GraphicalUIManager: {str(e)}")
            logger.error(traceback.format_exc())

    def set_thread_info(self, translation_thread, error_queue):
        """Set translation thread and error queue for monitoring."""
        self.translation_thread = translation_thread
        self.error_queue = error_queue

    def debug(self, message: str):
        """Thread-safe debug message."""
        self.signals.debug_signal.emit(message)

    def info(self, message: str):
        """Thread-safe info message."""
        self.signals.info_signal.emit(message)

    def warning(self, message: str):
        """Thread-safe warning message."""
        self.signals.warning_signal.emit(message)

    def error(self, message: str):
        """Thread-safe error message."""
        self.signals.error_signal.emit(message)

    def critical(self, message: str):
        """Thread-safe critical message."""
        self.signals.critical_signal.emit(message)

    def _debug(self, message: str):
        """Internal debug handler (runs in main thread)."""
        self.status_messages.append(f"[DEBUG] {message}")
        self._ui_update_console()

    def _info(self, message: str):
        """Internal info handler (runs in main thread)."""
        self.status_messages.append(f"[INFO] {message}")
        self._ui_update_console()

    def _warning(self, message: str):
        """Internal warning handler (runs in main thread)."""
        self.status_messages.append(f"[WARNING] {message}")
        self._ui_update_console()

    def _error(self, message: str):
        """Internal error handler (runs in main thread)."""
        self.status_messages.append(f"[ERROR] {message}")
        self._ui_update_console()

    def _critical(self, message: str):
        """Internal critical handler (runs in main thread)."""
        self.status_messages.append(f"[CRITICAL] {message}")
        self._ui_update_console()


    def set_translation_list(self, missing_translations: Dict[str, List[Dict[str, Any]]]):
        """Set the translation list."""

        new_items = []
        for lang, items in missing_translations.items():
            for item in items:
                item['lang'] = lang
                item['time'] = datetime.now().strftime("%H:%M:%S")
                item['status'] = StatusIcons.WAITING
                new_items.append(item)

        self.signals.set_translation_list_signal.emit(new_items)
    def _set_translation_list(self, new_items: List[Dict[str, Any]]):
        """Set the translation list."""
        self.translation_entries.extend(new_items)
        self._ui_repaint_all()

    def update_translation_item(self, item: Dict[str, Any]):
        """Update a single translation item."""
        self.signals.update_translation_item_signal.emit(item)
    def _update_translation_item(self, item: Dict[str, Any]):
        """Update a single translation item."""
        idx = self._find_item_index(item)
        if idx != -1:
            self._ui_update_table_row(idx, item)
            return
        
        self.critical(f"Failed to find translation item: {item}")


    def on_translation_started(self, item):
        """Thread-safe add translation entry."""
        self.signals.on_translation_started_signal.emit(item)
    def _on_translation_started(self, item):
        """Internal add translation handler (runs in main thread)."""
        item["status"] = StatusIcons.TRANSLATING + " Translating"
        self._update_translation_item(item)

    def on_translation_ended(self, item):
        """Thread-safe complete translation."""
        self.signals.on_translation_ended_signal.emit(item)
    def _on_translation_ended(self, item):
        """Internal complete translation handler (runs in main thread)."""
        is_error = item.get("error")
        item["status"] = StatusIcons.SUCCESS if not is_error else StatusIcons.FAILED
        self.overall_stats["total_attempts"] += 1
        if is_error:
            self.overall_stats["failed"] += 1
        else:
            self.overall_stats["successful"] += 1
        self._update_translation_item(item)

    def start_new_batch(self):
        """Thread-safe start new batch."""
        self.signals.start_new_batch_signal.emit()
    def _start_new_batch(self):
        # Internal start new batch handler (runs in main thread).
        pass

    def add_term_base_entry(self, term: str, lang: str, translation: str = "", comment: str = ""):
        """Thread-safe add term base entry."""
        return
        status = StatusIcons.INFO
        full_translation = translation
        if comment:
            full_translation += f" (Comment: {comment})"
        self.on_translation_started(term, lang, status, full_translation, [], "term_base")
        
    def _add_term_base_entry(self, term: str, lang: str, translation: str, comment: str):
        """Internal add term base entry handler (runs in main thread)."""
        return
        status = "📖"
        full_translation = translation
        if comment:
            full_translation += f" (Comment: {comment})"
            
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "term_base",
            "source": term,
            "lang": lang,
            "status": status,
            "translation": full_translation,
            "issues": []
        }
        #self._current_batch.append(entry)
        #self._update_display()
