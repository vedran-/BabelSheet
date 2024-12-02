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

logger = logging.getLogger(__name__)

class UISignals(QObject):
    """Signals for thread-safe UI updates."""
    debug_signal = pyqtSignal(str)
    info_signal = pyqtSignal(str)
    warning_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    critical_signal = pyqtSignal(str)
    add_translation_signal = pyqtSignal(str, str, str, str, str, object)  # source, lang, status, translation, entry_type, issues
    complete_translation_signal = pyqtSignal(str, str, str, str)
    start_new_batch_signal = pyqtSignal()
    add_term_base_signal = pyqtSignal(str, str, str, str)

class GraphicalUIManager:
    def __init__(self, max_history: int = 100, status_lines: int = 6, llm_handler: LLMHandler = None):
        """Initialize Graphical UI Manager."""
        self.app = QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("BabelSheet Translator")
        self.window.resize(1200, 800)
        
        # Create signals object in the main thread
        self.signals = UISignals()
        
        # Connect signals to slots
        self.signals.debug_signal.connect(self._debug, Qt.ConnectionType.QueuedConnection)
        self.signals.info_signal.connect(self._info, Qt.ConnectionType.QueuedConnection)
        self.signals.warning_signal.connect(self._warning, Qt.ConnectionType.QueuedConnection)
        self.signals.error_signal.connect(self._error, Qt.ConnectionType.QueuedConnection)
        self.signals.critical_signal.connect(self._critical, Qt.ConnectionType.QueuedConnection)
        self.signals.add_translation_signal.connect(self._add_translation_entry, Qt.ConnectionType.QueuedConnection)
        self.signals.complete_translation_signal.connect(self._complete_translation, Qt.ConnectionType.QueuedConnection)
        self.signals.start_new_batch_signal.connect(self._start_new_batch, Qt.ConnectionType.QueuedConnection)
        
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
        self.translation_history = deque(maxlen=max_history)
        self.status_messages = deque(maxlen=status_lines)
        self._current_batch = []
        self.llm_handler = llm_handler
        
        # Statistics
        self.overall_stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0
        }
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(250)  # Update every 250ms
        
    def _setup_stats_panel(self):
        """Setup the statistics panel."""
        self.stats_widget = QWidget()
        self.stats_layout = QGridLayout(self.stats_widget)
        
        self.total_label = QLabel("Total Attempts: 0")
        self.success_label = QLabel("Successful: 0 (0%)")
        self.failed_label = QLabel("Failed: 0 (0%)")
        
        self.stats_layout.addWidget(QLabel("📊 Statistics"), 0, 0, 1, 2)
        self.stats_layout.addWidget(self.total_label, 1, 0)
        self.stats_layout.addWidget(self.success_label, 1, 1)
        self.stats_layout.addWidget(self.failed_label, 2, 0)
        
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
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        self.layout.addWidget(self.table)
        
    def _create_table_item(self, text: str, multiline: bool = False) -> QTableWidgetItem:
        """Create a table item with proper formatting."""
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make item read-only
        if multiline:
            # Enable text wrapping for multiline items
            item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        return item
        
    def _setup_status_panel(self):
        """Setup the status messages panel."""
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.layout.addWidget(self.status_text)
        
    def _get_status_color(self, status: str, entry_type: str) -> QColor:
        """Get the color for a status."""
        if entry_type == "term_base":
            return QColor(0, 0, 255, 50)  # Blue with alpha
        elif status.startswith("✓"):
            return QColor(0, 255, 0, 50)  # Green with alpha
        elif status.startswith("❌"):
            return QColor(255, 0, 0, 50)  # Red with alpha
        elif status.startswith("⏳") or status.startswith("⌛"):
            return QColor(255, 255, 0, 50)  # Yellow with alpha
        else:
            return QColor(128, 128, 128, 50)  # Gray with alpha

    def _format_translation_text(self, translation: str, entry: dict) -> str:
        """Format translation text with issues and previous attempts."""
        text_parts = []
        
        # Add current translation or status
        if translation.startswith("⏳") or translation.startswith("⌛"):
            text_parts.append(translation)
        else:
            text_parts.append(translation)
            
            # Add current issues if any
            if entry.get("issues"):
                text_parts.append("\n" + "=" * 40)
                text_parts.append("Current Issues:")
                for issue in entry["issues"]:
                    text_parts.append(f"• {issue}")
            
            # Add previous attempts if any
            if entry.get("last_issues"):
                text_parts.append("\n" + "=" * 40)
                text_parts.append("Previous Attempts:")
                for attempt in entry["last_issues"]:
                    text_parts.append(f"\n▶ Translation: {attempt['translation']}")
                    if attempt['issues']:
                        text_parts.append("Issues:")
                        if isinstance(attempt['issues'], list):
                            for issue in attempt['issues']:
                                text_parts.append(f"• {issue}")
                        else:
                            text_parts.append(f"• {attempt['issues']}")
                    text_parts.append("-" * 40)
        
        return "\n".join(text_parts)

    def _update_display(self):
        """Update the display with current data."""
        # Update statistics
        total = self.overall_stats['total_attempts']
        if total > 0:
            success_rate = (self.overall_stats['successful'] / total) * 100
            fail_rate = (self.overall_stats['failed'] / total) * 100
        else:
            success_rate = fail_rate = 0.0
            
        self.total_label.setText(f"Total Attempts: {total}")
        self.success_label.setText(f"Successful: {self.overall_stats['successful']} ({success_rate:.1f}%)")
        self.failed_label.setText(f"Failed: {self.overall_stats['failed']} ({fail_rate:.1f}%)")
        
        # Update translation table
        all_entries = list(self._current_batch)
        all_entries.extend(self.translation_history)
        
        self.table.setRowCount(len(all_entries))
        
        for i, entry in enumerate(reversed(all_entries)):
            time_item = self._create_table_item(entry["time"])
            source_item = self._create_table_item(entry["source"], multiline=True)
            lang_item = self._create_table_item(entry["lang"])
            status_item = self._create_table_item(entry["status"])
            
            translation = entry["translation"]
            if isinstance(translation, str):
                trans_text = self._format_translation_text(translation, entry)
            else:
                trans_text = translation.plain
                
            translation_item = self._create_table_item(trans_text, multiline=True)
            
            # Set colors based on status
            color = self._get_status_color(entry["status"], entry["type"])
            for item in [time_item, source_item, lang_item, status_item, translation_item]:
                item.setBackground(color)
                
            self.table.setItem(i, 0, time_item)
            self.table.setItem(i, 1, source_item)
            self.table.setItem(i, 2, lang_item)
            self.table.setItem(i, 3, status_item)
            self.table.setItem(i, 4, translation_item)
            
            # Adjust row height if needed
            self.table.resizeRowToContents(i)
            
        # Update status panel
        status_text = ""
        for msg in self.status_messages:
            status_text += f"{msg}\n"
        self.status_text.setText(status_text)
        
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
            if self.app:
                self.app.quit()
        except Exception as e:
            logger.error(f"Error stopping GraphicalUIManager: {str(e)}")
            logger.error(traceback.format_exc())
        
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

    def add_translation_entry(self, source: str, lang: str, status: str = "⏳", translation: str = "", issues: list = None, entry_type: str = "translation"):
        """Thread-safe add translation entry."""
        if issues is None:
            issues = []
        self.signals.add_translation_signal.emit(source, lang, status, translation, entry_type, issues)

    def complete_translation(self, source: str, lang: str, translation: str, error: str = ""):
        """Thread-safe complete translation."""
        self.signals.complete_translation_signal.emit(source, lang, translation, error)

    def start_new_batch(self):
        """Thread-safe start new batch."""
        self.signals.start_new_batch_signal.emit()

    def add_term_base_entry(self, term: str, lang: str, translation: str = "", comment: str = ""):
        """Thread-safe add term base entry."""
        status = "📖"
        full_translation = translation
        if comment:
            full_translation += f" (Comment: {comment})"
        self.add_translation_entry(term, lang, status, full_translation, [], "term_base")

    def _debug(self, message: str):
        """Internal debug handler (runs in main thread)."""
        self.status_messages.append(f"[DEBUG] {message}")
        self._update_display()

    def _info(self, message: str):
        """Internal info handler (runs in main thread)."""
        self.status_messages.append(f"[INFO] {message}")
        self._update_display()

    def _warning(self, message: str):
        """Internal warning handler (runs in main thread)."""
        self.status_messages.append(f"[WARNING] {message}")
        self._update_display()

    def _error(self, message: str):
        """Internal error handler (runs in main thread)."""
        self.status_messages.append(f"[ERROR] {message}")
        self._update_display()

    def _critical(self, message: str):
        """Internal critical handler (runs in main thread)."""
        self.status_messages.append(f"[CRITICAL] {message}")
        self._update_display()

    def _add_translation_entry(self, source: str, lang: str, status: str, translation: str, entry_type: str, issues: list):
        """Internal add translation handler (runs in main thread)."""
        # Find existing entry for this source and language
        existing_entry = None
        for entry in self._current_batch:
            if entry["source"] == source and entry["lang"] == lang and entry["type"] == entry_type:
                existing_entry = entry
                break
                
        if existing_entry:
            # Store current state as previous attempt if there was a translation
            if existing_entry.get("translation"):
                if "last_issues" not in existing_entry:
                    existing_entry["last_issues"] = []
                existing_entry["last_issues"].append({
                    "translation": existing_entry["translation"],
                    "issues": existing_entry.get("issues", [])
                })
            # Update existing entry
            existing_entry.update({
                "status": status,
                "translation": translation,
                "issues": issues
            })
        else:
            # Create new entry
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": entry_type,
                "source": source,
                "lang": lang,
                "status": status,
                "translation": translation,
                "issues": issues,
                "last_issues": []
            }
            self._current_batch.append(entry)
            
        self._update_display()

    def _complete_translation(self, source: str, lang: str, translation: str, error: str = ""):
        """Internal complete translation handler (runs in main thread)."""
        for entry in self._current_batch:
            if entry["source"] == source and entry["lang"] == lang:
                entry["translation"] = translation
                entry["status"] = "✓" if not error else "❌"
                if error:
                    entry["issues"] = [error]
                self._update_display()
                return

    def _start_new_batch(self):
        """Internal start new batch handler (runs in main thread)."""
        self.translation_history.extend(self._current_batch)
        self._current_batch = []
        self._update_display()
        
    def _add_term_base_entry(self, term: str, lang: str, translation: str, comment: str):
        """Internal add term base entry handler (runs in main thread)."""
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
        self._current_batch.append(entry)
        self._update_display()
        
    def add_term_base_entry(self, term: str, lang: str, translation: str = "", comment: str = ""):
        """Add a term base entry."""
        status = "📖"
        full_translation = translation
        if comment:
            full_translation += f" (Comment: {comment})"
        self.add_translation_entry(term, lang, status, full_translation, entry_type="term_base") 