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
    add_translation_signal = pyqtSignal(str, str, str)
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
        # Move signals to the main thread
        self.signals.moveToThread(self.app.thread())
        
        # Connect signals to slots
        self.signals.debug_signal.connect(self._debug, Qt.ConnectionType.QueuedConnection)
        self.signals.info_signal.connect(self._info, Qt.ConnectionType.QueuedConnection)
        self.signals.warning_signal.connect(self._warning, Qt.ConnectionType.QueuedConnection)
        self.signals.error_signal.connect(self._error, Qt.ConnectionType.QueuedConnection)
        self.signals.critical_signal.connect(self._critical, Qt.ConnectionType.QueuedConnection)
        self.signals.add_translation_signal.connect(self._add_translation_entry, Qt.ConnectionType.QueuedConnection)
        self.signals.complete_translation_signal.connect(self._complete_translation, Qt.ConnectionType.QueuedConnection)
        self.signals.start_new_batch_signal.connect(self._start_new_batch, Qt.ConnectionType.QueuedConnection)
        self.signals.add_term_base_signal.connect(self._add_term_base_entry, Qt.ConnectionType.QueuedConnection)
        
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
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Time", "Source Text", "Language", "Translation"])
        
        # Set column stretching
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        self.layout.addWidget(self.table)
        
    def _setup_status_panel(self):
        """Setup the status messages panel."""
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.layout.addWidget(self.status_text)
        
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
            time_item = QTableWidgetItem(entry["time"])
            source_item = QTableWidgetItem(f"{entry['status']} {entry['source']}")
            lang_item = QTableWidgetItem(entry["lang"])
            
            translation = entry["translation"]
            if isinstance(translation, str):
                trans_text = translation
            else:
                trans_text = translation.plain
                
            translation_item = QTableWidgetItem(trans_text)
            
            # Set colors based on status
            if entry["status"].startswith("✓"):
                color = QColor(0, 255, 0, 50)  # Green with alpha
            elif entry["status"].startswith("❌"):
                color = QColor(255, 0, 0, 50)  # Red with alpha
            elif entry["type"] == "term_base":
                color = QColor(0, 0, 255, 50)  # Blue with alpha
            else:
                color = QColor(255, 255, 0, 50)  # Yellow with alpha
                
            for item in [time_item, source_item, lang_item, translation_item]:
                item.setBackground(color)
                
            self.table.setItem(i, 0, time_item)
            self.table.setItem(i, 1, source_item)
            self.table.setItem(i, 2, lang_item)
            self.table.setItem(i, 3, translation_item)
                
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

    def add_translation_entry(self, source: str, lang: str, status: str = "⏳"):
        """Thread-safe add translation entry."""
        self.signals.add_translation_signal.emit(source, lang, status)

    def complete_translation(self, source: str, lang: str, translation: str, error: str = ""):
        """Thread-safe complete translation."""
        self.signals.complete_translation_signal.emit(source, lang, translation, error)

    def start_new_batch(self):
        """Thread-safe start new batch."""
        self.signals.start_new_batch_signal.emit()

    def add_term_base_entry(self, term: str, lang: str, translation: str = "", comment: str = ""):
        """Thread-safe add term base entry."""
        self.signals.add_term_base_signal.emit(term, lang, translation, comment)

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

    def _add_translation_entry(self, source: str, lang: str, status: str):
        """Internal add translation handler (runs in main thread)."""
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "translation",
            "source": source,
            "lang": lang,
            "status": status,
            "translation": ""
        }
        self._current_batch.append(entry)
        self._update_display()

    def _complete_translation(self, source: str, lang: str, translation: str, error: str):
        """Internal complete translation handler (runs in main thread)."""
        for entry in self._current_batch:
            if entry["source"] == source and entry["lang"] == lang:
                entry["translation"] = translation
                entry["status"] = "✓" if not error else "❌"
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
            "translation": full_translation
        }
        self._current_batch.append(entry)
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
            "translation": full_translation
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