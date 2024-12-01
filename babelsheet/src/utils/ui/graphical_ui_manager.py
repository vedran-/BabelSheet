from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QTableWidget, QTableWidgetItem, QTextEdit, QLabel,
                          QGridLayout, QHeaderView)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from rich.text import Text
from ..llm_handler import LLMHandler
import threading

class GraphicalUIManager:
    def __init__(self, max_history: int = 100, status_lines: int = 6, llm_handler: LLMHandler = None):
        """Initialize Graphical UI Manager.
        
        Args:
            max_history: Maximum number of translation entries to keep in history
            status_lines: Number of status lines to show at the bottom
            llm_handler: LLMHandler instance
        """
        self.app = QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("BabelSheet Translator")
        self.window.resize(1200, 800)
        
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
        """Start the graphical interface in a separate thread."""
        self.window.show()
        # Run the Qt event loop in a separate thread
        self.qt_thread = threading.Thread(target=self.app.exec)
        self.qt_thread.daemon = True  # Make thread daemon so it exits when main program exits
        self.qt_thread.start()
        
    def stop(self):
        """Stop the graphical interface."""
        self.app.quit()
        if hasattr(self, 'qt_thread'):
            self.qt_thread.join()
        
    def add_translation_entry(self, source: str, lang: str, status: str = "⏳", 
                            translation: str = "", error: str = "", entry_type: str = "translation"):
        """Add a new translation entry."""
        time = datetime.now().strftime("%H:%M:%S")
        
        entry = {
            "time": time,
            "type": entry_type,
            "source": source,
            "lang": lang,
            "status": status,
            "translation": translation if translation else ""
        }
        
        # Update or add entry
        for i, batch_entry in enumerate(self._current_batch):
            if batch_entry["source"] == source and batch_entry["lang"] == lang:
                self._current_batch[i] = entry
                return
                
        self._current_batch.append(entry)
        
    def complete_translation(self, source: str, lang: str, translation: str, error: str = ""):
        """Mark a translation as complete."""
        status = "❌" if error else "✓"
        
        self.overall_stats['total_attempts'] += 1
        if error:
            self.overall_stats['failed'] += 1
        else:
            self.overall_stats['successful'] += 1
            
        self.add_translation_entry(source, lang, status, translation, error)
        
    def start_new_batch(self, llm_handler=None):
        """Move current batch to history and start a new one."""
        self.translation_history.extend(self._current_batch)
        self._current_batch = []
        
    def add_status(self, message: str, level: str = "info"):
        """Add a status message."""
        time = datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            "debug": "gray",
            "info": "white",
            "warning": "yellow",
            "error": "red",
            "critical": "darkred"
        }
        
        color = color_map.get(level, "white")
        formatted_message = f'<font color="{color}">[{time}] {message}</font><br>'
        self.status_text.append(formatted_message)
        
    def info(self, message: str):
        self.add_status(message, "info")
        
    def warning(self, message: str):
        self.add_status(message, "warning")
        
    def error(self, message: str):
        self.add_status(message, "error")
        
    def critical(self, message: str):
        self.add_status(message, "critical")
        
    def debug(self, message: str):
        self.add_status(message, "debug")
        
    def add_term_base_entry(self, term: str, lang: str, translation: str = "", comment: str = ""):
        """Add a term base entry."""
        status = "📖"
        full_translation = translation
        if comment:
            full_translation += f" (Comment: {comment})"
        self.add_translation_entry(term, lang, status, full_translation, entry_type="term_base") 