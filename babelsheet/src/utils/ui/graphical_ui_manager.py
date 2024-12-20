from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QTableWidget, QTableWidgetItem, QTextEdit, QLabel,
                          QGridLayout, QHeaderView, QProgressBar)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QPalette, QIcon
from ..llm_handler import LLMHandler
import logging
import traceback
import queue
import os
import platform

logger = logging.getLogger(__name__)

class StatusIcons:
    WAITING = " "
    TRANSLATING = "✍️"
    VALIDATING = "🔍"
    RETRYING = "⟳"
    SUCCESS = "✔️"
    FAILED = "❌"
    INFO = "💡"
    WARNING = "⚠️"


class UISignals(QObject):
    """Signals for thread-safe UI updates."""
    debug_signal = pyqtSignal(str)
    info_signal = pyqtSignal(str)
    warning_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    critical_signal = pyqtSignal(str)
    set_translation_list_signal = pyqtSignal(list) # new_items
    update_translation_item_signal = pyqtSignal(object) # item

    begin_table_update_signal = pyqtSignal()
    end_table_update_signal = pyqtSignal()

    on_translation_started_signal = pyqtSignal(object)  # item
    on_translation_ended_signal = pyqtSignal(object)

    start_new_batch_signal = pyqtSignal()
    add_term_base_signal = pyqtSignal(str, str, str, str)
    check_thread_signal = pyqtSignal()  # New signal for checking thread status

class GraphicalUIManager:
    def __init__(self, ctx, max_history: int = 100, status_lines: int = 10):
        """Initialize Graphical UI Manager."""
        self.ctx = ctx
        self.config = ctx.config
        self.llm_handler = ctx.llm_handler
        self.app = QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("BabelSheet Translator")
        self.window.resize(1200, 800)
        self._set_window_icon()
        
        # Time tracking
        self.start_time = datetime.now()
        self.translation_times = []
        self.current_translation_start = None
        self.failed_translation_times = []
        self.last_successful_translation = None
        self.interrupted_translations = []
        
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

        self.signals.begin_table_update_signal.connect(self._begin_table_update, Qt.ConnectionType.QueuedConnection)
        self.signals.end_table_update_signal.connect(self._end_table_update, Qt.ConnectionType.QueuedConnection)

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

    def _set_window_icon(self):
        """Set the application window icon with proper sizing for different platforms."""
        try:
            # Get the absolute path to the icon file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(current_dir, '..', '..', '..', 'assets', 'icon.png')
            
            if os.path.exists(icon_path):
                app_icon = QIcon(icon_path)
                
                # Set multiple sizes for better scaling
                for size in [16, 24, 32, 48, 64, 128, 256]:
                    app_icon.addFile(icon_path, QSize(size, size))
                
                self.window.setWindowIcon(app_icon)
                self.app.setWindowIcon(app_icon)
                
                # Special handling for Windows taskbar icon
                if platform.system() == 'Windows':
                    try:
                        from ctypes import windll
                        windll.shell32.SetCurrentProcessExplicitAppUserModelID('BabelSheet.Translator')
                    except Exception as e:
                        self.logger.debug(f"Windows-specific icon setting failed: {e}")
                
            else:
                self.logger.warning(f"Icon file not found at {icon_path}")
        except Exception as e:
            self.logger.error(f"Failed to set window icon: {e}")
    def _setup_stats_panel(self):
        """Setup the statistics panel."""
        self.stats_widget = QWidget()
        self.stats_layout = QGridLayout(self.stats_widget)
        
        self.llm_stats_label = QLabel("LLM Stats: 0 tokens, $0.00")
        self.llm_stats_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.languages_label = QLabel("📊 Languages: 'en' (source), 'es' (target)")
        self.time_stats_label = QLabel("⏱️ Time Stats: Running for 0:00:00")
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)
        
        self.stats_layout.addWidget(self.languages_label, 0, 0, 1, 1)
        self.stats_layout.addWidget(self.time_stats_label, 0, 1)
        self.stats_layout.addWidget(self.llm_stats_label, 0, 2)
        self.stats_layout.addWidget(self.progress_bar, 1, 0, 1, 3)
        
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
        self.status_text.setMaximumHeight(250)
        self.status_text.setAcceptRichText(True)

        # Set default font and styling
        #font = self.status_text.font()
        #font.setPointSize(10)
        #self.status_text.setFont(font)
        self.layout.addWidget(self.status_text)

    def _find_item_index(self, item) -> int:
        idx = item.get('idx', -1)
        if idx != -1:
            return idx
        
        raise Exception(f"Item {item} has no index!")

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
            or status.startswith(StatusIcons.RETRYING):
            return QColor(255, 255, 0, 50)  # Yellow with alpha
        else:
            return QColor(192, 192, 192, 50)  # Gray with alpha

    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta to a readable string without microseconds."""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _get_time_since_last_success(self) -> str:
        """Get formatted time since last successful translation."""
        if not self.last_successful_translation:
            return "N/A"
        time_since = datetime.now() - self.last_successful_translation
        return self._format_timedelta(time_since)

    def _ui_update_statistics(self):
        """Update the statistics display."""
        total = self.overall_stats['total_attempts']
        total_translations = len(self.translation_entries)

        if total > 0:
            success_rate = (self.overall_stats['successful'] / total) * 100
            fail_rate = (self.overall_stats['failed'] / total) * 100
            
            # Calculate average translation time (successful translations only)
            if self.translation_times:
                avg_time = sum(self.translation_times, timedelta()) / len(self.translation_times)
            else:
                avg_time = timedelta()
            
            # Include failed and interrupted translations in the average
            all_times = self.translation_times + self.failed_translation_times + self.interrupted_translations
            if all_times:
                overall_avg = sum(all_times, timedelta()) / len(all_times)
            else:
                overall_avg = avg_time
            
            remaining_translations = total_translations - self.overall_stats['successful']
            estimated_completion = avg_time * remaining_translations if self.translation_times else timedelta()
        else:
            success_rate = fail_rate = 0.0
            avg_time = overall_avg = timedelta()
            estimated_completion = timedelta()

        # Update progress bar
        if total_translations > 0:
            progress = (total / total_translations) * 100
            self.progress_bar.setValue(int(progress))
            self.progress_bar.setFormat(
                f"Progress: {progress:.1f}% ({total}/{total_translations}) - "
                f"{StatusIcons.SUCCESS} {self.overall_stats['successful']} ({success_rate:.1f}%), "
                f"{StatusIcons.FAILED} {self.overall_stats['failed']} ({fail_rate:.1f}%)"
            )
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Initializing...")

        # Update time stats with color coding
        runtime = datetime.now() - self.start_time
        time_stats = [
            f"⏱️ <b>Runtime:</b> <font color='#ffffaf'>{self._format_timedelta(runtime)}</font>"
        ]
        if total > 0 and remaining_translations > 0:
            time_stats.append(f"<b>ETA:</b> <font color='#ffffaf'>{self._format_timedelta(estimated_completion)}</font>")
        self.time_stats_label.setText(" | ".join(time_stats))
        
        if self.llm_handler:
            llm_stats = self.llm_handler.get_usage_stats()
            prompt_tokens = llm_stats.get("prompt_tokens", 0)
            completion_tokens = llm_stats.get("completion_tokens", 0)
            total_tokens = llm_stats.get("total_tokens", 0)
            total_cost = llm_stats.get("total_cost", 0)
            self.llm_stats_label.setText(f"<b>LLM tokens:</b> <font color='#ffffaf'><b>{total_tokens}</b></font> "
                                       f"(<font color='#ffffaf'><b>{prompt_tokens}</b></font> prompt, "
                                       f"<font color='#ffffaf'><b>{completion_tokens}</b></font> completion), "
                                       f"cost: <font color='#ffffaf'><b>${total_cost:.6f}</b></font>")
        else:
            self.llm_stats_label.setText("<b>LLM:</b> <font color='gray'>N/A</font>")

        if hasattr(self.ctx, 'source_lang') and self.ctx.source_lang:
            self.languages_label.setText(f"📊 <b>Translating from:</b> <font color='cyan'><b>{self.ctx.source_lang}</b></font> ➜ "
                                   f"<font color='cyan'><b>{', '.join(self.ctx.target_langs)}</b></font> (target)")
        else:
            # Possible icons: 🔍 (magnifying glass), ⚡ (lightning), 🔧 (wrench), ⚙️ (gear), 📝 (memo), ✨ (sparkles)
            self.languages_label.setText(f"🔍 <b>Checking spacing for:</b> <font color='cyan'><b>{', '.join(self.ctx.target_langs)}</b></font> (target)")

    def _ui_update_console(self):
        """Update the status messages display."""
        # Update status panel
        # Get current scroll position and maximum before updating text
        scrollbar = self.status_text.verticalScrollBar()
        old_value = scrollbar.value()
        was_at_bottom = old_value == scrollbar.maximum()
        
        # Update the text
        status_text = ""
        for msg in self.status_messages:
            status_text += f"{msg}<br>\n"
        self.status_text.setHtml(status_text)
        
        # Only auto-scroll if we were already at the bottom
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(old_value)

    def _ui_repaint_translation_table(self):
        """Update the translation table."""
        # Update translation table only if there are changes
        all_entries = self.translation_entries
        
        self._begin_table_update()
        # Temporarily disable auto-resizing
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        
        # Set row count if needed
        if self.table.rowCount() != len(all_entries):
            self.table.setRowCount(len(all_entries))
        
        # Update all rows
        for i, entry in enumerate(all_entries):
            self._ui_update_table_row(i, entry)
        
        # Re-enable and perform one-time resize
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._end_table_update()

    def _ui_update_table_row(self, row_idx: int, entry: dict):
        """Update a single table row."""
        # Prepare the translation text
        translation = f"{entry.get('translation', '')}".replace('\\n', '\n')

        override = entry.get("override", '')
        if override and override != '':
            translation += f"\n ●●● OVERRIDE validation, because: {override}"

        last_issues = entry.get("last_issues")
        if last_issues and len(last_issues) > 0:
            translation += f"\n ----- Previous {len(last_issues)} failed attempts -----"
            for attempt in entry["last_issues"]:
                translation += f"\n● >>> {attempt['translation'].replace('\\n', '\n')} <<<"
                if attempt['issues']:
                    translation += "\n    - " + "\n    -".join(attempt['issues'])

        # Define the content for each column
        contents = [
            (entry["time"], False),
            (entry["source_text"].replace('\\n', '\n'), True),
            (entry["lang"], False),
            (entry["status"], False),
            (translation, True)
        ]

        color = self._get_status_color(entry["status"])

        # Update each column
        for col, (content, multiline) in enumerate(contents):
            item = self.table.item(row_idx, col)
            if item is None:
                item = self._create_table_item(content, multiline=multiline)
                self.table.setItem(row_idx, col, item)
            else:
                item.setText(str(content))                
                if multiline:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setBackground(color)
        
        # Adjust row height if needed
        self.table.resizeRowToContents(row_idx)
        self.table.setRowHeight(row_idx, self.table.rowHeight(row_idx) + 4)

    def _ui_repaint_all(self):
        """Update the display with current data."""
        self._ui_update_statistics()
        self._ui_repaint_translation_table()
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
            # If there's an ongoing translation, mark it as interrupted
            if self.current_translation_start:
                translation_time = datetime.now() - self.current_translation_start
                self.interrupted_translations.append(translation_time)
                self.current_translation_start = None
            
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
    def _debug(self, message: str):
        """Internal debug handler (runs in main thread)."""
        runtime = datetime.now() - self.start_time
        self.status_messages.append(f"[{runtime}:DEBUG] {message}")
        self._ui_update_console()

    def info(self, message: str):
        """Thread-safe info message."""
        self.signals.info_signal.emit(message)
    def _info(self, message: str):
        """Internal info handler (runs in main thread)."""
        runtime = datetime.now() - self.start_time
        self.status_messages.append(f"[{runtime}:INFO] {message}")
        self._ui_update_console()

    def warning(self, message: str):
        """Thread-safe warning message."""
        self.signals.warning_signal.emit(message)
    def _warning(self, message: str):
        """Internal warning handler (runs in main thread)."""
        runtime = datetime.now() - self.start_time
        self.status_messages.append(f"[{runtime}:WARNING] {message}")
        self._ui_update_console()

    def error(self, message: str):
        """Thread-safe error message."""
        self.signals.error_signal.emit(message)
    def _error(self, message: str):
        """Internal error handler (runs in main thread)."""
        runtime = datetime.now() - self.start_time
        self.status_messages.append(f"[{runtime}:ERROR] {message}")
        self._ui_update_console()

    def critical(self, message: str):
        """Thread-safe critical message."""
        self.signals.critical_signal.emit(message)
    def _critical(self, message: str):
        """Internal critical handler (runs in main thread)."""
        runtime = datetime.now() - self.start_time
        self.status_messages.append(f"[{runtime}:CRITICAL] {message}")
        self._ui_update_console()

    def print_overall_stats(self):
        """Print overall statistics."""
        total = self.overall_stats['total_attempts']
        if total > 0:
            success_rate = (self.overall_stats['successful'] / total) * 100
            fail_rate = (self.overall_stats['failed'] / total) * 100
            
            if self.translation_times:
                avg_success_time = sum(self.translation_times, timedelta()) / len(self.translation_times)
            else:
                avg_success_time = timedelta()
                
            all_times = self.translation_times + self.failed_translation_times + self.interrupted_translations
            if len(all_times) > len(self.translation_times):
                avg_fail_time = sum(self.failed_translation_times, timedelta()) / len(self.failed_translation_times) if self.failed_translation_times else timedelta()
                avg_interrupt_time = sum(self.interrupted_translations, timedelta()) / len(self.interrupted_translations) if self.interrupted_translations else timedelta()
                overall_avg = sum(all_times, timedelta()) / len(all_times)
            else:
                avg_fail_time = avg_interrupt_time = timedelta()
                overall_avg = avg_success_time
        else:
            all_times = []
            success_rate = fail_rate = 0.0
            avg_success_time = avg_fail_time = avg_interrupt_time = overall_avg = timedelta()
            
        self.info("\n📊 Overall Translation Statistics")
        self.info("─" * 40)
        self.info(f"Total Runtime: {self._format_timedelta(datetime.now() - self.start_time)}")
        if self.last_successful_translation:
            self.info(f"Time Since Last Success: {self._get_time_since_last_success()}")
        self.info(f"Average Successful Translation Time: {self._format_timedelta(avg_success_time)}")
        if self.failed_translation_times:
            self.info(f"Average Failed Translation Time: {self._format_timedelta(avg_fail_time)}")
        if self.interrupted_translations:
            self.info(f"Average Interrupted Translation Time: {self._format_timedelta(avg_interrupt_time)}")
        if len(all_times) > len(self.translation_times):
            self.info(f"Overall Average Translation Time: {self._format_timedelta(overall_avg)}")
        self.info(f"Total Translation Attempts: {total}")
        self.info(f"Successful Translations: {self.overall_stats['successful']} ({success_rate:.1f}%)")
        self.info(f"Failed Translations: {self.overall_stats['failed']} ({fail_rate:.1f}%)")
        if self.interrupted_translations:
            self.info(f"Interrupted Translations: {len(self.interrupted_translations)}")

        if self.llm_handler:
            usage_stats = self.llm_handler.get_usage_stats()
            self.info("─" * 40)
            self.info(f"Total tokens used: {usage_stats['total_tokens']} ({usage_stats['prompt_tokens']} prompt + {usage_stats['completion_tokens']} completion)")
            self.info(f"Total cost: ${usage_stats['total_cost']:.4f}")

    def begin_table_update(self):
        """Start table updates."""
        self.signals.begin_table_update_signal.emit()
    def _begin_table_update(self):
        """Start table updates."""
        # Disable sorting and updates before making changes
        self.table.setSortingEnabled(False)
        self.table.setUpdatesEnabled(False)
        
        # Block signals and suspend layout updates
        self.table.blockSignals(True)
        self.table.setAutoScroll(False)
        self.table.horizontalHeader().setUpdatesEnabled(False)
        self.table.verticalHeader().setUpdatesEnabled(False)
        
        # Notify model of upcoming changes
        model = self.table.model()
        model.layoutAboutToBeChanged.emit()
        model.beginResetModel()

    def end_table_update(self):
        """Stop table updates."""
        self.signals.end_table_update_signal.emit()
    def _end_table_update(self):
        """Stop table updates."""
        # Re-enable updates and signals
        model = self.table.model()
        model.endResetModel()
        
        self.table.setUpdatesEnabled(True)
        self.table.blockSignals(False)
        self.table.setAutoScroll(True)
        #self.table.horizontalHeader().setUpdatesEnabled(True) 
        #self.table.verticalHeader().setUpdatesEnabled(True)
        
        # Force viewport update and re-enable sorting
        self.table.viewport().update()
        #self.table.setSortingEnabled(True)


    def set_translation_list(self, missing_translations: Dict[str, List[Dict[str, Any]]]):
        """Set the translation list."""

        idx = len(self.translation_entries)
        new_items = []
        for lang, items in missing_translations.items():
            for item in items:
                item['lang'] = lang
                item['time'] = datetime.now().strftime("%H:%M:%S")
                item['status'] = StatusIcons.WAITING
                item['idx'] = idx
                idx += 1
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
            item['idx'] = idx
            self.translation_entries[idx] = item
            self._ui_update_table_row(idx, item)
            return
        
        raise Exception(f"Failed to find translation item: {item}")


    def on_translation_started(self, item):
        """Thread-safe add translation entry."""
        self.current_translation_start = datetime.now()
        self.signals.on_translation_started_signal.emit(item)
    def _on_translation_started(self, item):
        """Internal add translation handler (runs in main thread)."""
        self._update_translation_item(item)

    def on_translation_ended(self, item):
        """Thread-safe complete translation."""
        if self.current_translation_start:
            translation_time = datetime.now() - self.current_translation_start
            if item.get("error"):
                self.failed_translation_times.append(translation_time)
            else:
                self.translation_times.append(translation_time)
                self.last_successful_translation = datetime.now()
            self.current_translation_start = None
        self.signals.on_translation_ended_signal.emit(item)
    def _on_translation_ended(self, item):
        """Internal complete translation handler (runs in main thread)."""
        is_error = item.get("error")
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

