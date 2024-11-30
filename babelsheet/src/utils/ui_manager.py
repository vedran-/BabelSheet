from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.text import Text
from collections import deque
from datetime import datetime
import logging

class UIManager:
    def __init__(self, max_history: int = 100, status_lines: int = 6):
        """Initialize UI Manager.
        
        Args:
            max_history: Maximum number of translation entries to keep in history
            status_lines: Number of status lines to show at the bottom
        """
        self.console = Console()
        self.translation_history = deque(maxlen=max_history)
        self.status_messages = deque(maxlen=status_lines)
        self.live: Optional[Live] = None
        self._current_batch: List[Dict] = []
        
    def _create_layout(self) -> Layout:
        """Create the main layout with translations at top and status at bottom."""
        layout = Layout()
        layout.split_column(
            Layout(name="translations", ratio=3),
            Layout(name="status", ratio=1, minimum_size=8)
        )
        return layout
        
    def _create_translation_table(self) -> Table:
        """Create the translation progress table."""
        table = Table(box=box.ROUNDED, expand=True, show_footer=False)
        table.add_column("Time", style="cyan")
        table.add_column("Source Text")
        table.add_column("Language", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Translation", overflow="fold")
        
        # Add historical entries
        for entry in self.translation_history:
            table.add_row(
                entry["time"],
                entry["source"],
                entry["lang"],
                entry["status"],
                entry["translation"]
            )
            
        # Add current batch entries
        for entry in self._current_batch:
            table.add_row(
                entry["time"],
                entry["source"],
                entry["lang"],
                entry["status"],
                entry["translation"]
            )
            
        return Panel(table, title="🔄 Translation Progress", border_style="blue")
        
    def _create_status_panel(self) -> Panel:
        """Create the status messages panel."""
        status_text = Text()
        for msg in self.status_messages:
            if isinstance(msg, Text):
                status_text.append(msg)
                status_text.append("\n")
            else:
                status_text.append(str(msg) + "\n")
            
        return Panel(status_text, title="📋 Status Messages", border_style="yellow")
        
    def _update_display(self):
        """Update the live display."""
        if self.live:
            layout = self._create_layout()
            layout["translations"].update(self._create_translation_table())
            layout["status"].update(self._create_status_panel())
            self.live.update(layout)
            
    def start(self):
        """Start the live display."""
        layout = self._create_layout()
        self.live = Live(layout, console=self.console, refresh_per_second=4)
        self.live.start()
        
    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None
            
    def add_translation_entry(self, source: str, lang: str, status: str = "⏳ Pending", 
                            translation: str = "", error: str = ""):
        """Add a new translation entry."""
        time = datetime.now().strftime("%H:%M:%S")
        translation_text = f"{translation} {error}" if error else translation
        
        entry = {
            "time": time,
            "source": source,
            "lang": lang,
            "status": status,
            "translation": translation_text.strip()
        }
        
        # If entry is in current batch, update it
        for i, batch_entry in enumerate(self._current_batch):
            if batch_entry["source"] == source and batch_entry["lang"] == lang:
                self._current_batch[i] = entry
                self._update_display()
                return
                
        # If entry not in current batch, add it
        self._current_batch.append(entry)
        self._update_display()
        
    def complete_translation(self, source: str, lang: str, translation: str, error: str = ""):
        """Mark a translation as complete."""
        status = "❌ Failed" if error else "✓ Done"
        self.add_translation_entry(source, lang, status, translation, error)
        
    def start_new_batch(self):
        """Move current batch to history and start a new one."""
        self.translation_history.extend(self._current_batch)
        self._current_batch = []
        self._update_display()
        
    def add_status(self, message: str, level: str = "info"):
        """Add a status message."""
        time = datetime.now().strftime("%H:%M:%S")
        style = {
            "debug": "dim",
            "info": "white",
            "warning": "yellow",
            "error": "red bold",
            "critical": "red bold reverse"
        }.get(level, "white")
        
        msg = Text()
        msg.append(f"[{time}] ", style="cyan")
        msg.append(message, style=style)
        
        self.status_messages.append(msg)
        self._update_display()
        
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