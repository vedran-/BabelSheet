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
from .llm_handler import LLMHandler

class UIManager:
    def __init__(self, max_history: int = 100, status_lines: int = 6, llm_handler: LLMHandler = None):
        """Initialize UI Manager.
        
        Args:
            max_history: Maximum number of translation entries to keep in history
            status_lines: Number of status lines to show at the bottom
            llm_handler: LLMHandler instance
        """
        self.console = Console()
        self.llm_handler = llm_handler
        self.translation_history = deque(maxlen=max_history)
        self.status_messages = deque(maxlen=status_lines)
        self.live: Optional[Live] = None
        self._current_batch: List[Dict] = []
        
        # Add overall statistics tracking
        self.overall_stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0
        }
        
    def _create_layout(self) -> Layout:
        """Create the main layout with translations at top and status at bottom."""
        layout = Layout()
        layout.split_column(
            Layout(name="overall_stats", ratio=1, minimum_size=4),
            Layout(name="translations", ratio=3),
            Layout(name="status", ratio=1, minimum_size=8)
        )
        return layout
        
    def _create_translation_table(self) -> Table:
        """Create the translation progress table."""
        table = Table(box=box.ROUNDED, expand=True, show_footer=False)
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Source Text")
        table.add_column("Language", style="magenta")
        table.add_column("Translation", overflow="fold")
        
        # Combine current batch and history, but reverse the order
        all_entries = list(self._current_batch)
        all_entries.extend(self.translation_history)
        
        # Show newest entries first
        for entry in reversed(all_entries):
            row_style = "blue" if entry.get("type") == "term_base" else None
            
            # Extract just the icon from status and combine with source text
            status = entry["status"]
            if isinstance(status, str):
                icon = status.split()[0]  # Get just the emoji
            else:
                icon = "⏳"  # Default icon
                
            source_with_status = Text()
            source_with_status.append(icon + " ")  # Add icon with a space
            source_with_status.append(entry["source"])
            
            # Handle translation text
            translation = entry["translation"]
            if isinstance(translation, str):
                # For backward compatibility with history entries
                lines = translation.splitlines()
                if len(lines) > 4:
                    # Keep first 4 lines and add ellipsis
                    translation = "\n".join(lines[:4]) + "\n..."
                elif len(translation) > 200:  # Also limit by character count
                    translation = translation[:197] + "..."
            elif isinstance(translation, Text):
                # For new entries with colored text
                if len(translation.plain) > 200:
                    # Create new Text object with truncated content
                    truncated = Text()
                    current_length = 0
                    
                    # Preserve all segments with their styles
                    for segment in translation.split("\n"):
                        # If this is not the first segment, add newline
                        if current_length > 0:
                            truncated.append("\n")
                            current_length += 1
                            
                        # If adding this segment would exceed the limit
                        if current_length + len(segment.plain) > 197:
                            remaining_space = 197 - current_length
                            if remaining_space > 3:  # Only add partial segment if we have space
                                truncated.append(Text(segment.plain[:remaining_space], style=segment.style))
                            truncated.append("...", style="dim")
                            break
                            
                        # Add the full segment with its original style
                        truncated.append(segment)
                        current_length += len(segment.plain)
                    
                    translation = truncated
            
            table.add_row(
                entry["time"],
                source_with_status,
                entry["lang"],
                translation,
                style=row_style
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
        
    def _create_overall_stats_panel(self) -> Panel:
        """Create the overall statistics panel."""
        total = self.overall_stats['total_attempts']
        if total > 0:
            success_rate = (self.overall_stats['successful'] / total) * 100
            fail_rate = (self.overall_stats['failed'] / total) * 100
        else:
            success_rate = fail_rate = 0.0
            
        stats_table = Table(box=None, expand=True, show_header=False)
        stats_table.add_column("Label", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row(
            "Total Translation Attempts:",
            str(total)
        )
        stats_table.add_row(
            "Successful Translations:",
            f"{self.overall_stats['successful']} ({success_rate:.1f}%)"
        )
        stats_table.add_row(
            "Failed Translations:",
            f"{self.overall_stats['failed']} ({fail_rate:.1f}%)"
        )
        
        return Panel(stats_table, title="📊 Overall Translation Statistics", border_style="green")
        
    def _update_display(self):
        """Update the live display."""
        if self.live:
            layout = self._create_layout()
            layout["overall_stats"].update(self._create_overall_stats_panel())
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
            
    def add_translation_entry(self, source: str, lang: str, status: str = "⏳", 
                            translation: str = "", error: str = "", entry_type: str = "translation"):
        """Add a new translation entry.
        
        Args:
            source: Source text
            lang: Target language
            status: Status emoji
            translation: Translated text or Text object
            error: Error message if any
            entry_type: Type of entry ("translation" or "term_base")
        """
        time = datetime.now().strftime("%H:%M:%S")
        
        # Create colored text for translation and error
        translation_text = Text()
        if translation:
            # Handle both string and Text objects
            if isinstance(translation, Text):
                translation_text = translation  # Use the Text object as is
            else:
                # Color only the translation text based on status
                color = "yellow"  # Default color for in-progress
                if status.startswith("✓"):  # Done
                    color = "green"
                elif status.startswith("❌"):  # Failed
                    color = "red"
                translation_text.append(translation, style=color)
            
        # Add error or comments in default color
        if error:
            if translation:
                translation_text.append("\n")
            translation_text.append(error)  # No style means default white color
        
        entry = {
            "time": time,
            "type": entry_type,
            "source": source,
            "lang": lang,
            "status": status,
            "translation": translation_text
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
        status = "❌" if error else "✓"
        
        # Update overall statistics
        self.overall_stats['total_attempts'] += 1
        if error:
            self.overall_stats['failed'] += 1
        else:
            self.overall_stats['successful'] += 1
            
        self.add_translation_entry(source, lang, status, translation, error)
        
    def start_new_batch(self, llm_handler):
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
        
    def add_term_base_entry(self, term: str, lang: str, translation: str = "", comment: str = ""):
        """Add a term base entry with special highlighting."""
        status = "📖"
        
        # Create text with colored translation and default color comment
        translation_text = Text()
        if translation:
            translation_text.append(translation, style="blue")  # Term base entries in blue
            
        if comment:
            if translation:
                translation_text.append(" ")
            translation_text.append("(Comment: " + comment + ")")  # Comment in default color
                
        # Pass the Text object directly to the entry
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "term_base",
            "source": term,
            "lang": lang,
            "status": status,
            "translation": translation_text
        }
        
        # If entry is in current batch, update it
        for i, batch_entry in enumerate(self._current_batch):
            if batch_entry["source"] == term and batch_entry["lang"] == lang:
                self._current_batch[i] = entry
                self._update_display()
                return
                
        # If entry not in current batch, add it
        self._current_batch.append(entry)
        self._update_display()

class SimpleUIManager:
    """A simpler UI manager that uses standard logging output."""
    
    def __init__(self, max_history: int = 100, status_lines: int = 6, llm_handler: LLMHandler = None):
        """Initialize Simple UI Manager.
        
        Args:
            max_history: Not used in simple mode
            status_lines: Not used in simple mode
            llm_handler: LLMHandler instance
        """
        self.logger = logging.getLogger(__name__)
        self.console = Console()  # For colored output
        self.llm_handler = llm_handler
        
        # Add overall statistics tracking
        self.overall_stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0
        }
        
    def _print_overall_stats(self):
        """Print overall statistics."""
        total = self.overall_stats['total_attempts']
        if total > 0:
            success_rate = (self.overall_stats['successful'] / total) * 100
            fail_rate = (self.overall_stats['failed'] / total) * 100
        else:
            success_rate = fail_rate = 0.0
            
        self.console.print("\n📊 Overall Translation Statistics", style="green bold")
        self.console.print("─" * 40, style="dim")
        self.console.print(f"Total Translation Attempts: {total}", style="cyan")
        self.console.print(f"Successful Translations: {self.overall_stats['successful']} ({success_rate:.1f}%)", style="cyan")
        self.console.print(f"Failed Translations: {self.overall_stats['failed']} ({fail_rate:.1f}%)", style="cyan")

        if self.llm_handler:
            usage_stats = self.llm_handler.get_usage_stats()
            self.console.print("─" * 40, style="dim")
            self.console.print(f"Total tokens used: {usage_stats['total_tokens']} ({usage_stats['prompt_tokens']} prompt + {usage_stats['completion_tokens']} completion)", style="cyan")
            self.console.print(f"Total cost: ${usage_stats['total_cost']:.4f}", style="cyan")
        
    def start(self):
        """Start logging - no-op in simple mode."""
        pass
        
    def stop(self):
        """Stop logging - no-op in simple mode."""
        pass
        
    def add_translation_entry(self, source: str, lang: str, status: str = "⏳", 
                            translation: str = "", error: str = "", entry_type: str = "translation"):
        """Log a translation entry.
        
        Args:
            source: Source text
            lang: Target language
            status: Status emoji
            translation: Translated text
            error: Error message if any
            entry_type: Type of entry ("translation" or "term_base")
        """

        if status == "⏳":  # Skip in-progress entries
            return

        # Create colored text for translation and error
        output = Text()
        
        # Add time
        output.append(f"[{datetime.now().strftime('%H:%M:%S')}] ", style="cyan")
        
        # Add status icon and source
        output.append(f"{status} ", style="bold")
        output.append(f"[{lang}] ", style="magenta")
        output.append(f"{source} ", style="rgb(135,206,235)")
        output.append(f"→ ")
        
        # Add translation with appropriate color
        if translation:
            color = "yellow"  # Default color for in-progress
            if status.startswith("✓"):  # Done
                color = "green"
            elif status.startswith("❌"):  # Failed
                color = "red"
            elif entry_type == "term_base":
                color = "blue"
                
            if isinstance(translation, Text):
                output.append(translation)  # Use existing Text object
            else:
                output.append(translation, style=color)
        
        # Add error in default color
        if error:
            if translation:
                output.append("\n    ")  # Indent error message
            output.append(error)  # No style means default white color
            
        # Print the output
        self.console.print(output)
        
    def complete_translation(self, source: str, lang: str, translation: str, error: str = ""):
        """Log a completed translation."""
        status = "❌" if error else "✓"
        
        # Update overall statistics
        self.overall_stats['total_attempts'] += 1
        if error:
            self.overall_stats['failed'] += 1
        else:
            self.overall_stats['successful'] += 1
            
        self.add_translation_entry(source, lang, status, translation, error)
        
    def start_new_batch(self):
        """Start a new batch - prints a separator and overall stats in simple mode."""
        self._print_overall_stats()
        self.console.print("─" * 80, style="dim")
        
    def add_status(self, message: str, level: str = "info"):
        """Add a status message using standard logging."""
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
        
        self.console.print(msg)
        
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
        """Log a term base entry."""
        status = "📖"
        
        # Create text with colored translation and default color comment
        translation_text = Text()
        if translation:
            translation_text.append(translation, style="blue")  # Term base entries in blue
            
        if comment:
            if translation:
                translation_text.append(" ")
            translation_text.append("(Comment: " + comment + ")")  # Comment in default color
                
        self.add_translation_entry(term, lang, status, translation_text, entry_type="term_base")

def create_ui_manager(config: Dict[str, Any], llm_handler: LLMHandler, max_history: int = 100, status_lines: int = 6) -> UIManager:
    """Create the appropriate UI manager based on configuration.
    
    Args:
        config: Configuration dictionary
        max_history: Maximum history entries
        status_lines: Number of status lines
        
    Returns:
        UIManager instance (either fancy or simple)
    """
    use_simple_output = config.get('ui', {}).get('simple_output', False)
    if use_simple_output:
        return SimpleUIManager(max_history, status_lines, llm_handler)
    return UIManager(max_history, status_lines, llm_handler)