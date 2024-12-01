import logging
from datetime import datetime
from rich.console import Console
from rich.text import Text
from ..llm_handler import LLMHandler
from typing import Dict, Any

class ConsoleUIManager:
    """A console-based UI manager that uses standard logging output."""
    
    def __init__(self, max_history: int = 100, status_lines: int = 6, llm_handler: LLMHandler = None):
        """Initialize Console UI Manager.
        
        Args:
            max_history: Not used in console mode
            status_lines: Not used in console mode
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
        
    def print_overall_stats(self):
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
        """Start logging - no-op in console mode."""
        pass
        
    def stop(self):
        """Stop logging - no-op in console mode."""
        pass
        
    def add_translation_entry(self, source: str, lang: str, status: str = "⏳", 
                            translation: str = "", error: str = "", entry_type: str = "translation"):
        """Log a translation entry."""
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
        
    def start_new_batch(self, llm_handler=None):
        """Start a new batch - prints a separator and overall stats in console mode."""
        self.print_overall_stats()
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