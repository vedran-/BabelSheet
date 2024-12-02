from typing import Dict, Any
from .base_ui_manager import UIManager
from .console_ui_manager import ConsoleUIManager
from .graphical_ui_manager import GraphicalUIManager
from ..llm_handler import LLMHandler

def create_ui_manager(config: Dict[str, Any], llm_handler: LLMHandler, max_history: int = 100, status_lines: int = 6) -> UIManager:
    """Create the appropriate UI manager based on configuration.
    
    Args:
        config: Configuration dictionary
        llm_handler: LLMHandler instance
        max_history: Maximum history entries
        status_lines: Number of status lines
        
    Returns:
        UIManager instance (either fancy, console, or graphical)
    """
    ui_type = config.get('ui', {}).get('type', 'fancy')
    if ui_type == 'console':
        return ConsoleUIManager(max_history, status_lines, llm_handler)
    elif ui_type == 'graphical':
        return GraphicalUIManager(max_history, status_lines, llm_handler)
    return UIManager(max_history, status_lines, llm_handler) 