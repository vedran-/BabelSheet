from typing import Dict, Any, Optional
import aiohttp
import json
import os

class LLMHandler:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4", temperature: float = 0.3):
        """Initialize the LLM Handler.
        
        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the API (default: OpenAI's URL)
            model: Model to use for translations
            temperature: Temperature parameter for generation
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.model = model
        self.temperature = temperature
        
    async def generate_completion(self, 
                                messages: list[Dict[str, str]], 
                                **kwargs) -> Dict[str, Any]:
        """Generate completion using the configured LLM service.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            Dict containing the API response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLM API call failed: {error_text}")
                    
                return await response.json()

    def extract_completion_text(self, response: Dict[str, Any]) -> str:
        """Extract the generated text from the API response."""
        try:
            return response['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to extract completion text from response: {e}") 