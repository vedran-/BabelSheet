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
                                json_schema: Optional[Dict] = None,
                                **kwargs) -> Dict[str, Any]:
        """Generate completion using the configured LLM service.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            json_schema: Optional JSON schema for structured output
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

        if json_schema:
            # Add system message to enforce JSON response
            messages.insert(0, {
                "role": "system",
                "content": f"You must respond with valid JSON matching this schema: {json.dumps(json_schema)}"
            })
            
            # For newer models that support response_format
            if self.model.startswith(("gpt-4-1106", "gpt-3.5-turbo-1106")):
                data["response_format"] = {"type": "json_object"}
            
            # Add function calling as fallback for older models
            else:
                data["functions"] = [{
                    "name": "process_response",
                    "description": "Process the structured response",
                    "parameters": json_schema
                }]
                data["function_call"] = {"name": "process_response"}
        
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

    def extract_structured_response(self, response: Dict[str, Any]) -> Any:
        """Extract and parse JSON response from the API response."""
        try:
            if "function_call" in response["choices"][0]["message"]:
                # Extract from function call for older models
                content = response["choices"][0]["message"]["function_call"]["arguments"]
            else:
                # Extract from content for newer models
                content = response["choices"][0]["message"]["content"].strip()
            
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to extract structured response: {e}")