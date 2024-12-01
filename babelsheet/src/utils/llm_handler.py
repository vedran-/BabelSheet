from typing import Dict, Any, Optional
import json
import os
from litellm import acompletion

class LLMHandler:
    # Class-level variables to track tokens across all instances
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-sonnet", 
                 temperature: float = 0.3, config: Optional[Dict[str, bool]] = None):
        """Initialize the LLM Handler.
        
        Args:
            api_key: API key for the LLM service
            model: Model to use for translations (with provider prefix)
            temperature: Temperature parameter for generation
            config: Configuration dictionary with keys:
                - save_requests: Whether to save API requests to files
                - save_responses: Whether to save API responses to files
        """
        self.model = model
        self.temperature = temperature
        self.config = config or {
            "save_requests": False,
            "save_responses": False
        }
        
        # Configure environment based on provider prefix
        provider = model.split('/')[0] if '/' in model else 'openai'
        
        if provider == 'anthropic':
            os.environ["ANTHROPIC_API_KEY"] = api_key
            if not model.split('/')[-1].startswith("claude-"):
                raise ValueError("Invalid model for Anthropic. Must start with 'claude-'")
        elif provider == 'azure':
            os.environ["AZURE_API_KEY"] = api_key
            if "/" not in model:
                raise ValueError("Azure model should be in format 'azure/deployment_name/model_name'")
        elif provider == 'local':  # LM Studio
            os.environ["OPENAI_API_KEY"] = "not-needed"
            os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
        elif provider == 'ollama':
            os.environ["OPENAI_API_KEY"] = "not-needed"
            os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
        else:  # Default to OpenAI
            os.environ["OPENAI_API_KEY"] = api_key

    @classmethod
    def get_token_usage(cls) -> Dict[str, int]:
        """Get the total token usage."""
        return {
            "prompt_tokens": cls.total_prompt_tokens,
            "completion_tokens": cls.total_completion_tokens,
            "total_tokens": cls.total_prompt_tokens + cls.total_completion_tokens
        }

    @classmethod
    def print_token_usage(cls) -> None:
        """Print the total token usage statistics."""
        usage = cls.get_token_usage()
        print(f">>> Token Usage: {usage['prompt_tokens']} (prompt) + {usage['completion_tokens']} (completion) = {usage['total_tokens']} (total)")

    def dump_json_to_file(self, data: Dict[str, Any], filename: str) -> None:
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        json_str = json_str.replace('\\n', '\n')
        with open(filename, "w", encoding='utf-8') as f:
            f.write(json_str)

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
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **kwargs
        }

        if json_schema:
            # Add JSON schema requirement to system message or create new one
            json_requirement = f"You must respond ONLY with a valid JSON matching this schema: {json.dumps(json_schema)}"
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = messages[0]["content"] + "\n\n" + json_requirement
            else:
                messages.insert(0, {
                    "role": "system", 
                    "content": json_requirement
                })
            
            # Only add response_format and functions for supported models
            if "openai" in self.model.lower():
                if self.model.startswith(("gpt-4-1106", "gpt-3.5-turbo-1106", "gpt-4o", "o1")):
                    data["response_format"] = {"type": "json_object"}
                else:
                    data["functions"] = [{
                        "name": "process_response",
                        "description": "Process the structured response",
                        "parameters": json_schema
                    }]
                    data["function_call"] = {"name": "process_response"}
        
        # Save request data if configured
        if self.config["save_requests"]:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_{timestamp}_request.json"
            self.dump_json_to_file(data, filename)

        try:
            # Use LiteLLM's async completion function
            response = await acompletion(**data)
            
            # Update token counters
            if response.usage:
                LLMHandler.total_prompt_tokens += response.usage.prompt_tokens
                LLMHandler.total_completion_tokens += response.usage.completion_tokens

            if self.config["save_responses"]:
                filename = f"llm_{timestamp}_response.json"
                # Convert response to dict for JSON serialization
                response_dict = {
                    "choices": [{
                        "message": {
                            "role": response.choices[0].message.role,
                            "content": response.choices[0].message.content
                        }
                    }],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None
                }
                self.dump_json_to_file(response_dict, filename)

            return response

        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")

    def extract_structured_response(self, response: Dict[str, Any]) -> Any:
        """Extract and parse JSON response from the API response."""
        try:
            if "function_call" in response["choices"][0]["message"]:
                # Extract from function call for older models
                content = response["choices"][0]["message"]["function_call"]["arguments"]
            else:
                # Extract from content for newer models
                content = response["choices"][0]["message"]["content"].strip()

            # Extract JSON block if present - some models return JSON in a code block
            json_block_start_idx = content.find("```json")
            if json_block_start_idx != -1:
                json_block_end_idx = content.rfind("```")
                if json_block_end_idx != -1:
                    content = content[json_block_start_idx + len("```json"):json_block_end_idx]

            content = content.strip()

            # Try to extract JSON from content if it starts with '{'
            if content.startswith('{'):
                # Find first complete JSON object by matching braces
                brace_count = 0
                in_string = False
                escape_next = False
                
                for i, char in enumerate(content):
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                content = content[:i+1]
                                break
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    escape_next = char == '\\' and not escape_next

            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to extract structured response: {e}")