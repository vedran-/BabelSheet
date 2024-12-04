from typing import Dict, Any, Optional
import json
import os

import litellm
os.environ["LITELLM_LOG"] = "ERROR"

# Disable all LiteLLM logging
#import logging
#logging.getLogger("litellm").setLevel(logging.ERROR)
#logging.getLogger("litellm.llm_provider").setLevel(logging.ERROR)
#logging.getLogger("litellm.utils").setLevel(logging.ERROR)
#logging.getLogger("openai").setLevel(logging.ERROR)

# Now import litellm
from litellm import acompletion, completion_cost

# Additional verbosity controls
#litellm.verbose = False
litellm.verbose=False
litellm.set_verbose=False
litellm.log_raw_request_response=False
litellm.success_callback = [] 

def my_custom_logging_fn(model_call_dict):
    #print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXXXX XXXXXXXXXXX\nXXXXXXXXXXX XXXXXXXXXX XXXXXXXX\nXXXXXXXXXX\nXXXXXXXXXX\nmodel call details: {model_call_dict}")
    pass


class LLMHandler:
    # Class-level variables to track tokens and costs across all instances
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-5-sonnet", 
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
        elif provider == 'lm_studio':  # LM Studio
            os.environ["OPENAI_API_KEY"] = "not-needed"
            os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
        elif provider == 'ollama':
            os.environ["OPENAI_API_KEY"] = "not-needed"
            os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
        elif api_key:  # Default to OpenAI
            os.environ["OPENAI_API_KEY"] = api_key

    @classmethod
    def get_usage_stats(cls) -> Dict[str, Any]:
        """Get the total token usage and cost statistics."""
        return {
            "prompt_tokens": cls.total_prompt_tokens,
            "completion_tokens": cls.total_completion_tokens,
            "total_tokens": cls.total_prompt_tokens + cls.total_completion_tokens,
            "total_cost": cls.total_cost
        }

    @classmethod
    def print_usage_stats(cls) -> None:
        """Print the total token usage and cost statistics."""
        usage = cls.get_usage_stats()
        print(f">>> Token Usage: {usage['prompt_tokens']} (prompt) + {usage['completion_tokens']} (completion) = {usage['total_tokens']} (total)")
        print(f">>> Total Cost: ${usage['total_cost']}")

    def dump_json_to_file(self, data: Dict[str, Any], filename: str) -> None:
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        json_str = json_str.replace('\\n', '\n')
        with open(filename, "w", encoding='utf-8') as f:
            f.write(json_str)

    async def generate_completion(self, 
                                messages: list[Dict[str, str]], 
                                json_schema: Optional[Dict] = None,
                                **kwargs) -> Dict[str, Any]:
        """Generate completion using the configured LLM service."""
        data = {
            "model": self.model,  # Use clean model name without provider prefix
            "messages": messages,
            "temperature": self.temperature,
            **kwargs
        }

        if json_schema:
            json_requirement = f"You must respond ONLY with a valid JSON matching this schema: {json.dumps(json_schema)}." \
                              "Make sure to escape all double quotes with \\\", and newlines with \\n."
            
            # For Anthropic, we need to format the system message differently
            if "anthropic" in self.model.lower():
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] = messages[0]["content"] + "\n\n" + json_requirement
                else:
                    messages.insert(0, {
                        "role": "system", 
                        "content": json_requirement
                    })
                
                # Convert messages to Anthropic's format
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                data["messages"] = formatted_messages
                
            # For OpenAI models
            #elif "openai" in self.model.lower():
            else:
                if messages[0]["role"] == "system":
                    messages[0]["content"] = messages[0]["content"] + "\n\n" + json_requirement
                else:
                    messages.insert(0, {
                        "role": "system", 
                        "content": json_requirement
                    })
                
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
            response = await acompletion(**data, logger_fn=my_custom_logging_fn)
            
            # Calculate cost
            try:
                cost = completion_cost(
                    model=self.model,
                    messages=messages,
                    completion=response.choices[0].message.content
                )
            except Exception as e:
                # If cost calculation fails (e.g., for local models), default to 0
                #print(f"Cost calculation failed for model {self.model}, will use price of 0.0")
                cost = 0.0
                
            LLMHandler.total_cost += cost
            
            # Update token counters
            if response.usage:
                LLMHandler.total_prompt_tokens += response.usage.prompt_tokens
                LLMHandler.total_completion_tokens += response.usage.completion_tokens

            if self.config["save_responses"]:
                # Convert to serializable format only for saving
                ret = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response.choices[0].message.content
                        }
                    }],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
                    },
                    "cost": round(cost, 4)
                }
                filename = f"llm_{timestamp}_response.json"
                self.dump_json_to_file(ret, filename)

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

            ret = json.loads(content)
            return ret
        
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to extract structured response: {e}")