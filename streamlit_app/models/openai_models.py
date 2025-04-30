from utils.helper_functions import load_config
import os
import yaml
from openai import OpenAI
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any
import json

# Load configuration file
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize client as None - will be created when needed
client = None

def get_client():
    """
    Get or create OpenAI client with API key.
    """
    global client
    if client is None:
        api_key = config.get('OPENAI_API_KEY') or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        client = OpenAI(api_key=api_key)
    return client

def get_default_model_settings() -> Dict[str, Any]:
    """
    Get default model settings from session state or config.
    
    Returns:
        Dict[str, Any]: Dictionary containing model settings
    """
    try:
        import streamlit as st
        return {
            "model": st.session_state.get("llm_model", "gpt-3.5-turbo"),
            "temperature": float(st.session_state.get("temperature", 0.0)),
            "server": st.session_state.get("server", "openai"),
            "model_endpoint": st.session_state.get("server_endpoint", None)
        }
    except ImportError:
        # Fallback to config file if not in Streamlit context
        return {
            "model": config.get("llm_model", "gpt-3.5-turbo"),
            "temperature": float(config.get("temperature", 0.0)),
            "server": config.get("server", "openai"),
            "model_endpoint": config.get("server_endpoint", None)
        }

class CustomOpenAIWrapper:
    """
    A wrapper around OpenAI API that mimics the ChatOpenAI interface but uses direct API calls.
    Updated to use the OpenAI v1.x client.
    """
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0, response_format: Optional[Dict] = None):
        settings = get_default_model_settings()
        self.model = model or settings["model"]
        self.temperature = temperature
        # Only set response_format for models that support it
        self.response_format = response_format if self.model in ["gpt-4-turbo-preview", "gpt-3.5-turbo-1106"] else None
        self.client = get_client()  # Get client when needed
    
    def invoke(self, messages):
        """
        Invoke the OpenAI API using the v1.x client with the given messages.
        """
        try:
            messages_dict = [
                {"role": "system" if i == 0 else "user", "content": msg["content"]}
                for i, msg in enumerate(messages)
            ]
            
            params = {
                "model": self.model,
                "messages": messages_dict,
                "temperature": self.temperature,
            }
            
            # Only add response_format if it's set
            if self.response_format:
                params["response_format"] = self.response_format
                
            # Use the v1.x client API
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            
            # If response_format was requested but not supported, try to parse JSON from text
            if self.response_format and self.response_format.get("type") == "json_object":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON from response", "content": content}
            return content
        except Exception as e:
            print(f"Error in CustomOpenAIWrapper.invoke: {e}")
            raise


def get_open_ai(temperature: Optional[float] = None, model: Optional[str] = None, model_endpoint: Optional[str] = None) -> CustomOpenAIWrapper:
    """
    Get a custom OpenAI wrapper that uses direct API calls.
    
    Args:
        temperature (float, optional): Controls randomness in responses
        model (str, optional): The OpenAI model to use
        model_endpoint (str, optional): Not used for OpenAI but kept for API consistency
        
    Returns:
        CustomOpenAIWrapper: An instance of the wrapper class
    """
    try:
        return CustomOpenAIWrapper(
            model=model,
            temperature=temperature
        )
    except Exception as e:
        print(f"Error initializing CustomOpenAIWrapper: {e}")
        raise


def get_open_ai_json(system_prompt: str = "", user_input: str = "", temperature: Optional[float] = None, model: Optional[str] = None, model_endpoint: Optional[str] = None) -> Dict[str, Any]:
    try:
        # Modify system prompt to explicitly request JSON
        json_system_prompt = f"{system_prompt}\nYou must respond with valid JSON only."
        
        # Modify user input to explicitly request JSON
        json_user_input = f"{user_input}\nRespond with JSON only."
        
        wrapper = CustomOpenAIWrapper(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": json_user_input}
        ]
        
        response = wrapper.invoke(messages)
        
        # Ensure we have a dictionary
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON response"}
        
        if not isinstance(response, dict):
            return {"error": "Response is not a dictionary"}
            
        return response
        
    except Exception as e:
        print(f"Error in get_open_ai_json: {e}")
        return {"error": str(e)}
