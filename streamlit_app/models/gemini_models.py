import requests
import json
import os
from utils.helper_functions import load_config
from langchain_core.messages.human import HumanMessage

class GeminiJSONModel:
    """
    A model class for interacting with Google's Gemini API, specifically for JSON-formatted responses.
    
    Attributes:
        api_key (str): The Gemini API key loaded from environment variables
        headers (dict): HTTP headers for API requests
        model_endpoint (str): The complete API endpoint URL
        temperature (float): Controls randomness in model outputs (0-1)
        model (str): The specific Gemini model identifier
    """
    def __init__(self, temperature=0, model=None):
        """
        Initialize the GeminiJSONModel with configuration settings.
        
        Args:
            temperature (float): Temperature setting for response generation (default: 0)
            model (str): The Gemini model identifier to use
        """
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.model_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        self.temperature = temperature
        self.model = model

    def invoke(self, messages):
        """
        Invoke the Gemini model with a request for JSON-formatted output.
        
        Args:
            messages (list): List of message dictionaries containing system and user prompts
        
        Returns:
            HumanMessage: A formatted response containing either JSON data or error information
        """
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"system:{system}. Your output must be JSON formatted. Just return the specified JSON format, do not prepend your response with anything.\n\nuser:{user}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": self.temperature
            },
        }

        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            print("REQUEST RESPONSE", request_response.status_code)
            
            request_response_json = request_response.json()

            if 'candidates' not in request_response_json or not request_response_json['candidates']:
                raise ValueError("No content in response")

            response_content = request_response_json['candidates'][0]['content']['parts'][0]['text']
            
            response = json.loads(response_content)
            response = json.dumps(response)

            response_formatted = HumanMessage(content=response)

            return response_formatted
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            response = {"error": error_message}
            response_formatted = HumanMessage(content=json.dumps(response))
            return response_formatted

class GeminiModel:
    """
    A model class for interacting with Google's Gemini API for general text responses.
    
    Attributes:
        api_key (str): The Gemini API key loaded from environment variables
        headers (dict): HTTP headers for API requests
        model_endpoint (str): The complete API endpoint URL
        temperature (float): Controls randomness in model outputs (0-1)
        model (str): The specific Gemini model identifier
    """
    def __init__(self, temperature=0, model=None):
        """
        Initialize the GeminiModel with configuration settings.
        
        Args:
            temperature (float): Temperature setting for response generation (default: 0)
            model (str): The Gemini model identifier to use
        """
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.model_endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={self.api_key}"
        self.temperature = temperature
        self.model = model

    def invoke(self, messages):
        """
        Invoke the Gemini model for general text generation.
        
        Args:
            messages (list): List of message dictionaries containing system and user prompts
        
        Returns:
            HumanMessage: A formatted response containing either the generated text or error information
        """
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"system:{system}\n\nuser:{user}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature
            },
        }

        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            print("REQUEST RESPONSE", request_response.status_code)
            
            request_response_json = request_response.json()

            if 'candidates' not in request_response_json or not request_response_json['candidates']:
                raise ValueError("No content in response")

            response_content = request_response_json['candidates'][0]['content']['parts'][0]['text']
            response_formatted = HumanMessage(content=response_content)

            return response_formatted
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            response = {"error": error_message}
            response_formatted = HumanMessage(content=json.dumps(response))
            return response_formatted
