# External HTTP requests library for making API calls
import requests
# JSON handling for request/response data
import json
# Abstract Syntax Trees for Python code parsing
import ast
# Message type for LangChain compatibility
from langchain_core.messages.human import HumanMessage

class OllamaJSONModel:
    """
    A class for interacting with Ollama models that enforces JSON output format.
    
    Attributes:
        headers (dict): HTTP headers for API requests
        model_endpoint (str): URL endpoint for Ollama API
        temperature (float): Temperature setting for model responses
        model (str): Name of the Ollama model to use
    """
    def __init__(self, temperature=0, model="llama3.2:latest"):
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = "http://localhost:11434/api/generate"
        self.temperature = temperature
        self.model = model

    def invoke(self, messages):
        """
        Invokes the Ollama model with JSON output formatting.

        Args:
            messages (list): List containing system and user messages, where each message
                           is a dictionary with 'content' key

        Returns:
            HumanMessage: A formatted response containing either JSON data or error information
        """
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
                "model": self.model,
                "prompt": user,
                "format": "json",
                "system": system,
                "stream": False,
                "temperature": 0,
            }
        
        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
                )
            
            print("REQUEST RESPONSE", request_response)
            request_response_json = request_response.json()
            response = json.loads(request_response_json['response'])
            response = json.dumps(response)

            response_formatted = HumanMessage(content=response)

            return response_formatted
        except requests.RequestException as e:
            response = {"error": f"Error in invoking model! {str(e)}"}
            response_formatted = HumanMessage(content=response)
            return response_formatted

class OllamaModel:
    """
    A class for interacting with Ollama models with standard text output.
    
    Attributes:
        headers (dict): HTTP headers for API requests
        model_endpoint (str): URL endpoint for Ollama API
        temperature (float): Temperature setting for model responses
        model (str): Name of the Ollama model to use
    """
    def __init__(self, temperature=0, model="llama3:instruct"):
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = "http://localhost:11434/api/generate"
        self.temperature = temperature
        self.model = model

    def invoke(self, messages):
        """
        Invokes the Ollama model with standard text output.

        Args:
            messages (list): List containing system and user messages, where each message
                           is a dictionary with 'content' key

        Returns:
            HumanMessage: A formatted response containing either text data or error information
        """
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
                "model": self.model,
                "prompt": user,
                "system": system,
                "stream": False,
                "temperature": 0,
            }
        
        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
                )
            
            print("REQUEST RESPONSE JSON", request_response)

            request_response_json = request_response.json()['response']
            response = str(request_response_json)
            
            response_formatted = HumanMessage(content=response)

            return response_formatted
        except requests.RequestException as e:
            response = {"error": f"Error in invoking model! {str(e)}"}
            response_formatted = HumanMessage(content=response)
            return response_formatted

