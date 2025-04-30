import requests
import json
from langchain_core.messages.human import HumanMessage

class VllmJSONModel:
    """
    A model class for making API calls to VLLM endpoints with JSON-formatted responses.
    
    Attributes:
        headers (dict): HTTP headers for API requests
        model_endpoint (str): Full URL endpoint for the API
        temperature (float): Temperature parameter for response generation
        model (str): Name of the model to use
        guided_json (Any): JSON guidance parameters
        stop (Any): Stop sequence for text generation
    """
    def __init__(self, temperature=0, model="llama3:instruct", model_endpoint=None, guided_json=None, stop=None):
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = model_endpoint + 'v1/chat/completions'
        self.temperature = temperature
        self.model = model
        self.guided_json = guided_json
        self.stop = stop

    def invoke(self, messages):
        """
        Invokes the model with the given messages and returns JSON-formatted response.

        Args:
            messages (list): List containing system and user messages

        Returns:
            HumanMessage: Formatted response containing JSON data or error information
        """
        system = messages[0]["content"]
        user = messages[1]["content"]

        prefix = self.model.split('/')[0]

        # MistralAI model does not require system and user prefix
        if prefix == "mistralai":
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"system:{system}\n\n user:{user}"
                    }
                ],
                "temperature": self.temperature,
                "stop": None,
                "guided_json": self.guided_json
            }
        else:
            payload = {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ],
                "temperature": self.temperature,
                "stop": self.stop,
                "guided_json": self.guided_json
            }
        
        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
                )
            
            print("REQUEST RESPONSE", request_response)
            request_response_json = request_response.json()
            response = json.loads(request_response_json['choices'][0]['message']['content'])
            response = json.dumps(response)

            response_formatted = HumanMessage(content=response)

            return response_formatted
        except requests.RequestException as e:
            response = {"error": f"Error in invoking model! {str(e)}"}
            response_formatted = HumanMessage(content=response)
            return response_formatted

class VllmModel:
    """
    A model class for making API calls to VLLM endpoints with text responses.
    
    Attributes:
        headers (dict): HTTP headers for API requests
        model_endpoint (str): Full URL endpoint for the API
        temperature (float): Temperature parameter for response generation
        model (str): Name of the model to use
        stop (Any): Stop sequence for text generation
    """
    def __init__(self, temperature=0, model="llama3:instruct", model_endpoint=None, stop=None):
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = model_endpoint + 'v1/chat/completions'
        self.temperature = temperature
        self.model = model
        self.stop = stop

    def invoke(self, messages):
        """
        Invokes the model with the given messages and returns text response.

        Args:
            messages (list): List containing system and user messages

        Returns:
            HumanMessage: Formatted response containing text data or error information
        """
        system = messages[0]["content"]
        user = messages[1]["content"]

        prefix = self.model.split('/')[0]

        # MistralAI model does not require system and user prefix
        if prefix == "mistralai":
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"system:{system}\n\n user:{user}"
                    }
                ],
                "temperature": self.temperature,
                "stop": None,
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ],
                "temperature": self.temperature,
                "stop": self.stop,
            }
        
        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
                )
            
            print("REQUEST RESPONSE", request_response)
            request_response_json = request_response.json()['choices'][0]['message']['content']
            response = str(request_response_json)
            
            response_formatted = HumanMessage(content=response)

            return response_formatted
        except requests.RequestException as e:
            response = {"error": f"Error in invoking model! {str(e)}"}
            response_formatted = HumanMessage(content=response)
            return response_formatted
