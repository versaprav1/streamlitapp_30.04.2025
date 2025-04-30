from typing import Dict, Any, Optional, List
import os
import json
from datetime import datetime
import traceback
import streamlit as st
from pydantic import BaseModel, Field

class AgentGraphState(BaseModel):
    """State container for the agent graph"""
    current_agent: str = "planner"
    user_question: str = ""
    selected_schema: Dict = {}
    tool_responses: Dict = {}
    execution_path: List[str] = []
    errors: Dict[str, str] = {}
    is_error_state: bool = False
    
    # Add any other fields that your state needs
    planner_response: Dict = {}
    selector_response: Dict = {}
    SQLGenerator_response: Dict = {}
    reviewer_response: Dict = {}
    router_response: Dict = {}
    final_report_response: Dict = {}
    final_report_data: Dict = {}
    
    # Additional state data
    schemas: Dict[str, Any] = Field(default_factory=dict)
    sql_query: str = Field(default="")
    sql_query_results: Dict[str, Any] = Field(default_factory=dict)
    previous_selections: List[str] = Field(default_factory=list)
    previous_reports: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Execution tracking
    start_time: datetime = Field(default_factory=datetime.now)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    
    # Error handling
    last_successful_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Control flags
    end_chain: bool = Field(default=False)

    # New fields from the code block
    current_node: str = Field(default="start")
    iteration_count: int = Field(default=0)
    error_count: int = Field(default=0)
    last_error: Optional[str] = Field(default=None)
    last_success: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    workflow_completed: bool = Field(default=False)
    completion_timestamp: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        """Convert state to dictionary"""
        return super().dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        """Convert state to JSON string"""
        return super().json(*args, **kwargs)

def get_agent_graph_state(initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get or create a new agent graph state.
    
    Args:
        initial_data (dict, optional): Initial state data
        
    Returns:
        dict: State dictionary
    """
    state = AgentGraphState(**(initial_data or {}))
    return state.dict()

def create_agent_state(initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a new state dictionary with initial data.
    
    Args:
        initial_data (dict, optional): Initial state data
        
    Returns:
        dict: New state dictionary
    """
    return get_agent_graph_state(initial_data)

def get_model_settings() -> Dict[str, Any]:
    """Get model settings directly from session state"""
    try:
        return {
            "model": st.session_state.get("llm_model"),
            "server": st.session_state.get("server"),
            "temperature": float(st.session_state.get("temperature", 0.0)),
            "model_endpoint": st.session_state.get("server_endpoint"),
            "stop_token": st.session_state.get("stop_token"),
        }
    except Exception as e:
        print(f"Error loading model settings: {e}")
        return {}

def set_model_settings(model: str = None, server: str = None, 
                      temperature: float = None, model_endpoint: str = None) -> None:
    """
    Set model settings in the configuration.
    
    Args:
        model (str, optional): Model name
        server (str, optional): Server type
        temperature (float, optional): Temperature setting
        model_endpoint (str, optional): Model endpoint
    """
    try:
        from utils.helper_functions import save_config
        current_settings = get_model_settings()
        
        if model is not None:
            current_settings["model"] = model
        if server is not None:
            current_settings["server"] = server
        if temperature is not None:
            current_settings["temperature"] = temperature
        if model_endpoint is not None:
            current_settings["model_endpoint"] = model_endpoint
            
        save_config(current_settings)
    except Exception as e:
        print(f"Error saving model settings: {e}")

def update_state(state: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """Update state dictionary with new value"""
    if not isinstance(state, dict):
        state = state.dict() if hasattr(state, 'dict') else dict(state)
    
    keys = key.split('.')
    current = state
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    current[keys[-1]] = value
    return state

def get_state_value(state: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get value from state dictionary"""
    if not isinstance(state, dict):
        state = state.dict() if hasattr(state, 'dict') else dict(state)
    
    try:
        current = state
        for k in key.split('.'):
            current = current[k]
        return current
    except (KeyError, TypeError):
        return default

def extract_agent_response(state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """Extract agent response from state"""
    return get_state_value(state, f"{agent_name}_response", {})
