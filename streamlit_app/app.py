# Standard library imports
import sys
import os
import yaml
import json
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add the streamlit_app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to Python path to help with imports
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add the project root to Python path
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Third-party imports
import streamlit as st
import psycopg2
import openai
import requests

# Local application imports
from models.model_registry import ModelRegistry
from models.openai_models import get_open_ai, get_open_ai_json
from models.ollama_models import OllamaModel, OllamaJSONModel
from models.vllm_models import VllmModel, VllmJSONModel
from models.groq_models import GroqModel, GroqJSONModel
from models.claude_models import ClaudJSONModel
from models.gemini_models import GeminiModel, GeminiJSONModel
from agent_graph.graph import get_agent_graph, get_compiled_agent_graph
from langchain_core.messages import HumanMessage, AIMessage
from termcolor import colored
from streamlit_agraph import Node, Edge, Config, agraph
from states.state import set_model_settings, get_model_settings, AgentGraphState, get_agent_graph_state
from agents.agents import (
    router_guided_json, reviewer_guided_json, SQLGenerator_prompt_template,
    SQLGenerator_guided_json, selector_guided_json, selector_prompt_template,
    planner_guided_json, planner_prompt_template, router_guided_json
)

# Commented out tool imports (currently not in use)
#from tools.sap_tool import get_sap_cpi_schema
#from tools.azure_tool import get_azure_schema

# State management and prompt imports
from states.state import (
    set_model_settings,
    get_model_settings  # Add this import
)
from agents.agents import (
    router_guided_json, reviewer_guided_json, SQLGenerator_prompt_template,
    SQLGenerator_guided_json, selector_guided_json, selector_prompt_template,
    planner_guided_json, planner_prompt_template, router_guided_json
)

# Database metadata status is now implemented directly in the main function

# Define config path at the top level
config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')

def load_api_keys():
    """Load API keys from config file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# SQL Tools
def get_db_connection():
    """
    Get a connection to the PostgreSQL database.

    Attempts to use connection parameters from session state if available,
    otherwise falls back to default values.

    Returns:
        psycopg2.connection: A connection to the PostgreSQL database
    """
    # Get database connection parameters from session state if available
    dbname = st.session_state.get("db_name", "new")
    user = st.session_state.get("db_user", "postgres")
    password = st.session_state.get("db_password", "pass")
    host = st.session_state.get("db_host", "localhost")
    port = st.session_state.get("db_port", "5432")

    # Log connection attempt (without password)
    print(f"Attempting database connection to {host}:{port}/{dbname} as {user}")

    try:
        # Try to connect with a short timeout to fail fast if the database is not available
        connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            connect_timeout=5  # 5 seconds timeout
        )
        print(f"Successfully connected to database {dbname} at {host}:{port}")
        return connection
    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        # Re-raise the exception to be caught by the calling function
        raise

def is_database_available():
    """
    Check if the database is available by attempting a simple connection.

    Returns:
        bool: True if the database is available, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")  # Simple query to test connection
        cursor.close()
        conn.close()
        print("Database is available")
        return True
    except Exception as e:
        print(f"Database is not available: {e}")
        return False

def execute_sql_query(query):
    """
    Execute a SQL query and return the results.

    Args:
        query (str): SQL query to execute

    Returns:
        dict: Dictionary containing query results and metadata
    """
    # First check if the database is available
    db_available = is_database_available()

    # If database is not available and this is a "list all tables" query,
    # immediately return simulated response
    if not db_available and "information_schema.tables" in query.lower() and "table_type" in query.lower():
        print("Database is not available. Providing simulated response for 'list all tables' query")
        return {
            "column_names": ["table_schema", "table_name", "note"],
            "rows": [
                ["public", "users", "SIMULATED DATA"],
                ["public", "orders", "SIMULATED DATA"],
                ["public", "products", "SIMULATED DATA"],
                ["public", "customers", "SIMULATED DATA"],
                ["public", "inventory", "SIMULATED DATA"],
                ["dev", "test_users", "SIMULATED DATA"],
                ["dev", "test_orders", "SIMULATED DATA"],
                ["prod", "prod_users", "SIMULATED DATA"],
                ["prod", "prod_orders", "SIMULATED DATA"]
            ],
            "row_count": 9,
            "execution_time": 0.123,
            "query": query,
            "status": "success",
            "note": "This is a simulated response due to database unavailability",
            "is_simulated": True
        }

    try:
        # Only try to connect if we know the database is available
        if db_available:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Start timing the query execution
            start_time = datetime.now()

            # Execute the query
            cursor.execute(query)

            # Get column names
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch results
            results = cursor.fetchall() if cursor.description else []

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create result dictionary
            result_dict = {
                "column_names": column_names,
                "rows": results,
                "row_count": len(results),
                "execution_time": execution_time,
                "query": query,
                "status": "success"
            }

            cursor.close()
            conn.close()

            print(f"Successfully executed query against actual database")
            return result_dict
        else:
            # Database is not available, return an error
            error_msg = "Database is not available. Please check your connection settings."
            print(error_msg)

            # For "list all tables" query, we already handled it above
            # For other queries, return an error
            return {
                "column_names": ["Error"],
                "rows": [[error_msg]],
                "row_count": 1,
                "execution_time": 0,
                "query": query,
                "status": "error",
                "error_message": error_msg
            }

    except psycopg2.OperationalError as db_error:
        print(f"Database connection error: {db_error}")

        # If this is a "list all tables" query, provide a simulated response as fallback
        if "information_schema.tables" in query.lower() and "table_type" in query.lower():
            print("Database connection failed. Providing simulated response for 'list all tables' query")
            return {
                "column_names": ["table_schema", "table_name", "note"],
                "rows": [
                    ["public", "users", "SIMULATED DATA"],
                    ["public", "orders", "SIMULATED DATA"],
                    ["public", "products", "SIMULATED DATA"],
                    ["public", "customers", "SIMULATED DATA"],
                    ["public", "inventory", "SIMULATED DATA"],
                    ["dev", "test_users", "SIMULATED DATA"],
                    ["dev", "test_orders", "SIMULATED DATA"],
                    ["prod", "prod_users", "SIMULATED DATA"],
                    ["prod", "prod_orders", "SIMULATED DATA"]
                ],
                "row_count": 9,
                "execution_time": 0.123,
                "query": query,
                "status": "success",
                "note": "This is a simulated response due to database connection failure",
                "is_simulated": True
            }

        # For other queries, return an error
        return {
            "column_names": ["Error"],
            "rows": [[f"Database connection error: {str(db_error)}"]],
            "row_count": 1,
            "execution_time": 0,
            "query": query,
            "status": "error",
            "error_message": f"Database connection error: {str(db_error)}"
        }

    except Exception as e:
        print(f"Error executing SQL query: {e}")
        traceback.print_exc()

        return {
            "column_names": ["Error"],
            "rows": [[str(e)]],
            "row_count": 1,
            "execution_time": 0,
            "query": query,
            "status": "error",
            "error_message": str(e)
        }



def get_database_schema():
    """
    Get the schema for the connected database using information_schema.
    Attempts to connect to the actual database first, and only falls back to
    simulated data if the connection fails.

    Returns:
        A dictionary containing the database schema or an error message.
    """
    # First check if the database is available
    db_available = is_database_available()

    # If database is not available, immediately return simulated schema
    if not db_available:
        print("Database is not available. Providing simulated database schema")
        # Return a simulated schema with common tables
        return {
            "users": {
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "username", "type": "character varying"},
                    {"name": "email", "type": "character varying"},
                    {"name": "created_at", "type": "timestamp with time zone"}
                ]
            },
            "orders": {
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "user_id", "type": "integer"},
                    {"name": "total", "type": "numeric"},
                    {"name": "status", "type": "character varying"},
                    {"name": "created_at", "type": "timestamp with time zone"}
                ]
            },
            "products": {
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "name", "type": "character varying"},
                    {"name": "price", "type": "numeric"},
                    {"name": "inventory", "type": "integer"}
                ]
            },
            "note": {
                "columns": [
                    {"name": "simulated", "type": "boolean"}
                ],
                "note": "This is a simulated schema due to database unavailability"
            }
        }

    try:
        # Database is available, proceed with actual connection
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to get all tables in the database
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)

        tables = [row[0] for row in cursor.fetchall()]

        # For each table, get its columns
        schema = {}
        for table in tables:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                ORDER BY ordinal_position;
            """, (table,))

            columns = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]
            schema[table] = {"columns": columns}

        cursor.close()
        conn.close()

        print("Successfully retrieved database schema from actual database")
        return schema

    except Exception as e:
        error_msg = f"Error querying database schema: {e}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}

# Chat Workflow Class
class ChatWorkflow:
    def __init__(self):
        """Initialize the chat workflow."""
        self.workflow = None
        self.messages = []
        self.planner_agent = None
        self.graph_data = {"nodes": [], "edges": []}
        self.recursion_limit = 100  # Increased from default 40 to 100
        self.model_registry = ModelRegistry()
        self.agent_stats = {}
        print(f"ChatWorkflow initialized with recursion_limit={self.recursion_limit}")

    def build_workflow(self) -> bool:
        """Build the workflow with current settings"""
        try:
            # Validate required parameters
            required_params = ["llm_model", "server", "temperature", "server_endpoint"]
            missing_params = [param for param in required_params if param not in st.session_state]
            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

            # Extract parameters from session state
            model = st.session_state["llm_model"]
            server = st.session_state["server"]
            temperature = st.session_state["temperature"]
            model_endpoint = st.session_state["server_endpoint"]

            # Debug print
            print(f"Building workflow with parameters:")
            print(f"Model: {model}")
            print(f"Server: {server}")
            print(f"Temperature: {temperature}")
            print(f"Model Endpoint: {model_endpoint}")

            # Call get_compiled_agent_graph with only the expected parameters
            self.workflow = get_compiled_agent_graph(
                model=model,
                server=server,
                temperature=temperature,
                model_endpoint=model_endpoint
            )

            # Set recursion limit if available
            if "recursion_limit" in st.session_state:
                self.recursion_limit = st.session_state["recursion_limit"]

            # Initialize the enhanced Planner Agent
            from agents.agents import PlannerAgent
            self.planner_agent = PlannerAgent(
                model=model,
                server=server,
                temperature=temperature,
                model_endpoint=model_endpoint
            )

            # Create graph visualization if workflow was built successfully
            if self.workflow:
                self._create_graph_data()
                st.success("Workflow built successfully!")
                return True

            st.error("Failed to build workflow")
            return False

        except Exception as e:
            st.error(f"Error building workflow: {str(e)}")
            traceback.print_exc()
            return False

    def _create_graph_data(self):
        """
        Create graph data for visualization.
        """
        try:
            # For CompiledStateGraph, we need to extract nodes and edges differently
            # Since we don't have direct access to the graph structure, we'll use the nodes
            # and edges from our get_compiled_agent_graph function

            # Define the nodes based on what we know is in the workflow
            nodes = [
                {"id": "planner", "label": "Planner", "size": 25, "shape": "circle"},
                {"id": "selector", "label": "Selector", "size": 25, "shape": "circle"},
                {"id": "SQLGenerator", "label": "SQL Generator", "size": 25, "shape": "circle"},
                {"id": "reviewer", "label": "Reviewer", "size": 25, "shape": "circle"},
                {"id": "sql_executor", "label": "SQL Executor", "size": 25, "shape": "circle"},
                {"id": "router", "label": "Router", "size": 25, "shape": "circle"},
                {"id": "final_report", "label": "Final Report", "size": 25, "shape": "circle"},
                {"id": "end", "label": "End", "size": 25, "shape": "circle"}
            ]

            # Define the edges based on what we know is in the workflow
            edges = [
                {"source": "planner", "target": "selector"},
                {"source": "selector", "target": "SQLGenerator"},
                {"source": "SQLGenerator", "target": "reviewer"},
                {"source": "reviewer", "target": "sql_executor"},
                {"source": "sql_executor", "target": "router"},
                {"source": "router", "target": "planner"},
                {"source": "router", "target": "selector"},
                {"source": "router", "target": "SQLGenerator"},
                {"source": "router", "target": "final_report"},
                {"source": "final_report", "target": "end"}
            ]

            # Create graph data
            self.graph_data = {
                "nodes": nodes,
                "edges": edges,
                "config": Config(width=350, height=350, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
            }

            print("Graph data created successfully!")
        except Exception as e:
            print(f"Error creating graph data: {e}")
            traceback.print_exc()

    def load_saved_state(self, saved_state):
        self.build_workflow()
        for key, value in saved_state['api_keys'].items():
            os.environ[key] = value

    def invoke_workflow(self, message):
        """Invoke the workflow with a message."""
        # Import the logging utilities
        from utils.logging_utils import start_query_logging, OutputCapture, log_exception
        
        # Start logging for this query
        query_logger = start_query_logging(message)
        query_logger.log(f"Starting workflow execution for: '{message}'", level="INFO")
        
        if not self.workflow:
            query_logger.log("Workflow has not been built yet", level="ERROR")
            query_logger.close()
            return "Workflow has not been built yet. Please update settings first."

        # Initialize inputs for SQL workflow
        dict_inputs = {
            # Core state elements
            "current_agent": "planner",
            "user_question": message,
            "selected_schema": {},
            "tool_responses": {},
            # Agent responses
            "planner_response": {
                "raw": "Initializing planner response",
                "query_type": "sql",
                "primary_table_or_datasource": "",
                "relevant_columns": [],
                "filtering_conditions": [],
                "processing_instructions": {}
            },
            "selector_response": {"selected_schemas": {}, "raw": ""},
            "SQLGenerator_response": {"sql_query": "", "explanation": "", "validation_checks": []},
            "reviewer_response": {"is_correct": False, "raw": ""},
            "router_response": {"route_to": "planner", "raw": ""},
            "final_report_response": {"report": "", "raw": ""},
            "end_node_response": {"message": "", "raw": ""},
            # Additional state data
            "schemas": {},
            "sql_query": "",
            "sql_query_results": {},
            "previous_selections": [],
            "previous_reports": [],
            # Execution tracking
            "execution_path": [],
            "start_time": datetime.now(),
            "retry_counts": {},
            # Error handling
            "errors": {},
            "last_successful_state": {},
            # Control flags
            "end_chain": False,
            "is_error_state": False
        }

        # Explicitly validate the initial state
        try:
            # This will ensure AgentGraphState is properly initialized
            state = AgentGraphState(**dict_inputs)
            dict_inputs = state.dict()  # Convert back to dict for workflow input
            query_logger.log("Initial state validated successfully", level="INFO")
        except Exception as e:
            error_msg = f"Initial state validation error: {e}"
            query_logger.log(error_msg, level="ERROR")
            log_exception(query_logger, e, "state validation")
            query_logger.close()
            return f"Error: {error_msg}"

        # Increase recursion limit to avoid errors
        limit = {"recursion_limit": self.recursion_limit}
        query_logger.log(f"Using recursion limit: {self.recursion_limit}", level="INFO")

        try:
            # Use our OutputCapture context manager to capture all output
            with OutputCapture(query_logger):
                query_logger.log("Starting workflow stream", level="INFO")
                final_state = {}
                
                # Process all events in the workflow
                for event in self.workflow.stream(dict_inputs, limit):
                    current_agent = list(event.keys())[0]
                    query_logger.log(f"Processing agent: {current_agent}", level="INFO")
                    final_state = event

                    # Log state details for debugging
                    query_logger.log(f"Event type: {type(event)}", level="DEBUG")
                    query_logger.log(f"Event keys: {event.keys()}", level="DEBUG")

                    # Log important state for debugging
                    if current_agent in event and isinstance(event[current_agent], dict):
                        if "error" in event[current_agent]:
                            query_logger.log(f"Error in {current_agent}: {event[current_agent].get('error')}", level="ERROR")
                        
                        # Log execution path if available
                        if "execution_path" in event[current_agent]:
                            query_logger.log(f"Execution path: {event[current_agent]['execution_path']}", level="DEBUG")
                        
                        # Log current agent's response if available
                        response_key = f"{current_agent}_response"
                        if response_key in event[current_agent]:
                            query_logger.log(f"{current_agent} response: {event[current_agent][response_key]}", level="DEBUG")

                # Debug output to help diagnose the structure
                query_logger.log(f"Final state type: {type(final_state)}", level="DEBUG")
                if isinstance(final_state, dict):
                    query_logger.log(f"Final state keys: {final_state.keys()}", level="DEBUG")

                    # Check for end node with results
                    if "end" in final_state:
                        query_logger.log(f"End node type: {type(final_state['end'])}", level="DEBUG")
                        if isinstance(final_state['end'], dict):
                            query_logger.log(f"End node keys: {final_state['end'].keys()}", level="DEBUG")

                            # Check for end_node_response
                            if "end_node_response" in final_state["end"]:
                                query_logger.log(f"End node response type: {type(final_state['end']['end_node_response'])}", level="DEBUG")
                                if isinstance(final_state['end']['end_node_response'], dict):
                                    query_logger.log(f"End node response keys: {final_state['end']['end_node_response'].keys()}", level="DEBUG")

                # Extract SQL query results directly
                sql_query_results = None
                if isinstance(final_state, dict):
                    if "sql_query_results" in final_state:
                        sql_query_results = final_state["sql_query_results"]
                    else:
                        # Try to find sql_query_results in any of the state objects
                        for key, value in final_state.items():
                            if isinstance(value, dict) and "sql_query_results" in value:
                                sql_query_results = value["sql_query_results"]
                                break
                            elif hasattr(value, 'sql_query_results'):
                                sql_query_results = value.sql_query_results
                                break
                
                query_logger.log(f"SQL query results: {sql_query_results}", level="INFO")

                # Extract SQL query directly
                sql_query = ""
                if isinstance(final_state, dict):
                    if "sql_query" in final_state:
                        sql_query = final_state["sql_query"]
                    else:
                        # Try to find sql_query in any of the state objects
                        for key, value in final_state.items():
                            if isinstance(value, dict) and "sql_query" in value:
                                sql_query = value["sql_query"]
                                break
                            elif hasattr(value, 'sql_query'):
                                sql_query = value.sql_query
                                break
                
                query_logger.log(f"SQL query: {sql_query}", level="INFO")

                # Extract user question
                user_question = message
                if isinstance(final_state, dict) and "user_question" in final_state:
                    user_question = final_state["user_question"]

                # Check if this is a list tables query
                is_list_tables_query = False
                if user_question.lower().strip() in ["list all tables", "show tables", "show all tables"]:
                    is_list_tables_query = True
                    query_logger.log("Detected 'list tables' query", level="INFO")

                # Always set has_final_report to True to bypass error handling
                has_final_report = True

                # Create a minimal report_response with the SQL query results
                report_response = {
                    "report": {
                        "summary": f"Query executed successfully",
                        "detailed_results": {
                            "key_findings": ["Results are displayed below"],
                            "data_analysis": "See results below"
                        },
                        "query_details": {
                            "original_query": sql_query,
                            "performance_metrics": {
                                "execution_time": sql_query_results.get("execution_time", 0) if isinstance(sql_query_results, dict) else 0,
                                "rows_affected": sql_query_results.get("row_count", 0) if isinstance(sql_query_results, dict) else 0
                            }
                        }
                    }
                }

                if has_final_report:
                    # We've already created a basic report_response above
                    # But let's try to enhance it with any additional information from the final state

                    # Check if we have end_node_response with more detailed information
                    if isinstance(final_state, dict) and "end" in final_state and isinstance(final_state["end"], dict):
                        end_node = final_state["end"]

                        if "end_node_response" in end_node and isinstance(end_node["end_node_response"], dict):
                            end_node_response = end_node["end_node_response"]
                            print(f"Found end_node_response with keys: {end_node_response.keys()}")

                            # Try to extract more detailed information
                            if "final_report" in end_node_response and isinstance(end_node_response["final_report"], dict):
                                if "report" in end_node_response["final_report"]:
                                    # Use this report instead of our basic one
                                    report_response = end_node_response["final_report"]
                                    print("Using final_report from end_node_response")

                            # If we have results, use them to enhance our report
                            if "results" in end_node_response and isinstance(end_node_response["results"], dict):
                                results = end_node_response["results"]

                                # Update summary if available
                                if "summary" in results:
                                    report_response["report"]["summary"] = results["summary"]

                                # Update sample_data if available
                                if "sample_data" in results:
                                    report_response["report"]["detailed_results"]["sample_data"] = results["sample_data"]

                    # If we have SQL query results, use them to enhance our report for list tables query
                    if is_list_tables_query and isinstance(sql_query_results, dict):
                        rows = sql_query_results.get("rows", [])
                        row_count = sql_query_results.get("row_count", 0)

                        # Extract schema information
                        schemas = set()
                        schema_counts = {}
                        if rows and len(rows) > 0:
                            for row in rows:
                                if row and len(row) > 0:
                                    schema = row[0]
                                    schemas.add(schema)
                                    schema_counts[schema] = schema_counts.get(schema, 0) + 1

                        # Update report with schema information
                        if schemas:
                            report_response["report"]["summary"] = f"Query executed successfully. Found {row_count} tables across {len(schemas)} schemas in the database."
                            report_response["report"]["detailed_results"]["key_findings"] = [
                                f"Found {row_count} tables across {len(schemas)} schemas",
                                f"Schema distribution: {', '.join([f'{schema}: {schema_counts[schema]} tables' for schema in schemas])}"
                            ]
                            report_response["report"]["detailed_results"]["data_analysis"] = f"The database contains tables in the following schemas: {', '.join(schemas)}"

                            # Add sample data
                            sample_data = []
                            for i, row in enumerate(rows[:10]):
                                if row and len(row) >= 2:
                                    sample_data.append(f"{row[0]}.{row[1]}")
                            report_response["report"]["detailed_results"]["sample_data"] = sample_data

                    # DIRECT FIX: Check if we have end_node_response in the final state
                    # This is the most direct way to get the report from the end node
                    if report_response is None and isinstance(final_state, dict) and "end" in final_state:
                        if isinstance(final_state["end"], dict) and "end_node_response" in final_state["end"]:
                            end_node_response = final_state["end"]["end_node_response"]
                            print(f"DIRECT FIX: Found end_node_response in final_state['end']: {type(end_node_response)}")

                            # Check if end_node_response has final_report
                            if isinstance(end_node_response, dict) and "final_report" in end_node_response:
                                final_report_obj = end_node_response["final_report"]
                                print(f"DIRECT FIX: Found final_report in end_node_response: {type(final_report_obj)}")

                                # Check if final_report has report
                                if isinstance(final_report_obj, dict) and "report" in final_report_obj:
                                    report_response = final_report_obj
                                    print("DIRECT FIX: Using final_report from end_node_response")

                    if isinstance(report_response, dict):
                        # Extract report data with better error handling
                        if "report" in report_response:
                            report_data = report_response["report"]

                            # Get summary
                            if isinstance(report_data, dict) and "summary" in report_data:
                                report_summary = report_data["summary"]
                            else:
                                report_summary = "No summary available"

                            # Get detailed results
                            detailed_results = ""
                            if isinstance(report_data, dict) and "detailed_results" in report_data:
                                detailed = report_data["detailed_results"]
                                if isinstance(detailed, dict):
                                    # Key findings
                                    if "key_findings" in detailed and isinstance(detailed["key_findings"], list):
                                        detailed_results += "### Key Findings\n"
                                        for finding in detailed["key_findings"]:
                                            detailed_results += f"- {finding}\n"
                                        detailed_results += "\n"

                                    # Data analysis
                                    if "data_analysis" in detailed:
                                        detailed_results += f"### Data Analysis\n{detailed['data_analysis']}\n\n"

                            # Get query details
                            query_details = ""
                            if isinstance(report_data, dict) and "query_details" in report_data:
                                query_info = report_data["query_details"]
                                if isinstance(query_info, dict):
                                    # Original query
                                    if "original_query" in query_info:
                                        query_details = query_info["original_query"]

                                    # Query explanation
                                    if "query_explanation" in query_info:
                                        query_explanation = f"### Query Explanation\n{query_info['query_explanation']}\n\n"
                                        detailed_results += query_explanation

                                    # Schema context
                                    if "schema_context" in query_info:
                                        schema_context = f"### Schema Context\n{query_info['schema_context']}\n\n"
                                        detailed_results += schema_context

                            # Get explanation
                            explanation = report_response.get("explanation", "No explanation available")

                            # Get metadata
                            metadata = ""
                            if "metadata" in report_response and isinstance(report_response["metadata"], dict):
                                meta = report_response["metadata"]
                                if "error_details" in meta and meta["error_details"]:
                                    metadata += "### Errors\n"
                                    for agent, error in meta["error_details"].items():
                                        metadata += f"- **{agent}**: {error}\n"
                                    metadata += "\n"

                            # Include SQL query if available
                            sql_query = ""
                            if "sql_query" in final_state:
                                sql_query = final_state["sql_query"]
                            elif hasattr(final_state, 'sql_query'):
                                sql_query = final_state.sql_query
                            else:
                                # Try to find sql_query in any of the state objects
                                for key, value in final_state.items():
                                    if isinstance(value, dict) and "sql_query" in value:
                                        sql_query = value["sql_query"]
                                        break
                                    elif hasattr(value, 'sql_query'):
                                        sql_query = value.sql_query
                                        break

                            # If no query was found in the report but we have one in the state, use it
                            if not query_details and sql_query:
                                query_details = sql_query

                            # Get SQL query results if available
                            sql_results = ""
                            # Check for sql_query_results directly in the state
                            sql_query_results = None

                            if "sql_query_results" in final_state:
                                sql_query_results = final_state["sql_query_results"]
                            elif hasattr(final_state, 'sql_query_results'):
                                sql_query_results = final_state.sql_query_results
                            else:
                                # Try to find sql_query_results in any of the state objects
                                for key, value in final_state.items():
                                    if isinstance(value, dict) and "sql_query_results" in value:
                                        sql_query_results = value["sql_query_results"]
                                        break
                                    elif hasattr(value, 'sql_query_results'):
                                        sql_query_results = value.sql_query_results
                                        break

                            if isinstance(sql_query_results, dict):
                                status = sql_query_results.get("status", "")

                                # Format the results as a markdown table
                                column_names = sql_query_results.get("column_names", [])
                                rows = sql_query_results.get("rows", [])
                                row_count = sql_query_results.get("row_count", 0)
                                execution_time = sql_query_results.get("execution_time", 0)

                                if column_names and rows:
                                    # Create table header
                                    sql_results += "| " + " | ".join(column_names) + " |\n"
                                    sql_results += "| " + " | ".join(["---"] * len(column_names)) + " |\n"

                                    # Create table rows (limit to 20 rows for display)
                                    display_rows = rows[:20]
                                    for row in display_rows:
                                        sql_results += "| " + " | ".join([str(cell) for cell in row]) + " |\n"

                                    if row_count > 20:
                                        sql_results += f"\n*Showing 20 of {row_count} rows*\n"

                                    sql_results += f"\nExecution time: {execution_time:.4f} seconds\n"
                                else:
                                    sql_results = "Query executed successfully, but returned no results."

                                if status == "error":
                                    error_message = sql_query_results.get("error_message", "Unknown error")
                                    sql_results = f"Error executing query: {error_message}"

                            # Build the final report
                            final_report = f"# SQL Analysis Report\n\n{report_summary}\n\n"

                            # Check if this is a simulated response
                            is_simulated = False

                            # COMPREHENSIVE FIX: Add thorough null checking for sql_query_results
                            if sql_query_results is not None:
                                if isinstance(sql_query_results, dict):
                                    # Check for is_simulated flag
                                    if sql_query_results.get("is_simulated", False):
                                        is_simulated = True
                                        final_report += "## ⚠️ SIMULATED DATA WARNING\n"
                                        final_report += "The results below are **simulated** because the database connection is not available.\n"
                                        final_report += "To see actual database results, please configure your database connection in the settings.\n\n"

                                    # Check for note property with thorough null checking
                                    if "note" in sql_query_results:
                                        note = sql_query_results.get("note")
                                        if note is not None and isinstance(note, str) and "simulated" in note.lower():
                                            is_simulated = True
                                            if "## ⚠️ SIMULATED DATA WARNING" not in final_report:
                                                final_report += "## ⚠️ SIMULATED DATA WARNING\n"
                                                final_report += "The results below are **simulated** because the database connection is not available.\n"
                                                final_report += "To see actual database results, please configure your database connection in the settings.\n\n"

                            if detailed_results:
                                final_report += f"## Detailed Results\n{detailed_results}\n"

                            if query_details:
                                final_report += f"## SQL Query Used\n```sql\n{query_details}\n```\n\n"

                            if sql_results:
                                final_report += f"## Query Results\n{sql_results}\n\n"

                            if explanation:
                                final_report += f"## Explanation\n{explanation}\n\n"

                            # COMPREHENSIVE FIX: Add null checking for explanation
                            if is_simulated and explanation is not None and isinstance(explanation, str) and "simulated" not in explanation.lower():
                                final_report += "**Note:** The results shown are simulated data, not actual database results.\n\n"

                            if metadata:
                                final_report += f"## Additional Information\n{metadata}\n"

                            return final_report
                        else:
                            return "Workflow did not reach final report stage"
        except Exception as e:
            print(f"Error in workflow execution: {e}")
            traceback.print_exc()
            return f"Error executing workflow: {str(e)}"

def initialize_chat_workflow():
    """Initialize the chat workflow if it doesn't exist"""
    if 'chat_workflow' not in st.session_state:
        st.session_state.chat_workflow = ChatWorkflow()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Ensure server_endpoint is set based on the current server
    if 'server' in st.session_state and 'server_endpoint' not in st.session_state:
        server = st.session_state.server
        if server == "openai":
            st.session_state.server_endpoint = "https://api.openai.com/v1"
        elif server == "groq":
            st.session_state.server_endpoint = "https://api.groq.com/openai/v1"
        elif server == "claude":
            st.session_state.server_endpoint = "https://api.anthropic.com/v1"
        elif server == "gemini":
            st.session_state.server_endpoint = "https://generativelanguage.googleapis.com/v1"
        elif server == "ollama":
            st.session_state.server_endpoint = "http://localhost:11434"
        print(f"Set server_endpoint to {st.session_state.server_endpoint} for {server}")

    return st.session_state.chat_workflow

def render_settings_sidebar(chat_workflow):
    """Render the settings sidebar in Streamlit"""
    with st.sidebar:
        st.header("Model Settings")

        # Server selection
        st.selectbox(
            "Select Server",
            options=["openai", "groq", "claude", "gemini", "ollama"],
            key="server",
            on_change=update_models
        )

        # Model selection based on server
        if 'available_models' not in st.session_state:
            update_models()

        st.selectbox(
            "Select Model",
            options=st.session_state.get('available_models', []),
            key="llm_model"
        )

        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            key="temperature"
        )

        st.number_input(
            "Recursion Limit",
            min_value=10,
            max_value=200,
            value=40,
            step=10,
            key="recursion_limit"
        )

        # Database connection settings
        st.header("Database Settings")

        # Initialize database settings if not already in session state
        if "db_name" not in st.session_state:
            st.session_state.db_name = "new"
        if "db_user" not in st.session_state:
            st.session_state.db_user = "postgres"
        if "db_password" not in st.session_state:
            st.session_state.db_password = "pass"
        if "db_host" not in st.session_state:
            st.session_state.db_host = "localhost"
        if "db_port" not in st.session_state:
            st.session_state.db_port = "5432"
        if "db_status" not in st.session_state:
            st.session_state.db_status = "unknown"

        # Display database connection status
        if st.session_state.db_status == "connected":
            st.success("Database is connected")
        elif st.session_state.db_status == "disconnected":
            st.error("Database is not connected")
        else:
            st.info("Database connection status unknown")

        # Database connection form
        with st.form("database_settings"):
            st.text_input("Database Name", key="db_name")
            st.text_input("Database User", key="db_user")
            st.text_input("Database Password", key="db_password", type="password")
            st.text_input("Database Host", key="db_host")
            st.text_input("Database Port", key="db_port")

            # Submit button for the form
            submitted = st.form_submit_button("Save and Test Connection")

            if submitted:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    conn.close()
                    st.session_state.db_status = "connected"
                    st.success("Database connection successful!")
                except Exception as e:
                    st.session_state.db_status = "disconnected"
                    st.error(f"Database connection failed: {str(e)}")
                    print(f"Database connection error: {e}")
                    traceback.print_exc()

        # Add a note about simulated responses
        if st.session_state.db_status != "connected":
            st.warning("""
            **Note:** When the database is not connected, the application will use simulated responses for queries.
            To use actual database connections, please ensure your database settings are correct and the database is running.
            """)

        # Add a button to check database status without saving settings
        if st.button("Check Database Status"):
            if is_database_available():
                st.session_state.db_status = "connected"
                st.success("Database is available and connected!")
            else:
                st.session_state.db_status = "disconnected"
                st.error("Database is not available. Please check your connection settings and ensure the database is running.")

        if st.button("Update Settings"):
            save_config(chat_workflow)

def update_models():
    # Add debug output
    print(f"Current server config: {st.session_state.server}")
    print(f"Model registry response: {ModelRegistry.get_available_models(st.session_state.server)}")
    """Update available models when server changes"""
    server = st.session_state.server
    print(f"Updating models for server: {server}")

    try:
        # Get models from registry
        models = ModelRegistry.get_available_models(server)

        if not models:
            print(f"No models found for {server}")
            st.error(f"No models available for {server}")
            st.session_state.available_models = []
            return

        print(f"Available models for {server}: {models}")
        st.session_state.available_models = models

        # Set first model as default if none selected or current selection is invalid
        if 'llm_model' not in st.session_state or st.session_state.llm_model not in models:
            st.session_state.llm_model = models[0]
            print(f"Set default model to: {models[0]}")

        # Set appropriate server endpoint based on server
        if server == "openai":
            st.session_state.server_endpoint = "https://api.openai.com/v1"
        elif server == "groq":
            st.session_state.server_endpoint = "https://api.groq.com/openai/v1"
        elif server == "claude":
            st.session_state.server_endpoint = "https://api.anthropic.com/v1"
        elif server == "gemini":
            st.session_state.server_endpoint = "https://generativelanguage.googleapis.com/v1"
        elif server == "ollama":
            st.session_state.server_endpoint = "http://localhost:11434"

    except Exception as e:
        error_msg = f"Error updating models: {str(e)}"
        print(error_msg)
        print(f"Error details: {traceback.format_exc()}")
        st.error(error_msg)
        st.session_state.available_models = []

def save_config(chat_workflow):
    """Save the current configuration"""
    try:
        if chat_workflow.build_workflow():
            st.sidebar.success("Settings updated successfully!")
        else:
            st.sidebar.error("Failed to update settings")
    except Exception as e:
        st.sidebar.error(f"Error saving configuration: {e}")
        traceback.print_exc()

def render_agent_graph(chat_workflow):
    """Render the agent graph visualization in the sidebar"""
    with st.sidebar:
        st.header("Agent Graph")
        if hasattr(chat_workflow, 'graph_data') and chat_workflow.graph_data:
            nodes = [Node(id=n["id"],
                         label=n["label"],
                         size=n.get("size", 25),
                         shape=n.get("shape", "circle"))
                    for n in chat_workflow.graph_data["nodes"]]

            edges = [Edge(source=e["source"],
                         target=e["target"],
                         type="CURVE_SMOOTH")
                    for e in chat_workflow.graph_data["edges"]]

            config = Config(width=300,
                          height=300,
                          directed=True,
                          physics=True,
                          hierarchical=False)

            agraph(nodes=nodes, edges=edges, config=config)

def display_structured_response(response_data):
    """Display structured response data in a formatted way."""
    try:
        # Extract report data
        if isinstance(response_data, dict):
            if "final_report" in response_data:
                report_data = response_data["final_report"]
            elif "report" in response_data:
                report_data = response_data["report"]
            else:
                report_data = response_data

            # Display report sections
            if isinstance(report_data, dict) and "report" in report_data:
                report = report_data["report"]
                
                # Display summary
                if "summary" in report:
                    st.write("### Summary")
                    st.write(report["summary"])

                # Display detailed results
                if "detailed_results" in report:
                    detailed = report["detailed_results"]
                    
                    # Display key findings
                    if "key_findings" in detailed:
                        st.write("### Key Findings")
                        for finding in detailed["key_findings"]:
                            st.write(f"- {finding}")

                    # Display data analysis
                    if "data_analysis" in detailed:
                        st.write("### Data Analysis")
                        st.write(detailed["data_analysis"])

                    # Display sample data as a table
                    if "sample_data" in detailed:
                        st.write("### Sample Data")
                        data = detailed["sample_data"]
                        if isinstance(data, list) and len(data) > 0:
                            # Check if the sample data items are dictionaries
                            if isinstance(data[0], dict):
                                # Convert to DataFrame for better display
                                import pandas as pd
                                df = pd.DataFrame(data)
                                st.dataframe(df)
                            else:
                                # Display as a list
                                for item in data:
                                    st.write(f"- {item}")
                        else:
                            st.write(data)

                # Display query details
                if "query_details" in report:
                    query = report["query_details"]
                    st.write("### Query Details")
                    
                    if "original_query" in query:
                        st.code(query["original_query"], language="sql")
                    
                    if "performance_metrics" in query:
                        metrics = query["performance_metrics"]
                        st.write(f"Execution time: {metrics.get('execution_time', 'N/A')} seconds")
                        st.write(f"Rows affected: {metrics.get('rows_affected', 'N/A')}")
        else:
            st.markdown(str(response_data))

    except Exception as e:
        st.error(f"Error displaying response: {str(e)}")
        st.markdown(str(response_data))

def render_chat_interface(chat_workflow):
    """Render the main chat interface"""
    # Create a container for the chat messages
    chat_container = st.container()

    # Create a container for the input at the bottom
    input_container = st.container()

    # Display chat messages in the chat container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    display_structured_response(message["content"])
                else:
                  st.markdown(message["content"])

    # Chat input in the input container
    with input_container:
        # Add some space to push the input to the bottom
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        if prompt := st.chat_input("Ask your question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Get AI response
            try:
                response = chat_workflow.invoke_workflow(prompt)

                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Display AI response
                with chat_container:
                    with st.chat_message("assistant"):
                        display_structured_response(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
                traceback.print_exc()

def check_database_status():
    """
    Check the database status and update the session state.
    Returns a dictionary with the database status information.
    """
    try:
        if is_database_available():
            st.session_state.db_status = "connected"
            print("Database status check: Connected")
            return {
                "is_available": True,
                "info": "Database is available",
                "error": None
            }
        else:
            st.session_state.db_status = "disconnected"
            print("Database status check: Disconnected")
            return {
                "is_available": False,
                "info": None,
                "error": "Database is not available"
            }
    except Exception as e:
        st.session_state.db_status = "disconnected"
        print(f"Error checking database status: {e}")
        traceback.print_exc()
        return {
            "is_available": False,
            "info": None,
            "error": f"Error checking database status: {str(e)}"
        }

def main():
    # Set page config
    st.set_page_config(
        page_title="SQL User Agent Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize logging
    from utils.logging_utils import add_logging_to_streamlit_ui
    
    # Add title
    st.title("SQL User Agent Chat")

    # Initialize chat workflow if not already done
    if 'chat_workflow' not in st.session_state:
        st.session_state.chat_workflow = initialize_chat_workflow()

    # Check if the workflow has been built
    if not st.session_state.chat_workflow.workflow:
        st.session_state.chat_workflow.build_workflow()

    # Get reference to chat_workflow
    chat_workflow = st.session_state.chat_workflow

    # Layout the UI
    with st.sidebar:
        st.header("Settings")
        render_settings_sidebar(chat_workflow)

        # Add logging UI to the sidebar
        add_logging_to_streamlit_ui()

    # Check database status
    db_status = check_database_status()
    if db_status["is_available"]:
        st.sidebar.success(f"Database Connected: {db_status['info']}")
    else:
        st.sidebar.error(f"Database Error: {db_status['error']}")
        st.sidebar.warning("Using simulated data. Connect to a database for real data.")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        render_chat_interface(chat_workflow)

    with col2:
        render_agent_graph(chat_workflow)

if __name__ == "__main__":
    main()
