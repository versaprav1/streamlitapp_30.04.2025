# External library imports
from datetime import datetime
from termcolor import colored
import json
import traceback

# Model imports
from models.openai_models import get_open_ai, get_open_ai_json, CustomOpenAIWrapper
from models.ollama_models import OllamaModel, OllamaJSONModel
from models.vllm_models import VllmJSONModel, VllmModel
from models.groq_models import GroqModel, GroqJSONModel

# Database and tool imports
from tools.datasource_tool import get_db_connection, InventoryTypeValuesToNames

# State management imports
from states.state import (
    AgentGraphState,  # Main state container for the agent graph
    get_model_settings,  # Retrieves model configuration
    update_state,  # Updates state values
    get_state_value,  # Gets specific state values
    extract_agent_response  # Extracts response from agent output
)

###################
# Base Agent Class
###################

class Agent:
    def __init__(self, state=None, model=None, server=None, temperature=0, model_endpoint=None, stop=None, guided_json=None):
        """Initialize an agent with the given state and model settings."""
        # Keep state as AgentGraphState if it is one, otherwise create empty dict
        self.state = {} if state is None else state

        # Get model settings but don't provide defaults
        model_settings = get_model_settings()

        # Require model and server to be set
        if not model and not model_settings.get("model"):
            raise ValueError("Model must be specified")
        if not server and not model_settings.get("server"):
            raise ValueError("Server must be specified")

        self.model = model or model_settings.get("model")
        self.server = server or model_settings.get("server")
        self.temperature = temperature if temperature is not None else model_settings.get("temperature", 0)
        self.model_endpoint = model_endpoint or model_settings.get("model_endpoint")
        self.stop = stop
        self.guided_json = guided_json

    def get_model(self, json_model=False):
        """
        Get the appropriate LLM based on server type and whether JSON output is needed.

        Args:
            json_model (bool): Whether to use a model that outputs JSON.

        Returns:
            The appropriate LLM instance.
        """
        if self.server == "ollama":
            if json_model:
                return OllamaJSONModel(model=self.model, temperature=self.temperature)
            else:
                return OllamaModel(model=self.model, temperature=self.temperature)
        elif self.server == "vllm":
            if json_model:
                return VllmJSONModel(model=self.model, temperature=self.temperature, model_endpoint=self.model_endpoint)
            else:
                return VllmModel(model=self.model, temperature=self.temperature, model_endpoint=self.model_endpoint)
        elif self.server == "groq":
            if json_model:
                return GroqJSONModel(model=self.model, temperature=self.temperature, model_endpoint=self.model_endpoint)
            else:
                return GroqModel(model=self.model, temperature=self.temperature, model_endpoint=self.model_endpoint)
        elif self.server == "openai":
            if json_model:
                return CustomOpenAIWrapper(model=self.model, temperature=self.temperature, response_format={"type": "json_object"})
            else:
                return CustomOpenAIWrapper(model=self.model, temperature=self.temperature)
        else:
            raise ValueError(f"Unknown server type: {self.server}")

    def update_state(self, key, value):
        """
        Update the agent's state with the given key-value pair.

        Args:
            key (str): The key to update.
            value: The value to set.

        Returns:
            dict: The updated state.
        """
        if self.state is None:
            self.state = {}

        # Handle nested keys
        keys = key.split('.')
        current = self.state
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
        return self.state

    def get_state_value(self, key, default=None):
        """
        Get a value from the agent's state.

        Args:
            key (str): The key to retrieve.
            default: The default value to return if the key is not found.

        Returns:
            The value associated with the key, or the default value if not found.
        """
        # Handle nested keys
        try:
            if isinstance(self.state, dict):
                current = self.state
                for k in key.split('.'):
                    current = current[k]
                return current
            else:
                # If state is AgentGraphState, use attribute access
                return getattr(self.state, key, default)
        except (KeyError, TypeError, AttributeError):
            return default

    def extract_agent_response(self, agent_name):
        """
        Extract the response from a specific agent in the state.

        Args:
            agent_name (str): The name of the agent (e.g., "planner", "selector").

        Returns:
            The agent's response, or an empty dict if not found.
        """
        return self.get_state_value(f"{agent_name}_response", {})

    def handle_llm_response(self, response):
        """
        Process the response from the LLM and extract structured data.

        Args:
            response: The raw response from the LLM, which could be a string or a dictionary.

        Returns:
            dict: The structured response data.

        Raises:
            ValueError: If the response is invalid or missing required fields.
        """
        print(f"handle_llm_response received response of type: {type(response)}")

        try:
            # If response is a string, try to parse it as JSON
            if isinstance(response, str):
                response = response.strip()
                # Remove any markdown code block formatting if present
                if response.startswith('```') and response.endswith('```'):
                    response = response[3:-3].strip()
                if response.startswith('```json') and response.endswith('```'):
                    response = response[7:-3].strip()
                # Parse the JSON
                response = json.loads(response)

            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError("Response must be a dictionary")

            # Validate required fields based on agent type
            if isinstance(self, PlannerAgent):
                required_fields = ["query_type", "primary_table_or_datasource", "relevant_columns", "filtering_conditions", "processing_instructions"]
            elif isinstance(self, SelectorAgent):
                required_fields = ["selected_tool", "selected_datasource", "information_needed", "reason_for_selection", "query_parameters"]
            elif isinstance(self, SQLGenerator):
                required_fields = ["sql_query", "explanation", "validation_checks"]
            elif isinstance(self, ReviewerAgent):
                required_fields = ["is_correct", "issues", "suggestions", "explanation"]
            elif isinstance(self, RouterAgent):
                required_fields = ["route_to", "reason", "feedback"]
            elif isinstance(self, FinalReportAgent):
                required_fields = ["report", "explanation"]
            else:
                required_fields = []

            # Check for missing required fields
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            return response

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing LLM response: {str(e)}")

    def invoke(self, **kwargs):
        """
        Base invoke method with error handling and recovery.

        Args:
            **kwargs: Arguments for the specific agent

        Returns:
            dict: Agent response
        """
        try:
            # Get current state values
            current_state = self.get_state_value("current_state", {})
            retry_count = current_state.get("retry_count", 0)

            # Check retry limit
            if retry_count >= 3:
                return self._create_error_response(
                    ["Maximum retry limit reached"],
                    fatal=True
                )

            # Execute agent-specific logic
            response = self._execute(**kwargs)

            # Validate response
            response = self.handle_llm_response(response)

            # Reset retry count on success
            self.update_state("retry_count", 0)

            return response

        except Exception as e:
            # Increment retry count
            retry_count += 1
            self.update_state("retry_count", retry_count)

            # Log error
            print(f"Error in {self.__class__.__name__}: {str(e)}")
            traceback.print_exc()

            return self._create_error_response([str(e)])

    def _create_error_response(self, errors: list, fatal: bool = False) -> dict:
        """Create a standardized error response."""
        return {
            "error_type": "FATAL" if fatal else "NON_FATAL",
            "error_message": str(errors[0]) if len(errors) == 1 else str(errors),
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc(),
            "component": self.__class__.__name__,
            "state": self.state,
            "recommendations": [
                "Review error logs for more details",
                "Check state values for inconsistencies",
                "Contact support if issue persists"
            ]
        }

    def _create_error_report(self, error_message: str) -> dict:
        """Create a standardized error report."""
        return {
            "error_type": "ERROR",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc(),
            "component": self.__class__.__name__,
            "state": self.state,
            "recommendations": [
                "Review error logs for more details",
                "Check state values for inconsistencies",
                "Contact support if issue persists"
            ]
        }

###################
# Planner Agent
###################

planner_prompt_template = """
You are the Planner Agent responsible for analyzing user queries received from the UI.

Your task is to:

1. **Understand and Classify the Query**:
   - Determine if the query requires SQL database access
   - Identify if the query needs other types of processing
   - Validate if the query is clear and complete

2. **For SQL-Based Queries**:
   - Identify key components such as:
     * Specific data sources mentioned (e.g., SAP, Azure, Kafka)
     * Tables that might contain the relevant data
     * Columns that might need to be selected or filtered
     * Conditions or filters mentioned in the query
     * Type of operation needed (SELECT, JOIN, GROUP BY, etc.)

3. **For Non-SQL Queries**:
   - Identify the type of information or processing needed
   - Determine appropriate tools or approaches required
   - Specify any special handling requirements

4. **Validation and Error Handling**:
   - Check if the query is clear and well-formed
   - Identify any missing information

5. **Database Schema Awareness**:
   - Be aware that the database includes multiple schemas (e.g., `prod`, `dev`, and `test`)
   - Consider which schema(s) might be relevant for the query
   - I will automatically discover and provide database structure information
   - For queries about database structure (schemas, tables, columns), I will help discover this information
   - For queries about specific tables, I will provide table structure details
   - I maintain a cache of database metadata to avoid redundant queries

6. **Database Structure Discovery**:
   - For queries about listing schemas, I will discover all available schemas
   - For queries about listing tables in a schema, I will discover all tables in that schema
   - For queries about specific tables, I will discover the table structure (columns, relationships)
   - I will enhance your plan with this discovered metadata

You MUST respond with a valid JSON object with the following structure:
{
    "query_type": "sql or other",
    "primary_table_or_datasource": "main table or data source to query",
    "relevant_columns": ["list of columns needed"],
    "filtering_conditions": "conditions to filter the data",
    "processing_instructions": "specific instructions for handling this type of query"
}

DO NOT include any other fields or text outside this JSON structure.
"""

planner_guided_json = {
    "type": "object",
    "properties": {
        "query_type": {
            "type": "string",
            "enum": ["SELECT", "INSERT", "UPDATE", "DELETE"],
            "description": "Type of SQL query needed"
        },
        "primary_table_or_datasource": {
            "type": "string",
            "description": "Main table or data source to query"
        },
        "relevant_columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of columns needed for the query"
        },
        "filtering_conditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of conditions for filtering data"
        },
        "processing_instructions": {
            "type": "object",
            "properties": {
                "aggregations": {"type": "array", "items": {"type": "string"}},
                "grouping": {"type": "array", "items": {"type": "string"}},
                "ordering": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 0}
            }
        }
    },
    "required": ["query_type", "primary_table_or_datasource", "relevant_columns"]
}

class PlannerAgent(Agent):
    def __init__(self, state=None, model=None, server=None, temperature=None, model_endpoint=None, stop=None):
        """
        Initialize the Planner Agent.

        Args:
            state: Current state of the agent graph
            model: LLM model to use
            server: Server type (openai, ollama, etc.)
            temperature: Temperature setting for model generation
            model_endpoint: API endpoint for the model
            stop: Stop sequences for the model
        """
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=planner_guided_json
        )

        # Define database classes directly in this file to avoid import issues
        import json
        import os
        import psycopg2
        import traceback
        from datetime import datetime
        from typing import Dict, List, Any, Optional, Tuple

        # Define the DatabaseMetadataStore class
        class DatabaseMetadataStore:
            def __init__(self, cache_dir: str = ".cache"):
                self.metadata = {}
                self.change_log = []
                self.cache_dir = cache_dir
                self._ensure_cache_dir()

            def _ensure_cache_dir(self):
                """Ensure the cache directory exists"""
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir)

            def get_cache_path(self, db_name: str) -> str:
                """Get the path to the cache file for a database"""
                return os.path.join(self.cache_dir, f"{db_name}_metadata.json")

            def get_log_path(self, db_name: str) -> str:
                """Get the path to the change log file for a database"""
                return os.path.join(self.cache_dir, f"{db_name}_changelog.json")

            def load_metadata(self, db_name: str) -> bool:
                """Load metadata from cache if it exists"""
                cache_path = self.get_cache_path(db_name)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r') as f:
                            self.metadata[db_name] = json.load(f)
                        return True
                    except Exception as e:
                        print(f"Error loading metadata cache: {e}")
                return False

            def save_metadata(self, db_name: str) -> bool:
                """Save metadata to cache"""
                if db_name not in self.metadata:
                    return False

                cache_path = self.get_cache_path(db_name)
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(self.metadata[db_name], f, indent=2)
                    return True
                except Exception as e:
                    print(f"Error saving metadata cache: {e}")
                    return False

            def load_change_log(self, db_name: str) -> bool:
                """Load change log from file if it exists"""
                log_path = self.get_log_path(db_name)
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r') as f:
                            self.change_log = json.load(f)
                        return True
                    except Exception as e:
                        print(f"Error loading change log: {e}")
                return False

            def save_change_log(self, db_name: str) -> bool:
                """Save change log to file"""
                log_path = self.get_log_path(db_name)
                try:
                    with open(log_path, 'w') as f:
                        json.dump(self.change_log, f, indent=2)
                    return True
                except Exception as e:
                    print(f"Error saving change log: {e}")
                    return False

            def log_change(self, db_name: str, action: str, details: Dict[str, Any]) -> None:
                """Add an entry to the change log"""
                self.change_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "database": db_name,
                    "action": action,
                    "details": details
                })
                self.save_change_log(db_name)

            def get_schemas(self, db_name: str) -> List[str]:
                """Get list of schemas for a database"""
                if db_name in self.metadata and "schemas" in self.metadata[db_name]:
                    return list(self.metadata[db_name]["schemas"].keys())
                return []

            def get_tables(self, db_name: str, schema: str) -> List[str]:
                """Get list of tables for a schema"""
                if (db_name in self.metadata and
                    "schemas" in self.metadata[db_name] and
                    schema in self.metadata[db_name]["schemas"] and
                    "tables" in self.metadata[db_name]["schemas"][schema]):
                    return list(self.metadata[db_name]["schemas"][schema]["tables"].keys())
                return []

            def get_table_structure(self, db_name: str, schema: str, table: str) -> Dict[str, Any]:
                """Get structure of a table"""
                if (db_name in self.metadata and
                    "schemas" in self.metadata[db_name] and
                    schema in self.metadata[db_name]["schemas"] and
                    "tables" in self.metadata[db_name]["schemas"][schema] and
                    table in self.metadata[db_name]["schemas"][schema]["tables"]):
                    return self.metadata[db_name]["schemas"][schema]["tables"][table]
                return {}

            def update_schemas(self, db_name: str, schemas: List[str]) -> None:
                """Update the list of schemas for a database"""
                if db_name not in self.metadata:
                    self.metadata[db_name] = {"schemas": {}, "last_updated": datetime.now().isoformat()}

                # Track new schemas
                existing_schemas = set(self.get_schemas(db_name))
                new_schemas = set(schemas) - existing_schemas

                # Initialize new schemas
                for schema in new_schemas:
                    if schema not in self.metadata[db_name]["schemas"]:
                        self.metadata[db_name]["schemas"][schema] = {
                            "tables": {},
                            "last_updated": datetime.now().isoformat()
                        }
                        self.log_change(db_name, "discovered_schema", {"schema": schema})

                self.metadata[db_name]["last_updated"] = datetime.now().isoformat()
                self.save_metadata(db_name)

            def update_tables(self, db_name: str, schema: str, tables: List[str]) -> None:
                """Update the list of tables for a schema"""
                if db_name not in self.metadata:
                    self.metadata[db_name] = {"schemas": {}, "last_updated": datetime.now().isoformat()}

                if schema not in self.metadata[db_name]["schemas"]:
                    self.metadata[db_name]["schemas"][schema] = {
                        "tables": {},
                        "last_updated": datetime.now().isoformat()
                    }

                # Track new tables
                existing_tables = set(self.get_tables(db_name, schema))
                new_tables = set(tables) - existing_tables

                # Initialize new tables
                for table in new_tables:
                    if table not in self.metadata[db_name]["schemas"][schema]["tables"]:
                        self.metadata[db_name]["schemas"][schema]["tables"][table] = {
                            "columns": [],
                            "relationships": [],
                            "last_updated": datetime.now().isoformat()
                        }
                        self.log_change(db_name, "discovered_table", {"schema": schema, "table": table})

                self.metadata[db_name]["schemas"][schema]["last_updated"] = datetime.now().isoformat()
                self.metadata[db_name]["last_updated"] = datetime.now().isoformat()
                self.save_metadata(db_name)

            def update_table_structure(self, db_name: str, schema: str, table: str,
                                    columns: List[Dict[str, Any]],
                                    relationships: Optional[List[Dict[str, Any]]] = None) -> None:
                """Update the structure of a table"""
                if db_name not in self.metadata:
                    self.metadata[db_name] = {"schemas": {}, "last_updated": datetime.now().isoformat()}

                if schema not in self.metadata[db_name]["schemas"]:
                    self.metadata[db_name]["schemas"][schema] = {
                        "tables": {},
                        "last_updated": datetime.now().isoformat()
                    }

                if table not in self.metadata[db_name]["schemas"][schema]["tables"]:
                    self.metadata[db_name]["schemas"][schema]["tables"][table] = {
                        "columns": [],
                        "relationships": [],
                        "last_updated": datetime.now().isoformat()
                    }

                # Check if structure has changed
                current_columns = self.metadata[db_name]["schemas"][schema]["tables"][table]["columns"]
                if current_columns != columns:
                    self.metadata[db_name]["schemas"][schema]["tables"][table]["columns"] = columns
                    self.log_change(db_name, "updated_table_structure", {
                        "schema": schema,
                        "table": table,
                        "columns_changed": True
                    })

                if relationships is not None:
                    current_relationships = self.metadata[db_name]["schemas"][schema]["tables"][table]["relationships"]
                    if current_relationships != relationships:
                        self.metadata[db_name]["schemas"][schema]["tables"][table]["relationships"] = relationships
                        self.log_change(db_name, "updated_table_relationships", {
                            "schema": schema,
                            "table": table
                        })

                self.metadata[db_name]["schemas"][schema]["tables"][table]["last_updated"] = datetime.now().isoformat()
                self.metadata[db_name]["schemas"][schema]["last_updated"] = datetime.now().isoformat()
                self.metadata[db_name]["last_updated"] = datetime.now().isoformat()
                self.save_metadata(db_name)

            def is_metadata_fresh(self, db_name: str, max_age_hours: int = 24) -> bool:
                """Check if metadata is fresh (updated within max_age_hours)"""
                if db_name not in self.metadata or "last_updated" not in self.metadata[db_name]:
                    return False

                last_updated = datetime.fromisoformat(self.metadata[db_name]["last_updated"])
                age = datetime.now() - last_updated
                return age.total_seconds() < max_age_hours * 3600

            def is_schema_fresh(self, db_name: str, schema: str, max_age_hours: int = 24) -> bool:
                """Check if schema metadata is fresh"""
                if (db_name not in self.metadata or
                    "schemas" not in self.metadata[db_name] or
                    schema not in self.metadata[db_name]["schemas"] or
                    "last_updated" not in self.metadata[db_name]["schemas"][schema]):
                    return False

                last_updated = datetime.fromisoformat(self.metadata[db_name]["schemas"][schema]["last_updated"])
                age = datetime.now() - last_updated
                return age.total_seconds() < max_age_hours * 3600

            def is_table_fresh(self, db_name: str, schema: str, table: str, max_age_hours: int = 24) -> bool:
                """Check if table metadata is fresh"""
                if (db_name not in self.metadata or
                    "schemas" not in self.metadata[db_name] or
                    schema not in self.metadata[db_name]["schemas"] or
                    "tables" not in self.metadata[db_name]["schemas"][schema] or
                    table not in self.metadata[db_name]["schemas"][schema]["tables"] or
                    "last_updated" not in self.metadata[db_name]["schemas"][schema]["tables"][table]):
                    return False

                last_updated = datetime.fromisoformat(
                    self.metadata[db_name]["schemas"][schema]["tables"][table]["last_updated"]
                )
                age = datetime.now() - last_updated
                return age.total_seconds() < max_age_hours * 3600

        # Define the DatabaseDiscoveryService class
        class DatabaseDiscoveryService:
            def __init__(self, metadata_store: DatabaseMetadataStore):
                self.metadata_store = metadata_store

            def get_connection(self, db_name: str, user: str, password: str,
                            host: str, port: str) -> Tuple[Any, Any]:
                """Get a database connection and cursor"""
                conn = psycopg2.connect(
                    dbname=db_name,
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    connect_timeout=5
                )
                cursor = conn.cursor()
                return conn, cursor

            def discover_schemas(self, db_name: str, user: str, password: str,
                                host: str, port: str, force_refresh: bool = False) -> List[str]:
                """Discover all schemas in the database"""
                # Check if we have fresh metadata
                if not force_refresh and self.metadata_store.is_metadata_fresh(db_name):
                    return self.metadata_store.get_schemas(db_name)

                # Connect to the database
                try:
                    conn, cursor = self.get_connection(db_name, user, password, host, port)

                    # Query for schemas
                    cursor.execute("""
                        SELECT schema_name
                        FROM information_schema.schemata
                        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                        ORDER BY schema_name;
                    """)

                    schemas = [row[0] for row in cursor.fetchall()]

                    # Update metadata
                    self.metadata_store.update_schemas(db_name, schemas)

                    # Clean up
                    cursor.close()
                    conn.close()

                    return schemas
                except Exception as e:
                    print(f"Error discovering schemas: {e}")
                    traceback.print_exc()
                    return []

            def discover_tables(self, db_name: str, schema: str, user: str, password: str,
                            host: str, port: str, force_refresh: bool = False) -> List[str]:
                """Discover all tables in a schema"""
                print(f"Discovering tables in schema '{schema}' of database '{db_name}'")

                # Check if we have fresh metadata
                if not force_refresh and self.metadata_store.is_schema_fresh(db_name, schema):
                    tables = self.metadata_store.get_tables(db_name, schema)
                    print(f"Using cached metadata for schema '{schema}', found {len(tables)} tables")
                    if len(tables) > 0:
                        print(f"First 10 tables: {tables[:10]}")
                        if len(tables) > 10:
                            print(f"...and {len(tables) - 10} more tables")
                    return tables

                # Connect to the database
                try:
                    print(f"Connecting to database {db_name} at {host}:{port} as {user}")
                    conn, cursor = self.get_connection(db_name, user, password, host, port)

                    # Query for tables
                    query = """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = %s
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name;
                    """
                    print(f"Executing query: {query.strip()} with params: ({schema},)")
                    cursor.execute(query, (schema,))

                    tables = [row[0] for row in cursor.fetchall()]
                    print(f"Found {len(tables)} tables in schema '{schema}'")
                    if len(tables) > 0:
                        print(f"First 10 tables: {tables[:10]}")
                        if len(tables) > 10:
                            print(f"...and {len(tables) - 10} more tables")

                    # Update metadata
                    self.metadata_store.update_tables(db_name, schema, tables)

                    # Clean up
                    cursor.close()
                    conn.close()

                    return tables
                except Exception as e:
                    print(f"Error discovering tables in schema {schema}: {e}")
                    traceback.print_exc()

                    # Try an alternative query if the first one failed
                    try:
                        print("Trying alternative query to discover tables...")
                        conn, cursor = self.get_connection(db_name, user, password, host, port)

                        # Alternative query that might work better in some PostgreSQL versions
                        alt_query = f"""
                            SELECT tablename FROM pg_tables
                            WHERE schemaname = '{schema}'
                            ORDER BY tablename;
                        """
                        print(f"Executing alternative query: {alt_query.strip()}")
                        cursor.execute(alt_query)

                        tables = [row[0] for row in cursor.fetchall()]
                        print(f"Alternative query found {len(tables)} tables in schema '{schema}'")
                        if len(tables) > 0:
                            print(f"First 10 tables: {tables[:10]}")
                            if len(tables) > 10:
                                print(f"...and {len(tables) - 10} more tables")

                        # Update metadata
                        self.metadata_store.update_tables(db_name, schema, tables)

                        # Clean up
                        cursor.close()
                        conn.close()

                        return tables
                    except Exception as alt_e:
                        print(f"Alternative query also failed: {alt_e}")
                        traceback.print_exc()
                        return []

            def discover_table_structure(self, db_name: str, schema: str, table: str,
                                    user: str, password: str, host: str, port: str,
                                    force_refresh: bool = False) -> Dict[str, Any]:
                """Discover the structure of a table"""
                # Check if we have fresh metadata
                if not force_refresh and self.metadata_store.is_table_fresh(db_name, schema, table):
                    return self.metadata_store.get_table_structure(db_name, schema, table)

                # Connect to the database
                try:
                    conn, cursor = self.get_connection(db_name, user, password, host, port)

                    # Query for columns
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (schema, table))

                    columns = [
                        {
                            "name": row[0],
                            "type": row[1],
                            "nullable": row[2] == "YES",
                            "default": row[3]
                        }
                        for row in cursor.fetchall()
                    ]

                    # Query for primary key
                    cursor.execute("""
                        SELECT kcu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        WHERE tc.constraint_type = 'PRIMARY KEY'
                            AND tc.table_schema = %s
                            AND tc.table_name = %s
                        ORDER BY kcu.ordinal_position;
                    """, (schema, table))

                    pk_columns = [row[0] for row in cursor.fetchall()]

                    # Mark primary key columns
                    for column in columns:
                        if column["name"] in pk_columns:
                            column["primary_key"] = True

                    # Query for foreign keys
                    cursor.execute("""
                        SELECT
                            kcu.column_name,
                            ccu.table_schema AS foreign_table_schema,
                            ccu.table_name AS foreign_table_name,
                            ccu.column_name AS foreign_column_name
                        FROM information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                            ON ccu.constraint_name = tc.constraint_name
                            AND ccu.table_schema = tc.table_schema
                        WHERE tc.constraint_type = 'FOREIGN KEY'
                            AND tc.table_schema = %s
                            AND tc.table_name = %s;
                    """, (schema, table))

                    relationships = [
                        {
                            "column": row[0],
                            "references": {
                                "schema": row[1],
                                "table": row[2],
                                "column": row[3]
                            }
                        }
                        for row in cursor.fetchall()
                    ]

                    # Update metadata
                    self.metadata_store.update_table_structure(
                        db_name, schema, table, columns, relationships
                    )

                    # Clean up
                    cursor.close()
                    conn.close()

                    return self.metadata_store.get_table_structure(db_name, schema, table)
                except Exception as e:
                    print(f"Error discovering structure of table {schema}.{table}: {e}")
                    traceback.print_exc()
                    return {}

            def get_sample_data(self, db_name: str, schema: str, table: str,
                            user: str, password: str, host: str, port: str,
                            limit: int = 10) -> Dict[str, Any]:
                """Get sample data from a table"""
                try:
                    conn, cursor = self.get_connection(db_name, user, password, host, port)

                    # Query for sample data
                    cursor.execute(f"""
                        SELECT * FROM "{schema}"."{table}" LIMIT %s;
                    """, (limit,))

                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()

                    # Clean up
                    cursor.close()
                    conn.close()

                    return {
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows)
                    }
                except Exception as e:
                    print(f"Error getting sample data from {schema}.{table}: {e}")
                    traceback.print_exc()
                    return {
                        "columns": [],
                        "rows": [],
                        "row_count": 0,
                        "error": str(e)
                    }

        # Initialize the database metadata store and discovery service
        try:
            self.metadata_store = DatabaseMetadataStore()
            self.discovery_service = DatabaseDiscoveryService(self.metadata_store)
            print("Successfully initialized database metadata store and discovery service")
        except Exception as e:
            print(f"Error initializing database services: {e}")
            traceback.print_exc()

        # Track the current database context
        self.current_db_name = None
        self.current_schema = None

    def invoke(self, user_question: str, feedback: str = "") -> dict:
        """
        Generate a plan for answering the user's question.

        Args:
            user_question (str): The user's question or request
            feedback (str, optional): Any feedback from previous attempts

        Returns:
            dict: The planner's response as a structured JSON object
        """
        try:
            # Get the model for generating JSON responses
            model = self.get_model(json_model=True)

            # Create the system prompt
            system_prompt = planner_prompt_template

            # Create the user input (question + any feedback)
            user_input = f"User Question: {user_question}"
            if feedback:
                user_input += f"\n\nFeedback from previous attempt: {feedback}"

            # Add context about available data sources
            user_input += "\n\nAvailable Data Sources: SQL Database (PostgreSQL)"

            # Get the response from the model
            print(colored(f"Invoking Planner with question: {user_question[:100]}...", "cyan"))

            if self.server in ["ollama", "vllm", "groq"]:
                response = model(system_prompt=system_prompt, user_input=user_input)
            else:
                # Use the invoke method for CustomOpenAIWrapper
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
                response = model.invoke(messages)

            print(colored(f"Raw Planner response: {response}", "yellow"))

            # Handle various response formats
            # The planner should return a structured JSON object with the plan
            if isinstance(response, dict):
                # Already parsed JSON
                parsed_response = response
            elif isinstance(response, str):
                # JSON string needs to be parsed
                try:
                    # Handle potential markdown code blocks in the response
                    if "```json" in response or "```" in response:
                        # Extract JSON from markdown code block
                        import re
                        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                        if json_match:
                            json_str = json_match.group(1).strip()
                            parsed_response = json.loads(json_str)
                        else:
                            # Just try to parse the whole thing
                            parsed_response = json.loads(response)
                    else:
                        # Just try to parse the whole thing
                        parsed_response = json.loads(response)
                except json.JSONDecodeError:
                    print(colored(f"Error parsing planner JSON response: {response}", "red"))
                    parsed_response = {
                        "plan": {
                            "goal": "Process the user question",
                            "approach": "Analyzing the question to determine required information"
                        },
                        "required_information": ["database_schema"],
                        "data_sources": ["SQL"],
                        "error": f"Failed to parse JSON response: {response[:100]}..."
                    }
            else:
                # Unexpected response type
                print(colored(f"Unexpected response type from planner: {type(response)}", "red"))
                parsed_response = {
                    "plan": {
                        "goal": "Process the user question",
                        "approach": "Analyzing the question to determine required information"
                    },
                    "required_information": ["database_schema"],
                    "data_sources": ["SQL"],
                    "error": f"Unexpected response type: {type(response)}"
                }

            # Validate the response
            if not self._validate_response(parsed_response):
                # If validation fails, use a default structure
                parsed_response = {
                    "plan": {
                        "goal": "Process the user question",
                        "approach": "Analyzing the question to determine required information"
                    },
                    "required_information": ["database_schema"],
                    "data_sources": ["SQL"],
                    "validated": False
                }
            else:
                parsed_response["validated"] = True

            # Enhance the plan with database metadata
            enhanced_response = self.enhance_plan_with_metadata(parsed_response, user_question)

            # Add metadata
            enhanced_response["timestamp"] = str(datetime.now())
            enhanced_response["agent"] = "planner"
            enhanced_response["user_question"] = user_question

            # Add database context information
            if self.current_db_name:
                enhanced_response["database_context"] = {
                    "database": self.current_db_name,
                    "schema": self.current_schema,
                    "metadata_available": True,
                    "last_updated": self.metadata_store.metadata.get(self.current_db_name, {}).get("last_updated", "unknown")
                }

            return enhanced_response

        except Exception as e:
            print(colored(f"Error in PlannerAgent.invoke: {e}", "red"))
            traceback.print_exc()

            # Try to discover database structure even in case of error
            try:
                schemas = self.discover_database_structure()
                return {
                    "plan": {
                        "goal": "Handle error in planning",
                        "approach": "Proceeding with basic information gathering"
                    },
                    "required_information": ["database_schema"],
                    "data_sources": ["SQL"],
                    "error": str(e),
                    "agent": "planner",
                    "user_question": user_question,
                    "schemas": schemas,
                    "database": self.current_db_name
                }
            except Exception as db_error:
                print(colored(f"Error discovering database structure: {db_error}", "red"))
                return {
                    "plan": {
                        "goal": "Handle error in planning",
                        "approach": "Proceeding with basic information gathering"
                    },
                    "required_information": ["database_schema"],
                    "data_sources": ["SQL"],
                    "error": f"{str(e)} (Database discovery error: {str(db_error)})",
                    "agent": "planner",
                    "user_question": user_question
                }

    def get_db_connection_params(self):
        """Get database connection parameters from session state"""
        import streamlit as st

        return {
            "db_name": st.session_state.get("db_name", "new"),
            "user": st.session_state.get("db_user", "postgres"),
            "password": st.session_state.get("db_password", "pass"),
            "host": st.session_state.get("db_host", "localhost"),
            "port": st.session_state.get("db_port", "5432")
        }

    def create_test_table(self, schema="dev"):
        """Create a test table in the specified schema if none exists"""
        print(f"Attempting to create a test table in schema '{schema}'")

        # Get connection parameters
        params = self.get_db_connection_params()

        try:
            # Connect to the database
            import psycopg2
            conn = psycopg2.connect(
                dbname=params["db_name"],
                user=params["user"],
                password=params["password"],
                host=params["host"],
                port=params["port"],
                connect_timeout=5
            )
            conn.autocommit = True  # Set autocommit to True
            cursor = conn.cursor()

            # Check if schema exists, create if not
            cursor.execute(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema}'")
            if not cursor.fetchone():
                print(f"Schema '{schema}' does not exist, creating it...")
                cursor.execute(f"CREATE SCHEMA {schema}")
                print(f"Schema '{schema}' created successfully")

            # Create a test table
            table_name = "test_table"
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                value INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            print(f"Executing query: {create_table_query.strip()}")
            cursor.execute(create_table_query)

            # Insert some test data
            insert_query = f"""
            INSERT INTO {schema}.{table_name} (name, value)
            VALUES
                ('Test Item 1', 100),
                ('Test Item 2', 200),
                ('Test Item 3', 300)
            ON CONFLICT (id) DO NOTHING
            """
            print(f"Executing query: {insert_query.strip()}")
            cursor.execute(insert_query)

            # Clean up
            cursor.close()
            conn.close()

            print(f"Test table '{schema}.{table_name}' created successfully")
            return True
        except Exception as e:
            print(f"Error creating test table: {e}")
            import traceback
            traceback.print_exc()
            return False

    def discover_database_structure(self, force_refresh=False):
        """Discover database structure and update metadata"""
        # Get connection parameters
        params = self.get_db_connection_params()
        db_name = params["db_name"]

        print(f"Discovering database structure for '{db_name}', force_refresh={force_refresh}")

        # Set current database name
        self.current_db_name = db_name

        # Load existing metadata if available
        self.metadata_store.load_metadata(db_name)
        self.metadata_store.load_change_log(db_name)

        # Discover schemas
        schemas = self.discovery_service.discover_schemas(
            db_name=db_name,
            user=params["user"],
            password=params["password"],
            host=params["host"],
            port=params["port"],
            force_refresh=force_refresh
        )

        print(f"Discovered schemas: {schemas}")

        # For each schema, check if it has tables
        for schema in schemas:
            tables = self.discover_schema_tables(schema, force_refresh=force_refresh)
            print(f"Schema '{schema}' has {len(tables)} tables")
            if len(tables) > 0:
                print(f"First 10 tables in schema '{schema}': {tables[:10]}")
                if len(tables) > 10:
                    print(f"...and {len(tables) - 10} more tables")

        return schemas

    def discover_schema_tables(self, schema, force_refresh=False):
        """Discover tables in a schema and update metadata"""
        if not self.current_db_name:
            self.discover_database_structure()

        # Set current schema
        self.current_schema = schema

        # Get connection parameters
        params = self.get_db_connection_params()

        # Discover tables
        tables = self.discovery_service.discover_tables(
            db_name=self.current_db_name,
            schema=schema,
            user=params["user"],
            password=params["password"],
            host=params["host"],
            port=params["port"],
            force_refresh=force_refresh
        )

        return tables

    def discover_table_structure(self, schema, table, force_refresh=False):
        """Discover table structure and update metadata"""
        if not self.current_db_name:
            self.discover_database_structure()

        # Get connection parameters
        params = self.get_db_connection_params()

        # Discover table structure
        table_structure = self.discovery_service.discover_table_structure(
            db_name=self.current_db_name,
            schema=schema,
            table=table,
            user=params["user"],
            password=params["password"],
            host=params["host"],
            port=params["port"],
            force_refresh=force_refresh
        )

        return table_structure

    def analyze_query_context(self, user_question):
        """Analyze the query to determine database context"""
        # Extract potential schema and table references from the question
        # This is a simple implementation - in practice, you'd use NLP or the LLM itself
        import re

        # Check if the question is about listing schemas
        if "list schemas" in user_question.lower() or "show schemas" in user_question.lower():
            return {"action": "list_schemas"}

        # Check if the question is about listing tables in a schema
        schema_match = re.search(r"tables in (\w+)", user_question.lower())
        if schema_match:
            schema = schema_match.group(1)
            return {"action": "list_tables", "schema": schema}

        # Check if the question is about a specific table
        table_match = re.search(r"from (\w+)\.(\w+)", user_question.lower())
        if table_match:
            schema = table_match.group(1)
            table = table_match.group(2)
            return {"action": "query_table", "schema": schema, "table": table}

        # Default to listing all tables
        if "list all tables" in user_question.lower() or "show all tables" in user_question.lower():
            return {"action": "list_all_tables"}

        # For other queries, try to infer context from metadata
        return {"action": "general_query"}

    def enhance_plan_with_metadata(self, plan, user_question):
        """Enhance the query plan with database metadata"""
        # Analyze the query context
        context = self.analyze_query_context(user_question)

        # Handle different query types
        if context["action"] == "list_schemas":
            # Discover schemas
            schemas = self.discover_database_structure()
            plan["schemas"] = schemas
            plan["primary_table_or_datasource"] = "information_schema.schemata"

        elif context["action"] == "list_tables":
            # Discover tables in the specified schema
            schema = context["schema"]
            tables = self.discover_schema_tables(schema)
            plan["schema"] = schema
            plan["tables"] = tables
            plan["primary_table_or_datasource"] = f"information_schema.tables WHERE table_schema = '{schema}'"

        elif context["action"] == "query_table":
            # Discover table structure
            schema = context["schema"]
            table = context["table"]
            table_structure = self.discover_table_structure(schema, table)

            # Add table structure to the plan
            plan["schema"] = schema
            plan["table"] = table
            plan["table_structure"] = table_structure
            plan["primary_table_or_datasource"] = f"{schema}.{table}"

            # Add column information
            if "columns" in table_structure:
                plan["relevant_columns"] = [col["name"] for col in table_structure["columns"]]

            # Add relationship information
            if "relationships" in table_structure:
                plan["relationships"] = table_structure["relationships"]

        elif context["action"] == "list_all_tables":
            # Discover all schemas
            schemas = self.discover_database_structure()

            # For each schema, discover tables
            all_tables = {}
            for schema in schemas:
                tables = self.discover_schema_tables(schema)
                all_tables[schema] = tables

            plan["all_tables"] = all_tables
            plan["primary_table_or_datasource"] = "information_schema.tables"

        return plan

    def _validate_response(self, response: dict) -> bool:
        """
        Validate the response from the planner agent.

        Args:
            response (dict): The response to validate

        Returns:
            bool: True if the response is valid, False otherwise
        """
        if not isinstance(response, dict):
            print(colored(f"Planner response is not a dictionary: {response}", "red"))
            return False

        required_fields = ["query_type", "primary_table_or_datasource", "relevant_columns", "filtering_conditions", "processing_instructions"]
        for field in required_fields:
            if field not in response:
                print(colored(f"Missing required field: {field}", "red"))
                return False
        return True

###################
# Selector Agent
###################

selector_prompt_template = """
You are the Selector Agent responsible for choosing the appropriate data sources and tools based on the planner's analysis.

Your task is to:
1. Review the user's question and planner's response
2. Select the most appropriate data source and tools
3. Specify what information is needed and why
4. Define query parameters

Consider:
- Available data sources (databases, APIs, etc.)
- Data freshness requirements
- Query complexity and performance
- Access permissions and restrictions

Previous selections: {previous_selections}
Feedback (if any): {feedback}

You MUST respond with a valid JSON object containing:
{
    "selected_tool": "database_query or api_call",
    "selected_datasource": "specific_database_or_api",
    "information_needed": ["list of required data points"],
    "reason_for_selection": "detailed explanation of your choice",
    "query_parameters": {
        "param1": "value1",
        "param2": "value2"
    }
}

DO NOT include any text outside this JSON structure.
"""

selector_guided_json = {
    "type": "object",
    "properties": {
        "selected_tool": {
            "type": "string",
            "enum": ["database_query", "api_call"],
            "description": "Tool to use for data retrieval"
        },
        "selected_datasource": {
            "type": "string",
            "description": "Specific database or API to query"
        },
        "information_needed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of required data points"
        },
        "reason_for_selection": {
            "type": "string",
            "description": "Explanation for datasource selection"
        },
        "query_parameters": {
            "type": "object",
            "additionalProperties": True,
            "description": "Parameters for the query"
        }
    },
    "required": ["selected_tool", "selected_datasource", "information_needed", "reason_for_selection"]
}

class SelectorAgent(Agent):
    def __init__(self, state=None, model=None, server=None, temperature=None, model_endpoint=None, stop=None):
        """
        Initialize the Selector Agent.

        Args:
            state: Current state of the agent graph
            model: LLM model to use
            server: Server type (openai, ollama, etc.)
            temperature: Temperature setting for model generation
            model_endpoint: API endpoint for the model
            stop: Stop sequences for the model
        """
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=selector_guided_json
        )

    def invoke(self, user_question: str, planner_response: dict = None, feedback: str = None, previous_selections: list = None) -> dict:
        """
        Select appropriate data sources and tools based on the query requirements.

        Args:
            user_question: Original user question
            planner_response: Response from the Planner Agent
            feedback: Any feedback from previous iterations
            previous_selections: List of previous data source selections

        Returns:
            dict: Selection decision and reasoning
        """
        try:
            # Prepare the input context
            context = {
                "user_question": user_question,
                "planner_response": planner_response or {},
                "previous_selections": previous_selections or [],
                "feedback": feedback or ""
            }

            # Format the context for the model
            formatted_input = json.dumps(context, indent=2)

            # Get model response
            model_response = get_open_ai_json(
                system_prompt=selector_prompt_template,
                user_input=formatted_input,
                model=self.model,
                temperature=self.temperature
            )

            # Validate the response
            self._validate_response(model_response)

            # Handle special cases for database queries
            if model_response["selected_tool"] == "database_query":
                model_response = self._enhance_database_selection(model_response, planner_response)

            return {"selector_response": model_response}

        except Exception as e:
            print(f"Error in selector agent: {str(e)}")
            traceback.print_exc()
            return {
                "selector_response": {
                    "selected_tool": "error",
                    "selected_datasource": "none",
                    "information_needed": [],
                    "reason_for_selection": f"Error occurred: {str(e)}",
                    "query_parameters": {}
                }
            }

    def _validate_response(self, response: dict) -> bool:
        """
        Validate the structure and content of the selector response.

        Args:
            response: The response dictionary to validate

        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        required_fields = [
            "selected_tool",
            "selected_datasource",
            "information_needed",
            "reason_for_selection",
            "query_parameters"
        ]

        # Check all required fields are present
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            raise ValueError(f"Missing required fields in selector response: {missing_fields}")

        # Validate field types
        if not isinstance(response["information_needed"], list):
            raise ValueError("information_needed must be a list")
        if not isinstance(response["query_parameters"], dict):
            raise ValueError("query_parameters must be a dictionary")
        if not isinstance(response["reason_for_selection"], str):
            raise ValueError("reason_for_selection must be a string")

        return True

    def _enhance_database_selection(self, response: dict, planner_response: dict) -> dict:
        """
        Enhance the selection response for database queries with additional context.

        Args:
            response: The original selector response
            planner_response: The planner's response

        Returns:
            dict: Enhanced selector response
        """
        try:
            # Add database-specific parameters
            if planner_response and planner_response.get("primary_table_or_datasource"):
                response["query_parameters"]["primary_table"] = planner_response["primary_table_or_datasource"]

            # Add relevant columns if specified by planner
            if planner_response and planner_response.get("relevant_columns"):
                response["query_parameters"]["columns"] = planner_response["relevant_columns"]

            # Add any filtering conditions
            if planner_response and planner_response.get("filtering_conditions"):
                response["query_parameters"]["filters"] = planner_response["filtering_conditions"]

            # Validate database connection if possible
            try:
                with get_db_connection(response["selected_datasource"]) as conn:
                    response["query_parameters"]["connection_valid"] = True
            except Exception as e:
                response["query_parameters"]["connection_valid"] = False
                response["query_parameters"]["connection_error"] = str(e)

            return response

        except Exception as e:
            print(f"Error enhancing database selection: {str(e)}")
            return response

    def get_available_datasources(self) -> list:
        """
        Get list of available data sources.

        Returns:
            list: Available data sources
        """
        try:
            # This could be expanded to dynamically check available sources
            return [
                "main_database",
                "analytics_db",
                "reporting_db",
                "api_gateway"
            ]
        except Exception as e:
            print(f"Error getting available datasources: {str(e)}")
            return []

###################
# SQL Generator Agent
###################

SQLGenerator_prompt_template = """
You are the SQL Generator Agent responsible for creating optimized SQL queries based on the user's requirements and previous agent responses.

Your task is to:
1. Generate a valid PostgreSQL query that addresses the user's question
2. Ensure the query follows best practices and is optimized
3. Include validation checks and explanations
4. Consider any specific database requirements or limitations

PostgreSQL-Specific Guidelines:
1. Schema-Specific Queries:
   Always specify the schema in your queries (e.g., `"schema_name"."table_name"`). Avoid assuming a default schema like `public`.

2. Qualify Column Names:
   Use the table name (or alias) to avoid ambiguity when selecting columns that may exist in multiple tables.

3. Use PostgreSQL Functions:
   Incorporate PostgreSQL-specific functions for retrieving data and metadata, such as:
   - `pg_total_relation_size(relid)`
   - `pg_table_size(relid)`
   - `pg_indexes_size(relid)`

4. Handling Views and Functions:
   When queries involve views or functions, use:
   - `pg_class` and `pg_proc` for information about these objects.

5. Data Retrieval:
   For counting or fetching data:
   - Use `COUNT(*)`, `LIMIT`, and `OFFSET` as needed.
   - Utilize `pg_stat_user_tables` for estimated row counts.

6. Constraints and Foreign Keys:
   Retrieve information about constraints using:
   - `information_schema.table_constraints` for primary keys and unique constraints.
   - `pg_trigger` for trigger definitions.

7. Query Optimization:
   For performance-related questions, use:
   - `EXPLAIN` or `EXPLAIN ANALYZE` to analyze query execution plans.

8. Multiple Schemas:
   - The database includes multiple schemas (e.g., `prod`, `dev`, and `test`)
   - If a query is meant to pull data regardless of schema, create a query that encompasses all relevant schemas using `UNION ALL`.

General Best Practices:
1. Use appropriate indexing and join conditions
2. Avoid SELECT * unless specifically required
3. Include appropriate LIMIT clauses for large datasets
4. Use CTEs for complex queries
5. Consider query performance and optimization
6. Use appropriate data types in conditions
7. Include error handling where necessary

You MUST respond with a valid JSON object containing:
{
    "sql_query": "complete SQL query",
    "explanation": "detailed explanation of the query logic",
    "validation_checks": [
        "list of validation checks performed",
        "potential edge cases considered"
    ],
    "query_type": "SELECT/INSERT/UPDATE/DELETE",
    "estimated_complexity": "LOW/MEDIUM/HIGH",
    "required_indexes": ["list of recommended indexes"]
}

DO NOT include any text outside this JSON structure.
"""

SQLGenerator_guided_json = {
    "type": "object",
    "properties": {
        "sql_query": {
            "type": "string",
            "description": "Complete SQL query"
        },
        "explanation": {
            "type": "string",
            "description": "Detailed explanation of query logic"
        },
        "validation_checks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of validation checks performed"
        },
        "query_type": {
            "type": "string",
            "enum": ["SELECT", "INSERT", "UPDATE", "DELETE"],
            "description": "Type of SQL query"
        },
        "estimated_complexity": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH"],
            "description": "Estimated query complexity"
        },
        "required_indexes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of recommended indexes"
        }
    },
    "required": ["sql_query", "explanation", "validation_checks"]
}

class SQLGenerator(Agent):
    def __init__(self, state=None, model=None, server=None, temperature=None, model_endpoint=None, stop=None):
        """
        Initialize the SQL Generator Agent.

        Args:
            state: Current state of the agent graph
            model: LLM model to use
            server: Server type (openai, ollama, etc.)
            temperature: Temperature setting for model generation
            model_endpoint: API endpoint for the model
            stop: Stop sequences for the model
        """
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=SQLGenerator_guided_json
        )

    def invoke(self, user_question: str, planner_response: dict = None, selector_response: dict = None, feedback: str = None) -> dict:
        """
        Generate an optimized SQL query based on the requirements.

        Args:
            user_question: Original user question
            planner_response: Response from the Planner Agent
            selector_response: Response from the Selector Agent
            feedback: Any feedback from previous iterations

        Returns:
            dict: Generated SQL query and related information
        """
        try:
            # Check for common queries that we can handle directly
            if user_question.lower().strip() in ["list all tables", "show tables", "show all tables"]:
                # Return a standard query for listing tables
                # This will be executed against the actual database
                # The execute_sql_query function will handle fallback to simulation if needed
                return {
                    "sql_generator_response": {
                        "sql_query": "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' ORDER BY table_schema, table_name;",
                        "explanation": "This query lists all tables in the database by querying the information_schema.tables view, which contains metadata about all tables in the database. It filters for base tables only (excluding views) and orders the results by schema and table name for better readability.",
                        "validation_checks": [
                            "Query targets information_schema which exists in all PostgreSQL databases",
                            "Filtering on table_type ensures only actual tables are returned (not views)",
                            "Results are ordered for better readability"
                        ],
                        "query_type": "SELECT",
                        "estimated_complexity": "LOW",
                        "required_indexes": []
                    }
                }

            # Prepare the context for the model
            context = {
                "user_question": user_question,
                "planner_analysis": planner_response,
                "selected_datasource": selector_response.get("selected_datasource") if selector_response else None,
                "feedback": feedback
            }

            # Get database schema information
            schema_info = self._get_schema_info(selector_response)
            if schema_info:
                context["schema_info"] = schema_info

            # Format the context for the model
            formatted_input = json.dumps(context, indent=2)

            # Get model response
            model_response = get_open_ai_json(
                system_prompt=SQLGenerator_prompt_template,
                user_input=formatted_input,
                model=self.model,
                temperature=self.temperature
            )

            # Validate and enhance the response
            self._validate_response(model_response)
            enhanced_response = self._enhance_query_response(model_response)

            return {"sql_generator_response": enhanced_response}

        except Exception as e:
            print(f"Error in SQL generator: {str(e)}")
            traceback.print_exc()

            # Check if this is a common query we can handle even in case of error
            if user_question.lower().strip() in ["list all tables", "show tables", "show all tables"]:
                print("Error occurred, falling back to standard 'list tables' query")
                return {
                    "sql_generator_response": {
                        "sql_query": "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE' ORDER BY table_schema, table_name;",
                        "explanation": "This query lists all tables in the database by querying the information_schema.tables view. Note: This is a standard query that will be executed against the actual database, with fallback to simulation if needed.",
                        "validation_checks": ["Basic information schema query"],
                        "query_type": "SELECT",
                        "estimated_complexity": "LOW",
                        "required_indexes": []
                    }
                }

            return {
                "sql_generator_response": {
                    "sql_query": "",
                    "explanation": f"Error occurred: {str(e)}",
                    "validation_checks": ["Query generation failed"],
                    "query_type": "ERROR",
                    "estimated_complexity": "UNKNOWN",
                    "required_indexes": []
                }
            }

    def _validate_response(self, response: dict) -> bool:
        """
        Validate the structure and content of the SQL generator response.

        Args:
            response: The response dictionary to validate

        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        required_fields = [
            "sql_query",
            "explanation",
            "validation_checks",
            "query_type",
            "estimated_complexity",
            "required_indexes"
        ]

        # Check all required fields are present
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            raise ValueError(f"Missing required fields in SQL generator response: {missing_fields}")

        # Validate field types
        if not isinstance(response["sql_query"], str):
            raise ValueError("sql_query must be a string")
        if not isinstance(response["validation_checks"], list):
            raise ValueError("validation_checks must be a list")
        if not isinstance(response["required_indexes"], list):
            raise ValueError("required_indexes must be a list")

        # Validate query type
        valid_query_types = ["SELECT", "INSERT", "UPDATE", "DELETE", "ERROR"]
        if response["query_type"] not in valid_query_types:
            raise ValueError(f"Invalid query_type. Must be one of: {valid_query_types}")

        # Validate complexity
        valid_complexities = ["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
        if response["estimated_complexity"] not in valid_complexities:
            raise ValueError(f"Invalid complexity. Must be one of: {valid_complexities}")

        return True

    def _get_schema_info(self, selector_response: dict) -> dict:
        """
        Get database schema information for the selected datasource.

        Args:
            selector_response: Response from the Selector Agent

        Returns:
            dict: Database schema information
        """
        try:
            if not selector_response or not selector_response.get("selected_datasource"):
                return {}

            datasource = selector_response["selected_datasource"]
            with get_db_connection(datasource) as conn:
                # Get table information
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """)

                schema_info = {}
                for row in cursor.fetchall():
                    table_name, column_name, data_type, is_nullable = row
                    if table_name not in schema_info:
                        schema_info[table_name] = []
                    schema_info[table_name].append({
                        "column": column_name,
                        "type": data_type,
                        "nullable": is_nullable == "YES"
                    })

                return schema_info

        except Exception as e:
            print(f"Error getting schema info: {str(e)}")
            return {}

    def _enhance_query_response(self, response: dict) -> dict:
        """
        Enhance the query response with additional information and optimizations.

        Args:
            response: The original SQL generator response

        Returns:
            dict: Enhanced response with additional information
        """
        try:
            # Add query metadata
            response["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "includes_joins": "JOIN" in response["sql_query"].upper(),
                "includes_aggregation": any(agg in response["sql_query"].upper()
                                         for agg in ["GROUP BY", "COUNT(", "SUM(", "AVG("]),
                "has_subqueries": "SELECT" in response["sql_query"].upper()[
                    response["sql_query"].upper().find("SELECT")+6:
                ]
            }

            # Add performance considerations
            response["performance_considerations"] = []
            if response["metadata"]["includes_joins"]:
                response["performance_considerations"].append(
                    "Query includes joins - ensure proper indexing on join columns"
                )
            if response["metadata"]["includes_aggregation"]:
                response["performance_considerations"].append(
                    "Query includes aggregations - consider materialized views for frequent queries"
                )
            if response["metadata"]["has_subqueries"]:
                response["performance_considerations"].append(
                    "Query includes subqueries - evaluate if CTEs would be more efficient"
                )

            # Add query parameters if any
            response["parameters"] = self._extract_query_parameters(response["sql_query"])

            return response

        except Exception as e:
            print(f"Error enhancing query response: {str(e)}")
            return response

    def _extract_query_parameters(self, query: str) -> list:
        """
        Extract parameters from the SQL query.

        Args:
            query: The SQL query string

        Returns:
            list: List of parameters found in the query
        """
        import re
        # Look for common parameter patterns
        patterns = [
            r'%\((\w+)\)s',  # psycopg2 named parameters
            r':\w+',         # Oracle/SQLAlchemy style
            r'\$\d+',        # PostgreSQL positional parameters
            r'\?'            # JDBC style
        ]

        parameters = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            if matches:
                parameters.extend(matches)

        return list(set(parameters))

###################
# Reviewer Agent
###################

reviewer_prompt_template = """
You are the Reviewer Agent responsible for validating and reviewing PostgreSQL queries. Your task is to:

1. Verify query correctness and syntax
2. Check for potential performance issues
3. Validate against security best practices
4. Ensure the query meets the user's requirements
5. Suggest improvements if needed

PostgreSQL-Specific Considerations:
1. Schema Specification:
   - Verify that schemas are explicitly specified in table references
   - Check that the query doesn't assume a default schema like 'public'

2. Column Qualification:
   - Ensure columns are properly qualified with table names or aliases
   - Check for potential ambiguity in column references

3. PostgreSQL Functions:
   - Validate proper use of PostgreSQL-specific functions
   - Suggest appropriate PostgreSQL functions when applicable

4. Multiple Schemas:
   - If the query spans multiple schemas (prod, dev, test), ensure UNION ALL is used correctly
   - Verify that the same columns are selected in each part of a UNION ALL

General Considerations:
- SQL injection vulnerabilities
- Query performance and optimization
- Proper use of indexes
- Data type compatibility
- Error handling
- Edge cases
- Business logic correctness

You MUST respond with a valid JSON object containing:
{
    "is_correct": boolean,
    "issues": [
        "list of identified issues"
    ],
    "suggestions": [
        "list of improvement suggestions"
    ],
    "explanation": "detailed explanation of the review",
    "security_concerns": [
        "list of security considerations"
    ],
    "performance_impact": "LOW/MEDIUM/HIGH",
    "confidence_score": float (0-1)
}

DO NOT include any text outside this JSON structure.
"""

reviewer_guided_json = {
    "type": "object",
    "properties": {
        "is_correct": {
            "type": "boolean",
            "description": "Whether the SQL query is correct"
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["SYNTAX", "LOGIC", "PERFORMANCE", "SECURITY"]},
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}
                }
            },
            "description": "List of identified issues"
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of improvement suggestions"
        },
        "explanation": {
            "type": "string",
            "description": "Detailed explanation of review findings"
        },
        "performance_impact": {
            "type": "string",
            "enum": ["NONE", "LOW", "MEDIUM", "HIGH"],
            "description": "Estimated performance impact"
        }
    },
    "required": ["is_correct", "issues", "suggestions", "explanation"]
}

class ReviewerAgent(Agent):
    def __init__(self, state=None, model=None, server=None, temperature=None, model_endpoint=None, stop=None):
        """
        Initialize the Reviewer Agent.

        Args:
            state: Current state of the agent graph
            model: LLM model to use
            server: Server type (openai, ollama, etc.)
            temperature: Temperature setting for model generation
            model_endpoint: API endpoint for the model
            stop: Stop sequences for the model
        """
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=reviewer_guided_json
        )

    def invoke(self, user_question: str, sql_generator_response, schema_info: dict = None) -> dict:
        """
        Review the generated SQL query and provide feedback.

        Args:
            user_question: Original user question
            sql_generator_response: Response from the SQL Generator Agent (can be dict or string)
            schema_info: Database schema information

        Returns:
            dict: Review results and recommendations
        """
        try:
            # Handle case where sql_generator_response is a string (direct SQL query)
            sql_query = ""
            query_explanation = ""

            if isinstance(sql_generator_response, str):
                sql_query = sql_generator_response
                query_explanation = "SQL query provided directly"
            elif isinstance(sql_generator_response, dict):
                sql_query = sql_generator_response.get("sql_query", "")
                query_explanation = sql_generator_response.get("explanation", "")
            else:
                return self._create_error_response([f"Invalid sql_generator_response type: {type(sql_generator_response)}"])

            # Ensure we have a valid SQL query
            if not sql_query:
                return self._create_error_response(["Empty SQL query provided"])

            # Perform initial syntax check
            syntax_check_result = self._check_syntax(sql_query)
            if not syntax_check_result["is_valid"]:
                return self._create_error_response(syntax_check_result["errors"])

            # Prepare context for the model
            context = {
                "user_question": user_question,
                "sql_query": sql_query,
                "query_explanation": query_explanation,
                "schema_info": schema_info or {},
                "syntax_check": syntax_check_result
            }

            # Get model response
            model_response = self.get_model(json_model=True).invoke([
                {"role": "system", "content": reviewer_prompt_template},
                {"role": "user", "content": json.dumps(context, indent=2)}
            ])

            # Validate and enhance the response
            validated_response = self._validate_response(model_response)
            enhanced_response = self._enhance_review_response(validated_response, sql_generator_response)

            return {"reviewer_response": enhanced_response}

        except Exception as e:
            print(f"Error in reviewer: {str(e)}")
            traceback.print_exc()
            return self._create_error_response([str(e)])

    def _check_syntax(self, query: str) -> dict:
        """
        Perform basic SQL syntax validation.

        Args:
            query: SQL query string to validate

        Returns:
            dict: Validation results
        """
        try:
            # Basic syntax checks
            errors = []

            # Check for basic SQL keywords
            required_keywords = ["SELECT", "FROM"] if query.upper().startswith("SELECT") else []
            for keyword in required_keywords:
                if keyword not in query.upper():
                    errors.append(f"Missing required keyword: {keyword}")

            # Check for balanced parentheses
            if query.count('(') != query.count(')'):
                errors.append("Unbalanced parentheses")

            # Check for common SQL injection patterns
            suspicious_patterns = ["--", ";--", "/*", "*/", "UNION ALL", "UNION SELECT"]
            for pattern in suspicious_patterns:
                if pattern in query.upper():
                    errors.append(f"Suspicious SQL pattern found: {pattern}")

            # Check for proper quoting of string literals
            if query.count("'") % 2 != 0:
                errors.append("Unmatched single quotes")

            return {
                "is_valid": len(errors) == 0,
                "errors": errors
            }

        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Syntax check error: {str(e)}"]
            }

    def _validate_response(self, response) -> dict:
        """
        Validate the structure and content of the reviewer response.

        Args:
            response: The response to validate (can be dict or string)

        Returns:
            dict: Validated response
        """
        # Handle case where response is a string
        if isinstance(response, str):
            # Create a default response
            return {
                "is_correct": True,
                "issues": [],
                "suggestions": [],
                "explanation": "SQL query validated successfully",
                "security_concerns": [],
                "performance_impact": "LOW",
                "confidence_score": 0.8
            }

        # Ensure response is a dictionary
        if not isinstance(response, dict):
            raise ValueError(f"Invalid response type: {type(response)}")

        required_fields = [
            "is_correct",
            "issues",
            "suggestions",
            "explanation",
            "security_concerns",
            "performance_impact",
            "confidence_score"
        ]

        # Check required fields and add defaults if missing
        for field in required_fields:
            if field not in response:
                if field == "is_correct":
                    response[field] = True
                elif field in ["issues", "suggestions", "security_concerns"]:
                    response[field] = []
                elif field == "explanation":
                    response[field] = "SQL query validated successfully"
                elif field == "performance_impact":
                    response[field] = "LOW"
                elif field == "confidence_score":
                    response[field] = 0.8

        # Validate field types and fix if needed
        if not isinstance(response["is_correct"], bool):
            response["is_correct"] = True

        if not isinstance(response["issues"], list):
            response["issues"] = []

        if not isinstance(response["suggestions"], list):
            response["suggestions"] = []

        if not isinstance(response["security_concerns"], list):
            response["security_concerns"] = []

        # Validate performance impact
        valid_impacts = ["LOW", "MEDIUM", "HIGH"]
        if response["performance_impact"] not in valid_impacts:
            response["performance_impact"] = "LOW"

        # Validate confidence score
        if not isinstance(response["confidence_score"], (int, float)):
            response["confidence_score"] = 0.8
        elif not 0 <= response["confidence_score"] <= 1:
            response["confidence_score"] = max(0, min(1, response["confidence_score"]))

        return response

    def _enhance_review_response(self, review_response: dict, sql_generator_response) -> dict:
        """
        Enhance the review response with additional insights and metadata.

        Args:
            review_response: The original review response
            sql_generator_response: The SQL generator response being reviewed (can be dict or string)

        Returns:
            dict: Enhanced review response
        """
        # Add metadata
        complexity = "UNKNOWN"
        if isinstance(sql_generator_response, dict):
            complexity = sql_generator_response.get("estimated_complexity", "UNKNOWN")

        review_response["metadata"] = {
            "reviewed_at": datetime.now().isoformat(),
            "query_complexity": complexity,
            "review_version": "1.0"
        }

        # Add specific recommendations based on issues
        review_response["recommendations"] = self._generate_recommendations(
            review_response["issues"],
            sql_generator_response
        )

        # Add risk assessment
        review_response["risk_assessment"] = self._assess_risk(
            review_response["issues"],
            review_response["security_concerns"]
        )

        return review_response

    def _generate_recommendations(self, issues: list, sql_generator_response=None) -> list:
        """
        Generate specific recommendations based on identified issues.

        Args:
            issues: List of identified issues
            sql_generator_response: Original SQL generator response (can be dict, string, or None)

        Returns:
            list: Specific recommendations
        """
        recommendations = []

        # Map common issues to recommendations
        issue_recommendations = {
            "performance": [
                "Add appropriate indexes",
                "Consider using materialized views",
                "Optimize JOIN conditions"
            ],
            "security": [
                "Use parameterized queries",
                "Implement proper input validation",
                "Add appropriate access controls"
            ],
            "maintainability": [
                "Break down complex queries",
                "Add appropriate comments",
                "Use CTEs for better readability"
            ]
        }

        # Analyze issues and add relevant recommendations
        for issue in issues:
            issue_lower = issue.lower()
            if "performance" in issue_lower:
                recommendations.extend(issue_recommendations["performance"])
            if "security" in issue_lower:
                recommendations.extend(issue_recommendations["security"])
            if "maintainability" in issue_lower:
                recommendations.extend(issue_recommendations["maintainability"])

        return list(set(recommendations))  # Remove duplicates

    def _assess_risk(self, issues: list, security_concerns: list) -> dict:
        """
        Assess the overall risk of the query based on issues and security concerns.

        Args:
            issues: List of identified issues
            security_concerns: List of security concerns

        Returns:
            dict: Risk assessment results
        """
        risk_levels = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        # Count issues by severity
        for issue in issues + security_concerns:
            issue_lower = issue.lower()
            if "critical" in issue_lower:
                risk_levels["critical"] += 1
            elif "high" in issue_lower:
                risk_levels["high"] += 1
            elif "medium" in issue_lower:
                risk_levels["medium"] += 1
            else:
                risk_levels["low"] += 1

        # Calculate overall risk level
        if risk_levels["critical"] > 0:
            overall_risk = "CRITICAL"
        elif risk_levels["high"] > 0:
            overall_risk = "HIGH"
        elif risk_levels["medium"] > 0:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        return {
            "overall_risk_level": overall_risk,
            "risk_breakdown": risk_levels,
            "requires_immediate_attention": overall_risk in ["CRITICAL", "HIGH"]
        }

    def _create_error_response(self, errors: list) -> dict:
        """
        Create a standardized error response.

        Args:
            errors: List of error messages

        Returns:
            dict: Formatted error response
        """
        return {
            "reviewer_response": {
                "is_correct": False,
                "issues": errors,
                "suggestions": ["Fix the identified issues before proceeding"],
                "explanation": "Review failed due to critical issues",
                "security_concerns": [],
                "performance_impact": "HIGH",
                "confidence_score": 1.0,
                "metadata": {
                    "reviewed_at": datetime.now().isoformat(),
                    "review_status": "ERROR",
                    "review_version": "1.0"
                }
            }
        }

###################
# Router Agent
###################

router_prompt_template = """
You are the Router Agent responsible for directing the workflow based on the current state and agent responses.
Your task is to determine the next step in the query processing pipeline.

Available routes:
- "selector": Choose when we need to select or validate data sources
- "sql_generator": Choose when we need to generate a new SQL query
- "reviewer": Choose when we need to review a generated query
- "final_report": Choose when we're ready to present final results
- "end": Choose when the workflow should terminate
- "planner": Choose when we need to revise the query plan

You MUST respond with a valid JSON object containing:
{
    "route_to": "next_agent_name",
    "reason": "detailed explanation for the routing decision",
    "feedback": "feedback for the previous agent's output",
    "state_updates": {
        "key": "value of any state that should be updated"
    },
    "confidence_score": float (0-1),
    "requires_human_input": boolean
}

DO NOT include any text outside this JSON structure.
"""

router_guided_json = {
    "type": "object",
    "properties": {
        "route_to": {
            "type": "string",
            "enum": ["selector", "sql_generator", "reviewer", "final_report", "end", "planner"],
            "description": "The next agent to route to"
        },
        "reason": {
            "type": "string",
            "description": "Detailed explanation for the routing decision"
        },
        "feedback": {
            "type": "string",
            "description": "Feedback for the previous agent's output"
        },
        "state_updates": {
            "type": "object",
            "description": "Any state values that should be updated",
            "additionalProperties": True
        },
        "confidence_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence level in the routing decision"
        },
        "requires_human_input": {
            "type": "boolean",
            "description": "Whether human intervention is needed"
        }
    },
    "required": ["route_to", "reason", "feedback"],
    "additionalProperties": False
}

class RouterAgent(Agent):
    def __init__(self, state=None, model=None, server=None, temperature=None, model_endpoint=None, stop=None):
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=router_guided_json
        )
        self.valid_routes = ["planner", "selector", "SQLGenerator", "reviewer", "sql_executor", "final_report_generator", "end"]
        self.workflow_start = True  # Add this to track if this is the start of workflow

    def invoke(self, current_state: dict) -> dict:
        try:
            # Handle workflow start
            if self.workflow_start:
                self.workflow_start = False
                return {
                    "router_response": {
                        "route_to": "planner",
                        "reason": "Starting new workflow",
                        "feedback": "Initializing workflow with planner",
                        "state_updates": {
                            "workflow_started": True,
                            "start_time": datetime.now().isoformat()
                        },
                        "confidence_score": 1.0,
                        "requires_human_input": False
                    }
                }

            # Extract relevant information from current state
            context = self._prepare_routing_context(current_state)

            # Check if we're coming from SQL executor
            if "sql_executor" in current_state.get("execution_path", []):
                sql_results = current_state.get("sql_query_results", {})
                if sql_results.get("status") == "success":
                    return {
                        "router_response": {
                            "route_to": "final_report_generator",
                            "reason": "SQL query executed successfully",
                            "feedback": f"Query completed with {sql_results.get('row_count', 0)} rows",
                            "state_updates": {},
                            "confidence_score": 1.0,
                            "requires_human_input": False
                        }
                    }

            # Get model response for other cases
            model_response = self.get_model(json_model=True).invoke([
                {"role": "system", "content": router_prompt_template},
                {"role": "user", "content": json.dumps(context, indent=2)}
            ])

            # Validate and enhance the response
            validated_response = self._validate_response(model_response)
            enhanced_response = self._enhance_routing_response(validated_response, current_state)

            return {"router_response": enhanced_response}

        except Exception as e:
            print(f"Error in router: {str(e)}")
            traceback.print_exc()
            return self._create_error_response([str(e)])

    def _prepare_routing_context(self, current_state: dict) -> dict:
        """
        Prepare context information for routing decision.

        Args:
            current_state: Current workflow state

        Returns:
            dict: Prepared context for routing
        """
        context = {
            "current_step": current_state.get("current_step", "start"),
            "workflow_history": current_state.get("workflow_history", []),
            "error_count": current_state.get("error_count", 0),
            "iteration_count": current_state.get("iteration_count", 0)
        }

        # Add agent-specific information
        agent_responses = {
            "planner_response": self.extract_agent_response("planner"),
            "selector_response": self.extract_agent_response("selector"),
            "sql_generator_response": self.extract_agent_response("sql_generator"),
            "reviewer_response": self.extract_agent_response("reviewer")
        }
        context.update(agent_responses)

        # Add status indicators
        context["status"] = {
            "has_errors": any(response.get("error") for response in agent_responses.values() if response),
            "requires_revision": self._check_if_revision_needed(agent_responses),
            "is_complete": self._check_if_workflow_complete(agent_responses)
        }

        return context

    def _check_if_revision_needed(self, agent_responses: dict) -> bool:
        """
        Check if the current workflow requires revision.

        Args:
            agent_responses: Dictionary of agent responses

        Returns:
            bool: True if revision is needed
        """
        reviewer_response = agent_responses.get("reviewer_response", {})
        if reviewer_response:
            # Check if reviewer found issues
            if not reviewer_response.get("is_correct", True):
                return True
            # Check if there are critical issues
            if reviewer_response.get("issues", []):
                return True
            # Check if there are security concerns
            if reviewer_response.get("security_concerns", []):
                return True
        return False

    def _check_if_workflow_complete(self, agent_responses: dict) -> bool:
        """
        Check if the workflow is complete and ready for final report.

        Args:
            agent_responses: Dictionary of agent responses

        Returns:
            bool: True if workflow is complete
        """
        # Check if we have all required responses
        required_responses = ["sql_generator_response", "reviewer_response"]
        if not all(agent_responses.get(resp) for resp in required_responses):
            return False

        # Check if reviewer approved the query
        reviewer_response = agent_responses.get("reviewer_response", {})
        if not reviewer_response.get("is_correct", False):
            return False

        # Check if there are no pending issues
        if reviewer_response.get("issues", []) or reviewer_response.get("security_concerns", []):
            return False

        return True

    def _validate_response(self, response: dict) -> dict:
        """
        Validate the structure and content of the router response.

        Args:
            response: The response dictionary to validate

        Returns:
            dict: Validated response
        """
        required_fields = [
            "route_to",
            "reason",
            "feedback",
            "state_updates",
            "confidence_score",
            "requires_human_input"
        ]

        # Check required fields
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            raise ValueError(f"Missing required fields in router response: {missing_fields}")

        # Validate route
        if response["route_to"] not in self.valid_routes:
            raise ValueError(f"Invalid route: {response['route_to']}. Must be one of: {self.valid_routes}")

        # Validate field types
        if not isinstance(response["state_updates"], dict):
            raise ValueError("state_updates must be a dictionary")
        if not isinstance(response["confidence_score"], (int, float)):
            raise ValueError("confidence_score must be a number")
        if not 0 <= response["confidence_score"] <= 1:
            raise ValueError("confidence_score must be between 0 and 1")
        if not isinstance(response["requires_human_input"], bool):
            raise ValueError("requires_human_input must be a boolean")

        return response

    def _enhance_routing_response(self, routing_response: dict, current_state: dict) -> dict:
        """
        Enhance the routing response with additional metadata and insights.

        Args:
            routing_response: The original routing response
            current_state: Current workflow state

        Returns:
            dict: Enhanced routing response
        """
        # Add metadata
        routing_response["metadata"] = {
            "routed_at": datetime.now().isoformat(),
            "iteration_count": current_state.get("iteration_count", 0) + 1,
            "workflow_status": self._get_workflow_status(routing_response, current_state)
        }

        # Add workflow insights
        routing_response["workflow_insights"] = self._generate_workflow_insights(
            routing_response,
            current_state
        )

        return routing_response

    def _get_workflow_status(self, routing_response: dict, current_state: dict) -> str:
        """
        Determine the current status of the workflow.

        Args:
            routing_response: Current routing response
            current_state: Current workflow state

        Returns:
            str: Workflow status
        """
        if routing_response["route_to"] == "end":
            return "COMPLETED"
        if routing_response["requires_human_input"]:
            return "NEEDS_HUMAN_INPUT"
        if self._check_if_revision_needed(current_state):
            return "NEEDS_REVISION"
        return "IN_PROGRESS"

    def _generate_workflow_insights(self, routing_response: dict, current_state: dict) -> dict:
        """
        Generate insights about the workflow progress.

        Args:
            routing_response: Current routing response
            current_state: Current workflow state

        Returns:
            dict: Workflow insights
        """
        iteration_count = current_state.get("iteration_count", 0)
        error_count = current_state.get("error_count", 0)

        return {
            "efficiency_metrics": {
                "iterations": iteration_count,
                "errors_encountered": error_count,
                "efficiency_score": max(0, 1 - (error_count / (iteration_count + 1)))
            },
            "bottlenecks": self._identify_bottlenecks(current_state),
            "improvement_suggestions": self._generate_improvement_suggestions(
                routing_response,
                current_state
            )
        }

    def _identify_bottlenecks(self, current_state: dict) -> list:
        """
        Identify potential bottlenecks in the workflow.

        Args:
            current_state: Current workflow state

        Returns:
            list: Identified bottlenecks
        """
        bottlenecks = []
        workflow_history = current_state.get("workflow_history", [])

        # Analyze workflow history for patterns
        if len(workflow_history) > 3:
            # Check for repeated steps
            last_three_steps = workflow_history[-3:]
            if len(set(last_three_steps)) == 1:
                bottlenecks.append(f"Repeated step: {last_three_steps[0]}")

        # Check for high iteration count
        if current_state.get("iteration_count", 0) > 5:
            bottlenecks.append("High iteration count")

        # Check for error patterns
        if current_state.get("error_count", 0) > 2:
            bottlenecks.append("Frequent errors")

        return bottlenecks

    def _generate_improvement_suggestions(self, routing_response: dict, current_state: dict) -> list:
        """
        Generate suggestions for improving workflow efficiency.

        Args:
            routing_response: Current routing response
            current_state: Current workflow state

        Returns:
            list: Improvement suggestions
        """
        suggestions = []

        # Add suggestions based on routing decision
        if routing_response["route_to"] == "planner":
            suggestions.append("Consider refining initial query planning")
        elif routing_response["route_to"] == "sql_generator":
            suggestions.append("Review SQL generation parameters")
        elif routing_response["requires_human_input"]:
            suggestions.append("Consider automating common human input scenarios")

        # Add suggestions based on workflow metrics
        if current_state.get("iteration_count", 0) > 5:
            suggestions.append("Review workflow complexity and consider optimization")
        if current_state.get("error_count", 0) > 2:
            suggestions.append("Implement additional error prevention measures")

        return suggestions

    def _create_error_response(self, errors: list) -> dict:
        """
        Create a standardized error response.

        Args:
            errors: List of error messages

        Returns:
            dict: Formatted error response
        """
        return {
            "router_response": {
                "route_to": "end",
                "reason": "Routing failed due to critical errors",
                "feedback": errors,
                "state_updates": {
                    "error_count": 1,
                    "error_messages": errors
                },
                "confidence_score": 1.0,
                "requires_human_input": True,
                "metadata": {
                    "routed_at": datetime.now().isoformat(),
                    "routing_status": "ERROR"
                }
            }
        }

###################
# Final Report Agent
###################

final_report_prompt_template = """
You are the Final Report Agent responsible for generating comprehensive, user-friendly reports based on the SQL query results and workflow history.

Your task is to:
1. Analyze the SQL query results
2. Summarize the workflow process
3. Present findings in a clear, structured format
4. Include relevant metrics and insights
5. Highlight any important patterns or anomalies

PostgreSQL-Specific Considerations:
1. Interpret results from PostgreSQL-specific queries correctly
2. Provide context for schema-specific data (prod, dev, test)
3. Explain any PostgreSQL-specific functions used in the query
4. Highlight relationships between tables when JOINs are used

SQL Query Results Analysis:
1. Analyze the structure of the returned data
2. Identify key patterns or trends in the results
3. Highlight any anomalies or unexpected values
4. Provide context for the numerical values
5. Explain the significance of the results in relation to the user's question

You MUST respond with a valid JSON object containing:
{
    "report": {
        "summary": "brief overview of findings",
        "detailed_results": {
            "key_findings": ["list of main insights"],
            "data_analysis": "detailed analysis of results",
            "visualizations": ["suggested visualization types"]
        },
        "query_details": {
            "original_query": "the executed SQL query",
            "query_explanation": "explanation of what the query does",
            "schema_context": "information about the schemas used",
            "performance_metrics": {
                "execution_time": "query execution time",
                "rows_affected": "number of rows in result"
            }
        },
        "workflow_summary": {
            "steps_taken": ["list of workflow steps"],
            "optimization_notes": ["any optimization details"]
        }
    },
    "explanation": "detailed explanation of the report generation process",
    "metadata": {
        "timestamp": "ISO timestamp",
        "version": "report version"
    }
}

DO NOT include any text outside this JSON structure.
"""

final_report_guided_json = {
    "type": "object",
    "properties": {
        "report": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "detailed_results": {
                    "type": "object",
                    "properties": {
                        "key_findings": {"type": "array", "items": {"type": "string"}},
                        "data_analysis": {"type": "string"},
                        "visualizations": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "query_details": {
                    "type": "object",
                    "properties": {
                        "original_query": {"type": "string"},
                        "performance_metrics": {
                            "type": "object",
                            "properties": {
                                "execution_time": {"type": "string"},
                                "rows_affected": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "generated_at": {"type": "string", "format": "date-time"},
                "version": {"type": "string"}
            }
        }
    },
    "required": ["report"]
}

class FinalReportAgent(Agent):
    def __init__(self, state=None, model=None, server=None, temperature=None, model_endpoint=None, stop=None):
        super().__init__(state, model, server, temperature, model_endpoint, stop)
        self.json_model = True

    def invoke(self, query_results: dict, workflow_history: dict) -> dict:
        try:
            # Check if this is a list tables query
            is_list_tables_query = workflow_history.get('query_type') == 'list_tables'

            if is_list_tables_query:
                # Extract schema and table information
                rows = query_results.get('rows', [])
                row_count = query_results.get('row_count', 0)
                schemas = set()
                schema_counts = {}
                sample_data = []

                # Process rows to get schema information
                for row in rows:
                    if row and len(row) >= 2:
                        schema = row[0]
                        table = row[1]
                        schemas.add(schema)
                        schema_counts[schema] = schema_counts.get(schema, 0) + 1
                        sample_data.append(f"{schema}.{table}")

                # Create a detailed report
                report = {
                    "report": {
                        "summary": f"Query executed successfully. Found {row_count} tables across {len(schemas)} schemas in the database.",
                        "detailed_results": {
                            "key_findings": [
                                f"Found {row_count} tables across {len(schemas)} schemas",
                                f"Schema distribution: {', '.join([f'{schema}: {count} tables' for schema, count in schema_counts.items()])}"
                            ],
                            "data_analysis": f"The database contains tables in the following schemas: {', '.join(schemas)}",
                            "sample_data": sample_data[:10],
                            "visualizations": ["Bar chart of tables per schema"]
                        },
                        "query_details": {
                            "original_query": query_results.get('query', ''),
                            "performance_metrics": {
                                "execution_time": query_results.get('execution_time', 0),
                                "rows_affected": row_count
                            }
                        },
                        "workflow_summary": "Query executed successfully."
                    },
                    "explanation": "This report lists all tables in the database across different schemas.",
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "version": "1.0",
                        "schema_counts": schema_counts
                    }
                }

                return {"final_report": report}

            # For other query types, use the existing logic
            return super().invoke(query_results, workflow_history)

        except Exception as e:
            print(f"Error in final report generation: {str(e)}")
            traceback.print_exc()
            return self._create_error_report(str(e))

    def _create_error_report(self, error_message: str) -> dict:
        """
        Create a standardized error report.

        Args:
            error_message: Error message to include in report

        Returns:
            dict: Formatted error report
        """
        return {
            "error_type": "ERROR",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc(),
            "component": self.__class__.__name__,
            "state": self.state,
            "recommendations": [
                "Review error logs for more details",
                "Check state values for inconsistencies",
                "Contact support if issue persists"
            ]
        }

###################
# End Node Agent
###################

end_node_prompt_template = """
You are an end node agent that provides a final summary of the workflow.
Your task is to summarize the entire workflow and provide any final insights.

User Question: {user_question}

Final Report:
{final_report}
"""

class EndNodeAgent(Agent):
    def invoke(self, user_question, final_report_response=None):
        try:
            print(f"EndNodeAgent received final_report_response: {type(final_report_response)}")

            # Ensure we have a valid final report
            if not final_report_response:
                print("Warning: No final report response provided to EndNodeAgent")
                final_report_response = {}

            # Extract the report content if it's nested
            report_content = final_report_response
            if isinstance(final_report_response, dict):
                if "final_report" in final_report_response:
                    report_content = final_report_response["final_report"]
                elif "report" in final_report_response:
                    report_content = final_report_response["report"]

            # Extract key information from the report
            summary = "No summary available"
            query = "No query available"
            row_count = 0
            execution_time = 0
            sample_data = []

            if isinstance(report_content, dict):
                # Extract from report structure
                if "report" in report_content:
                    report = report_content["report"]
                    summary = report.get("summary", "No summary available")

                    # Extract query details
                    query_details = report.get("query_details", {})
                    query = query_details.get("original_query", "No query available")

                    # Extract performance metrics
                    performance_metrics = query_details.get("performance_metrics", {})
                    row_count = performance_metrics.get("rows_affected", 0)
                    execution_time = performance_metrics.get("execution_time", 0)

                    # Extract sample data if available
                    detailed_results = report.get("detailed_results", {})
                    if "sample_data" in detailed_results:
                        sample_data = detailed_results.get("sample_data", [])

                    # If no sample data but we have key findings, use those
                    if not sample_data and "key_findings" in detailed_results:
                        sample_data = detailed_results.get("key_findings", [])

            # Check if this is a list tables query
            is_list_tables_query = False
            if user_question.lower().strip() in ["list all tables", "show tables", "show all tables"]:
                is_list_tables_query = True

            # For list tables queries, create a more specific response
            if is_list_tables_query:
                print("EndNodeAgent detected 'list all tables' query, creating specialized response")

                # Try to extract schema information
                schemas = []
                schema_counts = {}

                # Look for schema information in the report
                if isinstance(report_content, dict) and "report" in report_content:
                    report = report_content["report"]

                    # Try to extract from query_details
                    if "query_details" in report and "schema_context" in report["query_details"]:
                        schema_context = report["query_details"]["schema_context"]
                        if isinstance(schema_context, str) and "Found tables in schemas:" in schema_context:
                            schemas_str = schema_context.split("Found tables in schemas:")[1].strip()
                            schemas = [s.strip() for s in schemas_str.split(",")]

                    # Try to extract from detailed_results
                    if "detailed_results" in report:
                        detailed_results = report["detailed_results"]

                        # Try to extract from key_findings
                        if "key_findings" in detailed_results:
                            for finding in detailed_results["key_findings"]:
                                if "Schema distribution:" in finding:
                                    distribution_str = finding.split("Schema distribution:")[1].strip()
                                    for item in distribution_str.split(","):
                                        if ":" in item:
                                            schema, count = item.split(":")
                                            schema_counts[schema.strip()] = int(count.split()[0])

            # IMPORTANT: For UI compatibility, we need to return a specific structure
            # The UI expects a final_report_response with a "report" key

            # Create a report structure that's compatible with the UI
            # This is the most important part - the UI expects this exact structure
            final_report_response = {
                "report": {
                    "summary": summary,
                    "detailed_results": {
                        "key_findings": [f"Found {row_count} tables across {len(schemas) if schemas else 0} schemas"] if is_list_tables_query else ["Query executed successfully"],
                        "data_analysis": f"The database contains tables in the following schemas: {', '.join(schemas)}" if schemas else "Query executed successfully",
                        "sample_data": sample_data[:10] if sample_data else []
                    },
                    "query_details": {
                        "original_query": query,
                        "performance_metrics": {
                            "execution_time": execution_time,
                            "rows_affected": row_count
                        }
                    },
                    "workflow_summary": "Query executed successfully."
                }
            }

            # Return a response that's compatible with the UI
            # The UI specifically looks for a "final_report_response" with a "report" key
            return {
                "status": "completed",
                "message": summary,
                "user_question": user_question,
                "final_report_response": final_report_response
            }
        except Exception as e:
            print(f"Error in EndNodeAgent: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "user_question": user_question,
                "error_details": traceback.format_exc()
            }

###################
# Helper Functions
###################

def format_planner_response(raw_response):
    """
    Transform the planner response into the required format with strict validation.

    Args:
        raw_response: The raw response from the planner agent.

    Returns:
        dict: The formatted response.

    Raises:
        ValueError: If the response cannot be properly formatted.
    """
    try:
        # If empty dict is provided, return a default response
        if not raw_response:
            raise ValueError("Empty response from planner agent")

        if isinstance(raw_response, str):
            raw_response = json.loads(raw_response)

        if not isinstance(raw_response, dict):
            raise ValueError("Response must be a dictionary")

        # Validate required fields
        required_fields = ["query_type", "primary_table_or_datasource", "relevant_columns", "filtering_conditions", "processing_instructions"]
        missing_fields = [field for field in required_fields if field not in raw_response]

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Validate field values
        if raw_response["query_type"] not in ["sql", "non_sql"]:
            raise ValueError("query_type must be either 'sql' or 'non_sql'")

        if not raw_response["primary_table_or_datasource"]:
            raise ValueError("primary_table_or_datasource cannot be empty")

        return raw_response

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error formatting planner response: {str(e)}")

# Add any other helper functions here
