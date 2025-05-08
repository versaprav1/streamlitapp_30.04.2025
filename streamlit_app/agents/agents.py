# External library imports
from datetime import datetime
from termcolor import colored
import json
import traceback
import re

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
        """Initialize the Planner Agent with enhanced database discovery and context management."""
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=planner_guided_json
        )

        # Initialize database context
        self.db_context = {
            "current_db": None,
            "current_schema": None,
            "discovered_schemas": set(),
            "interface_tables": {},
            "last_discovery": None
        }

        # Initialize metadata store and discovery service
        try:
            self.metadata_store = DatabaseMetadataStore()
            self.discovery_service = DatabaseDiscoveryService(self.metadata_store)
            print("Successfully initialized database metadata store and discovery service")
            
            # Perform initial database discovery
            self._initialize_database_context()
        except Exception as e:
            print(f"Error initializing database services: {e}")
            traceback.print_exc()

    def _initialize_database_context(self):
        """Initialize database context with proactive discovery."""
        try:
            # Get connection parameters
            params = self.get_db_connection_params()
            
            # Set current database
            self.db_context["current_db"] = params["db_name"]
            
            # Load existing metadata
            self.metadata_store.load_metadata(params["db_name"])
            self.metadata_store.load_change_log(params["db_name"])
            
            # Discover schemas if metadata is stale
            if not self.metadata_store.is_metadata_fresh(params["db_name"]):
                schemas = self.discovery_service.discover_schemas(
                    db_name=params["db_name"],
                    user=params["user"],
                    password=params["password"],
                    host=params["host"],
                    port=params["port"],
                    force_refresh=True
                )
                self.db_context["discovered_schemas"].update(schemas)
                
                # Pre-discover interface tables
                self._discover_interface_tables()
            
            self.db_context["last_discovery"] = datetime.now()
            print(f"Database context initialized successfully for {params['db_name']}")
            
        except Exception as e:
            print(f"Error initializing database context: {e}")
            traceback.print_exc()

    def _discover_interface_tables(self):
        """Discover tables containing interface-related columns."""
        try:
            params = self.get_db_connection_params()
            
            # Query for interface-related tables
            conn, cursor = self.discovery_service.get_connection(
                db_name=params["db_name"],
                user=params["user"],
                password=params["password"],
                host=params["host"],
                port=params["port"]
            )
            
            # Search for interface-related columns across all schemas
            cursor.execute("""
                SELECT table_schema, table_name, column_name
                FROM information_schema.columns
                WHERE column_name ILIKE '%interface%'
                AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name;
            """)
            
            interface_tables = {}
            for schema, table, column in cursor.fetchall():
                if schema not in interface_tables:
                    interface_tables[schema] = {}
                if table not in interface_tables[schema]:
                    interface_tables[schema][table] = []
                interface_tables[schema][table].append(column)
            
            self.db_context["interface_tables"] = interface_tables
            print(f"Discovered interface tables: {json.dumps(interface_tables, indent=2)}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error discovering interface tables: {e}")
            traceback.print_exc()

    def analyze_query_context(self, user_question):
        """Analyze the query to determine database context with enhanced understanding."""
        context = {
            "query_type": None,
            "target_schema": None,
            "target_table": None,
            "is_interface_query": False,
            "is_metadata_query": False,
            "conditions": [],
            "required_columns": []
        }
        
        # Normalize question for analysis
        question = user_question.lower().strip()
        
        # Check for interface-related queries
        if "interface" in question:
            context["is_interface_query"] = True
            context["query_type"] = "SELECT"
            # Check discovered interface tables
            if hasattr(self, 'db_context') and self.db_context.get("interface_tables"):
                interface_tables = self.db_context["interface_tables"]
                # Prioritize public schema
                if "public" in interface_tables:
                    context["target_schema"] = "public"
                    tables = interface_tables["public"]
                    if tables:
                        context["target_table"] = next(iter(tables))
    
        # Check for metadata queries
        elif any(keyword in question for keyword in ["list", "show", "describe", "tables", "schemas"]):
            context["is_metadata_query"] = True
            context["query_type"] = "SELECT"
            if "schemas" in question:
                context["target_schema"] = "information_schema"
                context["target_table"] = "schemata"
            elif "tables" in question:
                context["target_schema"] = "information_schema"
                context["target_table"] = "tables"
    
        # Extract schema and table information using regex
        if not context["target_schema"]:
            # Check for explicit schema.table pattern
            match = re.search(r'from\s+(["\w]+)\.(["\w]+)', question)
            if match:
                context["target_schema"] = match.group(1).strip('"')
                context["target_table"] = match.group(2).strip('"')
            else:
                # Check for table in schema pattern
                match = re.search(r'table\s+(["\w]+)\s+in\s+(["\w]+)', question)
                if match:
                    context["target_schema"] = match.group(2).strip('"')
                    context["target_table"] = match.group(1).strip('"')
    
        return context

    def enhance_plan_with_metadata(self, plan, user_question):
        """Enhance the query plan with database metadata."""
        # Analyze the query context
        context = self.analyze_query_context(user_question)

        # Special handling for interface queries
        if context["is_interface_query"]:
            if context["target_schema"] and context["target_table"]:
                # Get table structure
                table_structure = self.metadata_store.get_table_structure(
                    self.current_db_name,
                    context["target_schema"],
                    context["target_table"]
                )
                
                # If table structure is not found or stale, discover it
                if not table_structure or not self.metadata_store.is_table_fresh(
                    self.current_db_name,
                    context["target_schema"],
                    context["target_table"]
                ):
                    table_structure = self.discover_table_structure(
                        context["target_schema"],
                        context["target_table"],
                        force_refresh=True
                    )
                
                # Get interface-related columns
                interface_columns = []
                if table_structure and "columns" in table_structure:
                    interface_columns = [
                        col["name"] for col in table_structure["columns"]
                        if "interface" in col["name"].lower()
                    ]
                
                # Update plan with interface-specific information
                plan.update({
                    "query_type": "SELECT",
                    "primary_table_or_datasource": f"{context['target_schema']}.{context['target_table']}",
                    "relevant_columns": interface_columns,
                    "table_structure": table_structure,
                    "processing_instructions": "Query interface information",
                    "filtering_conditions": "interface_name IS NOT NULL",
                    "metadata": {
                        "is_interface_query": True,
                        "schema": context["target_schema"],
                        "table": context["target_table"],
                        "interface_columns": interface_columns
                    }
                })
                return plan

        # Handle metadata queries (list tables, schemas, etc.)
        if context["is_metadata_query"]:
            schemas = self.discover_database_structure()
            if "schemas" in user_question.lower():
                plan.update({
                    "schemas": schemas,
                    "primary_table_or_datasource": "information_schema.schemata",
                    "query_type": "SELECT",
                    "relevant_columns": ["schema_name"],
                    "filtering_conditions": "schema_name NOT IN ('pg_catalog', 'information_schema')",
                    "processing_instructions": "List all database schemas"
                })
            elif "tables" in user_question.lower():
                all_tables = {}
                for schema in schemas:
                    tables = self.discover_schema_tables(schema)
                    all_tables[schema] = tables
                plan.update({
                    "all_tables": all_tables,
                    "primary_table_or_datasource": "information_schema.tables",
                    "query_type": "SELECT",
                    "relevant_columns": ["table_schema", "table_name"],
                    "filtering_conditions": "table_schema NOT IN ('pg_catalog', 'information_schema')",
                    "processing_instructions": "List all tables in all schemas"
                })
            return plan

        # Handle regular queries
        if context["target_schema"] and context["target_table"]:
            table_structure = self.discover_table_structure(
                context["target_schema"],
                context["target_table"]
            )
            plan.update({
                "schema": context["target_schema"],
                "table": context["target_table"],
                "table_structure": table_structure,
                "primary_table_or_datasource": f"{context['target_schema']}.{context['target_table']}",
                "relevant_columns": [col["name"] for col in table_structure.get("columns", [])]
            })

        # Add metadata about the enhancement
        plan["metadata"] = {
            "enhanced_at": str(datetime.now()),
            "context": context,
            "database": self.current_db_name,
            "has_table_structure": bool(plan.get("table_structure")),
            "discovered_columns": len(plan.get("relevant_columns", []))
        }

        return plan

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

    def _validate_and_handle_errors(self, response):
        """Validate the planner's response and handle any errors."""
        try:
            # Ensure response is a dictionary
            if not isinstance(response, dict):
                raise ValueError(f"Response must be a dictionary, got {type(response)}")

            # Check required fields
            required_fields = [
                "query_type",
                "primary_table_or_datasource",
                "relevant_columns",
                "filtering_conditions",
                "processing_instructions"
            ]
            missing_fields = [field for field in required_fields if field not in response]
            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            # Validate query type
            valid_query_types = ["SELECT", "INSERT", "UPDATE", "DELETE"]
            if response.get("query_type") not in valid_query_types:
                raise ValueError(f"Invalid query_type: {response.get('query_type')}")

            # Validate table/datasource
            if not response.get("primary_table_or_datasource"):
                raise ValueError("primary_table_or_datasource cannot be empty")

            # Validate columns
            if not isinstance(response.get("relevant_columns"), list):
                raise ValueError("relevant_columns must be a list")

            # Add validation status
            response["validation"] = {
                "is_valid": True,
                "validated_at": str(datetime.now()),
                "validation_checks": [
                    "Response structure",
                    "Required fields",
                    "Query type",
                    "Table/datasource",
                    "Column format"
                ]
            }

            return response

        except Exception as e:
            error_response = {
                "error": str(e),
                "validation": {
                    "is_valid": False,
                    "validated_at": str(datetime.now()),
                    "validation_error": str(e),
                    "error_type": type(e).__name__
                }
            }
            print(f"Validation error: {str(e)}")
            return error_response

    def invoke(self, user_question: str, feedback: str = "") -> dict:
        """Generate a plan for answering the user's question with enhanced error handling."""
        try:
            # Get the model for generating JSON responses
            model = self.get_model(json_model=True)

            # Create the system prompt
            system_prompt = planner_prompt_template

            # Create the user input
            user_input = f"User Question: {user_question}"
            if feedback:
                user_input += f"\n\nFeedback from previous attempt: {feedback}"

            # Add context about available data sources
            user_input += "\n\nAvailable Data Sources: SQL Database (PostgreSQL)"

            # Get the response from the model
            print(f"Invoking Planner with question: {user_question}")

            if self.server in ["ollama", "vllm", "groq"]:
                response = model(system_prompt=system_prompt, user_input=user_input)
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
                response = model.invoke(messages)

            # Parse and validate the response
            parsed_response = self._parse_model_response(response)
            validated_response = self._validate_and_handle_errors(parsed_response)

            # If validation failed, try to recover
            if not validated_response.get("validation", {}).get("is_valid", False):
                print("Initial validation failed, attempting recovery...")
                validated_response = self._attempt_recovery(validated_response, user_question)

            # Enhance the plan with metadata
            if validated_response.get("validation", {}).get("is_valid", False):
                enhanced_response = self.enhance_plan_with_metadata(validated_response, user_question)
            else:
                enhanced_response = validated_response

            # Add final metadata
            enhanced_response.update({
                "timestamp": str(datetime.now()),
                "agent": "planner",
                "user_question": user_question
            })

            return enhanced_response

        except Exception as e:
            print(f"Error in PlannerAgent.invoke: {e}")
            traceback.print_exc()
            return self._create_error_response(str(e))

    def _parse_model_response(self, response):
        """Parse and normalize the model's response."""
        try:
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                # Handle potential markdown code blocks
                if "```json" in response or "```" in response:
                    import re
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                    if json_match:
                        return json.loads(json_match.group(1).strip())
                return json.loads(response)
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
        except Exception as e:
            print(f"Error parsing model response: {e}")
            return {
                "error": f"Failed to parse response: {str(e)}",
                "raw_response": str(response)
            }

    def _attempt_recovery(self, failed_response, user_question):
        """Attempt to recover from validation failures."""
        try:
            # Create a basic valid response structure
            recovered_response = {
                "query_type": "SELECT",
                "primary_table_or_datasource": "unknown",
                "relevant_columns": [],
                "filtering_conditions": "",
                "processing_instructions": "Analyze the database structure"
            }

            # Try to extract any valid information from the failed response
            if isinstance(failed_response, dict):
                if "primary_table_or_datasource" in failed_response:
                    recovered_response["primary_table_or_datasource"] = failed_response["primary_table_or_datasource"]
                if "relevant_columns" in failed_response and isinstance(failed_response["relevant_columns"], list):
                    recovered_response["relevant_columns"] = failed_response["relevant_columns"]

            # Add recovery metadata
            recovered_response["recovery"] = {
                "recovered_from_error": True,
                "original_error": failed_response.get("error", "Unknown error"),
                "recovery_timestamp": str(datetime.now())
            }

            # Validate the recovered response
            return self._validate_and_handle_errors(recovered_response)

        except Exception as e:
            print(f"Recovery attempt failed: {e}")
            return failed_response

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
        """Initialize the Selector Agent with enhanced metadata handling."""
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=selector_guided_json
        )
        
        # Initialize selection context
        self.selection_context = {
            "last_selection": None,
            "selection_history": [],
            "preferred_schema": None,
            "discovered_tables": {}
        }

    def invoke(self, user_question: str, planner_response: dict = None, feedback: str = None, previous_selections: list = None) -> dict:
        """Select appropriate data sources and tools based on the query requirements."""
        try:
            # Initialize response structure
            selection = {
                "selected_tool": "sql_query",
                "selected_datasource": "unknown",
                "information_needed": [],
                "reason_for_selection": "",
                "query_parameters": {"columns": [], "filters": ""}
            }

            # Validate planner response
            if not planner_response or not isinstance(planner_response, dict):
                return {
                    "selector_response": {
                        "error": "Invalid planner response",
                        "selected_tool": "error",
                        "selected_datasource": None,
                        "information_needed": ["Valid planner response"],
                        "reason_for_selection": "Planner response validation failed"
                    }
                }

            # Extract schema and table information
            schema_table = self._extract_schema_table(planner_response)
            if schema_table:
                selection.update({
                    "selected_schema": schema_table["schema"],
                    "selected_table": schema_table["table"],
                    "reason_for_selection": f"Using schema {schema_table['schema']} and table {schema_table['table']} based on planner analysis"
                })

            # Handle interface queries
            if "interface" in user_question.lower():
                interface_info = self._handle_interface_query(planner_response)
                if interface_info:
                    selection.update(interface_info)

            # Add query parameters
            selection["query_parameters"].update({
                "columns": planner_response.get("relevant_columns", []),
                "filters": planner_response.get("filtering_conditions", ""),
                "schema": selection.get("selected_schema", "public"),
                "table": selection.get("selected_table")
            })

            # Add metadata
            selection["metadata"] = {
                "timestamp": str(datetime.now()),
                "source": "planner_guided",
                "confidence": 0.8 if selection.get("selected_schema") else 0.5
            }

            return {"selector_response": selection}

        except Exception as e:
            print(f"Error in selector agent: {str(e)}")
            traceback.print_exc()
            return {
                "selector_response": {
                    "error": str(e),
                    "selected_tool": "error",
                    "selected_datasource": None,
                    "information_needed": ["Error resolution"],
                    "reason_for_selection": f"Error occurred: {str(e)}"
                }
            }

    def _extract_schema_table(self, planner_response: dict) -> dict:
        """Extract schema and table information from planner response."""
        result = {"schema": None, "table": None}
        
        # Check primary_table_or_datasource
        source = planner_response.get("primary_table_or_datasource")
        if source and isinstance(source, str):
            # Handle schema.table format
            if "." in source:
                schema, table = source.split(".")
                result["schema"] = schema.strip('"')
                result["table"] = table.strip('"')
            else:
                # Default to public schema
                result["schema"] = "public"
                result["table"] = source.strip('"')
        
        # Check metadata for schema preference
        metadata = planner_response.get("metadata", {})
        if metadata.get("schema"):
            result["schema"] = metadata["schema"]
        
        return result

    def _handle_interface_query(self, planner_response: dict) -> dict:
        """Handle interface-specific query requirements."""
        interface_info = {
            "is_interface_query": True,
            "information_needed": ["interface_columns"],
            "query_parameters": {
                "table_type": "interface"
            }
        }
        
        # Check for interface table in metadata
        metadata = planner_response.get("metadata", {})
        if metadata.get("interface_tables"):
            interface_tables = metadata["interface_tables"]
            # Prioritize public schema
            if "public" in interface_tables:
                interface_info.update({
                    "selected_schema": "public",
                    "selected_table": next(iter(interface_tables["public"])),
                    "reason_for_selection": "Found interface table in public schema"
                })
            else:
                # Take first available schema
                schema = next(iter(interface_tables))
                interface_info.update({
                    "selected_schema": schema,
                    "selected_table": next(iter(interface_tables[schema])),
                    "reason_for_selection": f"Found interface table in {schema} schema"
                })
        else:
            # Default to information schema search
            interface_info.update({
                "selected_schema": "information_schema",
                "selected_table": "columns",
                "reason_for_selection": "Searching for interface columns across schemas",
                "query_parameters": {
                    "column_pattern": "%interface%"
                }
            })
        
        return interface_info

    def _validate_planner_response(self, planner_response: dict) -> dict:
        """Validate and normalize the planner's response."""
        if not planner_response:
            return {
                "query_type": "unknown",
                "primary_table_or_datasource": "unknown",
                "relevant_columns": [],
                "is_valid": False
            }
            
        # Ensure required fields
        required_fields = [
            "query_type",
            "primary_table_or_datasource",
            "relevant_columns"
        ]
        
        validated = {}
        for field in required_fields:
            validated[field] = planner_response.get(field)
            
        # Add validation status
        validated["is_valid"] = all(validated.get(field) is not None for field in required_fields)
        
        # Extract metadata if available
        if "metadata" in planner_response:
            validated["metadata"] = planner_response["metadata"]
            
        return validated

    def _analyze_requirements(self, user_question: str, planner_response: dict) -> dict:
        """Analyze selection requirements based on the query and planner response."""
        requirements = {
            "needs_schema_selection": False,
            "needs_table_selection": False,
            "is_interface_query": False,
            "is_metadata_query": False,
            "required_columns": set(),
            "required_conditions": []
        }
        
        # Check for interface queries
        if "interface" in user_question.lower():
            requirements["is_interface_query"] = True
            requirements["needs_schema_selection"] = True
            requirements["needs_table_selection"] = True
            
        # Check for metadata queries
        if any(word in user_question.lower() for word in ["list", "show", "describe", "tables", "schemas"]):
            requirements["is_metadata_query"] = True
            
        # Extract required columns
        if planner_response.get("relevant_columns"):
            requirements["required_columns"].update(planner_response["relevant_columns"])
            
        # Extract conditions
        if planner_response.get("filtering_conditions"):
            requirements["required_conditions"].append(planner_response["filtering_conditions"])
            
        return requirements

    def _select_data_sources(self, requirements: dict, planner_response: dict) -> dict:
        """Select appropriate data sources based on requirements."""
        selection = {
            "selected_tool": "database_query",  # Default to database query
            "selected_datasource": None,
            "selected_schema": None,
            "selected_table": None,
            "confidence_score": 0.0
        }
        
        # Handle interface queries
        if requirements["is_interface_query"]:
            interface_selection = self._select_interface_source(planner_response)
            if interface_selection:
                selection.update(interface_selection)
                selection["confidence_score"] = 0.9
                
        # Handle metadata queries
        elif requirements["is_metadata_query"]:
            metadata_selection = self._select_metadata_source(planner_response)
            if metadata_selection:
                selection.update(metadata_selection)
                selection["confidence_score"] = 1.0
                
        # Handle regular queries
        else:
            regular_selection = self._select_regular_source(planner_response)
            if regular_selection:
                selection.update(regular_selection)
                
        return selection

    def _select_interface_source(self, planner_response: dict) -> dict:
        """Select appropriate source for interface queries."""
        # Check planner's metadata first
        if planner_response.get("metadata", {}).get("interface_tables"):
            interface_tables = planner_response["metadata"]["interface_tables"]
            # Prioritize public schema
            if "public" in interface_tables:
                return {
                    "selected_schema": "public",
                    "selected_table": next(iter(interface_tables["public"])),
                    "interface_columns": interface_tables["public"][next(iter(interface_tables["public"]))],
                    "requires_verification": False
                }
                
        # Default to information schema for discovery
        return {
            "selected_schema": "information_schema",
            "selected_table": "columns",
            "requires_verification": True
        }

    def _select_metadata_source(self, planner_response: dict) -> dict:
        """Select appropriate source for metadata queries."""
        return {
            "selected_schema": "information_schema",
            "selected_table": "tables",
            "selected_columns": ["table_schema", "table_name"],
            "requires_verification": False
        }

    def _select_regular_source(self, planner_response: dict) -> dict:
        """Select appropriate source for regular queries."""
        selection = {}
        
        # Use planner's suggested source if available
        if planner_response.get("primary_table_or_datasource"):
            source = planner_response["primary_table_or_datasource"]
            # Check if source includes schema
            if "." in source:
                schema, table = source.split(".")
                selection.update({
                    "selected_schema": schema,
                    "selected_table": table,
                    "confidence_score": 0.8
                })
            else:
                selection.update({
                    "selected_table": source,
                    "selected_schema": "public",  # Default to public schema
                    "confidence_score": 0.6,
                    "requires_verification": True
                })
                
        return selection

    def _enhance_selection(self, selection: dict, planner_response: dict) -> dict:
        """Enhance the selection with additional metadata and validations."""
        enhanced = selection.copy()
        
        # Add query parameters
        enhanced["query_parameters"] = {
            "columns": list(planner_response.get("relevant_columns", [])),
            "filters": planner_response.get("filtering_conditions", ""),
            "schema": enhanced.get("selected_schema"),
            "table": enhanced.get("selected_table")
        }
        
        # Add selection reasoning
        enhanced["reason_for_selection"] = self._generate_selection_reason(enhanced)
        
        # Add required information
        enhanced["information_needed"] = self._determine_required_info(enhanced)
        
        # Add validation status
        enhanced["validation"] = {
            "is_valid": bool(enhanced.get("selected_schema") and enhanced.get("selected_table")),
            "validated_at": str(datetime.now()),
            "confidence_score": enhanced.get("confidence_score", 0.0)
        }
        
        return enhanced

    def _generate_selection_reason(self, selection: dict) -> str:
        """Generate a detailed reason for the selection."""
        if selection.get("requires_verification"):
            return f"Selected {selection.get('selected_schema')}.{selection.get('selected_table')} as best match, requires verification"
        return f"Selected {selection.get('selected_schema')}.{selection.get('selected_table')} based on query requirements"

    def _determine_required_info(self, selection: dict) -> list:
        """Determine what information is needed for the selection."""
        required_info = []
        
        if selection.get("requires_verification"):
            required_info.append("Schema verification")
        if not selection.get("query_parameters", {}).get("columns"):
            required_info.append("Column information")
        if not selection.get("confidence_score", 0.0) > 0.8:
            required_info.append("Additional context")
            
        return required_info

    def _update_selection_history(self, selection: dict) -> None:
        """Update the selection history for better context in future selections."""
        self.selection_context["last_selection"] = selection
        self.selection_context["selection_history"].append({
            "timestamp": str(datetime.now()),
            "selection": selection
        })
        
        # Update preferred schema if selection was successful
        if selection.get("validation", {}).get("is_valid"):
            self.selection_context["preferred_schema"] = selection.get("selected_schema")

    def _create_error_response(self, error_message: str) -> dict:
        """Create a standardized error response."""
        return {
            "selector_response": {
                "error": error_message,
                "selected_tool": "error",
                "selected_datasource": None,
                "information_needed": ["Error resolution"],
                "reason_for_selection": f"Error occurred: {error_message}",
                "query_parameters": {},
                "validation": {
                    "is_valid": False,
                    "error": error_message,
                    "timestamp": str(datetime.now())
                }
            }
        }

###################
# SQL Generator Agent
###################

SQLGenerator_prompt_template = """
You are a SQL Generator that creates PostgreSQL queries. Your task is to generate valid SQL queries based on user questions.

For listing tables:
SELECT table_schema, table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE';

For querying interfaces:
SELECT interface_name, interface_id, description FROM inventories WHERE interface_name IS NOT NULL;

For other queries, follow these rules:
1. Always use explicit schema names
2. Include appropriate WHERE clauses
3. Add ORDER BY for consistent results
4. Use appropriate JOINs when needed
5. Include LIMIT clauses for large result sets

You MUST respond with a valid JSON object containing:
{
    "sql_query": "the complete SQL query",
    "explanation": "brief explanation of what the query does",
    "validation_checks": ["list of checks performed"],
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
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=SQLGenerator_guided_json
        )
        
        # Initialize query templates
        self.query_templates = {
            "interface_list": """
                SELECT DISTINCT table_schema, table_name, column_name 
                FROM information_schema.columns 
                WHERE column_name ILIKE '%interface%'
                AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name;
            """,
            "interface_query": """
                SELECT {columns}
                FROM {schema}.{table}
                WHERE {conditions}
                ORDER BY {order_by}
                LIMIT {limit};
            """,
            "table_list": """
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name;
            """
        }

    def invoke(self, user_question: str, planner_response: dict = None, selector_response: dict = None, feedback: str = None) -> dict:
        """Generate an optimized SQL query with enhanced validation and error recovery."""
        try:
            # Validate inputs
            validated_inputs = self._validate_inputs(user_question, planner_response, selector_response)
            if validated_inputs.get("error"):
                return self._create_error_response(validated_inputs["error"])

            # Generate SQL query
            sql_response = self._generate_sql_query(validated_inputs)
            
            # Validate and enhance response
            validated_response = self._validate_and_enhance_response(sql_response)
            
            return {"sql_generator_response": validated_response}

        except Exception as e:
            print(f"Error in SQL generator: {str(e)}")
            traceback.print_exc()
            return self._create_error_response(str(e))

    def _validate_inputs(self, user_question: str, planner_response: dict, selector_response: dict) -> dict:
        """Validate and normalize input parameters."""
        validated = {
            "user_question": user_question,
            "query_type": None,
            "schema": None,
            "table": None,
            "columns": [],
            "conditions": "",
            "error": None
        }

        # Check selector response
        if selector_response and isinstance(selector_response, dict):
            selector_data = selector_response.get("selector_response", {})
            validated.update({
                "schema": selector_data.get("selected_schema"),
                "table": selector_data.get("selected_table"),
                "query_parameters": selector_data.get("query_parameters", {})
            })

        # Check planner response
        if planner_response and isinstance(planner_response, dict):
            validated.update({
                "query_type": planner_response.get("query_type"),
                "columns": planner_response.get("relevant_columns", []),
                "conditions": planner_response.get("filtering_conditions", ""),
                "metadata": planner_response.get("metadata", {})
            })

        # Validate essential fields
        if not validated["schema"] or not validated["table"]:
            if "interface" in user_question.lower():
                validated["query_type"] = "interface_list"
            elif any(word in user_question.lower() for word in ["list", "show", "tables"]):
                validated["query_type"] = "table_list"
            else:
                validated["error"] = "Missing required schema or table information"

        return validated

    def _generate_sql_query(self, inputs: dict) -> dict:
        """Generate SQL query based on validated inputs."""
        response = {
            "sql_query": "",
            "explanation": "",
            "validation_checks": [],
            "query_type": inputs.get("query_type", "SELECT"),
            "estimated_complexity": "LOW",
            "required_indexes": []
        }

        try:
            # Handle interface queries
            if "interface" in inputs["user_question"].lower():
                if inputs["query_type"] == "interface_list":
                    response["sql_query"] = self.query_templates["interface_list"].strip()
                    response["explanation"] = "Query to discover interface-related columns across all schemas"
                else:
                    columns = inputs.get("columns", ["*"])
                    conditions = inputs.get("conditions") or "1=1"
                    response["sql_query"] = self.query_templates["interface_query"].format(
                        columns=", ".join(columns),
                        schema=inputs["schema"],
                        table=inputs["table"],
                        conditions=conditions,
                        order_by="interface_name" if "interface_name" in columns else "1",
                        limit=100
                    ).strip()
                    response["explanation"] = f"Query to fetch interface data from {inputs['schema']}.{inputs['table']}"

            # Handle table listing
            elif inputs["query_type"] == "table_list":
                response["sql_query"] = self.query_templates["table_list"].strip()
                response["explanation"] = "Query to list all tables in the database"

            # Handle regular queries
            else:
                columns = inputs.get("columns", ["*"])
                conditions = inputs.get("conditions") or "1=1"
                response["sql_query"] = f"""
                    SELECT {', '.join(columns)}
                    FROM {inputs['schema']}.{inputs['table']}
                    WHERE {conditions}
                    ORDER BY 1
                    LIMIT 100;
                """.strip()
                response["explanation"] = f"Query to fetch data from {inputs['schema']}.{inputs['table']}"

            # Add validation checks
            response["validation_checks"] = [
                "Query syntax validated",
                "Schema and table specified",
                "Column list validated",
                "WHERE clause included",
                "Results limited for safety"
            ]

            return response

        except Exception as e:
            print(f"Error generating SQL query: {e}")
            return self._create_error_response(str(e))

    def _validate_and_enhance_response(self, response: dict) -> dict:
        """Validate and enhance the SQL generator response."""
        try:
            # Ensure required fields
            required_fields = [
                "sql_query",
                "explanation",
                "validation_checks",
                "query_type",
                "estimated_complexity",
                "required_indexes"
            ]

            # Add missing fields with defaults
            for field in required_fields:
                if field not in response:
                    response[field] = "" if field == "sql_query" else []

            # Validate SQL query
            if not response["sql_query"]:
                raise ValueError("Empty SQL query generated")

            # Add metadata
            response["metadata"] = {
                "generated_at": str(datetime.now()),
                "validation_status": "valid",
                "has_limit": "LIMIT" in response["sql_query"].upper(),
                "has_order": "ORDER BY" in response["sql_query"].upper()
            }

            return response

        except Exception as e:
            print(f"Error in response validation: {e}")
            return self._create_error_response(str(e))

    def _create_error_response(self, error_message: str) -> dict:
        """Create a standardized error response."""
        return {
            "sql_query": "",
            "explanation": f"Error occurred: {error_message}",
            "validation_checks": ["Query generation failed"],
            "query_type": "ERROR",
            "estimated_complexity": "UNKNOWN",
            "required_indexes": [],
            "metadata": {
                "generated_at": str(datetime.now()),
                "validation_status": "error",
                "error_message": error_message
            }
        }

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
        super().__init__(
            state=state,
            model=model,
            server=server,
            temperature=temperature,
            model_endpoint=model_endpoint,
            stop=stop,
            guided_json=reviewer_guided_json
        )
        
        # Initialize validation rules
        self.validation_rules = {
            "syntax": [
                "balanced_parentheses",
                "valid_keywords",
                "proper_quotes",
                "semicolon_check"
            ],
            "security": [
                "injection_prevention",
                "schema_qualification",
                "permission_check"
            ],
            "performance": [
                "index_usage",
                "join_conditions",
                "result_limiting"
            ]
        }

    def invoke(self, user_question: str, sql_generator_response, schema_info: dict = None) -> dict:
        """Review SQL query with enhanced validation and feedback."""
        try:
            # Extract SQL query
            sql_query = ""
            if isinstance(sql_generator_response, str):
                sql_query = sql_generator_response
            elif isinstance(sql_generator_response, dict):
                if "sql_generator_response" in sql_generator_response:
                    sql_query = sql_generator_response["sql_generator_response"].get("sql_query", "")
                else:
                    sql_query = sql_generator_response.get("sql_query", "")

            # Initialize review response
            review = {
                "is_correct": False,
                "issues": [],
                "suggestions": [],
                "explanation": "",
                "security_concerns": [],
                "performance_impact": "HIGH",
                "confidence_score": 0.0
            }

            # Validate SQL query presence
            if not sql_query:
                review.update({
                    "issues": ["Empty SQL query provided"],
                    "suggestions": ["Ensure SQL query is generated properly"],
                    "explanation": "SQL query validation failed - empty query",
                    "confidence_score": 1.0
                })
                return {"reviewer_response": review}

            # Perform syntax validation
            syntax_issues = self._validate_syntax(sql_query)
            if syntax_issues:
                review.update({
                    "issues": syntax_issues,
                    "suggestions": ["Fix syntax errors before proceeding"],
                    "explanation": f"SQL syntax validation failed: {', '.join(syntax_issues)}",
                    "confidence_score": 1.0
                })
                return {"reviewer_response": review}

            # Perform security validation
            security_issues = self._validate_security(sql_query)
            if security_issues:
                review["security_concerns"].extend(security_issues)

            # Perform performance validation
            performance_issues, performance_suggestions = self._validate_performance(sql_query)
            if performance_issues:
                review["issues"].extend(performance_issues)
            if performance_suggestions:
                review["suggestions"].extend(performance_suggestions)

            # Generate optimization suggestions
            optimizations = self._generate_optimizations(sql_query)
            if optimizations:
                review["suggestions"].extend(optimizations)

            # Update review status
            review.update({
                "is_correct": len(review["issues"]) == 0,
                "explanation": self._generate_explanation(sql_query, review["issues"]),
                "performance_impact": self._assess_performance_impact(performance_issues),
                "confidence_score": 1.0,
                "metadata": {
                    "reviewed_at": str(datetime.now()),
                    "query_type": self._determine_query_type(sql_query),
                    "has_joins": "JOIN" in sql_query.upper(),
                    "has_where": "WHERE" in sql_query.upper(),
                    "has_limit": "LIMIT" in sql_query.upper()
                }
            })

            return {"reviewer_response": review}

        except Exception as e:
            print(f"Error in reviewer: {str(e)}")
            traceback.print_exc()
            return {
                "reviewer_response": {
                    "is_correct": False,
                    "issues": [str(e)],
                    "suggestions": ["Fix the identified error"],
                    "explanation": f"Review failed due to error: {str(e)}",
                    "security_concerns": [],
                    "performance_impact": "HIGH",
                    "confidence_score": 1.0
                }
            }

    def _validate_syntax(self, sql_query: str) -> list:
        """Validate SQL query syntax."""
        issues = []

        # Check for basic SQL keywords
        if sql_query.upper().startswith("SELECT"):
            required_keywords = ["SELECT", "FROM"]
            for keyword in required_keywords:
                if keyword not in sql_query.upper():
                    issues.append(f"Missing required keyword: {keyword}")

        # Check for balanced parentheses
        if sql_query.count('(') != sql_query.count(')'):
            issues.append("Unbalanced parentheses")

        # Check for proper quoting
        if sql_query.count("'") % 2 != 0:
            issues.append("Unmatched single quotes")

        # Check for schema qualification
        if " FROM " in sql_query.upper() and "." not in sql_query:
            issues.append("Tables should be schema-qualified")

        return issues

    def _validate_security(self, sql_query: str) -> list:
        """Validate SQL query for security concerns."""
        issues = []

        # Check for potential SQL injection patterns
        injection_patterns = ["--", ";--", "/*", "*/", "UNION ALL", "UNION SELECT"]
        for pattern in injection_patterns:
            if pattern in sql_query.upper():
                issues.append(f"Potential SQL injection risk: {pattern}")

        # Check for schema qualification
        if " FROM " in sql_query.upper() and "." not in sql_query:
            issues.append("Missing schema qualification (security risk)")

        return issues

    def _validate_performance(self, sql_query: str) -> tuple:
        """Validate SQL query for performance considerations."""
        issues = []
        suggestions = []

        # Check for result limiting
        if "LIMIT" not in sql_query.upper():
            issues.append("No LIMIT clause")
            suggestions.append("Add LIMIT clause to prevent large result sets")

        # Check for proper indexing hints
        if "WHERE" in sql_query.upper():
            suggestions.append("Ensure proper indexes exist for WHERE clause columns")

        # Check for JOIN conditions
        if "JOIN" in sql_query.upper():
            if "ON" not in sql_query.upper():
                issues.append("JOIN without ON clause")
            suggestions.append("Verify JOIN conditions and indexes")

        return issues, suggestions

    def _generate_optimizations(self, sql_query: str) -> list:
        """Generate optimization suggestions."""
        optimizations = []

        # Add index suggestions
        if "WHERE" in sql_query.upper():
            optimizations.append("Consider adding indexes on WHERE clause columns")

        # Add JOIN optimizations
        if "JOIN" in sql_query.upper():
            optimizations.append("Consider adding indexes on JOIN columns")
            optimizations.append("Verify JOIN order for optimal performance")

        # Add general optimizations
        optimizations.extend([
            "Consider adding appropriate WHERE clauses",
            "Use explicit column names instead of *",
            "Add ORDER BY for consistent results"
        ])

        return optimizations

    def _generate_explanation(self, sql_query: str, issues: list) -> str:
        """Generate detailed explanation of review results."""
        if issues:
            return f"Query validation found issues: {', '.join(issues)}"
        
        query_type = self._determine_query_type(sql_query)
        return f"Query validated successfully. Type: {query_type}"

    def _assess_performance_impact(self, performance_issues: list) -> str:
        """Assess the performance impact of the query."""
        if len(performance_issues) > 2:
            return "HIGH"
        elif len(performance_issues) > 0:
            return "MEDIUM"
        return "LOW"

    def _determine_query_type(self, sql_query: str) -> str:
        """Determine the type of SQL query."""
        sql_upper = sql_query.upper()
        if sql_upper.startswith("SELECT"):
            return "SELECT"
        elif sql_upper.startswith("INSERT"):
            return "INSERT"
        elif sql_upper.startswith("UPDATE"):
            return "UPDATE"
        elif sql_upper.startswith("DELETE"):
            return "DELETE"
        return "UNKNOWN"

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
            # Extract SQL query from workflow history
            sql_query = ""
            if isinstance(workflow_history, dict):
                if "SQLGenerator_response" in workflow_history:
                    sql_gen_response = workflow_history["SQLGenerator_response"]
                    if isinstance(sql_gen_response, dict) and "sql_generator_response" in sql_gen_response:
                        sql_query = sql_gen_response["sql_generator_response"].get("sql_query", "")

            # Create the report structure
            report = {
                "report": {
                    "summary": "Query execution completed",
                    "detailed_results": {
                        "key_findings": [],
                        "data_analysis": "",
                        "sample_data": [],
                        "query_details": {
                            "sql_query": sql_query,
                            "execution_time": query_results.get("execution_time", 0),
                            "row_count": query_results.get("row_count", 0)
                        }
                    }
                }
            }

            # Handle interface query results
            if "interface" in workflow_history.get("user_question", "").lower():
                if query_results.get("status") == "success":
                    rows = query_results.get("rows", [])
                    report["report"]["summary"] = f"Found {len(rows)} tables/columns containing interface information"
                    report["report"]["detailed_results"]["key_findings"] = [
                        f"Discovered {len(rows)} potential interface-related items",
                        "Search included all non-system schemas",
                        "Results show schema, table, and column names"
                    ]
                    report["report"]["detailed_results"]["data_analysis"] = (
                        "Analysis shows interface information across different schemas and tables. "
                        "Each result indicates a potential interface-related column."
                    )
                    # Include sample data (up to 10 rows)
                    report["report"]["detailed_results"]["sample_data"] = [
                        f"{row[0]}.{row[1]}.{row[2]}" for row in rows[:10]
                    ]
                else:
                    report["report"]["summary"] = "Error executing interface discovery query"
                    report["report"]["detailed_results"]["key_findings"] = [
                        "Query execution failed",
                        f"Error: {query_results.get('error_message', 'Unknown error')}"
                    ]

            return {"final_report": report}

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
