import sqlite3
import json
from pathlib import Path

class DatabaseManager:
    """
    Manages SQLite database operations for storing workflow state.
    
    This class handles database connection, table creation, and CRUD operations
    for workflow state data. The database is stored in the 'data' directory
    relative to this file's location.
    """
    
    def __init__(self):
        """
        Initializes database connection and creates required tables.
        
        Creates a 'data' directory if it doesn't exist and establishes
        connection to SQLite database named 'workflow_state.db'.
        """
        try:
            # Ensure data directory exists
            db_path = Path(__file__).parent.parent / 'data'
            db_path.mkdir(exist_ok=True)
            
            # Create full path to database
            db_file = db_path / 'workflow_state.db'
            print(f"Database path: {db_file}")
            
            # Connect to database
            self.conn = sqlite3.connect(str(db_file), check_same_thread=False)
            self.create_tables()
            
            # Verify database creation
            if db_file.exists():
                print("Database file created successfully")
            else:
                print("Warning: Database file not created")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
            self.conn = None
            print("Database successfully created and tables initialized")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            self.conn = None

    def create_tables(self):
        """
        Creates the workflow_state table if it doesn't exist.
        
        The table stores configuration and state information for the workflow,
        including server settings, model parameters, and graph data.
        """
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS workflow_state (
                    id INTEGER PRIMARY KEY,
                    server TEXT,
                    model TEXT,
                    model_endpoint TEXT,
                    temperature REAL,
                    recursion_limit INTEGER,
                    stop_token TEXT,
                    api_keys TEXT,
                    graph_data TEXT
                )
            ''')

    def save_workflow_state(self, server, model, model_endpoint, temperature, recursion_limit, stop_token, api_keys, graph_data):
        """
        Saves workflow state to database, replacing any existing state.
        
        Args:
            server (str): Server type (e.g., 'openai', 'ollama')
            model (str): Model name
            model_endpoint (str): API endpoint URL
            temperature (float): Model temperature setting
            recursion_limit (int): Maximum recursion depth
            stop_token (str): Token to stop generation
            api_keys (dict): API keys for different services
            graph_data (dict): Workflow graph structure data
        """
        with self.conn:
            self.conn.execute('DELETE FROM workflow_state')  # Clear previous state
            self.conn.execute('''
                INSERT INTO workflow_state 
                (server, model, model_endpoint, temperature, recursion_limit, stop_token, api_keys, graph_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (server, model, model_endpoint, temperature, recursion_limit, stop_token, 
                 json.dumps(api_keys), json.dumps(graph_data)))

    def load_workflow_state(self):
        """
        Loads the most recent workflow state from database.
        
        Returns:
            dict: Workflow state data including server settings, model parameters,
                 and graph data, or None if no state exists
        """
        cursor = self.conn.execute('SELECT * FROM workflow_state')
        row = cursor.fetchone()
        if row:
            return {
                'server': row[1],
                'model': row[2],
                'model_endpoint': row[3],
                'temperature': row[4],
                'recursion_limit': row[5],
                'stop_token': row[6],
                'api_keys': json.loads(row[7]),
                'graph_data': json.loads(row[8])
            }
        return None

    def __del__(self):
        """Ensure proper cleanup of database connection"""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    def is_connected(self):
        """
        Checks if database connection is active.
        
        Returns:
            bool: True if connection exists, False otherwise
        """
        return self.conn is not None
