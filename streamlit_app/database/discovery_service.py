import psycopg2
import traceback
from typing import Dict, List, Any, Optional, Tuple
from .metadata_store import DatabaseMetadataStore

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
        # Check if we have fresh metadata
        if not force_refresh and self.metadata_store.is_schema_fresh(db_name, schema):
            return self.metadata_store.get_tables(db_name, schema)
        
        # Connect to the database
        try:
            conn, cursor = self.get_connection(db_name, user, password, host, port)
            
            # Query for tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """, (schema,))
            
            tables = [row[0] for row in cursor.fetchall()]
            
            # Update metadata
            self.metadata_store.update_tables(db_name, schema, tables)
            
            # Clean up
            cursor.close()
            conn.close()
            
            return tables
        except Exception as e:
            print(f"Error discovering tables in schema {schema}: {e}")
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
