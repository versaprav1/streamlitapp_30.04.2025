import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

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
