"""
Database tool module for managing interface inventory data.
Provides functions to interact with a PostgreSQL database containing interface information.
"""

import psycopg2
from typing import Dict, Any, List, Optional

def get_db_connection():
    """
    Creates and returns a connection to the PostgreSQL database.
    
    Returns:
        psycopg2.connection: A connection object to the PostgreSQL database.
    """
    return psycopg2.connect(
        dbname="new",
        user="postgres",
        password="pass",
        host="localhost",
        port="5432"
    )

# Mapping for inventory types
InventoryTypeValuesToNames = {
    "MLAPI": "MULE_API",
    "MLAPP": "MULE_APP",
    "GAP": "APIM",
    "AZA": "AZURE_APIM",
    "AZE": "AZURE_EVENTGRID",
    "AZC": "AZURE_LA_CON",
    "AZS": "AZURE_LA_STD",
    "AZQ": "AZURE_SB_QUEUE",
    "AZT": "AZURE_SB_TOPIC",
    "BAC": "BACKEND",
    "BRO": "BROKER",
    "ESB": "ESB",
    "OTH": "OTHER",
    "PLA": "PLANNED",
    "SAO": "SAP_ODATA",
    "SAS": "SAP_SOAP",
    "SAE": "SAP_EVENTMESH",
    "SAI": "SAP_IDOC",
    "SIA": "SAP_IS_APIM",
    "SIC": "SAP_IS_CI",
    "SAP": "SAP_PO",
    "EAM": "EAM",
}

def get_data_sources() -> List[str]:
    """
    Retrieves all unique data source types from the inventories table.
    
    Returns:
        List[str]: A list of full names of all available data sources.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT type FROM inventories;")
        short_forms = cursor.fetchall()
        data_sources = [InventoryTypeValuesToNames[short_form[0]] for short_form in short_forms if short_form[0] in InventoryTypeValuesToNames]
        return data_sources
    finally:
        cursor.close()
        conn.close()

def get_interfaces_by_datasource(datasource_name: str) -> List[Dict[str, Any]]:
    """
    Retrieves all interfaces for a specific data source type.
    
    Args:
        datasource_name (str): The full name of the data source type.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing interface details.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        short_form = None
        for key, value in InventoryTypeValuesToNames.items():
            if value == datasource_name:
                short_form = key
                break
        
        if not short_form:
            return []
        
        cursor.execute("""
            SELECT id, name, description, type, status, created_at, updated_at
            FROM inventories
            WHERE type = %s;
        """, (short_form,))
        
        interfaces = []
        for row in cursor.fetchall():
            interface = {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "type": InventoryTypeValuesToNames.get(row[3], row[3]),
                "status": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
                "updated_at": row[6].isoformat() if row[6] else None
            }
            interfaces.append(interface)
        
        return interfaces
    finally:
        cursor.close()
        conn.close()

def get_interface_details(interface_id: Optional[int] = None, interface_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves detailed information about a specific interface by ID or name.
    
    Args:
        interface_id (Optional[int]): The ID of the interface to retrieve.
        interface_name (Optional[str]): The name of the interface to retrieve.
    
    Returns:
        Dict[str, Any]: A dictionary containing interface details and additional information.
    """
    if not interface_id and not interface_name:
        return {}
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if interface_id:
            cursor.execute("""
                SELECT id, name, description, type, status, created_at, updated_at
                FROM inventories
                WHERE id = %s;
            """, (interface_id,))
        else:
            cursor.execute("""
                SELECT id, name, description, type, status, created_at, updated_at
                FROM inventories
                WHERE name = %s;
            """, (interface_name,))
        
        row = cursor.fetchone()
        if not row:
            return {}
        
        interface = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "type": InventoryTypeValuesToNames.get(row[3], row[3]),
            "status": row[4],
            "created_at": row[5].isoformat() if row[5] else None,
            "updated_at": row[6].isoformat() if row[6] else None
        }
        
        cursor.execute("""
            SELECT column_name, column_value
            FROM interface_details
            WHERE interface_id = %s;
        """, (interface["id"],))
        
        details = {}
        for detail_row in cursor.fetchall():
            details[detail_row[0]] = detail_row[1]
        
        interface["details"] = details
        
        return interface
    finally:
        cursor.close()
        conn.close()

def search_interfaces(keyword: str) -> List[Dict[str, Any]]:
    """
    Searches for interfaces by keyword in name and description fields.
    
    Args:
        keyword (str): The search term to look for in interface names and descriptions.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing matching interface information.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, name, description, type, status
            FROM inventories
            WHERE name ILIKE %s OR description ILIKE %s;
        """, (f'%{keyword}%', f'%{keyword}%'))
        
        interfaces = []
        for row in cursor.fetchall():
            interface = {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "type": InventoryTypeValuesToNames.get(row[3], row[3]),
                "status": row[4]
            }
            interfaces.append(interface)
        
        return interfaces
    finally:
        cursor.close()
        conn.close()
