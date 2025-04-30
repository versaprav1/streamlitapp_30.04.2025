"""
Database schema information for SQL generation.
This module contains structured information about the database tables,
their relationships, and common query patterns to guide SQL generation.

The module defines three main schema sections:
- AZURE_SCHEMA: Schema for Azure-related tables and relationships
- CORE_SCHEMA: Schema for core system tables and inventory tracking
- BTP_SCHEMA: Schema for SAP Business Technology Platform components

Each schema section contains:
- tables: Dictionary of table definitions with fields and their properties
- relationships: List of relationships between tables
"""

# Azure Resources Schema
AZURE_SCHEMA = {
    "tables": {
        "azure_tenants": {
            "description": "Azure tenant information",
            "fields": {
                "id": {"type": "Primary Key", "description": "Unique identifier for each tenant"},
                "data_source_id": {"type": "Foreign Key", "references": "data_sources.id", 
                                  "description": "Links each tenant to a data source"}
            }
        },
        "azure_subscriptions": {
            "description": "Azure subscription information",
            "fields": {
                "id": {"type": "Primary Key", "description": "Unique identifier for each subscription"},
                "tenant_id": {"type": "Foreign Key", "references": "azure_tenants.id", 
                             "description": "Links each subscription to a specific tenant"}
            }
        },
        # Add other Azure tables...
    },
    "relationships": [
        {"from": "azure_tenants.data_source_id", "to": "data_sources.id", 
         "type": "One-to-One", "description": "Each tenant links to a specific data source"},
        {"from": "azure_subscriptions.tenant_id", "to": "azure_tenants.id", 
         "type": "Many-to-One", "description": "Each subscription belongs to a single tenant"},
        # Add other relationships...
    ]
}

# Core Integration Schema - defines the main system and inventory tables
CORE_SCHEMA = {
    "tables": {
        "systems": {
            "description": "Core system information",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "System identifier"},
                "data_source_id": {"type": "integer", "foreign_key": {"references": "data_sources", "field": "id"}, 
                                  "description": "Associated data source"}
            }
        },
        "inventories": {
            "description": "Inventory tracking",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "Inventory identifier"},
                "create_time": {"type": "timestamp", "description": "Creation timestamp"},
                "change_time": {"type": "timestamp", "description": "Last modification time"},
                "vendor": {"type": "string", "description": "Vendor information"},
                "type": {"type": "string", "description": "Inventory type"},
                "name": {"type": "string", "description": "Inventory name"},
                "description": {"type": "string", "description": "Description"},
                "object_id": {"type": "string", "description": "Object identifier"},
                "object_url": {"type": "string", "description": "Object URL"},
                "object_args": {"type": "string", "description": "Object arguments"},
                "sender_name": {"type": "string", "description": "Sender name"},
                "receiver_name": {"type": "string", "description": "Receiver name"},
                "data_source_id": {"type": "integer", "foreign_key": {"references": "data_sources", "field": "id"}, 
                                  "description": "Associated data source"},
                "sender_id": {"type": "integer", "foreign_key": {"references": "systems", "field": "id"}, 
                             "description": "Sender system"},
                "receiver_id": {"type": "integer", "foreign_key": {"references": "systems", "field": "id"}, 
                               "description": "Receiver system"}
            }
        },
    },
    "common_queries": {
        "flow_analysis": {
            "description": "Analyze data flows between systems",
            "tables": ["data_flows", "systems"],
            "joins": [
                {"from": "data_flows.sender_id", "to": "systems.id"},
                {"from": "data_flows.receiver_id", "to": "systems.id"}
            ]
        },
    }
}

# BTP (Business Technology Platform) Schema - defines SAP BTP related tables
BTP_SCHEMA = {
    "tables": {
        "btp_cloud_integration_packages": {
            "description": "SAP Cloud Integration packages",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "Package identifier"},
                "name": {"type": "string", "description": "Package name"},
                "description": {"type": "string", "description": "Package description"},
                "version": {"type": "string", "description": "Package version"},
                "data_source_id": {"type": "integer", "foreign_key": {"references": "data_sources", "field": "id"}, 
                                   "description": "Associated data source"}
            }
        },
        "btp_cloud_integration_artefacts": {
            "description": "SAP Cloud Integration artefacts",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "Artefact identifier"},
                "name": {"type": "string", "description": "Artefact name"},
                "description": {"type": "string", "description": "Artefact description"},
                "version": {"type": "string", "description": "Artefact version"},
                "package_id": {"type": "integer", "foreign_key": {"references": "btp_cloud_integration_packages", "field": "id"}, 
                                 "description": "Parent package"},
                "inventory_id": {"type": "integer", "foreign_key": {"references": "inventories", "field": "id"}, 
                                   "description": "Associated inventory"}
            }
        },
        "btp_api_management_providers": {
            "description": "SAP API Management providers",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "Provider identifier"},
                "name": {"type": "string", "description": "Provider name"},
                "data_source_id": {"type": "integer", "foreign_key": {"references": "data_sources", "field": "id"}, 
                                   "description": "Associated data source"}
            }
        },
        "btp_api_management_proxies": {
            "description": "SAP API Management proxies",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "Proxy identifier"},
                "name": {"type": "string", "description": "Proxy name"},
                "provider_id": {"type": "integer", "foreign_key": {"references": "btp_api_management_providers", "field": "id"}, 
                                  "description": "Parent provider"},
                "inventory_id": {"type": "integer", "foreign_key": {"references": "inventories", "field": "id"}, 
                                   "description": "Associated inventory"}
            }
        },
        "btp_event_mesh_queues": {
            "description": "SAP Event Mesh queues",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "Queue identifier"},
                "name": {"type": "string", "description": "Queue name"},
                "data_source_id": {"type": "integer", "foreign_key": {"references": "data_sources", "field": "id"}, 
                                   "description": "Associated data source"},
                "inventory_id": {"type": "integer", "foreign_key": {"references": "inventories", "field": "id"}, 
                                   "description": "Associated inventory"}
            }
        },
        "btp_event_mesh_topics": {
            "description": "SAP Event Mesh topics",
            "fields": {
                "id": {"type": "integer", "primary_key": True, "description": "Topic identifier"},
                "name": {"type": "string", "description": "Topic name"},
                "data_source_id": {"type": "integer", "foreign_key": {"references": "data_sources", "field": "id"}, 
                                   "description": "Associated data source"},
                "inventory_id": {"type": "integer", "foreign_key": {"references": "inventories", "field": "id"}, 
                                   "description": "Associated inventory"}
            }
        }
    },
    "relationships": [
        {"from": "btp_cloud_integration_artefacts.package_id", "to": "btp_cloud_integration_packages.id", 
         "type": "Many-to-One", "description": "Each artefact belongs to a single package"},
        {"from": "btp_cloud_integration_artefacts.inventory_id", "to": "inventories.id", 
         "type": "One-to-One", "description": "Each artefact links to a specific inventory"},
        {"from": "btp_api_management_proxies.provider_id", "to": "btp_api_management_providers.id", 
         "type": "Many-to-One", "description": "Each proxy belongs to a single provider"},
        {"from": "btp_api_management_proxies.inventory_id", "to": "inventories.id", 
         "type": "One-to-One", "description": "Each proxy links to a specific inventory"},
        {"from": "btp_event_mesh_queues.inventory_id", "to": "inventories.id", 
         "type": "One-to-One", "description": "Each queue links to a specific inventory"},
        {"from": "btp_event_mesh_topics.inventory_id", "to": "inventories.id", 
         "type": "One-to-One", "description": "Each topic links to a specific inventory"}
    ]
}

# Combined schema dictionary containing all schema definitions
DATABASE_SCHEMA = {
    "azure": AZURE_SCHEMA,
    "core": CORE_SCHEMA,
    "btp": BTP_SCHEMA
}

# Mapping of interface type codes to their human-readable descriptions
INTERFACE_TYPES = {
    "SIC": "SAP Cloud Integration artifact",
    "SIA": "API Management",
    "SAE": "Event Mesh"
}

# Predefined SQL query templates for common operations
COMMON_QUERY_PATTERNS = {
    "interfaces_by_type": """
    SELECT name, type 
    FROM {schema}.inventories
    WHERE type = '{type}'
    """,
    
    "interfaces_with_traffic": """
    SELECT i.name 
    FROM {schema}.inventories AS i
    LEFT JOIN {schema}.metadata AS m 
    ON i.id = m.inventory_id
    WHERE m.name = 'Last Traffic' 
    AND TO_TIMESTAMP(m.value, 'YYYY-MM-DD HH24:MI:SS') > now() - interval '{days} day'
    """,
    
    "interfaces_by_system": """
    SELECT i.name 
    FROM {schema}.inventories AS i
    FULL OUTER JOIN {schema}.systems AS s
    ON i.sender_id = s.id OR i.receiver_id = s.id
    WHERE s.name LIKE '%{system_name}%'
    """,
    
    "interfaces_with_metadata": """
    SELECT i.name 
    FROM {schema}.inventories AS i
    LEFT JOIN {schema}.metadata AS m 
    ON i.id = m.inventory_id
    WHERE m.name = '{metadata_name}' 
    AND m.value {operator} '{metadata_value}'
    """
}

def get_schema_info():
    """
    Return the complete database schema information.
    
    Returns:
        dict: Complete database schema containing Azure, Core, and BTP schemas
    """
    return DATABASE_SCHEMA

def get_common_query_pattern(pattern_name, **kwargs):
    """
    Get a common query pattern with parameters filled in.
    
    Args:
        pattern_name (str): Name of the pattern to retrieve
        **kwargs: Parameters to fill into the pattern. Expected parameters depend on the pattern:
            - interfaces_by_type: schema, type
            - interfaces_with_traffic: schema, days
            - interfaces_by_system: schema, system_name
            - interfaces_with_metadata: schema, metadata_name, operator, metadata_value
        
    Returns:
        str: Formatted SQL query string, or None if pattern_name not found
    """
    if pattern_name in COMMON_QUERY_PATTERNS:
        return COMMON_QUERY_PATTERNS[pattern_name].format(**kwargs)
    return None
