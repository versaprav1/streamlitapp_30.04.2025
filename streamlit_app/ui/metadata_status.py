import streamlit as st
import json
import sys
import os
from datetime import datetime

# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_app_dir = os.path.dirname(current_dir)
if streamlit_app_dir not in sys.path:
    sys.path.insert(0, streamlit_app_dir)

def render_database_metadata_status(chat_workflow):
    """Render database metadata status in the sidebar"""
    if hasattr(chat_workflow, 'planner_agent') and hasattr(chat_workflow.planner_agent, 'metadata_store'):
        metadata_store = chat_workflow.planner_agent.metadata_store
        db_name = st.session_state.get("db_name", "new")

        # Load metadata if available
        metadata_store.load_metadata(db_name)

        # Use st.sidebar for all UI elements
        st.sidebar.subheader("Database Metadata Status")

        # Check if we have metadata for this database
        if db_name in metadata_store.metadata:
            st.sidebar.success(f"Metadata available for database: {db_name}")

            # Show schemas
            if "schemas" in metadata_store.metadata[db_name]:
                schemas = list(metadata_store.metadata[db_name]["schemas"].keys())
                st.sidebar.write(f"Discovered schemas: {', '.join(schemas)}")

                # Show tables for a selected schema
                selected_schema = st.sidebar.selectbox(
                    "Select schema to view tables",
                    options=schemas
                )

                if selected_schema in metadata_store.metadata[db_name]["schemas"]:
                    schema_data = metadata_store.metadata[db_name]["schemas"][selected_schema]
                    if "tables" in schema_data:
                        tables = list(schema_data["tables"].keys())
                        if tables:
                            st.sidebar.write(f"Tables in {selected_schema}: {', '.join(tables)}")

                            # Show table structure for a selected table
                            selected_table = st.sidebar.selectbox(
                                "Select table to view structure",
                                options=tables
                            )

                            if selected_table in schema_data["tables"]:
                                table_data = schema_data["tables"][selected_table]

                                # Display columns
                                if "columns" in table_data:
                                    st.sidebar.write("**Columns:**")
                                    for column in table_data["columns"]:
                                        primary_key = "🔑 " if column.get("primary_key", False) else ""
                                        nullable = "NULL" if column.get("nullable", True) else "NOT NULL"
                                        st.sidebar.write(f"{primary_key}{column['name']} ({column['type']}, {nullable})")

                                # Display relationships
                                if "relationships" in table_data and table_data["relationships"]:
                                    st.sidebar.write("**Relationships:**")
                                    for rel in table_data["relationships"]:
                                        st.sidebar.write(f"{rel['column']} → {rel['references']['schema']}.{rel['references']['table']}.{rel['references']['column']}")

                                # Display last updated time
                                if "last_updated" in table_data:
                                    last_updated = datetime.fromisoformat(table_data["last_updated"])
                                    st.sidebar.write(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            st.sidebar.info(f"No tables discovered in schema {selected_schema}")
                    else:
                        st.sidebar.info(f"No table metadata for schema {selected_schema}")
            else:
                st.sidebar.info("No schema metadata available")

            # Add refresh button
            if st.sidebar.button("Refresh Metadata"):
                if hasattr(chat_workflow, 'planner_agent'):
                    chat_workflow.planner_agent.discover_database_structure(force_refresh=True)
                    st.sidebar.success("Metadata refreshed!")
                    st.rerun()
        else:
            st.sidebar.info(f"No metadata available for database: {db_name}")

            # Add discover button
            if st.sidebar.button("Discover Database Structure"):
                if hasattr(chat_workflow, 'planner_agent'):
                    chat_workflow.planner_agent.discover_database_structure(force_refresh=True)
                    st.sidebar.success("Database structure discovered!")
                    st.rerun()

        # Add change log viewer
        if st.sidebar.checkbox("View Change Log"):
            metadata_store.load_change_log(db_name)
            if metadata_store.change_log:
                st.sidebar.write("**Change Log:**")
                for entry in metadata_store.change_log[-10:]:  # Show last 10 entries
                    timestamp = datetime.fromisoformat(entry["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
                    st.sidebar.write(f"{timestamp}: {entry['action']}")
                    with st.sidebar.expander("Details"):
                        st.sidebar.json(entry["details"])
            else:
                st.sidebar.info("No change log entries available")
