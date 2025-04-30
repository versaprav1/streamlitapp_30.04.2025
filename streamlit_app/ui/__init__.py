# UI package initialization
try:
    from .metadata_status import render_database_metadata_status
except ImportError:
    import streamlit as st

    # Define a fallback function if the import fails
    def render_database_metadata_status(chat_workflow):
        st.warning("Database metadata status UI is not available.")
        if st.button("Initialize Database Discovery"):
            if hasattr(chat_workflow, 'planner_agent'):
                chat_workflow.planner_agent.discover_database_structure(force_refresh=True)
                st.success("Database structure discovery initiated!")
