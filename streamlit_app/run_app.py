import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys

    app_path = os.path.join(project_root, "app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.port=8502"]
    sys.exit(stcli.main())
