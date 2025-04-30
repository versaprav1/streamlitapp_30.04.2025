import os
import sys

# Add the streamlit_app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to Python path to help with imports
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add the project root to Python path
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Python path:")
for path in sys.path:
    print(f"  - {path}")

print("\nTrying to import database modules...")
try:
    from database.metadata_store import DatabaseMetadataStore
    from database.discovery_service import DatabaseDiscoveryService
    print("✅ Successfully imported database modules!")
except ImportError as e:
    print(f"❌ Error importing database modules: {e}")
    
    print("\nTrying alternative import paths...")
    try:
        from streamlit_app.database.metadata_store import DatabaseMetadataStore
        from streamlit_app.database.discovery_service import DatabaseDiscoveryService
        print("✅ Successfully imported database modules using streamlit_app prefix!")
    except ImportError as e:
        print(f"❌ Error importing with streamlit_app prefix: {e}")
        
        print("\nChecking if database directory exists...")
        db_dir = os.path.join(current_dir, "database")
        if os.path.exists(db_dir):
            print(f"✅ Database directory exists at {db_dir}")
            print("Files in database directory:")
            for file in os.listdir(db_dir):
                print(f"  - {file}")
        else:
            print(f"❌ Database directory does not exist at {db_dir}")
