import os
import yaml
import json
import logging
from datetime import datetime
import traceback
from typing import Dict, Any, Optional, List
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# for loading configs to environment variables
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add timestamp to config
    config['timestamp'] = datetime.now().isoformat()
    return config
