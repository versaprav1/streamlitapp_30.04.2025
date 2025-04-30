import os
import sys
import time
import logging
import datetime
import io
import contextlib
import traceback
import threading
import streamlit as st
from typing import Dict, List, Optional, Any, Union

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Dictionary to store logs by query ID
query_logs: Dict[str, List[str]] = {}

# Create a custom logger
logger = logging.getLogger('sql_workflow')
logger.setLevel(logging.DEBUG)

# Create a global file handler for all logs
all_logs_file = os.path.join(logs_dir, f'all_logs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(all_logs_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Lock for thread safety
log_lock = threading.Lock()

class QueryLogger:
    """Logger for capturing output related to a specific query."""
    
    def __init__(self, query_id: str, query_text: str):
        self.query_id = query_id
        self.query_text = query_text
        self.start_time = datetime.datetime.now()
        self.log_file = os.path.join(logs_dir, f'query_{query_id}_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log')
        self.log_buffer = []
        
        # Create a file handler for this query
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(file_formatter)
        logger.addHandler(self.file_handler)
        
        # Initialize log with query information
        self.log(f"QUERY START: {query_text}", level="INFO")
        self.log(f"QUERY ID: {query_id}", level="INFO")
        self.log(f"START TIME: {self.start_time}", level="INFO")
        
    def log(self, message: str, level: str = "DEBUG"):
        """Log a message both to file and to the in-memory buffer."""
        with log_lock:
            # Add to buffer
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"{timestamp} - {level} - {message}"
            self.log_buffer.append(formatted_message)
            
            # Log using the logger
            if level == "DEBUG":
                logger.debug(message)
            elif level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
            elif level == "CRITICAL":
                logger.critical(message)
    
    def get_logs(self) -> List[str]:
        """Get all logs for this query."""
        with log_lock:
            return self.log_buffer.copy()
    
    def close(self):
        """Close the logger and finalize logs."""
        end_time = datetime.datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.log(f"QUERY END: {self.query_text}", level="INFO")
        self.log(f"END TIME: {end_time}", level="INFO")
        self.log(f"DURATION: {duration:.2f} seconds", level="INFO")
        
        # Remove the file handler
        logger.removeHandler(self.file_handler)
        self.file_handler.close()
        
        # Add to the global dictionary
        with log_lock:
            query_logs[self.query_id] = self.log_buffer.copy()


class OutputCapture:
    """Context manager for capturing stdout and stderr."""
    
    def __init__(self, query_logger: QueryLogger):
        self.query_logger = query_logger
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def __enter__(self):
        sys.stdout = self._create_redirector(self.stdout_capture, "STDOUT")
        sys.stderr = self._create_redirector(self.stderr_capture, "STDERR")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        if exc_type is not None:
            self.query_logger.log(f"Exception during output capture: {exc_type.__name__}: {exc_val}", level="ERROR")
            self.query_logger.log(f"Traceback: {''.join(traceback.format_tb(exc_tb))}", level="ERROR")
    
    def _create_redirector(self, buffer, prefix):
        """Create a file-like object that redirects output to both buffer and logger."""
        original = getattr(sys, prefix.lower())
        
        class Redirector:
            def write(self_, string):
                if string.strip():  # Skip empty strings
                    self.query_logger.log(f"{prefix}: {string.rstrip()}")
                buffer.write(string)
                # Also write to the original stream for real-time viewing
                original.write(string)
            
            def flush(self_):
                buffer.flush()
                original.flush()
        
        return Redirector()


def start_query_logging(query_text: str) -> QueryLogger:
    """Start logging for a new query."""
    query_id = f"{int(time.time())}-{hash(query_text) % 1000000:06d}"
    return QueryLogger(query_id, query_text)


def get_query_logs(query_id: str) -> List[str]:
    """Get logs for a specific query."""
    with log_lock:
        return query_logs.get(query_id, [])


def get_all_query_ids() -> List[str]:
    """Get all query IDs."""
    with log_lock:
        return list(query_logs.keys())


def log_exception(query_logger: Optional[QueryLogger], error: Exception, context: str = ""):
    """Log an exception with full traceback."""
    error_message = f"ERROR in {context}: {type(error).__name__}: {str(error)}"
    tb = traceback.format_exc()
    
    if query_logger:
        query_logger.log(error_message, level="ERROR")
        query_logger.log(f"Traceback: {tb}", level="ERROR")
    else:
        logger.error(error_message)
        logger.error(f"Traceback: {tb}")


def add_logging_to_streamlit_ui():
    """Add a logging section to the Streamlit UI."""
    if 'show_logs' not in st.session_state:
        st.session_state.show_logs = False
    
    if st.sidebar.checkbox("Show Debug Logs", value=st.session_state.show_logs):
        st.session_state.show_logs = True
        st.sidebar.markdown("### Query Logs")
        
        # Get all query IDs
        query_ids = get_all_query_ids()
        
        if not query_ids:
            st.sidebar.info("No logs available yet.")
            return
        
        # Allow selecting a query to view logs
        selected_query = st.sidebar.selectbox("Select Query:", query_ids)
        
        if selected_query:
            logs = get_query_logs(selected_query)
            
            if logs:
                log_text = "\n".join(logs)
                
                # Create a download button for logs
                log_bytes = log_text.encode('utf-8')
                st.sidebar.download_button(
                    label="Download Logs",
                    data=log_bytes,
                    file_name=f"query_logs_{selected_query}.txt",
                    mime="text/plain"
                )
                
                # Display logs in a scrollable area
                st.sidebar.text_area("Query Logs", log_text, height=400)
            else:
                st.sidebar.info("No logs found for this query.")
    else:
        st.session_state.show_logs = False 