#!/usr/bin/env python
"""
Diagnostic script for analyzing and fixing LangGraph workflow issues
"""
import os
import sys
import argparse
import glob
import json

# Add the streamlit_app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.workflow_fix import analyze_workflow_execution, increase_recursion_limit, fix_route_decision
from utils.logging_utils import logger


def list_log_files():
    """List all log files in the logs directory."""
    logs_dir = os.path.join(parent_dir, 'logs')
    log_files = glob.glob(os.path.join(logs_dir, '*.log'))
    return sorted(log_files, key=os.path.getmtime, reverse=True)


def find_session_state_file():
    """Find the Streamlit session state file."""
    # Streamlit stores session state in .streamlit/config.toml
    # Look in common locations
    possible_paths = [
        os.path.expanduser("~/.streamlit/config.toml"),
        os.path.join(parent_dir, '.streamlit/config.toml'),
        os.path.join(os.path.dirname(parent_dir), '.streamlit/config.toml')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def fix_recursion_issues(args):
    """Run diagnostics and fix recursion issues."""
    if args.list_logs:
        # List all log files
        log_files = list_log_files()
        if log_files:
            print("Available log files:")
            for i, file in enumerate(log_files):
                print(f"{i+1}. {os.path.basename(file)} - {os.path.getmtime(file)}")
        else:
            print("No log files found")
        return
    
    if args.fix_route:
        # Print fix instructions for route_decision
        print(fix_route_decision())
        return
    
    if args.increase_limit:
        # Find the session state file
        state_file = find_session_state_file()
        if state_file:
            # Increase the recursion limit
            result = increase_recursion_limit(state_file, args.limit)
            if result:
                print(f"Successfully increased recursion limit to {args.limit}")
            else:
                print("Failed to increase recursion limit")
        else:
            print("Session state file not found")
        return
    
    if args.log_file:
        # Analyze the specified log file
        log_file = args.log_file
        if not os.path.exists(log_file):
            # Check if it's an index
            try:
                index = int(log_file) - 1
                log_files = list_log_files()
                if 0 <= index < len(log_files):
                    log_file = log_files[index]
                else:
                    print(f"Invalid log file index: {log_file}")
                    return
            except ValueError:
                print(f"Log file not found: {log_file}")
                return
        
        # Analyze the log file
        analysis = analyze_workflow_execution(log_file)
        
        # Print the analysis
        print(f"Analysis of {os.path.basename(log_file)}:")
        print(f"Total transitions: {analysis.get('total_transitions', 0)}")
        print(f"Unique agents: {analysis.get('unique_agents', 0)}")
        print(f"Max path length: {analysis.get('max_path_length', 0)}")
        
        # Print agent frequencies
        if "agent_frequencies" in analysis:
            print("\nAgent frequencies:")
            for agent, count in analysis["agent_frequencies"].items():
                print(f"  {agent}: {count}")
        
        # Print detected cycles
        if "detected_cycles" in analysis and analysis["detected_cycles"]:
            print("\nDetected cycles:")
            for cycle in analysis["detected_cycles"]:
                print(f"  {' -> '.join(cycle)}")
        
        # Print errors
        if "errors" in analysis and analysis["errors"]:
            print("\nErrors:")
            for error in analysis["errors"][:10]:  # Limit to 10 errors
                print(f"  {error}")
            
            if len(analysis["errors"]) > 10:
                print(f"  ... and {len(analysis['errors']) - 10} more errors")
        
        # Print fix suggestions
        if "fix_suggestions" in analysis:
            print("\nSuggested fixes:")
            for suggestion in analysis["fix_suggestions"]:
                print(f"  - {suggestion}")
        
        return
    
    # If no specific action specified, show help
    print("No action specified. Use --help to see available options.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Diagnostic tool for LangGraph workflow issues")
    parser.add_argument("--list-logs", action="store_true", help="List all log files")
    parser.add_argument("--log-file", type=str, help="Log file to analyze (path or index)")
    parser.add_argument("--fix-route", action="store_true", help="Print fix instructions for route_decision")
    parser.add_argument("--increase-limit", action="store_true", help="Increase recursion limit")
    parser.add_argument("--limit", type=int, default=100, help="New recursion limit value")
    
    args = parser.parse_args()
    fix_recursion_issues(args)


if __name__ == "__main__":
    main() 