"""
Utility script to fix workflow recursion issues by providing 
tools to diagnose infinite loops and fix state transitions
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

# Add the streamlit_app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.logging_utils import logger

def analyze_workflow_execution(log_file: str) -> Dict[str, Any]:
    """
    Analyze a workflow execution log file to identify recursion issues.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dict with analysis results
    """
    # Check if file exists
    if not os.path.exists(log_file):
        return {"error": f"Log file not found: {log_file}"}
    
    # Read log file
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Track agent transitions
    transitions = []
    agent_counts = {}
    execution_paths = []
    errors = []
    
    # Find patterns indicating recursion
    for line in lines:
        # Extract agent transitions
        if "Processing agent:" in line:
            parts = line.split("Processing agent:")
            if len(parts) > 1:
                agent = parts[1].strip()
                transitions.append(agent)
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Extract execution paths
        if "Execution path:" in line:
            parts = line.split("Execution path:")
            if len(parts) > 1:
                try:
                    path = json.loads(parts[1].strip())
                    execution_paths.append(path)
                except:
                    # If we can't parse JSON, just store the string
                    execution_paths.append(parts[1].strip())
        
        # Extract errors
        if " - ERROR - " in line:
            errors.append(line.strip())
    
    # Detect cycles
    cycles = detect_cycles(transitions)
    
    # Calculate stats
    stats = {
        "total_transitions": len(transitions),
        "unique_agents": len(agent_counts),
        "agent_frequencies": agent_counts,
        "detected_cycles": cycles,
        "max_path_length": max([len(path) for path in execution_paths]) if execution_paths else 0,
        "errors": errors
    }
    
    # Suggest fixes based on the analysis
    if cycles:
        stats["fix_suggestions"] = [
            f"Detected cycle between {' -> '.join(cycle)}. Check route_decision function.",
            "Check state transitions in route_decision function",
            "Ensure final_report_generator node properly transitions to end node",
            "Check for circular references in agent responses"
        ]
    
    if any("router" in agent for agent in agent_counts) and agent_counts.get("router", 0) > 10:
        stats["fix_suggestions"] = stats.get("fix_suggestions", []) + [
            "Router node is being called excessively. Ensure proper state management.",
            "Check router_node implementation for infinite loops"
        ]
    
    return stats

def detect_cycles(sequence: List[str]) -> List[List[str]]:
    """
    Detect cycles in a sequence of agent transitions.
    
    Args:
        sequence: List of agent names in order of execution
    
    Returns:
        List of detected cycles
    """
    cycles = []
    seen = {}
    
    # Look for cycles of length 2-5 (most common in LangGraph)
    for cycle_length in range(2, 6):
        for i in range(len(sequence) - cycle_length + 1):
            pattern = tuple(sequence[i:i+cycle_length])
            
            # Check if this pattern repeats
            if pattern in seen:
                # If we see the pattern again and it's not already in cycles
                if pattern not in cycles and pattern[::-1] not in cycles:
                    cycles.append(pattern)
            else:
                seen[pattern] = i
    
    return cycles

def increase_recursion_limit(state_file: str, new_limit: int = 100) -> bool:
    """
    Update the recursion limit in the ChatWorkflow state file.
    
    Args:
        state_file: Path to the state file
        new_limit: New recursion limit
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the current state
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Update the recursion limit
        if 'chat_workflow' in state and 'recursion_limit' in state['chat_workflow']:
            old_limit = state['chat_workflow']['recursion_limit']
            state['chat_workflow']['recursion_limit'] = new_limit
            
            # Write back the updated state
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Updated recursion limit from {old_limit} to {new_limit}")
            return True
        else:
            logger.error("Could not find recursion_limit in state file")
            return False
            
    except Exception as e:
        logger.error(f"Error updating recursion limit: {e}")
        return False

def fix_route_decision():
    """
    Print instructions to fix the route_decision function.
    """
    instructions = """
To fix the route_decision function, update it as follows:

1. Make sure all route_to values are properly mapped to edge names
2. Add a recursion guard to prevent infinite loops
3. Ensure the final_report_generator node transitions to end node
4. Log all transitions for debugging

Example implementation:

```python
def route_decision(state: AgentGraphState) -> str:
    # Get the router_response
    router_response = state.router_response if hasattr(state, 'router_response') else {}
    
    # Get the current execution path
    execution_path = state.execution_path if hasattr(state, 'execution_path') else []
    
    # Add recursion detection (prevent more than 3 visits to the same node)
    node_counts = {}
    for node in execution_path:
        node_counts[node] = node_counts.get(node, 0) + 1
        
        # If we've visited any node more than 3 times, go to end
        if node_counts[node] > 3:
            print(f"RECURSION GUARD: Node {node} visited {node_counts[node]} times. Forcing to end.")
            return "end"
    
    # If execution path is too long, go to end
    if len(execution_path) > 20:
        print(f"RECURSION GUARD: Execution path too long ({len(execution_path)}). Forcing to end.")
        return "end"
    
    # Get the route_to value
    route_to = router_response.get("route_to", "final_report_generator") if isinstance(router_response, dict) else "final_report_generator"
    
    # Map route_to values to valid edge names
    route_mapping = {
        "planner": "to_planner",
        "selector": "to_selector",
        "SQLGenerator": "to_SQLGenerator", 
        "reviewer": "to_reviewer",
        "final_report_generator": "to_final_report",
        "end": "end"
    }
    
    # Get the mapped route
    result = route_mapping.get(route_to, "to_final_report")
    print(f"ROUTE DECISION: {route_to} -> {result}")
    
    return result
```

Apply this fix to your route_decision function in agent_graph/graph.py
    """
    print(instructions)
    return instructions

if __name__ == "__main__":
    # If this script is run directly, print the fix instructions
    print("Workflow Fix Utility")
    print("===================\n")
    fix_route_decision() 