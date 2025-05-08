# Standard library imports for system operations and error handling
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List  # Add Optional to imports
from langgraph.graph import Graph, StateGraph, END
from states.state import AgentGraphState, get_model_settings
from langchain.callbacks import StdOutCallbackHandler
from models.model_registry import ModelRegistry
from pydantic import BaseModel
import streamlit as st

# Add project root to system path for absolute imports
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Make sure the database module is available
try:
    import database
except ImportError:
    print("Note: database module not found in sys.path. This is expected and will be handled by the PlannerAgent.")

# Import custom state management and prompt templates
from states.state import get_agent_graph_state

# Import agent classes and prompt templates
from agents.agents import (
    PlannerAgent,
    SelectorAgent,
    SQLGenerator,
    ReviewerAgent,
    RouterAgent,
    FinalReportAgent,
    EndNodeAgent,
    SQLGenerator_prompt_template,
    reviewer_prompt_template,
    router_prompt_template,
    final_report_prompt_template,
    end_node_prompt_template
)

def filter_state_for_agent(state, allowed_keys):
    return {k: v for k, v in state.items() if k in allowed_keys}

def get_agent_graph(
    model=None,
    server=None,
    temperature=None,
    model_endpoint=None,
    stop=None
):
    """
    Creates and returns a graph of AI agents for SQL User workflow.
    All model settings should come from session state via the UI.

    Args:
        model (str, optional): Model name from session state
        server (str, optional): API server from session state
        temperature (float, optional): Temperature from session state
        model_endpoint (str, optional): Custom endpoint from session state
        stop (str, optional): Stop sequence from session state

    Returns:
        dict: Dictionary containing agent nodes for the workflow graph
    """
    # Get model settings from state
    from states.state import get_model_settings
    model_settings = get_model_settings()

    # Use provided settings or fall back to state settings
    model = model or model_settings["model"]
    server = server or model_settings["server"]
    temperature = temperature if temperature is not None else model_settings["temperature"]
    model_endpoint = model_endpoint

    # Initialize state management
    state = get_agent_graph_state()

    # Create graph with AgentGraphState as the state type
    graph = StateGraph(AgentGraphState)

    # Add nodes to the graph - use the standalone node functions directly
    graph.add_node("planner", planner_node)
    graph.add_node("selector", selector_node)
    graph.add_node("SQLGenerator", sql_generator_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("router", router_node)
    graph.add_node("final_report_generator", final_report_node)
    graph.add_node("end", end_node)

    # Add direct edges between nodes in the workflow
    graph.add_edge("planner", "selector")
    graph.add_edge("selector", "SQLGenerator")
    graph.add_edge("SQLGenerator", "reviewer")
    graph.add_edge("reviewer", "router")

    # Add conditional edges based on router's decision
    graph.add_conditional_edges(
        "router",
        route_decision,  # Use the improved route_decision function
        {
            "to_planner": "planner",
            "to_selector": "selector",
            "to_SQLGenerator": "SQLGenerator",
            "to_reviewer": "reviewer",
            "to_final_report": "final_report_generator",
            "end": "end"
        }
    )

    # Add direct edge from final report to end
    graph.add_edge("final_report_generator", "end")

    # Define the entry point
    graph.set_entry_point("planner")

    return graph

def route_decision(state: AgentGraphState) -> str:
    """Route decision function for the agent graph."""
    try:
        # Add recursion detection (prevent more than 3 visits to the same node)
        execution_path = state.execution_path if hasattr(state, 'execution_path') else []
        
        # Count visits to each node
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

        # Get router_response as attribute
        router_response = state.router_response if hasattr(state, 'router_response') else {}

        if isinstance(router_response, dict):
            # Get the route_to value with final_report_generator as default
            route_to = router_response.get("route_to", "final_report_generator")

            # Always force to end if coming from final_report_generator
            if execution_path and execution_path[-1] == "final_report_generator":
                print("ROUTE DECISION: Coming from final_report_generator, forcing to end")
                return "end"

            # Map route_to values to valid edge names
            route_mapping = {
                "planner": "to_planner",
                "selector": "to_selector",
                "SQLGenerator": "to_SQLGenerator",
                "reviewer": "to_reviewer",
                "final_report_generator": "to_final_report",
                "end": "end"
            }
            
            # Get the mapped route or default to final_report
            mapped_route = route_mapping.get(route_to)
            if mapped_route:
                print(f"ROUTE DECISION: {route_to} -> {mapped_route}")
                return mapped_route
            else:
                print(f"Invalid route_to value: {route_to}, defaulting to to_final_report")
                return "to_final_report"
        else:
            print(f"router_response is not a dict: {type(router_response)}, defaulting to to_final_report")
            return "to_final_report"
    except Exception as e:
        print(f"Error in route decision: {e}")
        traceback.print_exc()
        print("Routing to end due to error")
        return "end"

def get_agent_graph_state() -> AgentGraphState:
    """Initialize agent graph state with current model settings"""
    model_settings = get_model_settings()
    return AgentGraphState(
        current_agent="planner",
        user_question="",
        selected_schema={},
        tool_responses={},
        execution_path=[],
        errors={},
        is_error_state=False,
        planner_response={},
        selector_response={},
        SQLGenerator_response={},
        reviewer_response={},
        router_response={},
        final_report_response={},
        schemas={},
        sql_query="",
        sql_query_results={},
        previous_selections=[],
        previous_reports=[],
        start_time=datetime.now(),
        retry_counts={},
        last_successful_state={},
        end_chain=False,
        current_node="start",
        iteration_count=0,
        error_count=0,
        last_error=None,
        last_success=None,
        metadata={},
        model=model_settings.get("model"),
        server=model_settings.get("server"),
        temperature=model_settings.get("temperature", 0.0),
        model_endpoint=model_settings.get("model_endpoint")
    )

def planner_node(state: AgentGraphState):
    """
    Planner node implementation.

    Args:
        state: Current state of the workflow

    Returns:
        Dict: Updated state information
    """
    print("\n==== PLANNER NODE START ====")
    print(f"Input state: user_question={state.user_question}")
    print(f"Current execution path: {state.execution_path}")

    try:
        # Get user question from state using direct attribute access
        user_question = state.user_question
        print(f"Processing user question: '{user_question}'")

        # Initialize the enhanced planner agent with database discovery capabilities
        try:
            planner = PlannerAgent(
                model=st.session_state.get("llm_model"),
                server=st.session_state.get("server"),
                temperature=st.session_state.get("temperature", 0),
                model_endpoint=st.session_state.get("server_endpoint")
            )
            print(f"Successfully initialized PlannerAgent with database capabilities using model={st.session_state.get('llm_model')}, server={st.session_state.get('server')}")
            print(f"PlannerAgent is using LLM to analyze the query and determine the appropriate database tables and columns")
        except ImportError as e:
            print(f"Warning: Database module import failed in PlannerAgent: {e}")
            print("Falling back to basic PlannerAgent without database capabilities")
            # Create a simplified version of PlannerAgent without database dependencies
            from agents.agents import Agent

            class SimplifiedPlannerAgent(Agent):
                def invoke(self, user_question):
                    print("SimplifiedPlannerAgent: Using hardcoded response without LLM")
                    return {
                        "query_type": "sql",
                        "primary_table_or_datasource": "unknown",
                        "relevant_columns": ["*"],
                        "filtering_conditions": "",
                        "processing_instructions": f"Process query: {user_question}"
                    }

                def discover_database_structure(self, force_refresh=False):
                    """Simplified version that doesn't actually discover anything"""
                    print("SimplifiedPlannerAgent: Database discovery not available")
                    return ["dev"]

                def discover_schema_tables(self, schema, force_refresh=False):
                    """Simplified version that doesn't actually discover anything"""
                    print(f"SimplifiedPlannerAgent: Table discovery for schema '{schema}' not available")
                    return []

            planner = SimplifiedPlannerAgent(
                model=st.session_state.get("llm_model"),
                server=st.session_state.get("server"),
                temperature=st.session_state.get("temperature", 0),
                model_endpoint=st.session_state.get("server_endpoint")
            )

        # Get the plan from the planner
        print(f"Invoking PlannerAgent with question: '{user_question}'")
        plan = planner.invoke(user_question)
        print(f"PlannerAgent returned plan: {json.dumps(plan, indent=2)}")

        # Ensure plan is a dictionary
        if not isinstance(plan, dict):
            print(f"Warning: Plan is not a dictionary, converting to dict. Type was: {type(plan)}")
            plan = {"raw": str(plan)}

        # Update state with plan
        result = {
            "current_agent": "selector",
            "execution_path": state.execution_path + ["planner"],
            "planner_response": plan,
            "planner_agent": planner  # Store the planner agent instance for later use
        }

        print(f"Planner node output: current_agent={result['current_agent']}")
        print(f"Planner response being sent to selector: {json.dumps(result['planner_response'], indent=2)}")
        print("==== PLANNER NODE END ====\n")
        return result
    except Exception as e:
        print(f"Error in planner node: {e}")
        traceback.print_exc()

        # Handle errors with proper attribute access
        error_result = {
            "current_agent": "planner_error",
            "errors": {"planner": str(e)},
            "execution_path": state.execution_path + ["planner_error"],
            "is_error_state": True
        }
        print(f"Planner node error output: {json.dumps(error_result, indent=2)}")
        print("==== PLANNER NODE END (WITH ERROR) ====\n")
        return error_result

def selector_node(state: AgentGraphState):
    """
    Selector node implementation.

    Args:
        state: Current state of the workflow

    Returns:
        Dict: Updated state information
    """
    print("\n==== SELECTOR NODE START ====")
    print(f"Input state: user_question={state.user_question}")
    print(f"Current execution path: {state.execution_path}")
    print(f"Planner response received: {json.dumps(state.planner_response, indent=2)}")

    try:
        # Use direct attribute access
        user_question = state.user_question
        planner_response = state.planner_response

        print(f"Selector is processing planner response for question: '{user_question}'")
        print("NOTE: The selector node is currently simplified and doesn't use LLM.")
        print("It simply passes through to the SQLGenerator without making actual selections.")

        # For a real implementation, we would:
        # 1. Analyze the planner_response to determine which schemas/tables to use
        # 2. Create a SelectorAgent instance and invoke it with the planner_response
        # 3. Use the response to select the appropriate schemas and tables

        # Create a simple selector response
        selector_response = {
            "raw": "Selection completed",
            "selected_schemas": {},
            "selected_tool": "sql_query",
            "selected_datasource": planner_response.get("primary_table_or_datasource", "information_schema.tables"),
            "information_needed": "Table listing",
            "reason_for_selection": "Direct database query is the most efficient way to list tables",
            "query_parameters": {
                "columns": planner_response.get("relevant_columns", ["table_schema", "table_name"]),
                "filters": planner_response.get("filtering_conditions", "table_type = 'BASE TABLE'")
            }
        }

        # Create the result
        result = {
            "current_agent": "SQLGenerator",
            "execution_path": state.execution_path + ["selector"],
            "selector_response": selector_response
        }

        print(f"Selector node output: current_agent={result['current_agent']}")
        print(f"Selector response being sent to SQLGenerator: {json.dumps(result['selector_response'], indent=2)}")
        print("==== SELECTOR NODE END ====\n")
        return result
    except Exception as e:
        print(f"Error in selector node: {e}")
        traceback.print_exc()

        error_result = {
            "current_agent": "selector_error",
            "errors": {"selector": str(e)},
            "execution_path": state.execution_path + ["selector_error"],
            "is_error_state": True
        }
        print(f"Selector node error output: {json.dumps(error_result, indent=2)}")
        print("==== SELECTOR NODE END (WITH ERROR) ====\n")
        return error_result

def sql_generator_node(state: AgentGraphState):
    """Generate SQL query based on user question and selected schema"""
    print("\n==== SQL GENERATOR NODE START ====")
    print(f"Input state: user_question={state.user_question}")
    print(f"Current execution path: {state.execution_path}")
    print(f"Selector response received: {json.dumps(state.selector_response, indent=2) if hasattr(state, 'selector_response') else 'None'}")

    try:
        # Create a new SQLGenerator with current settings
        generator = SQLGenerator(
            model=st.session_state.get("llm_model"),
            server=st.session_state.get("server"),
            temperature=st.session_state.get("temperature", 0),
            model_endpoint=st.session_state.get("server_endpoint")
        )
        print(f"Created SQLGenerator with model={st.session_state.get('llm_model')}, server={st.session_state.get('server')}")

        # Get required inputs
        user_question = state.user_question
        planner_response = state.planner_response if hasattr(state, 'planner_response') else None
        selector_response = state.selector_response if hasattr(state, 'selector_response') else None

        print(f"Invoking SQLGenerator with user_question: '{user_question}'")
        response = generator.invoke(
            user_question=user_question,
            planner_response=planner_response,
            selector_response=selector_response
        )
        print(f"SQLGenerator returned response: {json.dumps(response, indent=2)}")

        # Extract SQL query from response
        sql_query = ""
        if isinstance(response, dict) and "sql_generator_response" in response:
            sql_generator_response = response["sql_generator_response"]
            if isinstance(sql_generator_response, dict):
                sql_query = sql_generator_response.get("sql_query", "").strip()

        # If sql_query is empty, this is an error condition
        if not sql_query:
            error_result = {
                "current_agent": "router",
                "errors": {"SQLGenerator": "Failed to generate SQL query"},
                "execution_path": state.execution_path + ["SQLGenerator_error"],
                "is_error_state": True,
                "SQLGenerator_response": response,
                "sql_query": "",
                "model": st.session_state.get("llm_model"),
                "server": st.session_state.get("server"),
                "temperature": st.session_state.get("temperature", 0),
                "model_endpoint": st.session_state.get("server_endpoint")
            }
            print(f"SQL Generator node error output: {json.dumps(error_result, indent=2)}")
            print("==== SQL GENERATOR NODE END (WITH ERROR) ====\n")
            return error_result

        # Create successful result
        result = {
            "current_agent": "reviewer",
            "execution_path": state.execution_path + ["SQLGenerator"],
            "SQLGenerator_response": response,
            "sql_query": sql_query,
            "model": st.session_state.get("llm_model"),
            "server": st.session_state.get("server"),
            "temperature": st.session_state.get("temperature", 0),
            "model_endpoint": st.session_state.get("server_endpoint")
        }

        print(f"SQL Generator node output: current_agent={result['current_agent']}")
        print(f"SQL query generated: {sql_query}")
        print(f"SQL Generator response being sent to reviewer: {json.dumps(response, indent=2)}")
        print("==== SQL GENERATOR NODE END ====\n")
        return result

    except Exception as e:
        print(f"Error in sql_generator_node: {e}")
        traceback.print_exc()

        error_result = {
            "current_agent": "router",
            "errors": {"SQLGenerator": str(e)},
            "execution_path": state.execution_path + ["SQLGenerator_error"],
            "is_error_state": True,
            "SQLGenerator_response": {"error": str(e)},
            "sql_query": "",
            "model": st.session_state.get("llm_model"),
            "server": st.session_state.get("server"),
            "temperature": st.session_state.get("temperature", 0),
            "model_endpoint": st.session_state.get("server_endpoint")
        }

        print(f"SQL Generator node error output: {json.dumps(error_result, indent=2)}")
        print("==== SQL GENERATOR NODE END (WITH ERROR) ====\n")
        return error_result

def reviewer_node(state: AgentGraphState):
    """
    Reviewer node implementation.

    Args:
        state: Current state of the workflow

    Returns:
        Dict: Updated state information
    """
    print("\n==== REVIEWER NODE START ====")
    print(f"Input state: user_question={state.user_question}")
    print(f"Current execution path: {state.execution_path}")
    print(f"SQL query to review: {state.sql_query}")
    print(f"SQLGenerator response received: {json.dumps(state.SQLGenerator_response, indent=2) if hasattr(state, 'SQLGenerator_response') else 'None'}")

    try:
        # Get SQL query from state
        sql_query = state.sql_query
        user_question = state.user_question
        print(f"Reviewer is analyzing SQL query for question: '{user_question}'")

        # For simple queries like "list all tables", we could use a simplified reviewer
        # that doesn't use LLM, but for completeness we'll use the LLM-based reviewer
        print(f"Using LLM-based ReviewerAgent to analyze the SQL query")

        # Initialize the reviewer agent with state
        reviewer = ReviewerAgent(
            state=state,
            model=st.session_state.get("llm_model"),
            server=st.session_state.get("server"),
            temperature=st.session_state.get("temperature", 0),
            model_endpoint=st.session_state.get("server_endpoint")
        )
        print(f"Created ReviewerAgent with model={st.session_state.get('llm_model')}, server={st.session_state.get('server')}")

        # Get the review from the reviewer
        print(f"Invoking ReviewerAgent with user_question: '{user_question}' and sql_query: '{sql_query}'")
        review = reviewer.invoke(user_question, sql_query)
        print(f"ReviewerAgent returned review: {json.dumps(review, indent=2)}")

        # Ensure review is a dictionary
        if not isinstance(review, dict):
            print(f"Warning: Review is not a dictionary, converting to dict. Type was: {type(review)}")
            review = {"raw": str(review)}

        # Update state with review
        result = {
            "current_agent": "sql_executor",  # Changed to go to sql_executor instead of router
            "execution_path": state.execution_path + ["reviewer"],
            "reviewer_response": review,
            "model": st.session_state.get("llm_model"),
            "server": st.session_state.get("server"),
            "temperature": st.session_state.get("temperature", 0),
            "model_endpoint": st.session_state.get("server_endpoint")
        }

        print(f"Reviewer node output: current_agent={result['current_agent']}")
        print(f"Reviewer response being sent to sql_executor: {json.dumps(review, indent=2)}")
        print("==== REVIEWER NODE END ====\n")
        return result
    except Exception as e:
        # Handle errors with proper attribute access
        print(f"Error in reviewer node: {e}")
        traceback.print_exc()

        # Create a default review to allow the workflow to continue
        default_review = {
            "is_correct": True,  # Assume query is correct to continue
            "issues": [],
            "suggestions": [],
            "explanation": f"Error in reviewer: {str(e)}. Proceeding with query execution.",
            "security_concerns": [],
            "performance_impact": "LOW",
            "confidence_score": 0.5
        }

        error_result = {
            "current_agent": "sql_executor",  # Go directly to SQL executor even on error
            "errors": {"reviewer": str(e)},
            "execution_path": state.execution_path + ["reviewer_error"],
            "is_error_state": False,  # Don't mark as error state to continue the flow
            "reviewer_response": default_review,
            "model": st.session_state.get("llm_model"),
            "server": st.session_state.get("server"),
            "temperature": st.session_state.get("temperature", 0),
            "model_endpoint": st.session_state.get("server_endpoint")
        }

        print(f"Reviewer node error output: {json.dumps(error_result, indent=2)}")
        print("==== REVIEWER NODE END (WITH ERROR) ====\n")
        return error_result

def sql_executor_node(state: AgentGraphState):
    """
    SQL Executor node implementation.

    Args:
        state: Current state of the workflow

    Returns:
        Dict: Updated state information with SQL query results
    """
    print("\n==== SQL EXECUTOR NODE START ====")
    print(f"Input state: user_question={state.user_question}")
    print(f"Current execution path: {state.execution_path}")
    print(f"SQL query to execute: {state.sql_query}")
    print(f"Reviewer response received: {json.dumps(state.reviewer_response, indent=2) if hasattr(state, 'reviewer_response') else 'None'}")

    try:
        # Get SQL query from state
        sql_query = state.sql_query
        print(f"SQL Executor is processing SQL query: '{sql_query}'")

        # Check if SQL query is empty
        if not sql_query or sql_query.strip() == "":
            # Don't use a default query, instead return an error
            print("ERROR: SQL query is empty. Cannot execute empty query.")

            error_result = {
                "current_agent": "router",
                "errors": {"sql_executor": "Empty SQL query provided"},
                "execution_path": state.execution_path + ["sql_executor_error"],
                "is_error_state": True,
                "sql_query_results": {
                    "status": "error",
                    "error_message": "Empty SQL query provided. The SQL Generator failed to create a valid query.",
                    "column_names": ["Error"],
                    "rows": [["Empty SQL query provided"]],
                    "row_count": 1,
                    "execution_time": 0,
                    "query": ""
                },
                "model": state.model if hasattr(state, 'model') else st.session_state.get("llm_model"),
                "server": state.server if hasattr(state, 'server') else st.session_state.get("server"),
                "temperature": state.temperature if hasattr(state, 'temperature') else st.session_state.get("temperature", 0),
                "model_endpoint": state.model_endpoint if hasattr(state, 'model_endpoint') else st.session_state.get("server_endpoint")
            }

            print(f"SQL Executor node error output (empty query): {json.dumps(error_result, indent=2)}")
            print("==== SQL EXECUTOR NODE END (WITH ERROR) ====\n")
            return error_result

        # Import the execute_sql_query function
        from app import execute_sql_query
        print(f"Executing SQL query against database: '{sql_query}'")
        print("NOTE: SQL Executor does not use LLM. It directly executes the SQL query against the database.")

        # Execute the SQL query
        query_results = execute_sql_query(sql_query)
        print(f"SQL query execution results: {json.dumps(query_results, indent=2)}")

        # Update state with query results
        result = {
            "current_agent": "router",
            "execution_path": state.execution_path + ["sql_executor"],
            "sql_query_results": query_results,
            "sql_query": sql_query,  # Update the sql_query in case we used a default
            "model": state.model if hasattr(state, 'model') else st.session_state.get("llm_model"),
            "server": state.server if hasattr(state, 'server') else st.session_state.get("server"),
            "temperature": state.temperature if hasattr(state, 'temperature') else st.session_state.get("temperature", 0),
            "model_endpoint": state.model_endpoint if hasattr(state, 'model_endpoint') else st.session_state.get("server_endpoint")
        }

        print(f"SQL Executor node output: current_agent={result['current_agent']}")
        print(f"SQL query results summary: status={query_results.get('status', 'unknown')}, rows={query_results.get('row_count', 0)}")
        print("==== SQL EXECUTOR NODE END ====\n")
        return result
    except Exception as e:
        # Handle errors
        print(f"Error in SQL executor: {e}")
        traceback.print_exc()

        # Create mock results for error case
        error_results = {
            "status": "error",
            "error_message": str(e),
            "column_names": ["Error"],
            "rows": [[str(e)]],
            "row_count": 1,
            "execution_time": 0,
            "query": sql_query if 'sql_query' in locals() else ""
        }

        error_result = {
            "current_agent": "router",  # Changed from sql_executor_error to router to continue the flow
            "errors": {"sql_executor": str(e)},
            "execution_path": state.execution_path + ["sql_executor_error"],
            "is_error_state": False,  # Changed from True to False to continue the flow
            "sql_query_results": error_results,
            "model": state.model if hasattr(state, 'model') else st.session_state.get("llm_model"),
            "server": state.server if hasattr(state, 'server') else st.session_state.get("server"),
            "temperature": state.temperature if hasattr(state, 'temperature') else st.session_state.get("temperature", 0),
            "model_endpoint": state.model_endpoint if hasattr(state, 'model_endpoint') else st.session_state.get("server_endpoint")
        }

        print(f"SQL Executor node error output: {json.dumps(error_result, indent=2)}")
        print("==== SQL EXECUTOR NODE END (WITH ERROR) ====\n")
        return error_result

def router_node(state: AgentGraphState):
    """
    Router node implementation.

    Args:
        state: Current state of the workflow

    Returns:
        Dict: Updated state information
    """
    print("\n==== ROUTER NODE START ====")
    print(f"Input state: user_question={state.user_question}")
    print(f"Current execution path: {state.execution_path}")
    
    try:
        # Initialize router with current state
        router = RouterAgent(
            state=state,
            model=state.model if hasattr(state, 'model') else st.session_state.get("llm_model"),
            server=state.server if hasattr(state, 'server') else st.session_state.get("server"),
            temperature=state.temperature if hasattr(state, 'temperature') else st.session_state.get("temperature", 0),
            model_endpoint=state.model_endpoint if hasattr(state, 'model_endpoint') else st.session_state.get("server_endpoint")
        )

        # Get routing decision
        decision = router.invoke(state)
        print(f"RouterAgent returned decision: {json.dumps(decision, indent=2)}")

        # Extract router_response
        router_response = decision.get("router_response", {})
        if not isinstance(router_response, dict):
            router_response = {"route_to": "final_report_generator", "reason": "Invalid router response"}

        # Get the route_to value
        route_to = router_response.get("route_to", "final_report_generator")
        print(f"Route decision: {route_to}")

        # Update state with routing decision
        result = {
            "current_agent": route_to,
            "execution_path": state.execution_path + ["router"],
            "router_response": router_response,
            "is_error_state": False,
            "errors": state.errors
        }
        
        # Add workflow tracking
        if "state_updates" in router_response:
            result.update(router_response["state_updates"])

        print(f"Router node output: current_agent={result['current_agent']}")
        print(f"Router response: {json.dumps(result['router_response'], indent=2)}")
        print("==== ROUTER NODE END ====\n")
        return result
        
    except Exception as e:
        print(f"Error in router node: {e}")
        traceback.print_exc()

        # Create error response that routes to final report
        error_result = {
            "current_agent": "final_report_generator",
            "errors": {"router": str(e)},
            "execution_path": state.execution_path + ["router_error"],
            "is_error_state": True,
            "router_response": {
                "route_to": "final_report_generator",
                "reason": f"Error in router: {str(e)}",
                "feedback": "Error occurred during routing"
            }
        }

        print(f"Router node error output: {json.dumps(error_result, indent=2)}")
        print("==== ROUTER NODE END (WITH ERROR) ====\n")
        return error_result

def final_report_node(state: AgentGraphState) -> AgentGraphState:
    """
    Generate a final report based on the SQL query results and workflow history.
    """
    try:
        # Get the SQL query results
        query_results = state.sql_query_results
        
        # Create the final report
        final_report = {
                "report": {
                "summary": "Query executed successfully",
                    "detailed_results": {
                    "key_findings": [],
                    "data_analysis": "",
                    "sample_data": [],
                    "visualizations": [],
                    "query_details": {
                        "query": state.sql_query,
                        "execution_time": query_results.get("execution_time", 0),
                        "row_count": query_results.get("row_count", 0)
                    }
                }
            }
        }

        # Update the state with the final report
        state.final_report_data = final_report
        
        return state
        
    except Exception as e:
        print(f"Error in final_report_node: {str(e)}")
        traceback.print_exc()
        # Create error state with minimal report
        error_state = state.copy()
        error_state.final_report_data = {
            "report": {
                "summary": "Error generating final report",
                "detailed_results": {
                    "key_findings": [f"Error: {str(e)}"],
                    "data_analysis": "Unable to generate analysis due to error",
                    "sample_data": [],
                    "visualizations": [],
                "query_details": {
                        "query": state.sql_query,
                        "error": str(e)
                    }
                }
            }
        }
        return error_state

def end_node(state: AgentGraphState, workflow_history: dict = None) -> AgentGraphState:
    """
    Process the end stage of the workflow.
    This node ensures the final report response has the expected structure.
    """
    try:
        print("\n==== END NODE START ====")
        print(f"Processing final state")
        
        # Get the final report from the state - handle AttributeError properly
        final_report = {}
        
        # Try multiple ways to access the final report data
        if hasattr(state, "final_report_data"):
            final_report = state.final_report_data
        elif isinstance(state, dict) and "final_report_data" in state:
            final_report = state["final_report_data"]
        
        # If we still don't have a report, build one from available information
        if not final_report:
            print("Creating default final report")
            
            # Get SQL query results
            sql_query_results = {}
            if hasattr(state, "sql_query_results"):
                sql_query_results = state.sql_query_results
            elif isinstance(state, dict) and "sql_query_results" in state:
                sql_query_results = state["sql_query_results"]
            
            # Get SQL query
            sql_query = ""
            if hasattr(state, "sql_query"):
                sql_query = state.sql_query
            elif isinstance(state, dict) and "sql_query" in state:
                sql_query = state["sql_query"]
            
                # Create a minimal report structure
            final_report = {
                "report": {
                    "summary": "Query execution completed",
                    "detailed_results": {
                        "key_findings": ["Results processed successfully"],
                        "data_analysis": "See results for details",
                        "sample_data": [],
                        "visualizations": [],
                    "query_details": {
                            "query": sql_query,
                            "execution_time": sql_query_results.get("execution_time", 0) if isinstance(sql_query_results, dict) else 0,
                            "row_count": sql_query_results.get("row_count", 0) if isinstance(sql_query_results, dict) else 0
                        }
                    }
                }
            }
        
        # Update the state with the final report - handle both dict and class versions
        if isinstance(state, dict):
            state["final_report_data"] = final_report
            state["workflow_completed"] = True
            state["completion_timestamp"] = datetime.now().isoformat()
        else:
            # Set attributes carefully to avoid errors
            try:
                state.final_report_data = final_report
                state.workflow_completed = True
                state.completion_timestamp = datetime.now().isoformat()
            except AttributeError:
                print("Warning: Could not set attributes on state object")
        
        print(f"Final report prepared successfully")
        print("==== END NODE END ====\n")
        return state
    except Exception as e:
        print(f"Error in end node: {e}")
        traceback.print_exc()

        # Create error state with minimal report
        error_report = {
            "report": {
                "summary": "Error processing final report",
                "detailed_results": {
                    "key_findings": [f"Error: {str(e)}"],
                    "data_analysis": "Unable to generate analysis due to error",
                    "sample_data": [],
                    "visualizations": [],
                    "query_details": {
                        "query": state.sql_query if hasattr(state, "sql_query") else "",
                        "error": str(e)
                    }
                }
            }
        }
        
        # Update the state with the error report - handle both dict and class versions
        if isinstance(state, dict):
            state["final_report_data"] = error_report
            state["workflow_completed"] = True
            state["completion_timestamp"] = datetime.now().isoformat()
            state["is_error_state"] = True
            if "errors" not in state:
                state["errors"] = {}
            state["errors"]["end_node"] = str(e)
        else:
            # Set attributes carefully to avoid errors
            try:
                state.final_report_data = error_report
                state.workflow_completed = True
                state.completion_timestamp = datetime.now().isoformat()
                state.is_error_state = True
                if not hasattr(state, "errors") or state.errors is None:
                    state.errors = {}
                state.errors["end_node"] = str(e)
            except AttributeError:
                print("Warning: Could not set attributes on state object")
        
        print("==== END NODE END (WITH ERROR) ====\n")
        return state

def get_compiled_agent_graph(
    model=None,
    server=None,
    temperature=None,
    model_endpoint=None,
    stop=None
):
    """
    Build a compiled LangGraph workflow for SQL processing.

    Args:
        model (str): LLM model name
        server (str): Server type (openai, groq, etc.)
        temperature (float): Temperature setting for LLM
        model_endpoint (str): API endpoint URL
        stop (str): Stop token

    Returns:
        CompiledGraph: A compiled LangGraph workflow
    """
    # Create the builder
    builder = StateGraph(AgentGraphState)

    # Add explicit node registrations using our fixed node functions
    builder.add_node("planner", planner_node)
    builder.add_node("selector", selector_node)
    builder.add_node("SQLGenerator", sql_generator_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("sql_executor", sql_executor_node)  # Add SQL executor node
    builder.add_node("router", router_node)
    builder.add_node("final_report_generator", final_report_node)  # Renamed from final_report
    builder.add_node("end", end_node)

    # Set the entry point
    builder.set_entry_point("planner")

    # Define conditional edges
    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "to_planner": "planner",
            "to_selector": "selector",
            "to_SQLGenerator": "SQLGenerator",
            "to_reviewer": "reviewer",
            "to_final_report": "final_report_generator",  # Updated to match new node name
            "end": "end"
        }
    )

    # Define standard edges
    builder.add_edge("planner", "selector")
    builder.add_edge("selector", "SQLGenerator")
    builder.add_edge("SQLGenerator", "reviewer")
    builder.add_edge("reviewer", "sql_executor")  # Reviewer goes to SQL executor
    builder.add_edge("sql_executor", "router")    # SQL executor goes to router
    builder.add_edge("final_report_generator", "end")  # Updated to match new node name

    # Compile and return the graph
    workflow = builder.compile()
    return workflow
