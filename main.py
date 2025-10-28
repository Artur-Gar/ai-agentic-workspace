import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import json

from agents.sql_agent import SQLAgent
from agents.visualization_agent import VisualizationAgent
from utils.orchestrator import Orchestrator
from utils.config import Config


### Helpers
def format_memory(history: List[Dict[str, str]], max_turns: int = 10, max_assistant_chars: int = 600) -> str:
    """Render last N exchanges into a compact, model-friendly string."""
    if not history:
        return "No recent context."
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        user = (m.get("user") or "").strip()
        assistant = (m.get("assistant") or "").strip()
        if len(assistant) > max_assistant_chars:
            assistant = assistant[:max_assistant_chars] + "..."
        lines.append(f"User: {user}\nAssistant: {assistant}")
    return "\n\n".join(lines)


def augment_task_with_memory(task: str, memory_text: str, extra_guidance: str = "") -> str:
    """Compose an augmented instruction that includes memory context without changing agent APIs."""
    guidance = extra_guidance.strip()
    guidance_block = f"\n\n### guidance\n{guidance}" if guidance else ""
    return (
        f"{task.strip()}\n\n"
        f"### relevant recent context\n{memory_text}"
        f"{guidance_block}\n\n"
        f"### instruction\n"
        f"If this request refines or modifies a previous result (e.g., filters, exclusions, formatting changes), "
        f"reuse the prior context and behave incrementally rather than starting from scratch."
    )


# Shared State Definition
class WorkspaceState(BaseModel):
    """State for the agentic workspace workflow"""
    task: str
    current_step: str = ""
    agent_plan: List[str] = []
    results: Dict[str, Any] = {}
    final_output: str = ""
    current_dataframe: Any = None
    step_idx: int = 0
    history: List[Dict[str, str]] = []


# Create orchestrator
orchestrator = Orchestrator()

# Initialize agents with orchestrator
sql_agent = SQLAgent(orchestrator)
viz_agent = VisualizationAgent(orchestrator)


# Nodes
def sql_agent_node(state: WorkspaceState):
    """Execute SQL queries"""
    print("[SQL AGENT] - Running SQL query...")
    state.current_step = "sql_agent"

    memory_text = format_memory(state.history)
    sql_guidance = (
        """
        Prefer reusing table names, filters, or joins implied by the recent context. 
        If the new request is a refinement (e.g., 'exclude US'), apply it as an additional WHERE clause. 
        Return concise results or a brief summary if result is very large.
        """
    )
    sql_task = augment_task_with_memory(state.task, memory_text, sql_guidance)

    try:
        result = sql_agent.run_whole_pipeline(sql_task)
        result = result.get('output', str(result))
        state.results["sql_result"] = result
        state.final_output += f"\n SQL Results:\n{result}\n"
    except Exception as e:
        state.results["sql_error"] = str(e)
        state.final_output += f"\n SQL Error: {str(e)}\n"
    
    state.step_idx += 1
    return state

def visualization_agent_node(state: WorkspaceState):
    """Create visualizations"""
    print("[VISUALIZATION AGENT]  Creating visualizations...")
    state.current_step = "visualization_agent"
    
    memory_text = format_memory(state.history)
    viz_guidance = (
        "If the previous step or context referenced a specific plot, modify that chart accordingly "
        "(e.g., filter out a country, add labels, change axis). "
        "Use the same column mappings as before unless instructed otherwise. "
        "If code is generated, ensure it executes quickly and renders in the current environment."
    )
    viz_task = augment_task_with_memory(state.task, memory_text, viz_guidance)

    try:        
        result = viz_agent.visualize(viz_task)
        result = result.get('output', str(result))
        state.results["visualization_result"] = result
        state.final_output += f"\n Visualization Results:\n{result}\n"
    except Exception as e:
        state.results["viz_error"] = str(e)
        state.final_output += f"\n Visualization Error: {str(e)}\n"
    
    state.step_idx += 1
    return state

def summary_agent_node(state: WorkspaceState):
    """Create final summary"""
    print("[SUMMARY AGENT] - Generating final report...")
    state.current_step = "summary_agent"
    
    # Create a cohesive summary from all results
    summary_parts = ["Summary:"]
    
    if "sql_result" in state.results:
        summary_parts.append(f"Database Query Results: {state.results['sql_result'][:500]}...")
    
    if "visualization_result" in state.results:
        summary_parts.append(f"Visualizations Results: {state.results['visualization_result'][:60]}...")
    
    # Add any errors
    errors = [f"{key}: {value}" for key, value in state.results.items() if "error" in key]
    if errors:
        summary_parts.append("Issues encountered: " + "; ".join(errors))
    
    state.final_output = "\n\n".join(summary_parts)
    state.step_idx += 1
    return state


# Router Node (Agent Selection)
def router_node(state: WorkspaceState):
    print("[ROUTER] - Planning agent sequence...")

    router_llm = ChatOpenAI(
        model=Config.PLANNER_OPENAI_MODEL, 
        api_key=Config.OPENAI_API_KEY,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an intelligent router for a modular data workspace.

        Your goal is to analyze the user's current request and decide
        which agents should run next. Always consider both the new task
        and the recent conversation history to maintain continuity.

        ### Available Agents
        - sql_agent — for database queries, SQL operations, and data retrieval.
        - visualization_agent — for creating charts, plots, graphs, and visualizations.
        - summary_agent — for summarizing final outputs (always include this last).

        ### Conversation Memory
        The following is the last 10 turns of interaction between the user and the system:
        {memory_text}

        Use this memory to:
        - Detect when the new task refers to previous data or visualizations.
        - Update or refine previous results (e.g., "exclude US", "add labels", "filter last chart").
        - Reuse context instead of starting from scratch.

        ### Current User Task
        {user_task}

        ### Output Format
        Return ONLY valid JSON in this structure:
        {{
            "plan": ["agent1", "agent2", ..., "summary_agent"]
        }}
        """
    )
    formatted = prompt.format(memory_text=state.history, user_task=state.task)

    try:
        response = router_llm.invoke(formatted)
        plan_data = json.loads(response.content)
        state.agent_plan = plan_data["plan"]
        print(f"[ROUTER] - Planned sequence: {state.agent_plan}")
    except Exception as e:
        print(f"[ROUTER] - Using fallback plan due to error: {e}")

    return state


# Conditional Edge Logic
def decide_next_step(state: WorkspaceState) -> str:
    """Determine the next node to execute based on the plan"""
    if state.step_idx < len(state.agent_plan):
        next_agent = state.agent_plan[state.step_idx]
        print(f"[DECIDER] - Next agent: {next_agent}")
        return next_agent
    else:
        return "end"


# Build Graph
def create_workflow():
    """Create and compile the LangGraph workflow"""
    graph = StateGraph(WorkspaceState)
    
    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("visualization_agent", visualization_agent_node)
    graph.add_node("summary_agent", summary_agent_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Define conditional edges from router
    agent_edges = {
    "sql_agent": "sql_agent",
    "visualization_agent": "visualization_agent",
    "summary_agent": "summary_agent",
    "end": END
    }
    
    # Define edges between agent nodes
    for node in ["router", "sql_agent", "visualization_agent", "summary_agent"]:
        graph.add_conditional_edges(node, decide_next_step, agent_edges)
    
    return graph.compile()


# Main Workspace Class
class AgenticWorkspace:
    """Main workspace with LangGraph orchestration"""

    def __init__(self):
        self.workflow = create_workflow()
        self.memory = []   # rolling chat memory

    def _update_memory(self, user_input: str, agent_output: str):
        """Maintain the last 10 exchanges"""
        self.memory.append({"user": user_input, "assistant": agent_output})
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

    def process_task(self, task: str) -> Dict[str, Any]:
        print(f"\n Processing task: {task}")

        try:
            # Inject memory into the workflow state
            initial_state = WorkspaceState(task=task, history=self.memory)
            final_state_dict = self.workflow.invoke(initial_state)
            final_state = WorkspaceState(**final_state_dict)

            output = {
                "task": task,
                "agent_plan": final_state.agent_plan,
                "final_output": final_state.final_output,
                "results": final_state.results,
                "success": True
            }

            # Update rolling memory
            self._update_memory(task, final_state.final_output)

            return output

        except Exception as e:
            return {
                "task": task,
                "agent_plan": [],
                "final_output": f"Error processing task: {str(e)}",
                "results": {},
                "success": False
            }