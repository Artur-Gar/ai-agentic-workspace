import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

from agents.sql_agent import SQLAgent
from agents.visualization_agent import VisualizationAgent
from utils.config import Config


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


# Initialize Agents
sql_agent = SQLAgent()
viz_agent = VisualizationAgent()


# Nodes
def sql_agent_node(state: WorkspaceState):
    """Execute SQL queries"""
    print("[SQL AGENT] - Running SQL query...")
    state.current_step = "sql_agent"
    
    try:
        result = sql_agent.query(state.task)
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
    
    try:        
        result = viz_agent.visualize(state.task)
        state.results["visualization"] = result
        state.final_output += f"\n Visualization:\n{result}\n"
    except Exception as e:
        state.results["viz_error"] = str(e)
        state.final_output += f"\n Visualization Error: {str(e)}\n"
    
    state.step_idx += 1
    return state

def summary_agent_node(state: WorkspaceState):
    """Create final summary"""
    print("[SUMMARY AGENT] → Generating final report...")
    state.current_step = "summary_agent"
    
    # Create a cohesive summary from all results
    summary_parts = [f"Task: {state.task}"]
    
    if "sql_result" in state.results:
        summary_parts.append(f"Database Query Results: {state.results['sql_result'][:200]}...")
    
    
    if "visualization" in state.results:
        summary_parts.append(f"Visualizations Created: Check generated charts and plots")
    
    # Add any errors
    errors = [f"{key}: {value}" for key, value in state.results.items() if "error" in key]
    if errors:
        summary_parts.append("Issues encountered: " + "; ".join(errors))
    
    state.final_output = "\n\n".join(summary_parts)
    state.step_idx += 1
    return state


# Router Node (Intelligent Agent Selection)
def router_node(state: WorkspaceState):
    print("[ROUTER] - Planning agent sequence...")

    router_llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=0.1)

    # 👇 Prepare the memory text
    memory_text = ""
    if state.history:
        memory_text = "\nRecent conversation (last 10 turns):\n"
        for m in state.history[-10:]:
            memory_text += f"User: {m['user']}\nAssistant: {m['assistant']}\n"

    prompt = f"""
    You are an intelligent router for a data workspace. Analyze the user's task and determine which agents to use.
    {memory_text}

    Available Agents:
    - sql_agent: For database queries, SQL operations, data retrieval from databases
    - visualization_agent: For creating charts, plots, graphs, visualizations
    - summary_agent: For creating final summaries (always include this last)

    User Task: "{state.task}"

    Return ONLY a JSON object with this structure:
    {{
        "plan": ["agent1", "agent2", ..., "summary_agent"]
    }}
    """

    try:
        response = router_llm.invoke(prompt)
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
        self.task_history = []
        self.memory = []   # rolling chat memory

    def _update_memory(self, user_input: str, agent_output: str):
        """Maintain the last 10 exchanges"""
        self.memory.append({"user": user_input, "assistant": agent_output})
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

    def process_task(self, task: str) -> Dict[str, Any]:
        print(f"\n Processing task: {task}")
        self.task_history.append(task)

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

            # 👈 Update rolling memory
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


################################## DEMO
def main():
    """Demo the agentic workspace with LangGraph orchestration"""
    workspace = AgenticWorkspace()
    
    print("🚀 Agentic Workspace: Graph-Based Orchestration")
    print("=" * 60)
    
    # Demo tasks that will trigger different agent sequences
    demo_tasks = [
        "Query the database for all customers and create a bar chart of their countries",
        "I meant not considering US",
        #"Load student data and plot a histogram of final grades",
        #"Show me summary statistics of available CSV files",
        #"Create a scatter plot of study time vs final grades from the student dataset",
        #"What tables are available in the database?",
    ]
    
    for i, task in enumerate(demo_tasks, 1):
        print(f"\n{'='*50}")
        print(f"Task {i}: {task}")
        print('='*50)
        
        result = workspace.process_task(task)
        
        print(f"🤖 Agent Plan: {result['agent_plan']}")
        print(f"📊 Final Output: {result['final_output']}")
        print(f"✅ Success: {result['success']}")

if __name__ == "__main__":
    main()