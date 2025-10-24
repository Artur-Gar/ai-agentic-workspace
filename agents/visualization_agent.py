from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from utils.config import Config


# Directory for storing plots
import matplotlib.pyplot as plt
import os
import datetime
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


class VisualizationAgent:
    """Visualization agent using pandas dataframe agent"""
    
    def __init__(self, orchestrator=None):
        """orchestrator, optional - Central orchestrator instance for sharing state"""
        self.orchestrator = orchestrator
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
    
    def visualize(self, question: str) -> str:
        """Create visualizations using pandas dataframe agent"""
        df = None
        
        if self.orchestrator:
            df = self.orchestrator.get_dataframe()

        if df is None:
            return {"output": "No data loaded. Please load data first using load_data()."}

        try:
            # Create agent with the current dataframe
            agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=df,
                verbose=False,
                allow_dangerous_code=True,  # Required for matplotlib code execution
                max_execution_time=30,
                return_intermediate_steps=True,
                handle_parsing_errors=True
            )
            
            result = agent.invoke(question+". If a plot is buils - save final one")
            
            # Extract both the output and any generated code
            output = result.get('output', str(result))

            # Automatically save any figure after execution
            if plt.get_fignums():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(PLOTS_DIR, f"auto_saved_{timestamp}.png")
                plt.savefig(filepath, bbox_inches="tight")
                plt.close("all")
                output += f"\n\n✅ Auto-saved plot at: {filepath}"

            intermediate_steps = result.get('intermediate_steps', [])
            
            # If there are intermediate steps, we can extract the code used
            if intermediate_steps:
                last_step = intermediate_steps[-1]
                if hasattr(last_step[0], 'tool_input'):
                    code_used = last_step[0].tool_input
                    output += f"\n\nGenerated Code:\n{code_used}"
            
            return {"output": output}
            
        except Exception as e:
            return {"output": f"Error creating visualization: {str(e)}"} 

