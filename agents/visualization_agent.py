import matplotlib.pyplot as plt
import os
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate

from utils.config import Config

# Directory for storing plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


class VisualizationAgent:
    """Visualization agent using pandas dataframe agent"""
    
    def __init__(self, orchestrator=None):
        """orchestrator, optional - Central orchestrator instance for sharing state"""
        self.orchestrator = orchestrator
        self.llm = ChatOpenAI(
            model=Config.VISUALISATION_OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
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
                allow_dangerous_code=True,  # needed for matplotlib code execution
                max_execution_time=60,
                return_intermediate_steps=True,
                handle_parsing_errors=True
            )

            prompt = ChatPromptTemplate.from_template(
                """
                Execute the user request, using tools from pandas, matplotlib or seaborn packages. 
                As for the data source use ONLY provided DataFrame.
                If a plot is required - build it.
                
                User request: {question}
                """
            )
            formatted = prompt.format(question=question)
            
            result = agent.invoke(formatted)
            
            # Extract both the output and any generated code
            output = result.get('output', str(result))

            # Automatically save any figure after execution
            if plt.get_fignums():
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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

