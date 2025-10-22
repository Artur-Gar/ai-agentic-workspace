import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from utils.config import Config

class VisualizationAgent:
    """Visualization agent using pandas dataframe agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.current_dataframe = None
    
    def load_data(self, data_source: str) -> str:
        """Load data from file path"""
        try:
            self.current_dataframe = pd.read_csv(data_source)
            
            # Cache the dataframe
            Config.DATAFRAME_CACHE[data_source] = self.current_dataframe
            
            return f"Data loaded successfully: {self.current_dataframe.shape[0]} rows, {self.current_dataframe.shape[1]} columns"
        except Exception as e:
            return f"Error loading data: {str(e)}"
    
    def visualize(self, question: str) -> str:
        """Create visualizations using pandas dataframe agent"""
        if self.current_dataframe is None:
            return "No data loaded. Please load data first using load_data()."
        
        try:
            # Create agent with the current dataframe
            agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.current_dataframe,
                verbose=False,
                allow_dangerous_code=True,  # Required for matplotlib code execution
                return_intermediate_steps=True
            )
            
            result = agent.invoke(question)
            
            # Extract both the output and any generated code
            output = result.get('output', '')
            intermediate_steps = result.get('intermediate_steps', [])
            
            # If there are intermediate steps, we can extract the code used
            if intermediate_steps:
                last_step = intermediate_steps[-1]
                if hasattr(last_step[0], 'tool_input'):
                    code_used = last_step[0].tool_input
                    output += f"\n\nGenerated Code:\n```python\n{code_used}\n```"
            
            return output
            
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set a pandas dataframe directly"""
        self.current_dataframe = dataframe