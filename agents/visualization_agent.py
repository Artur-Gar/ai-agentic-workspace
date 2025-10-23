import pandas as pd
import json
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from utils.config import Config

import matplotlib.pyplot as plt
import io
import contextlib

from typing import List, Any
from pydantic import BaseModel, ValidationError

class TableSchema(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

class VisualizationAgent:
    """Visualization agent using pandas dataframe agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.current_dataframe = None

    def from_sql_output(self, sql_text: str) -> str:
        """Convert raw SQLAgent text output into a pandas DataFrame using LLM for CSV conversion"""
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                The following text is a raw SQL query output.
                Convert ONLY the tabular data portion into valid JSON with this structure:
                {{
                    "columns": ["col1", "col2", ...],
                    "rows": [
                        [val11, val12, ...],
                        [val21, val22, ...]
                    ]
                }}

                Ensure column names are meaningful (based on the SQL output context).
                Do NOT include explanations or code fencing — output ONLY the JSON.

                SQL Output:
                ```
                {sql_text}
                ```
                """
            )
            
            formatted = prompt.format(sql_text=sql_text)
            response = self.llm.invoke(formatted)

            # Clean up and parse
            json_text = response.content.strip().strip('`')
            data = json.loads(json_text)

            try:
                schema = TableSchema(**data)
                df = pd.DataFrame(schema.rows, columns=schema.columns)

                self.current_dataframe = df
                return f"✅ Converted SQL output into DataFrame with columns: {df.columns.tolist()} (shape {df.shape})."
            except Exception:
                return "⚠️ LLM did not return valid CSV format."
        
        except Exception as e:
            return f"Error converting SQL output: {str(e)}"
    
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
                max_execution_time=30,
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
                    output += f"\n\nGenerated Code:\n{code_used}"
            
            ## Try executing generated code if it contains matplotlib
            #if code_used and "plt" in code_used:
            #    try:
            #        local_vars = {"df": self.current_dataframe, "plt": plt, "pd": pd}
            #        with contextlib.redirect_stdout(io.StringIO()):
            #            exec(code_used, {}, local_vars)
            #        plt.show()
            #    except Exception as plot_error:
            #        output += f"\n Could not execute plot code: {plot_error}"
            
            return output
            
        except Exception as e:
            return f"Error creating visualization: {str(e)}"