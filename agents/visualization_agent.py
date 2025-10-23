import pandas as pd
import json

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from utils.config import Config

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
        prompt = ChatPromptTemplate.from_template(
                """
                The following text with a raw text SQL query output.
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
        
        try:    
            response = self.llm.invoke(formatted)

            # Clean up and parse
            json_text = response.content.strip().strip('`')
            data = json.loads(json_text)

            schema = TableSchema(**data)
            df = pd.DataFrame(schema.rows, columns=schema.columns)

            self.current_dataframe = df
            return {"output": sql_text}
        
        except Exception as e:
            return {"output": f"Error converting SQL output: {str(e)}"}
    
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
            
            return output
            
        except Exception as e:
            return f"Error creating visualization: {str(e)}"

    # LCEL PIPELINE
    def run_whole_pipeline(self, question: str):
        convert_stage = RunnableLambda(lambda x: self.from_sql_output(x["query"]))
        visualize_stage = RunnableLambda(lambda x: self.visualize(x["output"]))
        
        universal_chain = (
            convert_stage
            | visualize_stage
        )

        result = universal_chain.invoke({
            "query": question
        })

        return  result