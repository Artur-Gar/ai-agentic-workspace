import pandas as pd
import json
from typing import List, Any
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

from utils.config import Config


class TableSchema(BaseModel):
    columns: List[str]
    rows: List[List[Any]]


class SQLAgent:
    """SQL query agent for database operations"""
    
    def __init__(self, orchestrator=None):
        """orchestrator, optional - Central orchestrator instance for sharing state"""
        self.orchestrator = orchestrator
        self.llm = ChatOpenAI(
            model=Config.SQL_OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.agent = None
        self.db = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Initialize SQL agent with database connection"""
        try:
            mysql_uri = (
                f'mysql+mysqlconnector://{Config.DB_USERNAME}:{Config.DB_PASSWORD}'
                f'@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}'
            )
            self.db = SQLDatabase.from_uri(mysql_uri)
            
            self.agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                verbose=False,
                handle_parsing_errors=True,
                agent_type=AgentType.OPENAI_FUNCTIONS
            )
        except Exception as e:
            print(f"SQL agent setup failed: {e}")

    def query(self, question: str) -> str:
        """Execute SQL query using natural language"""
        prompt = ChatPromptTemplate.from_template(
            """
            Given the user request, generate and execute a SQL query. 
            Do NOT try to generate graphs, charts or any other kind of visualisation.
            
            User request: {question}
            """
        )
        formatted = prompt.format(question=question)
        
        try:
            result = self.agent.invoke(formatted)
            output = result.get('output', str(result))

            return {"output": output}
        
        except Exception as e:
            return {"output": f"Error executing query: {str(e)}"}

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
            Do NOT try to generate graphs, charts or any other kind of visualisation.

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

            self.orchestrator.set_dataframe(df)

            return {"output": sql_text}
        
        except Exception as e:
            return {"output": f"Error converting SQL output: {str(e)}"}


    # LCEL PIPELINE
    def run_whole_pipeline(self, query: str):
        sql_call_stage = RunnableLambda(lambda x: self.query(question=x["query"]))
        convert_stage = RunnableLambda(lambda x: self.from_sql_output(sql_text=x["output"]))
        
        universal_chain = (
            sql_call_stage
            | convert_stage
        )

        result = universal_chain.invoke({
            "query": query
        })

        output = result.get('output', str(result))

        return  {"output": output}