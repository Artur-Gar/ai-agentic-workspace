from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from utils.config import Config

class SQLAgent:
    """SQL query agent for database operations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.agent = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Initialize SQL agent with database connection"""
        try:
            mysql_uri = f'mysql+mysqlconnector://{Config.DB_USERNAME}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}'
            db = SQLDatabase.from_uri(mysql_uri)
            
            self.agent = create_sql_agent(
                llm=self.llm,
                db=db,
                verbose=False,
                handle_parsing_errors=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            )
        except Exception as e:
            print(f"SQL agent setup failed: {e}")
    
    def query(self, question: str) -> str:
        """Execute SQL query using natural language"""
        if not self.agent:
            return "SQL agent not available. Check database connection."
        
        try:
            result = self.agent.invoke(question)
            return result.get('output', str(result))
        except Exception as e:
            return f"Error executing query: {str(e)}"