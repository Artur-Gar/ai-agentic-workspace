from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.data_tools import list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method
from utils.config import Config

class DataAgent:
    """Data analysis agent for CSV operations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.agent = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Initialize data analysis agent with tools"""
        tools = [list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis assistant. Use available tools to analyze datasets.
            Your tasks include loading data, describing distributions, providing insights about data structure.
            Be concise and focus on actionable insights."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    
    def analyze(self, question: str) -> str:
        """Analyze data using natural language"""
        try:
            result = self.agent.invoke({"input": question})
            return result.get('output', str(result))
        except Exception as e:
            return f"Error in data analysis: {str(e)}"