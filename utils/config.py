import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the Agentic Workspace"""
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    PLANNER_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    SQL_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    VISUALISATION_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = os.getenv("TEMPERATURE", 0.1)
    MAX_TOKENS = os.getenv("MAX_TOKENS", 1024)
    
    # Database settings
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")