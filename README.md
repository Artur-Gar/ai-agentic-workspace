# Agentic Workspace 🚀

A modular LLM operating system for data tasks that automatically routes natural language requests to specialized agents.

## Features

- **SQL Agent**: Natural language database queries 
- **Visualization Agent**: Automated chart generation using LangChain's pandas agent
- **Streamlit UI**: User-friendly web interface
- **Modular Architecture**: Easy to extend with new agents

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd agentic-workspace
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL="your-model_name-here"
```

4. Run the demo:
```bash
python main.py
```

5. Run the web interface:
```bash
streamlit run streamlit_app.py
```

Project Structure:
agentic-workspace/
│
├── agents/
│   ├── __init__.py
│   ├── sql_agent.py
│   ├── data_agent.py
│   └── visualization_agent.py
│
├── tools/
│   ├── __init__.py
│   └── data_tools.py
│
├── utils/
│   ├── __init__.py
│   └── config.py
│
├── main.py
├── streamlit_app.py
├── requirements.txt
├── .env                   
└── README.md

Usage Examples
- "Plot a histogram of customer ages"
- "Query the database for top products"
- "Show summary statistics of sales data"
- "Create a scatter plot of price vs rating"