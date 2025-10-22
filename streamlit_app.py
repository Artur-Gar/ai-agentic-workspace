import streamlit as st
import pandas as pd
from main import AgenticWorkspace

st.set_page_config(page_title="Agentic Workspace", page_icon="🔧", layout="wide")

st.title("🔧 Agentic Workspace")
st.subheader("Modular LLM Operating System for Data Tasks")

# Initialize workspace
if 'workspace' not in st.session_state:
    st.session_state.workspace = AgenticWorkspace()
    st.session_state.task_history = []

# Sidebar
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    import os
    os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.header("Quick Actions")
if st.sidebar.button("📋 View Task History"):
    if st.session_state.task_history:
        for i, task in enumerate(st.session_state.task_history, 1):
            st.sidebar.write(f"{i}. {task}")
    else:
        st.sidebar.info("No tasks yet")

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("New Task")
    task_input = st.text_area(
        "Enter your data task:",
        placeholder="e.g., 'Create a bar plot of sales by category', 'Query the database for top customers', 'Show summary statistics'",
        height=100
    )
    
    if st.button("Execute Task", type="primary"):
        if task_input:
            with st.spinner("🤖 Processing task..."):
                result = st.session_state.workspace.process_task(task_input)
                st.session_state.task_history.append(task_input)
                
                st.subheader("Results")
                st.write(f"**Agent Used:** {result['agent_used']}")
                st.write(f"**Success:** {result['success']}")
                
                st.text_area("Output", result['output'], height=300)
        
        else:
            st.warning("Please enter a task")

with col2:
    st.header("Sample Tasks")
    
    sample_tasks = [
        "List CSV files",
        "Show database tables", 
        "Load student data",
        "Plot gender distribution",
        "Create grade histogram",
        "Summary statistics"
    ]
    
    for task in sample_tasks:
        if st.button(f"🚀 {task}"):
            st.session_state.last_sample = task
            st.rerun()

# Handle sample task execution
if 'last_sample' in st.session_state:
    task_map = {
        "List CSV files": "List all available CSV files in the directory",
        "Show database tables": "What are the tables available in the database?",
        "Load student data": "Load the student performance dataset and show me the first 5 rows",
        "Plot gender distribution": "Create a bar plot showing gender distribution",
        "Create grade histogram": "Generate a histogram of final grades", 
        "Summary statistics": "What are the summary statistics of the dataset?"
    }
    
    task_text = task_map.get(st.session_state.last_sample)
    if task_text:
        with st.spinner(f"Executing: {st.session_state.last_sample}"):
            result = st.session_state.workspace.process_task(task_text)
            st.session_state.task_history.append(task_text)
            
            st.subheader("Results")
            st.write(f"**Agent Used:** {result['agent_used']}")
            st.text_area("Output", result['output'], height=300)
    
    # Clear the sample task
    del st.session_state.last_sample

# Footer
st.markdown("---")
st.markdown("Built with LangChain + OpenAI + Streamlit")