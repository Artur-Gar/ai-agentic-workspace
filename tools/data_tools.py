import pandas as pd
import glob
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from utils.config import Config

@tool
def list_csv_files() -> Optional[List[str]]:
    """List all CSV file names in the current directory"""
    csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
    if not csv_files:
        return None
    return [os.path.basename(file) for file in csv_files]

@tool
def preload_datasets(paths: List[str]) -> str:
    """Load CSV files into cache for efficient processing"""
    loaded = []
    cached = []
    for path in paths:
        if path not in Config.DATAFRAME_CACHE:
            Config.DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    
    return (
        f"Loaded datasets: {loaded}\n"
        f"Already cached: {cached}"
    )

@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple CSV files and return metadata summaries"""
    summaries = []

    for path in dataset_paths:
        if path not in Config.DATAFRAME_CACHE:
            Config.DATAFRAME_CACHE[path] = pd.read_csv(path)
        
        df = Config.DATAFRAME_CACHE[path]

        summary = {
            "file_name": path,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "shape": df.shape
        }

        summaries.append(summary)

    return summaries

@tool
def call_dataframe_method(file_name: str, method: str) -> str:
    """Execute pandas DataFrame methods like head, describe, info"""
    if file_name not in Config.DATAFRAME_CACHE:
        try:
            Config.DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return f"DataFrame '{file_name}' not found in cache or on disk."
        except Exception as e:
            return f"Error loading '{file_name}': {str(e)}"
   
    df = Config.DATAFRAME_CACHE[file_name]
    func = getattr(df, method, None)
    if not callable(func):
        return f"'{method}' is not a valid method of DataFrame."
    try:
        result = func()
        return str(result)
    except Exception as e:
        return f"Error calling '{method}' on '{file_name}': {str(e)}"