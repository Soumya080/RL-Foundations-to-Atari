import os
import json
import csv
import logging
import numpy as np

def setup_logger(name="REINFORCE-Framework", log_file=None, level=logging.INFO):
    """
    Configure and return a standard Logger object that writes to stdout and optionally to a log file.
    """
    logger = logging.getLogger(name)
    # Check if logger is already configured to prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def save_json_results(data, filepath):
    """
    Save dictionary or list data to a JSON file. Handles NumPy types automatically.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Custom encoder to serialize numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)

def load_json_results(filepath):
    """
    Load data from a JSON file.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_csv_trajectory(headers, rows, filepath):
    """
    Save trajectories or lists of rows to a CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
