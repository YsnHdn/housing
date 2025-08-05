import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.datasets import fetch_california_housing
from src.utils.config import CONFIG

def load_california_housing():
    """Load the California housing dataset from sklearn."""
    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    
    # Create DataFrame
    df = housing.frame
    
    # Save to raw data folder
    raw_data_path = Path(CONFIG['data']['raw_data_path'])
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(raw_data_path / "california_housing.csv", index=False)
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Saved to: {raw_data_path / 'california_housing.csv'}")
    
    return df

if __name__ == "__main__":
    load_california_housing()