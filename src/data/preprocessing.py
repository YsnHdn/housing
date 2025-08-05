"""Basic data preprocessing for housing data."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import CONFIG

def preprocess_data():
    """Load and preprocess the housing data."""
    # Load raw data
    raw_data_path = Path(CONFIG['data']['raw_data_path'])
    df = pd.read_csv(raw_data_path / "california_housing.csv")
    
    # Basic info
    print(f"Original data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Split features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Train-test split
    test_size = CONFIG['data']['dataset']['test_size']
    random_state = CONFIG['data']['dataset']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save processed data
    processed_path = Path(CONFIG['data']['processed_data_path'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(processed_path / "X_train.csv", index=False)
    X_test.to_csv(processed_path / "X_test.csv", index=False)
    y_train.to_csv(processed_path / "y_train.csv", index=False)
    y_test.to_csv(processed_path / "y_test.csv", index=False)
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print("Data preprocessed and saved!")

if __name__ == "__main__":
    preprocess_data()