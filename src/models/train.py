"""Train models with MLflow tracking."""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import CONFIG

def load_processed_data():
    """Load preprocessed training data."""
    processed_path = Path(CONFIG['data']['processed_data_path'])
    
    X_train = pd.read_csv(processed_path / "X_train.csv")
    y_train = pd.read_csv(processed_path / "y_train.csv").squeeze()
    X_test = pd.read_csv(processed_path / "X_test.csv")
    y_test = pd.read_csv(processed_path / "y_test.csv").squeeze()
    
    return X_train, X_test, y_train, y_test

def train_models():
    """Train models and log with MLflow."""
    # Setup MLflow
    mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
    mlflow.set_experiment(CONFIG['mlflow']['experiment_name'])
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Models to train
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(
            n_estimators=CONFIG['model']['random_forest']['n_estimators'],
            random_state=CONFIG['model']['random_forest']['random_state']
        )
    }
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5  # Calculate RMSE manually
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

if __name__ == "__main__":
    train_models()