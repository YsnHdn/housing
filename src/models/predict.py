"""Load trained model and make predictions."""

import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import CONFIG

def load_latest_model():
    """Load the latest model from MLflow."""
    mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
    
    # Get latest run from experiment
    experiment = mlflow.get_experiment_by_name(CONFIG['mlflow']['experiment_name'])
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        raise ValueError("No trained models found!")
    
    # Get best run (lowest MAE)
    best_run = runs.loc[runs['metrics.mae'].idxmin()]
    
    # Load model
    model_uri = f"runs:/{best_run.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    print(f"Loaded model from run: {best_run.run_id}")
    print(f"Model MAE: {best_run['metrics.mae']:.2f}")
    
    return model

def predict_price(model, features):
    """Make prediction for house features."""
    # Convert to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    
    prediction = model.predict(features)
    return prediction[0] if len(prediction) == 1 else prediction

if __name__ == "__main__":
    # Test prediction
    model = load_latest_model()
    
    # Sample house features
    sample_house = {
        'MedInc': 8.3252,
        'HouseAge': 41.0,
        'AveRooms': 6.984,
        'AveBedrms': 1.023,
        'Population': 322.0,
        'AveOccup': 2.555,
        'Latitude': 37.88,
        'Longitude': -122.23
    }
    
    price = predict_price(model, sample_house)
    print(f"Predicted price: ${price * 100000:.2f}")