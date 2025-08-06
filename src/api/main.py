import logging

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.utils.logging_config import setup_logging

# --- 1. SETUP ---
# Initialize logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Iris Model API",
    description="API for serving the Iris classification model.",
    version="0.1.0"
)

# Instrument the app with Prometheus metrics (for /metrics endpoint)
Instrumentator().instrument(app).expose(app)

# --- 2. MODEL LOADING ---
# Define model name and stage
MODEL_NAME = "IrisClassifier"
MODEL_STAGE = "Production"
model = None

@app.on_event("startup")
def load_model():
    """Load the model from MLflow Model Registry at startup."""
    global model
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded model '{MODEL_NAME}' version from stage '{MODEL_STAGE}'")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None # Ensure model is None if loading fails

# --- 3. API DATA MODELS (Pydantic Schemas) ---
# This provides automatic data validation and documentation
class IrisInput(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2
            }
        }
        
class PredictionOut(BaseModel):
    prediction: int
    class_name: str

# --- 4. API ENDPOINTS ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "Iris model API is running."}

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict(request: Request, data: IrisInput):
    """
    Accepts Iris flower features and returns the predicted species.
    - 0: Setosa
    - 1: Versicolor
    - 2: Virginica
    """
    if model is None:
        logger.error("Model is not loaded. Cannot make prediction.")
        return {"error": "Model not available"}

    # Log incoming request
    client_host = request.client.host
    logger.info(f"Received prediction request from {client_host}: {data.dict()}")

    # Convert input to DataFrame in the correct format
    # The feature names must match what the model was trained on
    feature_names = [
        "sepal length (cm)", "sepal width (cm)", 
        "petal length (cm)", "petal width (cm)"
    ]
    input_data = pd.DataFrame([[
        data.sepal_length_cm,
        data.sepal_width_cm,
        data.petal_length_cm,
        data.petal_width_cm
    ]], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = int(prediction[0])
    
    # Map prediction index to class name
    class_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_class_name = class_map.get(predicted_class, "Unknown")
    
    # Log the output
    logger.info(f"Prediction result: {predicted_class} ({predicted_class_name})")

    return PredictionOut(prediction=predicted_class, class_name=predicted_class_name)