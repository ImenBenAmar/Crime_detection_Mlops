import os
import sys
import pickle
import numpy as np
from dotenv import load_dotenv
import mlflow
import dagshub
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
from typing import Optional

# Ensure we can import from backend/src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Feature Store class
from feature_store import CrimeFeatureStore

load_dotenv()

# --- Config ---
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME')
EXPERIMENT_NAME = "Crime_MLOPS1"

# Global state
ml_components = {
    "model": None, 
    "store": None, 
    "model_name": "Unknown"
}

# --- Pydantic Schemas ---
class CrimeInput(BaseModel):
    # Field aliases allow the user to send keys like "DATE OCC" (from CSV)
    date_occ: str = Field(..., alias="DATE OCC")
    time_occ: int = Field(..., alias="TIME OCC")
    area: int = Field(..., alias="AREA")
    # Optional fields (might be missing in input)
    rpt_dist_no: Optional[int] = Field(None, alias="Rpt Dist No")
    part_1_2: Optional[int] = Field(None, alias="Part 1-2")
    crm_cd: Optional[int] = Field(None, alias="Crm Cd")
    mocodes: Optional[str] = Field(None, alias="Mocodes")
    vict_age: Optional[float] = Field(None, alias="Vict Age")
    vict_sex: Optional[str] = Field(None, alias="Vict Sex")
    vict_descent: Optional[str] = Field(None, alias="Vict Descent")
    premis_cd: Optional[float] = Field(None, alias="Premis Cd")
    premis_desc: Optional[str] = Field(None, alias="Premis Desc")
    weapon_used_cd: Optional[float] = Field(None, alias="Weapon Used Cd")
    weapon_desc: Optional[str] = Field(None, alias="Weapon Desc")
    status: Optional[str] = Field(None, alias="Status")
    location: Optional[str] = Field(None, alias="LOCATION")
    lat: Optional[float] = Field(None, alias="LAT")
    lon: Optional[float] = Field(None, alias="LON")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float]
    model_info: str

# --- Helper Functions ---

def setup_mlflow():
    """Configures MLflow tracking uri and authentication."""
    
    # Load variables
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_TOKEN')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')
    # The repo owner might be different from the user running the script
    # If YomnaJL owns the repo, hardcode it or add a new env var
    repo_owner = "YomnaJL" 

    if all([username, token, repo_name]):
        # Set auth variables for MLflow
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        
        # Construct Tracking URI
        mlflow_tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        
        # Initialize DagsHub (handles auth under the hood)
        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        except Exception as e:
            print(f"⚠️ dagshub.init failed: {e}. Falling back to manual config.")
        
        # Set URI explicitly
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"✅ MLflow connected: {mlflow_tracking_uri}")
    else:
        print("⚠️ Missing DagsHub credentials in .env. MLflow loading might fail.")

def load_best_model():
    """Loads best model from MLflow based on F1 Score."""
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
            return None, None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_weighted DESC"],
            max_results=1
        )
        if runs.empty:
            print("❌ No runs found.")
            return None, None
        
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        model_name = best_run['tags.model_name']
        stage = best_run['tags.stage']
        
        print(f"🏆 Best Model: {model_name} ({stage}) - F1: {best_run['metrics.f1_weighted']:.4f}")

        # Construct filename based on training script convention
        artifact_path = f"{model_name}_{stage}.pkl"
        
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        
        with open(local_path, 'rb') as f:
            model = pickle.load(f)
            
        return model, f"{model_name}_{stage}"
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        return None, None

# --- Lifecycle Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚙️ Initializing API...")
    
    # 1. Setup Environment
    setup_mlflow()
    
    # 2. Initialize Feature Store (loads encoders/scaler)
    store = CrimeFeatureStore(processors_path="processors")
    store.load_artifacts()
    ml_components["store"] = store
    
    # 3. Load Model
    model, name = load_best_model()
    if model:
        ml_components["model"] = model
        ml_components["model_name"] = name
        print("🚀 Server Ready.")
    else:
        print("⚠️ No model loaded. Endpoints will fail.")
        
    yield
    print("🛑 Shutting down.")

# --- FastAPI App ---
app = FastAPI(title="Crime API", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Crime Prediction API Running"}

@app.get("/health")
def health():
    if not ml_components["model"]:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy", "model": ml_components["model_name"]}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: CrimeInput):
    if not ml_components["model"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        # Convert Pydantic object to dict
        data = payload.model_dump(by_alias=True)
        
        # 1. Feature Store Processing
        # This handles cleaning, defaults for missing cols, encoding, and scaling
        store = ml_components["store"]
        X_input = store.get_online_features(data)
        
        # 2. Prediction
        model = ml_components["model"]
        pred_idx = model.predict(X_input)[0]
        
        # 3. Decoding
        pred_label = store.decode_target(pred_idx)
        
        # 4. Confidence (if supported)
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            confidence = float(np.max(probs))
            
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "model_info": ml_components["model_name"]
        }
    except Exception as e:
        # Print error to console for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)