import os
import sys
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import mlflow
import dagshub
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

# ==========================================
# 0. PATH SETUP & IMPORT PREPROCESSING
# ==========================================
# Ensure we can import from backend/src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import preprocessing as pp
except ImportError:
    raise ImportError("Could not import 'preprocessing.py'. Ensure it exists in ../src/")

# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()

DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME')
EXPERIMENT_NAME = "Crime_MLOPS1"

PROCESSORS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "processors")

# Global state storage
ml_components = {
    "model": None,
    "target_encoder": None,
    "target_mapping": None,  # Fallback for older artifacts
    "feature_encoders": None,
    "scaler": None,
    "config": None
}

# ==========================================
# 2. PYDANTIC SCHEMAS (Data Validation)
# ==========================================

class CrimeInput(BaseModel):
    """
    Defines the expected input schema. 
    Uses aliases so users can send JSON keys matching the Raw CSV headers (e.g., "Vict Sex").
    """
    date_occ: str = Field(..., alias="DATE OCC", description="Date of occurrence e.g. '01/01/2020 12:00:00 AM'")
    time_occ: int = Field(..., alias="TIME OCC", description="Time of occurrence e.g. 1330")
    area: int = Field(..., alias="AREA")
    rpt_dist_no: Optional[int] = Field(None, alias="Rpt Dist No")
    part_1_2: int = Field(..., alias="Part 1-2")
    crm_cd: int = Field(..., alias="Crm Cd")
    mocodes: Optional[str] = Field(None, alias="Mocodes")
    vict_age: float = Field(..., alias="Vict Age")
    vict_sex: str = Field(..., alias="Vict Sex")
    vict_descent: str = Field(..., alias="Vict Descent")
    premis_cd: float = Field(..., alias="Premis Cd")
    premis_desc: str = Field(..., alias="Premis Desc")
    weapon_used_cd: Optional[float] = Field(None, alias="Weapon Used Cd")
    weapon_desc: Optional[str] = Field(None, alias="Weapon Desc")
    status: str = Field(..., alias="Status")
    location: str = Field(..., alias="LOCATION")
    lat: float = Field(..., alias="LAT")
    lon: float = Field(..., alias="LON")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow"  # Allow extra fields if the CSV has them
    )

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float]
    model_info: str

# ==========================================
# 3. MLFLOW & LOADING LOGIC
# ==========================================

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
    """Finds and downloads the best model from MLflow."""
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
        
        print(f"🏆 Best Run: {model_name} ({stage}) - F1: {best_run['metrics.f1_weighted']:.4f}")
        
        # Download artifact
        artifact_filename = f"{model_name}_{stage}.pkl"
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_filename)
        
        with open(local_path, 'rb') as f:
            model = pickle.load(f)
            
        return model, f"{model_name}_{stage}"

    except Exception as e:
        print(f"❌ Error loading model from MLflow: {e}")
        return None, None

def load_processors():
    """Loads local preprocessing artifacts (Scalers, Encoders)."""
    try:
        # Try to load target_label_encoder first (preferred)
        target_encoder_path = os.path.join(PROCESSORS_DIR, "target_label_encoder.pkl")
        if os.path.exists(target_encoder_path):
            with open(target_encoder_path, "rb") as f:
                ml_components["target_encoder"] = pickle.load(f)
        
        # Also try to load target_mapping as fallback (for older artifacts)
        target_mapping_path = os.path.join(PROCESSORS_DIR, "target_mapping.pkl")
        if os.path.exists(target_mapping_path):
            with open(target_mapping_path, "rb") as f:
                ml_components["target_mapping"] = pickle.load(f)
        
        with open(os.path.join(PROCESSORS_DIR, "feature_label_encoders.pkl"), "rb") as f:
            ml_components["feature_encoders"] = pickle.load(f)
        with open(os.path.join(PROCESSORS_DIR, "robust_scaler.pkl"), "rb") as f:
            ml_components["scaler"] = pickle.load(f)
        with open(os.path.join(PROCESSORS_DIR, "features_config.pkl"), "rb") as f:
            ml_components["config"] = pickle.load(f)
        print("✅ Preprocessors loaded.")
    except Exception as e:
        print(f"❌ Error loading processors: {e}")
        raise RuntimeError("Failed to load preprocessing artifacts. Run preprocessing.py first.")

# ==========================================
# 4. DATA PROCESSING ADAPTER
# ==========================================

def prepare_input_for_model(input_dict: Dict[str, Any]):
    """
    Uses functions imported from preprocessing.py to transform single-row JSON.
    """
    # 1. Create DataFrame
    df = pd.DataFrame([input_dict])
    
    # 2. Standard Cleaning (Imported)
    df = pp.clean_column_names(df)
    
    # 3. Feature Engineering (Imported)
    df = pp.feature_engineering_temporal(df)
    
    # 4. Missing Values (Imported)
    df = pp.handle_missing_values_and_text(df)
    
    # 5. Encoding & Scaling (Use preprocessing.encode_features if available)
    # This handles both LabelEncoding and One-Hot encoding (for vict_sex)
    encoders = ml_components["feature_encoders"]
    df, _ = pp.encode_features(df, encoders=encoders)
    
    # 6. Feature Selection / Ordering
    required_features = ml_components["config"]['final_feature_order']
    
    # Ensure columns exist (handle missing One-Hot columns)
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    
    # Reindex to match exact feature order from training
    df_final = df.reindex(columns=required_features, fill_value=0)
    
    # 7. Scaling
    X_scaled = ml_components["scaler"].transform(df_final)
    
    return X_scaled

# ==========================================
# 5. FASTAPI APP DEFINITION
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and Shutdown logic."""
    # --- Startup ---
    print("⚙️ Starting API Service...")
    setup_mlflow()
    load_processors()
    
    model, name = load_best_model()
    if model:
        ml_components["model"] = model
        ml_components["model_name"] = name
        print("🚀 Server Ready.")
    else:
        print("⚠️ Server started without a model (Health check will fail).")
    
    yield
    # --- Shutdown ---
    print("🛑 Shutting down API.")

app = FastAPI(title="Crime Prediction API", version="1.0", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Crime Prediction API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    if ml_components["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy", 
        "model": ml_components.get("model_name", "Unknown"),
        "processors_loaded": ml_components["scaler"] is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_crime(payload: CrimeInput):
    if not ml_components["model"]:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Convert Pydantic model to dict (using aliases to match CSV headers)
        input_data = payload.model_dump(by_alias=True)
        
        # Preprocess
        X_input = prepare_input_for_model(input_data)
        
        # Inference
        model = ml_components["model"]
        pred_idx = model.predict(X_input)[0]
        
        # Decode Label (prefer target_encoder, fallback to target_mapping)
        if ml_components["target_encoder"] is not None:
            pred_label = ml_components["target_encoder"].inverse_transform([pred_idx])[0]
        elif ml_components["target_mapping"] is not None:
            pred_label = ml_components["target_mapping"].get("code_to_class", {}).get(int(pred_idx), str(pred_idx))
        else:
            pred_label = str(pred_idx)
        
        # Confidence
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            confidence = float(np.max(probs))
            
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "model_info": ml_components.get("model_name", "Unknown")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 6. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
#http://127.0.0.1:5000/docs#/
