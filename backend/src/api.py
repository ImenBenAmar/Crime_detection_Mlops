import os
import sys
import pickle
import numpy as np
from dotenv import load_dotenv
import mlflow
import dagshub
import dagshub.auth # Import nécessaire pour l'auth silencieuse
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
REPO_OWNER = "YomnaJL"  # Propriétaire du repo

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
    """Configures MLflow tracking uri and authentication safely for Docker."""
    
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_TOKEN')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')
    
    if not token or not repo_name:
        print("⚠️ Variables d'environnement DAGSHUB manquantes.")
        return

    # 1. Force l'authentification DagsHub (Empêche l'ouverture du navigateur)
    try:
        dagshub.auth.add_app_token(token)
        print("✅ DagsHub Token ajouté.")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'ajout du token DagsHub: {e}")

    # 2. Configurer les variables d'environnement MLflow EXPLICITEMENT
    # Cela garantit que le client MLflow a les identifiants même si dagshub.init échoue
    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    tracking_uri = f"https://dagshub.com/{REPO_OWNER}/{repo_name}.mlflow"
    
    # 3. Tenter l'initialisation DagsHub, mais avec un fallback
    try:
        dagshub.init(repo_owner=REPO_OWNER, repo_name=repo_name, mlflow=True)
        print(f"✅ DagsHub initialized. Tracking URI: {mlflow.get_tracking_uri()}")
    except Exception as e:
        print(f"⚠️ dagshub.init a échoué ({e}). Passage en configuration manuelle MLflow.")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ MLflow Tracking URI forcé manuellement : {tracking_uri}")

def load_best_model():
    """Loads best model from MLflow based on F1 Score."""
    try:
        print(f"🔍 Recherche de l'expérience : {EXPERIMENT_NAME}...")
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if not experiment:
            # Essayer de récupérer par ID si le nom échoue ou lister tout
            print(f"❌ Expérience '{EXPERIMENT_NAME}' introuvable. Vérifiez le nom exact sur DagsHub.")
            return None, None
        
        print(f"✅ Expérience trouvée ID: {experiment.experiment_id}")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_weighted DESC"],
            max_results=1
        )
        
        if runs.empty:
            print("❌ Aucun run trouvé dans cette expérience.")
            return None, None
        
        best_run = runs.iloc[0]
        run_id = best_run.run_id
        
        # Gestion des tags manquants
        model_name = best_run.get('tags.model_name', 'UnknownModel')
        stage = best_run.get('tags.stage', 'UnknownStage')
        f1_score = best_run.get('metrics.f1_weighted', 0.0)
        
        print(f"🏆 Meilleur Modèle trouvé : {model_name} ({stage}) - F1: {f1_score:.4f}")
        print(f"📥 Téléchargement de l'artefact depuis le Run ID: {run_id}...")

        # Construct filename based on training script convention
        artifact_path = f"{model_name}_{stage}.pkl"
        
        # Download artifact
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        
        with open(local_path, 'rb') as f:
            model = pickle.load(f)
            
        return model, f"{model_name}_{stage}"
    except Exception as e:
        print(f"❌ Erreur critique lors du chargement du modèle : {e}")
        # Pour le debug, afficher plus de détails si nécessaire
        import traceback
        traceback.print_exc()
        return None, None

# --- Lifecycle Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚙️ Initializing API...")
    
    # 1. Setup Environment
    setup_mlflow()
    
    # 2. Initialize Feature Store (loads encoders/scaler)
    print("📦 Chargement du Feature Store...")
    try:
        store = CrimeFeatureStore(processors_path="processors")
        store.load_artifacts()
        ml_components["store"] = store
        print("✅ Feature Store chargé.")
    except Exception as e:
        print(f"❌ Erreur Feature Store: {e}")
    
    # 3. Load Model
    model, name = load_best_model()
    if model:
        ml_components["model"] = model
        ml_components["model_name"] = name
        print("🚀 Server Ready and Model Loaded.")
    else:
        print("⚠️ ATTENTION: Aucun modèle chargé. L'API répondra, mais /predict échouera.")
        
    yield
    print("🛑 Shutting down.")

# --- FastAPI App ---
app = FastAPI(title="Crime API", lifespan=lifespan)

@app.get("/")
def root():
    status = "Ready" if ml_components["model"] else "Model Missing"
    return {"message": "Crime Prediction API Running", "status": status}

@app.get("/health")
def health():
    if not ml_components["model"]:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy", "model": ml_components["model_name"]}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: CrimeInput):
    if not ml_components["model"]:
        raise HTTPException(status_code=503, detail="Model not initialized on server.")
    
    try:
        # Convert Pydantic object to dict
        data = payload.model_dump(by_alias=True)
        
        # 1. Feature Store Processing
        store = ml_components["store"]
        if not store:
             raise HTTPException(status_code=500, detail="Feature Store not initialized")

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
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)