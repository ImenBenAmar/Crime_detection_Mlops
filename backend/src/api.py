import os
import sys
import shutil
import pickle
import numpy as np
from dotenv import load_dotenv
import mlflow
import dagshub
import dagshub.auth
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
from typing import Optional
from mlflow.tracking import MlflowClient

# Ensure we can import from backend/src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_store import CrimeFeatureStore

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME')
REPO_OWNER = "YomnaJL"

# Le nom EXACT défini dans train.py
REGISTERED_MODEL_NAME = "Crime_Prediction_Model"

# Dossier temporaire pour garantir la fraîcheur des fichiers (Stateless Docker)
LOCAL_ARTIFACTS_DIR = "/tmp/downloaded_processors"

# Global state
ml_components = {
    "model": None, 
    "store": None, 
    "model_name": "Unknown",
    "version": "Unknown"
}

# ==========================================
# SCHEMAS PYDANTIC
# ==========================================
class CrimeInput(BaseModel):
    date_occ: str = Field(..., alias="DATE OCC")
    time_occ: int = Field(..., alias="TIME OCC")
    area: int = Field(..., alias="AREA")
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

# ==========================================
# HELPER FUNCTIONS (MLflow Registry)
# ==========================================

def setup_mlflow():
    """Authentification et Configuration MLflow"""
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_TOKEN')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')
    
    if not token or not repo_name:
        print("⚠️ Variables DAGSHUB manquantes. Vérifiez votre .env ou Docker.")
        return

    # Auth silencieuse
    try:
        dagshub.auth.add_app_token(token)
    except Exception:
        pass

    os.environ['MLFLOW_TRACKING_USERNAME'] = username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    tracking_uri = f"https://dagshub.com/{REPO_OWNER}/{repo_name}.mlflow"
    
    try:
        dagshub.init(repo_owner=REPO_OWNER, repo_name=repo_name, mlflow=True)
        print(f"✅ DagsHub connecté. URI: {mlflow.get_tracking_uri()}")
    except Exception:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ MLflow URI forcé : {tracking_uri}")

def download_model_from_registry():
    """
    Télécharge le modèle marqué comme 'Production' (ou fallback) ET ses artifacts.
    """
    client = MlflowClient()
    
    try:
        print(f"🔍 Interrogation du Registry pour : {REGISTERED_MODEL_NAME}...")
        
        # 1. Chercher les versions disponibles
        # On cherche 'Production' en priorité, sinon 'None' (la dernière uploadée)
        latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production", "None"])
        
        if not latest_versions:
            print(f"❌ Aucun modèle trouvé dans le Registry sous le nom '{REGISTERED_MODEL_NAME}'.")
            return None, None, None

        # Priorité à la Production
        target_version = None
        for v in latest_versions:
            if v.current_stage == "Production":
                target_version = v
                break
        
        # Fallback
        if not target_version:
            target_version = latest_versions[-1] # La plus récente (souvent V1 ou V2 non promue)
            print(f"⚠️ Pas de modèle en stage 'Production'. Utilisation de la version {target_version.version} (Stage: {target_version.current_stage})")
        else:
            print(f"✅ Modèle trouvé en 'Production' : Version {target_version.version}")

        run_id = target_version.run_id
        model_version = target_version.version
        
        # 2. Télécharger le Modèle
        print(f"📥 Téléchargement du Modèle V{model_version}...")
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
        # On utilise pyfunc pour charger de manière générique (XGBoost, Sklearn, Catboost...)
        model = mlflow.pyfunc.load_model(model_uri)

        # 3. Télécharger les Processors (Synchronisation Drift)
        print(f"📥 Téléchargement des Processors associés (Run {run_id})...")
        
        # Nettoyage du dossier temporaire
        if os.path.exists(LOCAL_ARTIFACTS_DIR):
            shutil.rmtree(LOCAL_ARTIFACTS_DIR)
        
        mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path="processors", 
            dst_path=LOCAL_ARTIFACTS_DIR
        )
        
        # Gestion de la structure de dossier (parfois artifacts/processors/processors...)
        final_processors_path = os.path.join(LOCAL_ARTIFACTS_DIR, "processors")
        if not os.path.exists(os.path.join(final_processors_path, "robust_scaler.pkl")):
            # Si les fichiers sont directement à la racine du téléchargement
            final_processors_path = LOCAL_ARTIFACTS_DIR

        print(f"✅ Synchronisation réussie : Modèle V{model_version} + Processors.")
        return model, f"{REGISTERED_MODEL_NAME}_v{model_version}", final_processors_path

    except Exception as e:
        print(f"❌ Erreur lors du chargement MLflow : {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==========================================
# LIFECYCLE MANAGER (STARTUP)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚙️ Démarrage de l'API...")
    
    setup_mlflow()
    
    # CHARGEMENT DYNAMIQUE DEPUIS LE REGISTRY
    model, name, processors_path = download_model_from_registry()
    
    if model and processors_path:
        ml_components["model"] = model
        ml_components["model_name"] = name
        
        print(f"📦 Initialisation du Feature Store avec les processors téléchargés...")
        try:
            store = CrimeFeatureStore(processors_path=processors_path)
            store.load_artifacts()
            ml_components["store"] = store
            print("🚀 API PRÊTE et SYNCHRONISÉE.")
        except Exception as e:
            print(f"❌ Erreur critique Feature Store : {e}")
    else:
        print("⚠️ ECHEC MLFLOW : Tentative de fallback sur fichiers locaux...")
        # Fallback local (si internet coupé ou erreur DagsHub)
        if os.path.exists("processors"):
             print("⚠️ Utilisation des processors locaux (Risque de version mismatch).")
             store = CrimeFeatureStore(processors_path="processors")
             store.load_artifacts()
             ml_components["store"] = store
        else:
            print("❌ Aucun processeur disponible. L'API ne pourra pas prédire.")

    yield
    print("🛑 Arrêt de l'API.")

# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(title="Crime API", lifespan=lifespan)

@app.get("/")
def root():
    status = "Ready" if ml_components["model"] else "Not Ready"
    return {
        "message": "Crime Prediction API", 
        "status": status, 
        "loaded_model": ml_components["model_name"]
    }

@app.get("/health")
def health():
    if not ml_components["model"]:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy", "model": ml_components["model_name"]}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: CrimeInput):
    if not ml_components["model"]:
        raise HTTPException(status_code=503, detail="Model not initialized.")
    
    try:
        # 1. Préparation
        data = payload.model_dump(by_alias=True)
        store = ml_components["store"]
        X_input = store.get_online_features(data)
        
        # 2. Prédiction (Classe)
        model = ml_components["model"]
        prediction_result = model.predict(X_input)
        
        if isinstance(prediction_result, (list, np.ndarray)):
            pred_idx = prediction_result[0]
        else:
            pred_idx = prediction_result

        # 3. Décodage
        pred_label = store.decode_target(pred_idx)
        
        # 4. Confiance (Probabilité) - VERSION ROBUSTE
        confidence = 0.0
        try:
            # On tente d'accéder à l'objet sous-jacent (Sklearn/XGBoost natif)
            raw_model = model
            
            # Si c'est un wrapper PyFunc générique
            if hasattr(model, "unwrap_python_model"):
                try:
                    raw_model = model.unwrap_python_model()
                except:
                    pass # Ce n'était pas un PythonModel, on continue
            
            # Si c'est un wrapper Flavor natif (XGBoost/Sklearn)
            if hasattr(model, "_model_impl"):
                raw_model = model._model_impl
            
            # Maintenant on cherche predict_proba sur le vrai objet
            if hasattr(raw_model, "predict_proba"):
                probs = raw_model.predict_proba(X_input)[0]
                confidence = float(np.max(probs))
            else:
                # Fallback : certains modèles XGBoost natifs n'ont pas predict_proba mais predict renvoie des probas
                # si l'objectif était multi:softprob. Mais ici on a déjà la classe en résultat, 
                # donc on suppose que predict_proba est la voie.
                print("⚠️ Pas de méthode predict_proba trouvée sur le modèle interne.")
                confidence = 0.0
                
        except Exception as e:
            print(f"⚠️ Erreur calcul confiance : {e}")
            confidence = 0.0
            
        return {
            "prediction": str(pred_label),
            "confidence": confidence,
            "model_info": ml_components["model_name"]
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)