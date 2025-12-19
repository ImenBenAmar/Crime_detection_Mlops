import os
import shutil
import pickle
import json
import mlflow
import dagshub
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# Import des modèles
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocessing2 import run_preprocessing_pipeline, ARTIFACTS_PATH

# ==========================================
# CONFIGURATION
# ==========================================
EXPERIMENT_NAME = "Crime_MLOPS1"  
REGISTERED_MODEL_NAME = "Crime_Prediction_Model"
DATA_VERSION = "v1"  # Ta version de donnée demandée
DATA_PATH = "../../data/crime_v1.csv" # Chemin vers ta donnée v1

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_USERNAME", "YomnaJL")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "MLOPS_Project")

def setup_mlflow():
    try:
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
        print(f"✅ Connecté à DagsHub MLflow.")
    except Exception as e:
        print(f"⚠️ Erreur init DagsHub: {e}. Utilisation de MLflow local ou env vars.")

def get_best_run_config():
    """
    Cherche le meilleur run basé sur f1_weighted.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if not experiment:
        raise ValueError(f"❌ Expérience '{EXPERIMENT_NAME}' introuvable !")

    # Recherche basée sur f1_weighted DESC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_weighted DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("❌ Aucun run trouvé dans l'expérience MLflow.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    run_name = best_run.data.tags.get("mlflow.runName", "Unknown")
    f1_score_val = best_run.data.metrics.get('f1_weighted', 0)
    params = best_run.data.params 
    
    print(f"\n🏆 MEILLEUR RUN SOURCE : {run_name} (ID: {run_id})")
    print(f"📊 F1-Weighted actuel : {f1_score_val:.4f}")

    # --- IDENTIFICATION DE L'ALGO ---
    algo_type = "unknown"
    name_upper = run_name.upper()
    keys = params.keys()

    if any(x in name_upper for x in ["XGB", "XGBOOST"]): algo_type = "xgboost"
    elif any(x in name_upper for x in ["CAT", "CATBOOST"]): algo_type = "catboost"
    elif any(x in name_upper for x in ["LGB", "LIGHTGBM"]): algo_type = "lightgbm"
    elif any(x in name_upper for x in ["FOREST", "RF"]): algo_type = "randomforest"
    
    # Fallback par analyse de paramètres
    if algo_type == "unknown":
        if 'iterations' in keys: algo_type = "catboost"
        elif 'num_leaves' in keys: algo_type = "lightgbm"
        elif 'max_depth' in keys and 'learning_rate' not in keys: algo_type = "randomforest"
        else: algo_type = "xgboost"
            
    print(f"🕵️ Algorithme détecté pour re-training : {algo_type.upper()}")
    return algo_type, params

def instantiate_model(algo_type, params, y_train):
    """
    Nettoie les paramètres MLflow et instancie le bon modèle.
    """
    clean_params = {}
    for k, v in params.items():
        # On ignore les tags mlflow loggués comme params
        if k.startswith("mlflow."): continue
        try:
            if v.lower() == 'none': clean_params[k] = None
            elif v.lower() == 'true': clean_params[k] = True
            elif v.lower() == 'false': clean_params[k] = False
            elif '.' in v: clean_params[k] = float(v)
            else: clean_params[k] = int(v)
        except:
            clean_params[k] = v

    # Paramètres de base pour tous les modèles
    clean_params.pop('verbose', None)
    clean_params.pop('n_jobs', None)
    
    if algo_type == "xgboost":
        num_class = len(np.unique(y_train))
        clean_params['objective'] = 'multi:softprob'
        clean_params['num_class'] = num_class
        return XGBClassifier(**clean_params, n_jobs=-1, random_state=42)
    
    elif algo_type == "catboost":
        return CatBoostClassifier(**clean_params, verbose=0, random_state=42)
    
    elif algo_type == "lightgbm":
        return LGBMClassifier(**clean_params, n_jobs=-1, random_state=42, verbose=-1)
    
    elif algo_type == "randomforest":
        return RandomForestClassifier(**clean_params, n_jobs=-1, random_state=42)
    
    raise ValueError(f"Modèle {algo_type} non supporté.")

def train_and_register():
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. RÉCUPÉRATION DE LA CONFIG DU MEILLEUR MODÈLE
    algo_type, best_params = get_best_run_config()

    # 2. RUN PREPROCESSING (Mode Train pour régénérer les processeurs frais)
    print(f"⚙️ Exécution du preprocessing sur la donnée {DATA_VERSION}...")
    run_preprocessing_pipeline(data_path=DATA_PATH, mode="train")

    # Chargement des données pré-traitées
    data_file = os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    X_train, y_train = data["X_train_scaled"], data["y_train"]
    X_test, y_test = data["X_test_scaled"], data["y_test"]

    # 3. ENTRAÎNEMENT DANS MLFLOW
    with mlflow.start_run(run_name=f"Retrain_{algo_type}_{DATA_VERSION}") as run:
        # Logging des informations de versioning
        mlflow.set_tag("data_version", DATA_VERSION)
        mlflow.set_tag("model_status", "retrained")
        mlflow.log_param("dataset_version", DATA_VERSION)
        mlflow.log_param("algo_family", algo_type)

        # Instanciation et Fit
        model = instantiate_model(algo_type, best_params, y_train)
        print(f"🚀 Ré-entraînement du modèle {algo_type} en cours...")
        model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        
        print(f"📊 Résultats Retraining -> F1 Weighted: {f1:.4f} | Accuracy: {acc:.4f}")
        
        # Log des metrics et params
        mlflow.log_params(best_params)
        mlflow.log_metrics({"f1_weighted": f1, "accuracy": acc})

        # 4. LOG DES PROCESSORS (Les artefacts du preprocessing)
        # On log tout le dossier 'processors' pour qu'il soit lié à CE modèle précis
        mlflow.log_artifacts(ARTIFACTS_PATH, artifact_path="processors")
        print(f"📁 Processors sauvegardés comme artefacts.")

        # 5. ENREGISTREMENT DANS LE MODEL REGISTRY
        model_info = mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="model", 
                registered_model_name=REGISTERED_MODEL_NAME,
                pyfunc_predict_fn="predict" # Force predict par défaut
            )
        # 6. PROMOTION EN PRODUCTION
        # On définit un seuil minimal pour la promotion automatique
        MIN_F1_THRESHOLD = 0.65 
        client = MlflowClient()
        new_version = model_info.registered_model_version

        if f1 >= MIN_F1_THRESHOLD:
            print(f"✅ Seuil F1 dépassé ({f1:.4f}). Promotion en 'Production' de la v{new_version}...")
            
            # Archivage des anciennes versions en production
            latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
            for v in latest_versions:
                if v.version != new_version:
                    client.transition_model_version_stage(
                        name=REGISTERED_MODEL_NAME, version=v.version, stage="Archived"
                    )

            # Passage en Production
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME, version=new_version, stage="Production"
            )
            print(f"🚀 Modèle {REGISTERED_MODEL_NAME} version {new_version} est maintenant en PRODUCTION.")
        else:
            print(f"⚠️ F1 trop faible ({f1:.4f}). Modèle enregistré en 'None' (Staging requis).")

if __name__ == "__main__":
    train_and_register()