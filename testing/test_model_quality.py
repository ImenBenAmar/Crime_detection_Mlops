import sys
import os
import pickle
import pandas as pd
import numpy as np
import mlflow
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation
from dotenv import load_dotenv

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../backend/src')))
ARTIFACTS_PATH = "processors"

# Chargement des variables (DagsHub)
load_dotenv()

def load_data_and_config():
    """Reconstruit des DataFrames pandas à partir des arrays numpy pour Deepchecks"""
    print("📦 Chargement des données...")
    with open(os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    with open(os.path.join(ARTIFACTS_PATH, "features_config.pkl"), "rb") as f:
        config = pickle.load(f)
        
    columns = config['final_feature_order']
    
    # Reconstruction des DataFrames (Deepchecks a besoin de noms de colonnes)
    df_train = pd.DataFrame(data['X_train_scaled'], columns=columns)
    df_train['target'] = data['y_train']
    
    df_test = pd.DataFrame(data['X_test_scaled'], columns=columns)
    df_test['target'] = data['y_test']
    
    return df_train, df_test

def get_best_model():
    """Récupère le modèle champion depuis MLflow"""
    print("🔍 Récupération du modèle depuis MLflow...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    runs = mlflow.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(os.getenv("DAGSHUB_REPO_NAME", "Crime_MLOPS1")).experiment_id],
        order_by=["metrics.f1_weighted DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise ValueError("Aucun modèle trouvé dans MLflow.")
        
    run_id = runs.iloc[0].run_id
    print(f"🏆 Meilleur Run ID: {run_id}")
    
    # Téléchargement
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "model/model.pkl")
    
    with open(local_path, "rb") as f:
        model = pickle.load(f)
        
    return model

def run_quality_check():
    try:
        # 1. Préparation
        df_train, df_test = load_data_and_config()
        model = get_best_model()
        
        # 2. Création des Datasets Deepchecks
        # cat_features=[] car les données sont déjà encodées/scalées
        ds_train = Dataset(df_train, label='target', cat_features=[])
        ds_test = Dataset(df_test, label='target', cat_features=[])
        
        # 3. Lancement de la Suite "Model Evaluation"
        # Vérifie : Performance, Overfitting, Biais, etc.
        print("🚀 Lancement de l'analyse Deepchecks (cela peut prendre un moment)...")
        suite = model_evaluation()
        result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)
        
        # 4. Sauvegarde du rapport
        report_file = "model_quality_report.html"
        result.save_as_html(report_file)
        print(f"✅ Rapport généré : {report_file}")
        
        # 5. Vérification des conditions (Quality Gate)
        # On peut être strict (result.passed()) ou permissif (juste générer le rapport)
        # Ici, on affiche juste les résultats en JSON pour les logs Jenkins
        print(result.to_json())
        
        # Pour bloquer le pipeline en cas d'échec critique, décommentez ceci :
        # if not result.passed():
        #     print("❌ La qualité du modèle est insuffisante !")
        #     sys.exit(1)
            
    except Exception as e:
        print(f"❌ Erreur lors du test de qualité : {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_quality_check()