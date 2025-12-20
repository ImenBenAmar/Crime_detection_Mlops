import os
import sys
import pickle
import pandas as pd
import numpy as np
import mlflow
import shutil
from dotenv import load_dotenv
import traceback

# --- Imports Deepchecks ---
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

# 1. Configuration des chemins pour importer le code du backend
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../backend/src')))

# 2. Import des fonctions API pour MLflow
from api import setup_mlflow, download_model_from_registry, REGISTERED_MODEL_NAME

# 3. Chargement des variables du fichier .env
load_dotenv()

def run_quality_check_from_mlflow():
    print("\n" + "="*60)
    print(f"📡 EXTRACTION MLFLOW & TEST QUALITÉ : {REGISTERED_MODEL_NAME}")
    print("="*60)

    # A. Connexion à MLflow/DagsHub
    setup_mlflow()

    try:
        # B. Extraction automatique depuis MLflow
        model, model_name, extracted_path = download_model_from_registry()

        if not model or not extracted_path:
            print("❌ Erreur : Impossible d'extraire les données depuis MLflow.")
            return

        print(f"\n✅ Version Cloud extraite : {model_name}")
        print(f"📂 Chemin de l'extraction : {extracted_path}")

        # C. Chargement des fichiers .pkl
        data_file = os.path.join(extracted_path, "preprocessed_data.pkl")
        config_file = os.path.join(extracted_path, "features_config.pkl")

        if not os.path.exists(data_file):
            # Fallback si structure imbriquée
            data_file = os.path.join(extracted_path, "processors", "preprocessed_data.pkl")
            config_file = os.path.join(extracted_path, "processors", "features_config.pkl")

        with open(data_file, "rb") as f:
            data = pickle.load(f)
        with open(config_file, "rb") as f:
            config = pickle.load(f)

        cols = config['final_feature_order']

        # Préparation des DataFrames pour Deepchecks
        df_train = pd.DataFrame(data['X_train_scaled'], columns=cols)
        df_train['target'] = data['y_train']

        df_test = pd.DataFrame(data['X_test_scaled'], columns=cols)
        df_test['target'] = data['y_test']

        # D. Création des datasets Deepchecks
        ds_train = Dataset(df_train, label='target', cat_features=[])
        ds_test = Dataset(df_test, label='target', cat_features=[])

        # E. Exécution de la suite d'évaluation
        print("🚀 Lancement de l'évaluation Deepchecks (model_evaluation)...")
        suite = model_evaluation()
        result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)

        # F. Sauvegarde du rapport HTML avec fallback
        report_filename = f"quality_report_{model_name}.html"
        try:
            result.save_as_html(report_filename, open_browser=False)
        except TypeError:
            # Fallback pour anciennes versions
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(result.to_html())

        print(f"✨ Rapport généré : {os.path.abspath(report_filename)}")

        # Optionnel : sauvegarde en notebook pour éviter page vide
        notebook_filename = f"quality_report_{model_name}.ipynb"
        try:
            result.save_as_notebook(notebook_filename)
            print(f"📓 Notebook généré : {os.path.abspath(notebook_filename)}")
        except Exception:
            pass

        print("\n🏆 TEST DE QUALITÉ TERMINÉ AVEC SUCCÈS !")

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE : {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    run_quality_check_from_mlflow()
