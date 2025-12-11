import sys
import os
import pickle
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

# --- Vérification des dépendances graphiques ---
try:
    import anywidget
    import ipywidgets
except ImportError as e:
    print("❌ ERREUR : Des librairies graphiques manquent.")
    print(f"Détail : {e}")
    print("👉 Exécutez : pip install anywidget ipywidgets nbformat")
    sys.exit(1)

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ajout du backend au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..', 'backend', 'src')))

# Chemin vers le dossier processors
ARTIFACTS_PATH = os.path.abspath(os.path.join(current_dir, '..', 'backend', 'src', 'processors'))
print(f"📂 Chemin des artefacts configuré : {ARTIFACTS_PATH}")

# Chargement du .env
dotenv_path = os.path.join(current_dir, '..', '.env')
load_dotenv(dotenv_path)

# ==========================================
# 2. CHARGEMENT DES DONNÉES ET CONFIG
# ==========================================
def load_data_and_config():
    """Reconstruit des DataFrames pandas à partir des arrays numpy"""
    print("📦 Chargement des données locales...")

    data_path = os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl")
    config_path = os.path.join(ARTIFACTS_PATH, "features_config.pkl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Fichier introuvable : {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    columns = config['final_feature_order']

    # Reconstruction des DataFrames
    df_train = pd.DataFrame(data['X_train_scaled'], columns=columns)
    df_train['target'] = data['y_train']

    df_test = pd.DataFrame(data['X_test_scaled'], columns=columns)
    df_test['target'] = data['y_test']

    return df_train, df_test

# ==========================================
# 3. CHARGEMENT DU MEILLEUR MODÈLE MLflow
# ==========================================
def get_best_model():
    """Récupère le modèle champion depuis MLflow (Auto-détection)"""
    print("🔍 Connexion à MLflow...")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("❌ MLFLOW_TRACKING_URI manquant dans le .env")

    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment_name = os.getenv("EXPERIMENT_NAME", "Crime_MLOPS1")
        experiment = mlflow.get_experiment_by_name(experiment_name)

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id] if experiment else None,
            order_by=["metrics.f1_weighted DESC"],
            max_results=1
        )

        if runs.empty:
            raise ValueError("Aucun run trouvé dans MLflow.")

        run_id = runs.iloc[0].run_id
        print(f"🏆 Meilleur Run ID trouvé : {run_id}")

        # Récupération du fichier .pkl
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)

        model_filename = next((art.path for art in artifacts if art.path.endswith('.pkl')), None)

        if not model_filename:
            try:
                artifacts_sub = client.list_artifacts(run_id, "model")
                model_filename = next((f"model/{art.path}" for art in artifacts_sub if art.path.endswith('.pkl')), None)
            except:
                pass

        if not model_filename:
            raise FileNotFoundError("❌ Aucun fichier .pkl trouvé dans les artefacts du Run.")

        print(f"   ⬇️ Téléchargement de : '{model_filename}'...")
        local_path = client.download_artifacts(run_id, model_filename)
        print(f"   ✅ Modèle téléchargé ici : {local_path}")

        with open(local_path, "rb") as f:
            model = pickle.load(f)

        return model

    except Exception as e:
        print(f"❌ Erreur MLflow : {e}")
        raise e

# ==========================================
# 4. TEST DE QUALITÉ DU MODÈLE
# ==========================================
def run_quality_check():
    try:
        # 1. Préparation
        df_train, df_test = load_data_and_config()
        model = get_best_model()

        # 2. Création des Datasets Deepchecks
        print("⚙️ Préparation des Datasets Deepchecks...")
        ds_train = Dataset(df_train, label='target', cat_features=[])
        ds_test = Dataset(df_test, label='target', cat_features=[])

        # 3. Lancement de la Suite
        print("🚀 Lancement de l'analyse Deepchecks (Cela peut prendre 1-2 minutes)...")
        suite = model_evaluation()
        result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)

        # 4. Sauvegarde du rapport
        report_file = "model_quality_report.html"
        full_report_path = os.path.abspath(report_file)

        # Compatible toutes versions : méthode to_html fallback
        try:
            result.save_as_html(report_file, open_browser=False)
        except TypeError:
            # Anciennes versions Deepchecks
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(result.to_html())

        print(f"\n✨ SUCCÈS ! Rapport généré : {full_report_path}")

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE DANS LE SCRIPT DE TEST : {e}")
        sys.exit(1)

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    run_quality_check()
