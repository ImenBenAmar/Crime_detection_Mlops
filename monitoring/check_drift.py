import pandas as pd
import sys
import os
import pickle
import warnings
import shutil
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestShareOfDriftedColumns
)

# ==========================================================
# 0. GLOBAL CONFIG
# ==========================================================
warnings.filterwarnings("ignore")
load_dotenv()

# ==========================================================
# 1. MLFLOW / DAGSHUB CONFIG
# ==========================================================
REPO_OWNER = "YomnaJL"
REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "MLOPS_Project")
REGISTERED_MODEL_NAME = "Crime_Prediction_Model"

# ==========================================================
# 2. MONITORING OUTPUT CONFIG
# ==========================================================
MONITORING_DIR = "monitoring"
MONITORING_TMP_DIR = "/tmp/monitoring_artifacts"

REPORT_HTML = os.path.join(MONITORING_DIR, "monitoring_drift_report.html")
TEST_HTML = os.path.join(MONITORING_DIR, "test_results.html")
TRIGGER_FILE = os.path.join(MONITORING_DIR, "drift_detected")

# ==========================================================
# 3. MLFLOW AUTH
# ==========================================================
def setup_mlflow():
    """Authentification DagsHub + MLflow"""
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not token:
        print("❌ DAGSHUB_TOKEN manquant")
        sys.exit(1)

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    dagshub.init(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        mlflow=True
    )

    print(f"✅ MLflow connecté : {mlflow.get_tracking_uri()}")

# ==========================================================
# 4. DOWNLOAD ARTEFACTS FROM REGISTRY (PRODUCTION)
# ==========================================================
def download_reference_from_mlflow():
    """Télécharge les artefacts du modèle en Production"""
    client = MlflowClient()

    print(f"🔍 Recherche du modèle '{REGISTERED_MODEL_NAME}' en Production")

    versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME,
        stages=["Production"]
    )

    if not versions:
        print("⚠️ Aucun modèle en Production. Fallback dernière version.")
        versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME,
            stages=["None"]
        )

    if not versions:
        print("❌ Aucun modèle trouvé dans le Registry")
        sys.exit(1)

    run_id = versions[0].run_id
    print(f"📥 Téléchargement des artefacts depuis run {run_id}")

    if os.path.exists(MONITORING_TMP_DIR):
        shutil.rmtree(MONITORING_TMP_DIR)

    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="processors",
        dst_path=MONITORING_TMP_DIR
    )

    path = os.path.join(MONITORING_TMP_DIR, "processors")
    if not os.path.exists(path):
        path = MONITORING_TMP_DIR

    return path

# ==========================================================
# 5. LOAD DATA
# ==========================================================
def load_data(processors_path):
    """Charge les données preprocessées"""
    data_path = os.path.join(processors_path, "preprocessed_data.pkl")
    config_path = os.path.join(processors_path, "features_config.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    columns = config["final_feature_order"]

    reference_df = pd.DataFrame(
        data["X_train_scaled"], columns=columns
    )
    reference_df["target"] = data["y_train"]

    current_df = pd.DataFrame(
        data["X_test_scaled"], columns=columns
    )
    current_df["target"] = data["y_test"]

    return reference_df, current_df

# ==========================================================
# 6. EVIDENTLY MONITORING
# ==========================================================
def run_evidently_analysis(reference, current):
    """Analyse Drift + Quality Gate"""

    os.makedirs(MONITORING_DIR, exist_ok=True)

    print("📊 Analyse Evidently en cours...")

    # ---------- Report ----------
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(REPORT_HTML)

    print(f"✅ Rapport enregistré : {REPORT_HTML}")

    # ---------- Tests ----------
    tests = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=0.3),
        TestNumberOfColumnsWithMissingValues(eq=0)
    ])
    tests.run(reference_data=reference, current_data=current)
    tests.save_html(TEST_HTML)

    print(f"✅ Tests enregistrés : {TEST_HTML}")

    # ---------- Decision ----------
    results = tests.as_dict()

    if not results["summary"]["all_passed"]:
        print("🚨 DRIFT DÉTECTÉ")
        with open(TRIGGER_FILE, "w") as f:
            f.write("drift=true")
    else:
        print("✅ Données stables")
        if os.path.exists(TRIGGER_FILE):
            os.remove(TRIGGER_FILE)

# ==========================================================
# 7. MAIN
# ==========================================================
if __name__ == "__main__":
    setup_mlflow()

    processors_path = download_reference_from_mlflow()

    reference_df, current_df = load_data(processors_path)

    run_evidently_analysis(reference_df, current_df)
