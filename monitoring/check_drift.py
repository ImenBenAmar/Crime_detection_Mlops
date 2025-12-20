import sys
import os
import pickle
import pandas as pd
import numpy as np

# Evidently imports
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns

# ==========================================
# 1. CONFIGURATION ET CHEMINS (Version Jenkins)
# ==========================================
# On part du principe que le script est dans /tests/
current_dir = os.path.dirname(os.path.abspath(__file__))
# Racine du projet (un cran au dessus de /tests/)
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '..'))

# Chemin vers backend/src pour importer feature_store.py
BACKEND_SRC_PATH = os.path.join(PROJECT_ROOT, 'backend', 'src')
sys.path.insert(0, BACKEND_SRC_PATH)

# Chemin vers les processeurs (.pkl)
ARTIFACTS_PATH = os.path.join(BACKEND_SRC_PATH, 'processors')

# Chemin vers les nouvelles données (Simulation de prod)
# Dans Jenkins, on peut passer ce chemin en variable d'env
NEW_DATA_PATH = os.getenv("PROD_DATA_PATH", os.path.join(PROJECT_ROOT, 'data', 'last_rows.csv'))

try:
    from feature_store import CrimeFeatureStore
except ImportError:
    print(f"❌ Erreur d'import : feature_store.py introuvable dans {BACKEND_SRC_PATH}")
    sys.exit(1)

def get_reference_data():
    """Charge les données d'entraînement (Reference)"""
    data_path = os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl")
    config_path = os.path.join(ARTIFACTS_PATH, "features_config.pkl")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Artefact manquant : {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    with open(config_path, "rb") as f:
        config = pickle.load(f)
        
    cols = config['final_feature_order']
    return pd.DataFrame(data['X_train_scaled'], columns=cols)

def process_current_data_with_store(df_raw, store):
    """Transforme les données brutes via le Feature Store"""
    print(f"⚙️ Transformation de {len(df_raw)} nouvelles lignes...")
    input_records = df_raw.to_dict(orient='records')
    processed_rows = [store.get_online_features(record)[0] for record in input_records]
    return pd.DataFrame(processed_rows, columns=store.required_features)

def run_drift_check():
    try:
        # 1. Init Store
        store = CrimeFeatureStore(processors_path=ARTIFACTS_PATH)
        store.load_artifacts()
        
        # 2. Reference Data
        reference_data = get_reference_data()
        
        # 3. Current Data (Nouvelles données)
        if not os.path.exists(NEW_DATA_PATH):
            print(f"⚠️ {NEW_DATA_PATH} introuvable. Simulation avec un sample de reference.")
            current_data = reference_data.sample(min(500, len(reference_data)))
        else:
            df_new_raw = pd.read_csv(NEW_DATA_PATH)
            # On prend les 500 dernières lignes pour le test de drift
            current_data = process_current_data_with_store(df_new_raw.tail(500), store)
        
        # 4. Evidently Test Suite
        print("🚀 Analyse de dérive (Data Drift) via Evidently...")
        # Seuil : Si plus de 30% des colonnes ont dérivé, le test échoue
        drift_suite = TestSuite(tests=[
            TestShareOfDriftedColumns(lt=0.3) 
        ])
        
        drift_suite.run(reference_data=reference_data, current_data=current_data)
        
        # Sauvegarde du rapport HTML
        report_path = os.path.join(PROJECT_ROOT, "drift_report.html")
        drift_suite.save_html(report_path)
        print(f"✅ Rapport généré : {report_path}")
        
        # 5. Verdict pour Jenkins
        result = drift_suite.as_dict()
        if not result['summary']['all_passed']:
            print("🚨 ALERTE DRIFT : Les données de production ont trop changé !")
            # On retourne 1 pour stopper le pipeline Jenkins si drift critique
            # (Ou 0 si on veut juste un warning, selon ta stratégie)
            sys.exit(1) 
        else:
            print("✅ Pas de drift significatif détecté.")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Erreur critique Drift : {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_drift_check()