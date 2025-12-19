import pytest
import pandas as pd
import numpy as np
import os
import shutil
import time
import pickle
import sys
from unittest.mock import patch

# ==========================================
# 1. SETUP DES CHEMINS (Dynamique & Robuste)
# ==========================================
# Dossier courant : 
current_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le code source : 
# On remonte d'un cran (..) puis on va dans backend/src
backend_src_path = os.path.abspath(os.path.join(current_dir, '..', 'backend', 'src'))

# Ajout au système pour que Python trouve le fichier
if backend_src_path not in sys.path:
    sys.path.insert(0, backend_src_path)

print(f"✅ Recherche du module dans : {backend_src_path}")

try:
    # IMPORTANT : On importe 'preprocessing2' car c'est le nom de votre fichier
    import preprocessing2 as preprocessing
    # On récupère les variables nécessaires
    from preprocessing2 import ARTIFACTS_PATH
except ImportError as e:
    pytest.fail(f"❌ Impossible d'importer 'preprocessing2.py'. Vérifiez qu'il est bien dans {backend_src_path}. Erreur : {e}")

# ==========================================
# 2. FIXTURES (Données de simulation)
# ==========================================

@pytest.fixture
def sample_raw_df():
    """Crée un DataFrame qui imite le CSV brut."""
    data = {
        'DR_NO': [1, 2, 3, 4, 5],
        'Date Rptd': ['01/01/2020']*5,
        'DATE OCC': ['01/01/2020 12:00:00 AM']*5,
        'TIME OCC': [100, 1200, 1300, 2200, 800],
        'AREA': [1, 2, 3, 4, 5],
        'Rpt Dist No': [101, 102, 103, 104, 105],
        'Part 1-2': [1, 2, 1, 2, 1],
        'Crm Cd': [100, 200, 300, 400, 500],
        'Crm Cd Desc': [
            'VEHICLE - STOLEN',           
            'RAPE, FORCIBLE',             
            'BATTERY - SIMPLE ASSAULT',   
            'THEFT OF IDENTITY',          
            'VANDALISM - FELONY'          
        ],
        'Mocodes': ['0100', '0200', np.nan, '0400', '0500'],
        'Vict Age': [25, 30, -5, 120, 40],
        'Vict Sex': ['M', 'F', 'X', 'H', np.nan], 
        'Vict Descent': ['W', 'B', 'H', '-', np.nan],
        'Premis Cd': [101.0, 102.0, np.nan, 104.0, 105.0],
        'Premis Desc': ['STREET', 'SIDEWALK', np.nan, 'PARK', 'ALLEY'],
        'Weapon Used Cd': [100.0, np.nan, 300.0, 400.0, 500.0],
        'Weapon Desc': ['GUN', np.nan, 'KNIFE', 'HAND', 'PIPE'],
        'Status': ['AA', 'IC', 'AO', 'AA', np.nan],
        # Colonnes inutiles
        'Crm Cd 1': [1]*5, 'Cross Street': [np.nan]*5, 'Unnamed: 0': [0, 1, 2, 3, 4]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_file(tmp_path, sample_raw_df):
    """Crée un fichier CSV temporaire."""
    file_path = tmp_path / "test_crime_data.csv"
    sample_raw_df.to_csv(file_path, index=False) 
    return str(file_path)

@pytest.fixture
def cleanup_artifacts():
    """Nettoie le dossier processors après le test."""
    yield
    # Le dossier processors est créé là où le script est lancé
    local_artifacts = os.path.join(os.getcwd(), ARTIFACTS_PATH)
    if os.path.exists(local_artifacts): 
        try:
            shutil.rmtree(local_artifacts)
        except Exception:
            pass

# ==========================================
# 3. TESTS UNITAIRES (Adaptés à preprocessing2.py)
# ==========================================

def test_clean_column_names(sample_raw_df):
    df = preprocessing.clean_column_names(sample_raw_df.copy())
    assert 'date_rptd' in df.columns
    assert 'Date Rptd' not in df.columns

def test_load_and_clean_initial(temp_data_file):
    df = preprocessing.load_and_clean_initial(temp_data_file)
    assert 'dr_no' not in df.columns
    assert 'crm_risk' in df.columns # preprocessing2 renomme part_1_2 en crm_risk
    assert 'unnamed:_0' not in df.columns

def test_feature_engineering_temporal(sample_raw_df):
    df = preprocessing.clean_column_names(sample_raw_df.copy())
    df = preprocessing.feature_engineering_temporal(df)
    
    assert 'year' in df.columns
    assert 'hour_bin' in df.columns
    # Dans votre preprocessing2, Time OCC 100 -> 1h matin -> Night
    assert df.loc[0, 'hour_bin'] == 'Night'

def test_handle_missing_values_and_text(sample_raw_df):
    df = preprocessing.clean_column_names(sample_raw_df.copy())
    df['vict_age'] = pd.to_numeric(df['vict_age'], errors='coerce')
    
    df = preprocessing.handle_missing_values_and_text(df)
    
    # Sexe : H et NaN doivent devenir X
    assert df['vict_sex'].iloc[3] == 'X'
    assert df['vict_sex'].iloc[4] == 'X'
    
    # Age : Moyenne appliquée (preprocessing2 logic)
    assert df['vict_age'].iloc[2] > 0 

def test_process_target(sample_raw_df):
    df = preprocessing.clean_column_names(sample_raw_df.copy())
    df, encoder = preprocessing.process_target(df)
    
    # Preprocessing2 crée 'target_enc' et non 'Crime_Class_Enc'
    assert 'target_enc' in df.columns
    assert 'crime_class' in df.columns
    assert encoder is not None

def test_encode_features(sample_raw_df):
    df = preprocessing.clean_column_names(sample_raw_df.copy())
    
    if 'part_1_2' in df.columns: df = df.rename(columns={'part_1_2': 'crm_risk'})
    df['vict_sex'] = df['vict_sex'].fillna('X')
    
    df, encoders = preprocessing.encode_features(df)
    
    # Preprocessing2 utilise .lower() donc vict_sex_f (minuscule)
    assert 'vict_sex_f' in df.columns
    assert 'vict_sex_m' in df.columns
    
    assert 'crm_risk' in encoders

# ==========================================
# 4. TESTS D'INTÉGRATION
# ==========================================

@pytest.mark.usefixtures("cleanup_artifacts")
def test_pipeline_execution_fit_mode(temp_data_file):
    """Test le mode Entraînement (Fit)."""
    
    # On appelle la fonction avec les arguments (spécifique à preprocessing2)
    preprocessing.run_preprocessing_pipeline(data_path=temp_data_file, mode="train")
    
    # Vérification des fichiers
    local_artifacts = os.path.join(os.getcwd(), ARTIFACTS_PATH)
    
    assert os.path.exists(os.path.join(local_artifacts, "preprocessed_data.pkl"))
    assert os.path.exists(os.path.join(local_artifacts, "target_label_encoder.pkl"))
    
    with open(os.path.join(local_artifacts, "preprocessed_data.pkl"), "rb") as f:
        data = pickle.load(f)
    
    assert data["X_train_scaled"].shape[0] > 0
    # ==========================================
# 5. TESTS SUPPLÉMENTAIRES (Logic Spécifique preprocessing2)
# ==========================================

def test_find_data_file(tmp_path):
    """Vérifie la logique de recherche automatique du fichier CSV."""
    
    # Cas 1 : Chemin explicite fourni qui existe
    f = tmp_path / "explicit.csv"
    f.touch()
    assert preprocessing.find_data_file(str(f)) == str(f)
    
    # Cas 2 : Chemin explicite qui n'existe pas -> Erreur
    # CORRECTION ICI : On utilise 'patch' pour forcer os.path.exists à renvoyer False
    # Cela empêche le script de trouver votre vrai fichier data/crime_v1.csv par hasard
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            preprocessing.find_data_file("chemin/imaginaire.csv")
        
    # Cas 3 : Recherche automatique (Fallback)
    # On crée un faux fichier dans le dossier courant pour simuler 'data/crime_v1.csv'
    fake_data_dir = tmp_path / "data"
    fake_data_dir.mkdir()
    fake_csv = fake_data_dir / "crime_v1.csv"
    fake_csv.touch()
    
    # On change le dossier courant pour le test
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Ici, os.path.exists fonctionne normalement, donc il va trouver le faux fichier
        found = preprocessing.find_data_file()
        assert "data" in found and "crime_v1.csv" in found
    finally:
        os.chdir(original_cwd)
    
    # On change le dossier courant pour le test (monkeypatching context)
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Comme data/crime_v1.csv existe dans tmp_path, il doit le trouver
        found = preprocessing.find_data_file()
        assert "data" in found and "crime_v1.csv" in found
    finally:
        os.chdir(original_cwd)

@pytest.mark.usefixtures("cleanup_artifacts")
def test_pipeline_transform_mode_missing_artifacts(temp_data_file):
    """Vérifie que le mode 'transform' plante proprement si on n'a pas entraîné avant."""
    
    # On s'assure que le dossier processors est vide/inexistant
    local_artifacts = os.path.join(os.getcwd(), ARTIFACTS_PATH)
    if os.path.exists(local_artifacts):
        shutil.rmtree(local_artifacts)
        
    # On lance en mode 'transform' sans avoir fait de 'train'
    # Cela doit lever une FileNotFoundError (comme codé dans preprocessing2.py)
    with pytest.raises(FileNotFoundError) as excinfo:
        preprocessing.run_preprocessing_pipeline(data_path=temp_data_file, mode="transform")
    
    assert "aucun processeur trouvé" in str(excinfo.value)

@pytest.mark.usefixtures("cleanup_artifacts")
def test_pipeline_execution_load_mode(temp_data_file):
    """Test le mode Transformation (Load)."""
    
    # 1. On entraîne pour créer les fichiers
    preprocessing.run_preprocessing_pipeline(data_path=temp_data_file, mode="train")
    
    # 2. On transforme (ne doit pas planter)
    preprocessing.run_preprocessing_pipeline(data_path=temp_data_file, mode="transform")
    
    assert True # Si on arrive ici sans erreur, c'est bon
