import pytest
import pandas as pd
import numpy as np
import sys
import os

# --- CONFIGURATION DES CHEMINS D'IMPORTATION ---
# Permet de trouver 'preprocessing.py' dans le dossier '../backend/source'
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '../backend/source')
sys.path.insert(0, source_dir)

try:
    from preprocessing import categorize_crime, preprocess_data
except ImportError as e:
    pytest.fail(f"ERREUR CRITIQUE : Impossible d'importer 'preprocessing' depuis {source_dir}. Détail : {e}")

# --- FIXTURE (Données de test) ---
@pytest.fixture
def df_raw():
    """
    Crée un DataFrame factice (Mock) injecté automatiquement 
    dans les fonctions de test qui demandent l'argument 'df_raw'.
    """
    return pd.DataFrame({
        'DR_NO': [1, 2, 3, 4, 5],
        'Date Rptd': ['01/01/2023 10:00:00 AM', '01/02/2023 11:00:00 AM', '01/03/2023 12:00:00 PM', '01/04/2023 01:00:00 PM', '01/05/2023 02:00:00 PM'],
        'DATE OCC': ['01/01/2023 09:00:00 AM', '01/02/2023 10:00:00 AM', '01/03/2023 11:00:00 AM', '01/04/2023 12:00:00 PM', '01/05/2023 01:00:00 PM'],
        'TIME OCC': [900, 1000, 1100, 1200, 1300],
        'AREA': [1, 2, 3, 1, 2],
        'Rpt Dist No': [101, 202, 303, 101, 202],
        'Part 1-2': [1, 1, 2, 1, 2],
        'Crm Cd': [510, 330, 624, 510, 740],
        'Crm Cd Desc': ['VEHICLE - STOLEN', 'BURGLARY FROM VEHICLE', 'BATTERY - SIMPLE ASSAULT', 'VEHICLE - STOLEN', 'VANDALISM - FELONY'],
        'Mocodes': ['0100', np.nan, '0200', '0100', '0300'],
        # Cas limites pour le sexe : 'H', '-' et NaN doivent devenir 'X'
        'Vict Age': [25, 30, 35, 40, 45],
        'Vict Sex': ['M', 'F', '-', np.nan, 'H'],
        'Vict Descent': ['H', 'W', '-', np.nan, 'B'],
        'Premis Cd': [101.0, 102.0, np.nan, 101.0, 103.0],
        'Premis Desc': ['STREET', 'SIDEWALK', np.nan, 'STREET', 'PARKING LOT'],
        # Cas limites pour les armes : NaN doit devenir 0.0 ou 'NO WEAPON'
        'Weapon Used Cd': [np.nan, 400.0, np.nan, np.nan, 500.0],
        'Weapon Desc': [np.nan, 'STRONG-ARM', np.nan, np.nan, 'UNKNOWN WEAPON'],
        'Status': ['AA', 'IC', 'AO', 'AA', np.nan],
        # Colonnes à supprimer
        'Crm Cd 1': [510.0, 330.0, 624.0, 510.0, 740.0],
        'Crm Cd 2': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Crm Cd 3': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Crm Cd 4': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'Cross Street': [np.nan, np.nan, 'MAIN ST', np.nan, np.nan],
        'LAT': [34.05, 34.06, 34.07, 34.05, 34.06],
        'LON': [-118.24, -118.25, -118.26, -118.24, -118.25]
    })

# --- TESTS UNITAIRES ---

def test_categorize_crime():
    """Test de la logique de regroupement des crimes (Test Complet)."""
    # 1. Test Catégorie : Vol
    assert categorize_crime('VEHICLE - STOLEN') == 'السرقة والسطو / Theft and Burglary'
    
    # 2. Test Catégorie : Violence
    assert categorize_crime('BATTERY - SIMPLE ASSAULT') == 'العنف والاعتداء / Violence and Assault'
    
    # 3. Test Catégorie : Vandalisme
    assert categorize_crime('VANDALISM - FELONY') == 'التخريب والتدمير / Vandalism and Destruction'
    
    # 4. Test Catégorie : Fraude
    assert categorize_crime('CREDIT CARDS') == 'الاحتيال والتزوير / Fraud and Forgery'
    
    # 5. Test Valeur Inconnue ou Manquante
    assert categorize_crime('UNKNOWN CRIME TYPE') == 'جرائم متنوعة / Miscellaneous Crimes'
    assert categorize_crime(np.nan) == 'جرائم متنوعة / Miscellaneous Crimes'

def test_preprocess_data_columns_drop(df_raw):
    """Vérifie que les colonnes inutiles sont bien supprimées."""
    X, y, encoders = preprocess_data(df_raw)
    
    # Note: 'dr_no' est converti en minuscule par le script avant d'être supprimé
    dropped_cols = ['dr_no', 'crm_cd_1', 'crm_cd_2', 'crm_cd_3', 'crm_cd_4', 'cross_street']
    
    for col in dropped_cols:
        assert col not in X.columns, f"La colonne {col} aurait dû être supprimée"

def test_preprocess_data_date_features(df_raw):
    """Vérifie la création des features temporelles (sans Weekday/is_weekend)."""
    X, y, encoders = preprocess_data(df_raw)
    
    expected_date_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
    
    for col in expected_date_cols:
        assert col in X.columns, f"La colonne temporelle {col} est manquante"
        
    # On vérifie que les colonnes non désirées sont absentes
    assert 'Weekday' not in X.columns
    assert 'is_weekend' not in X.columns

    # Vérification de la valeur (1ère ligne = 2023)
    assert X['Year'].iloc[0] == 2023

def test_preprocess_data_missing_values(df_raw):
    """Vérifie que les valeurs manquantes critiques sont gérées."""
    X, y, encoders = preprocess_data(df_raw)
    
    # Les colonnes numériques ne doivent pas avoir de NaN
    assert not X['premis_cd'].isnull().any()
    assert not X['weapon_used_cd'].isnull().any()
    
    # Vérification spécifique : la ligne 0 avait NaN pour l'arme, doit être 0.0
    assert X['weapon_used_cd'].iloc[0] == 0.0

def test_preprocess_data_encoding(df_raw):
    """Vérifie le One-Hot Encoding du sexe et le Label Encoding de la cible"""
    X, y, encoders = preprocess_data(df_raw)
    
    # One-Hot Encoding : Sexe (M, F, X attendus)
    assert 'vict_sex_M' in X.columns
    assert 'vict_sex_F' in X.columns
    assert 'vict_sex_X' in X.columns
    
    # Vérif ligne 0 (Homme) -> M=1
    assert X['vict_sex_M'].iloc[0] == 1
    # Vérif ligne 2 ('-') -> X=1
    assert X['vict_sex_X'].iloc[2] == 1

    # Target Encoding
    assert len(y) == 5
    # Vérifie que y contient des entiers
    assert pd.api.types.is_integer_dtype(y)

def test_preprocess_data_shape(df_raw):
    """Vérifie la dimension finale"""
    X, y, encoders = preprocess_data(df_raw)
    assert X.shape[0] == 5
    assert y.shape[0] == 5