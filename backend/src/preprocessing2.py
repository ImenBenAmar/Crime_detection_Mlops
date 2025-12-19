import pandas as pd
import numpy as np
import re
import os
import pickle
import sys
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ==========================================
# CONFIGURATION
# ==========================================
ARTIFACTS_PATH = "processors"

# 1. Stratégie de chemin de données "Intelligente"
# Priorité : Variable d'environnement (Jenkins/Docker) > Argument > Relatif Local
ENV_DATA_PATH = os.getenv("DATA_PATH")
DEFAULT_LOCAL_PATH = "../../data/crime_v1.csv" # Pour le dev local dans backend/src

REQUIRED_ARTIFACTS = [
    "robust_scaler.pkl",
    "target_label_encoder.pkl",
    "feature_label_encoders.pkl",
    "features_config.pkl"
]

DEFAULT_SELECTED_FEATURES = [
    'mocodes', 'premis_cd', 'location', 'weapon_used_cd', 'vict_age',
    'day', 'area', 'crm_risk', 'month', 'vict_descent', 'status',
    'weekday', 'hour_bin', 'year', 'vict_sex_f', 'vict_sex_m', 'vict_sex_x'
]

CATEGORICAL_COLS_TO_ENCODE = [
    'crm_risk', 'mocodes', 'vict_descent', 
    'status', 'location', 'hour_bin'
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def categorize_crime(crime):
    if not isinstance(crime, str): return 'جرائم متنوعة / Miscellaneous Crimes'
    crime = crime.upper()
    if any(x in crime for x in ['CREDIT CARDS', 'EMBEZZLEMENT', 'FORGERY']): return 'الاحتيال والتزوير / Fraud and Forgery'
    elif any(x in crime for x in ['ASSAULT', 'BATTERY', 'ROBBERY', 'HOMICIDE']): return 'العنف والاعتداء / Violence and Assault'
    elif any(x in crime for x in ['VANDALISM', 'ARSON', 'DAMAGE']): return 'التخريب والتدمير / Vandalism and Destruction'
    elif any(x in crime for x in ['VEHICLE - STOLEN', 'BURGLARY', 'THEFT']): return 'السرقة والسطو / Theft and Burglary'
    elif any(x in crime for x in ['COURT', 'WEAPON', 'TRESPASSING']): return 'المخالفات القانونية والجرائم المتعلقة بالأسلحة / Legal Offences & Weapons'
    elif any(x in crime for x in ['RAPE', 'SEX', 'TRAFFICKING']): return 'الجرائم الجنسية والاتجار / Sexual Crimes & Exploitation'
    return 'جرائم متنوعة / Miscellaneous Crimes'

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('/', '_')
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )
    return df

def find_data_file(provided_path=None):
    """
    Logique robuste pour trouver le fichier CSV n'importe où (Docker, Jenkins, Local).
    """
    # 1. Si un chemin est fourni (argument ou env var), on teste
    if provided_path and os.path.exists(provided_path):
        return provided_path
    
    # 2. Sinon, on teste le défaut local relatif
    if os.path.exists(DEFAULT_LOCAL_PATH):
        return DEFAULT_LOCAL_PATH
    
    # 3. Fallback : On cherche dans le dossier courant ou dossier parent (structure Jenkins)
    candidates = [
        "data/crime_v1.csv",
        "../data/crime_v1.csv",
        "../../data/crime_v1.csv",
        "/app/data/crime_v1.csv" # Standard Docker path
    ]
    
    for path in candidates:
        if os.path.exists(path):
            print(f"⚠️ Chemin deviné automatiquement : {path}")
            return path
            
    # Si on arrive ici, c'est critique
    if provided_path:
        raise FileNotFoundError(f"❌ Fichier introuvable au chemin demandé : {provided_path}")
    else:
        raise FileNotFoundError(f"❌ Impossible de trouver crime_v1.csv dans les chemins standards.")

def load_and_clean_initial(filepath):
    print(f"📂 Chargement des données depuis : {os.path.abspath(filepath)}")
    df = pd.read_csv(filepath)
    if "Unnamed: 0" in df.columns: df.drop("Unnamed: 0", axis=1, inplace=True)
    df = clean_column_names(df)
    if "dr_no" in df.columns: df.drop("dr_no", axis=1, inplace=True)
    df = df.rename(columns={"part_1_2": "crm_risk"})
    df = df.drop_duplicates()
    return df

def feature_engineering_temporal(df):
    print("🛠️ Feature Engineering...")
    df['date_occ'] = pd.to_datetime(df['date_occ'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    df['year'] = df['date_occ'].dt.year
    df['month'] = df['date_occ'].dt.month
    df['day'] = df['date_occ'].dt.day
    df['weekday'] = df['date_occ'].dt.weekday
    df['hour'] = df['time_occ'] // 100
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['hour_bin'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False).astype(str)
    df = df.drop(columns=['time_occ', 'date_rptd', 'date_occ'], errors='ignore')
    return df

def handle_missing_values_and_text(df):
    print("🛠️ Gestion des valeurs manquantes...")
    df['vict_descent'] = df['vict_descent'].fillna('UNKNOWN').replace({'-': 'UNKNOWN'})
    df['vict_sex'] = df['vict_sex'].fillna('X').replace({'H': 'X', '-': 'X'})
    for col in ['mocodes', 'premis_cd', 'status']:
        if col in df.columns: df[col] = df[col].fillna('0')
    df['weapon_used_cd'] = df['weapon_used_cd'].fillna(0.0)
    df.loc[(df['vict_age'] < 0) | (df['vict_age'] > 100), 'vict_age'] = np.nan
    df['vict_age'] = df['vict_age'].fillna(df['vict_age'].mean() if not df['vict_age'].isnull().all() else 30)
    return df

def process_target(df, encoder=None):
    df['crime_class'] = df['crm_cd_desc'].apply(categorize_crime)
    if encoder:
        df['target_enc'] = encoder.transform(df['crime_class'])
    else:
        encoder = LabelEncoder()
        df['target_enc'] = encoder.fit_transform(df['crime_class'])
    return df, encoder

def encode_features(df, encoders=None):
    # One-Hot Encoding manuel pour garantir les colonnes
    df['vict_sex'] = df['vict_sex'].str.lower()
    df['vict_sex_f'] = (df['vict_sex'] == 'f').astype(int)
    df['vict_sex_m'] = (df['vict_sex'] == 'm').astype(int)
    df['vict_sex_x'] = (~df['vict_sex'].isin(['f', 'm'])).astype(int)
    
    if encoders:
        # Mode Transform : Utilise les mappings existants
        for col, le in encoders.items():
            if col in df.columns:
                # Map unknown to 0/First Class safely
                df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    else:
        # Mode Train : Apprend les mappings
        encoders = {}
        for col in CATEGORICAL_COLS_TO_ENCODE:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    return df, encoders

# ==========================================
# MAIN PIPELINE
# ==========================================

def run_preprocessing_pipeline(data_path=None, mode="train"):
    """
    Pipeline principal.
    mode='train' -> Apprend Scalers/Encoders et les sauvegarde (Pour Retraining/Drift).
    mode='transform' -> Utilise les Scalers/Encoders existants (Pour Test/Validation).
    """
    
    # 1. Résolution intelligente du chemin
    # Si data_path est None, on regarde ENV_DATA_PATH, sinon DEFAULT_LOCAL_PATH
    target_path = data_path if data_path else ENV_DATA_PATH
    final_path = find_data_file(target_path)
    
    # 2. Chargement & Nettoyage
    df = load_and_clean_initial(final_path)
    df = feature_engineering_temporal(df)
    df = handle_missing_values_and_text(df)
    
    # 3. Gestion des Artefacts selon le mode
    target_encoder, feature_encoders, scaler = None, None, None
    
    if mode == "transform":
        if os.path.exists(os.path.join(ARTIFACTS_PATH, "target_label_encoder.pkl")):
            print("[INFO] Mode Transform: Chargement des processeurs existants...")
            with open(os.path.join(ARTIFACTS_PATH, "target_label_encoder.pkl"), "rb") as f: target_encoder = pickle.load(f)
            with open(os.path.join(ARTIFACTS_PATH, "feature_label_encoders.pkl"), "rb") as f: feature_encoders = pickle.load(f)
            with open(os.path.join(ARTIFACTS_PATH, "robust_scaler.pkl"), "rb") as f: scaler = pickle.load(f)
        else:
            raise FileNotFoundError("❌ Mode 'transform' demandé mais aucun processeur trouvé dans processors/")
    else:
        print("[INFO] Mode Train: Initialisation de nouveaux processeurs...")
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # 4. Encodage
    df, target_encoder = process_target(df, encoder=target_encoder)
    df, feature_encoders = encode_features(df, encoders=feature_encoders)
    
    # 5. Sélection Features
    for col in DEFAULT_SELECTED_FEATURES:
        if col not in df.columns: df[col] = 0
            
    X = df[DEFAULT_SELECTED_FEATURES]
    y = df['target_enc']
    
    # 6. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if scaler:
        print("⚖️ Scaling (Transform)...")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("⚖️ Scaling (Fit & Transform)...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # 7. Sauvegarde Données
    print("💾 Sauvegarde preprocessed_data.pkl...")
    data_package = {"X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled, "y_train": y_train.values, "y_test": y_test.values}
    with open(os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl"), "wb") as f: pickle.dump(data_package, f)

    # 8. Sauvegarde Processors (Seulement en mode Train)
    if mode == "train":
        with open(os.path.join(ARTIFACTS_PATH, "target_label_encoder.pkl"), "wb") as f: pickle.dump(target_encoder, f)
        with open(os.path.join(ARTIFACTS_PATH, "feature_label_encoders.pkl"), "wb") as f: pickle.dump(feature_encoders, f)
        with open(os.path.join(ARTIFACTS_PATH, "robust_scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
        with open(os.path.join(ARTIFACTS_PATH, "features_config.pkl"), "wb") as f: pickle.dump({"final_feature_order": DEFAULT_SELECTED_FEATURES}, f)
        print(f"✅ Nouveaux processeurs sauvegardés dans {ARTIFACTS_PATH}/")

    print(f"✨ Preprocessing terminé avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Chemin du CSV")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "transform"], help="Mode d'exécution")
    args = parser.parse_args()
    
    run_preprocessing_pipeline(data_path=args.data_path, mode=args.mode)