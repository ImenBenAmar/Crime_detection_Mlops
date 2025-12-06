import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ==========================================
# CONFIGURATION
# ==========================================
ARTIFACTS_PATH = "processors"
DATA_PATH = "../../data/crime_v1.csv" # Ensure this points to your file

REQUIRED_ARTIFACTS = [
    "robust_scaler.pkl",
    "target_label_encoder.pkl",
    "feature_label_encoders.pkl",
    "features_config.pkl"
]

DEFAULT_SELECTED_FEATURES = [
    'mocodes', 'premis_cd', 'location', 'weapon_used_cd', 'vict_age',
    'day', 'area', 'crm_risk', 'month', 'vict_descent', 'status',
    'weekday', 'hour_bin', 'year', 'vict_sex_F', 'vict_sex_M', 'vict_sex_X'
]

CATEGORICAL_COLS_TO_ENCODE = [
    'crm_risk', 'mocodes', 'vict_descent', 
    'status', 'status_desc', 'location', 'hour_bin'
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def categorize_crime(crime):
    if not isinstance(crime, str):
        return 'جرائم متنوعة / Miscellaneous Crimes'
    crime = crime.upper()
    # Check fraud first (more specific conditions before general ones)
    if any(x in crime for x in ['CREDIT CARDS', 'EMBEZZLEMENT', 'DEFRAUDING', 'FORGERY', 'IDENTITY']):
        return 'الاحتيال والتزوير / Fraud and Forgery'
    elif any(x in crime for x in ['VEHICLE - STOLEN', 'BURGLARY', 'THEFT', 'BUNCO', 'PICKPOCKET', 'SHOPLIFTING']):
        return 'السرقة والسطو / Theft and Burglary'
    elif any(x in crime for x in ['ASSAULT', 'BATTERY', 'ROBBERY', 'HOMICIDE', 'KIDNAPPING', 'STALKING', 'ABUSE', 'THREATS']):
        return 'العنف والاعتداء / Violence and Assault'
    elif any(x in crime for x in ['VANDALISM', 'ARSON', 'SHOTS FIRED', 'THROWING OBJECT', 'DAMAGE', 'BOMB']):
        return 'التخريب والتدمير / Vandalism and Destruction'
    elif any(x in crime for x in ['COURT', 'CONTEMPT', 'VIOLATION', 'TRESPASSING', 'WEAPON', 'FIREARM']):
        return 'المخالفات القانونية والجرائم المتعلقة بالأسلحة / Legal Offences & Weapons'
    elif any(x in crime for x in ['RAPE', 'SEX', 'LEWD', 'PIMPING', 'TRAFFICKING', 'INCEST']):
        return 'الجرائم الجنسية والاتجار / Sexual Crimes & Exploitation'
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

def load_and_clean_initial(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # --- FIX: Remove the index column saved by to_csv ---
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
        
    df = clean_column_names(df)
    
    if "dr_no" in df.columns:
        df.drop("dr_no", axis=1, inplace=True)
    
    df = df.rename(columns={"part_1_2": "crm_risk"})
    
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_len - len(df)} duplicates.")
    return df

def feature_engineering_temporal(df):
    print("Engineering temporal features...")
    df['date_occ'] = pd.to_datetime(df['date_occ'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    
    # Lowercase column names to match the clean_column_names standard
    df['year'] = df['date_occ'].dt.year
    df['month'] = df['date_occ'].dt.month
    df['day'] = df['date_occ'].dt.day
    df['weekday'] = df['date_occ'].dt.weekday
    
    df['hour'] = df['time_occ'] // 100
    df['minute'] = df['time_occ'] % 100 
    
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['hour_bin'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False).astype(str)
    
    df = df.drop(columns=['time_occ', 'date_rptd', 'date_occ'], errors='ignore')
    return df

def handle_missing_values_and_text(df):
    print("Handling missing values...")
    
    df['vict_descent'] = df['vict_descent'].fillna('UNKNOWN').replace({'-': 'UNKNOWN'})
    df['vict_sex'] = df['vict_sex'].fillna('X').replace({'H': 'X', '-': 'X'})
    
    # Only fill modes if columns exist
    for col in ['mocodes', 'premis_cd', 'premis_desc', 'status']:
        if col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    df['weapon_desc'] = df['weapon_desc'].fillna('NO WEAPON')
    df['weapon_used_cd'] = df['weapon_used_cd'].fillna(0.0)
    
    df.loc[(df['vict_age'] < 0) | (df['vict_age'] > 100), 'vict_age'] = np.nan
    mean_age = df['vict_age'].mean()
    df['vict_age'] = df['vict_age'].fillna(mean_age)

    if 'premis_desc' in df.columns:
        df['premis_desc'] = df['premis_desc'].str.title().str.replace(r'\(.*?\)', '', regex=True).str.strip()
    
    cols_to_drop = ['crm_cd_1', 'crm_cd_2', 'crm_cd_3', 'crm_cd_4', 'cross_street']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    return df

def process_target(df, encoder=None):
    print("Processing target variable...")
    df['Crime_Class'] = df['crm_cd_desc'].apply(categorize_crime)
    
    if encoder:
        print("Using loaded Target Encoder.")
        df['Crime_Class_Enc'] = encoder.transform(df['Crime_Class'])
    else:
        print("Fitting new Target Encoder.")
        encoder = LabelEncoder()
        df['Crime_Class_Enc'] = encoder.fit_transform(df['Crime_Class'])
    
    df.drop(columns=['Crime_Class', 'crm_cd_desc', 'area_name', 'premis_desc', 'weapon_desc'], inplace=True, errors='ignore')
    return df, encoder

def encode_features(df, encoders=None):
    print("Encoding features...")
    
    # One-Hot Encoding for Sex
    df = pd.get_dummies(df, columns=['vict_sex'], prefix='vict_sex', dtype=int)
    
    if encoders:
        print(f"Using {len(encoders)} loaded Feature Encoders.")
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unseen labels by mapping them to -1
                # Convert to string to match training type
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError as e:
                    # Handle unseen labels by mapping them to -1
                    print(f"Warning: Unseen labels in {col}, mapping to -1. Error: {str(e)}")
                    # Create a mask for known vs unknown labels
                    known_mask = df[col].astype(str).isin(le.classes_)
                    df.loc[known_mask, col] = le.transform(df.loc[known_mask, col].astype(str))
                    df.loc[~known_mask, col] = -1  # Default value for unseen labels
    else:
        print("Fitting new Feature Encoders.")
        encoders = {}
        for col in CATEGORICAL_COLS_TO_ENCODE:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            
    return df, encoders

def check_artifacts_exist():
    for f in REQUIRED_ARTIFACTS:
        if not os.path.exists(os.path.join(ARTIFACTS_PATH, f)):
            return False
    return True

# ==========================================
# MAIN PIPELINE FUNCTION
# ==========================================

def run_preprocessing_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    
    # 1. Load & Clean
    df = load_and_clean_initial(DATA_PATH)
    
    # 2. Engineer & Impute
    df = feature_engineering_temporal(df)
    df = handle_missing_values_and_text(df)
    
    # 3. Determine Mode (Load vs Fit)
    artifacts_exist = check_artifacts_exist()
    target_encoder = None
    feature_encoders = None
    scaler = None
    selected_features = [f.lower() for f in DEFAULT_SELECTED_FEATURES] # Ensure lowercase matches clean_column_names
    
    if artifacts_exist:
        print("\n[INFO] Artifacts found. Loading processors...")
        with open(os.path.join(ARTIFACTS_PATH, "target_label_encoder.pkl"), "rb") as f:
            target_encoder = pickle.load(f)
        with open(os.path.join(ARTIFACTS_PATH, "feature_label_encoders.pkl"), "rb") as f:
            feature_encoders = pickle.load(f)
        with open(os.path.join(ARTIFACTS_PATH, "robust_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(ARTIFACTS_PATH, "features_config.pkl"), "rb") as f:
            config = pickle.load(f)
            selected_features = [f.lower() for f in config.get("final_feature_order", DEFAULT_SELECTED_FEATURES)]
    else:
        print("\n[INFO] Artifacts not found. Starting Training Mode (Fitting processors)...")
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    # 4. Process Target & Encode
    df, target_encoder = process_target(df, encoder=target_encoder)
    df, feature_encoders = encode_features(df, encoders=feature_encoders)
    
    # 5. Select Features
    print(f"Selecting {len(selected_features)} features...")
    # Ensure all selected features exist (handle potential missing One-Hot columns)
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
            
    X = df[selected_features]
    y = df['Crime_Class_Enc']
    
    # 6. Split
    print("Splitting data...")
    # Check if stratification is possible (each class needs at least 2 samples)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    can_stratify = np.all(class_counts >= 2)

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        print("Warning: Some classes have too few samples for stratification. Using regular split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # 7. Scale
    if artifacts_exist:
        print("Applying loaded Scaler...")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("Fitting and Applying Scaler...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # 8. Save
    print("\nSaving processed data...")
    data_package = {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values
    }
    with open(os.path.join(ARTIFACTS_PATH, "preprocessed_data.pkl"), "wb") as f:
        pickle.dump(data_package, f)

    if not artifacts_exist:
        print("Saving new processors...")
        with open(os.path.join(ARTIFACTS_PATH, "target_label_encoder.pkl"), "wb") as f:
            pickle.dump(target_encoder, f)
        
        # Save encoders only for columns present in selected features
        relevant_encoders = {col: enc for col, enc in feature_encoders.items() if col in selected_features}
        with open(os.path.join(ARTIFACTS_PATH, "feature_label_encoders.pkl"), "wb") as f:
            pickle.dump(relevant_encoders, f)
            
        with open(os.path.join(ARTIFACTS_PATH, "robust_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
            
        target_mapping = {"code_to_class": {code: cls for code, cls in enumerate(target_encoder.classes_)}}
        with open(os.path.join(ARTIFACTS_PATH, "target_mapping.pkl"), "wb") as f:
            pickle.dump(target_mapping, f)
            
        feature_config = {"final_feature_order": selected_features}
        with open(os.path.join(ARTIFACTS_PATH, "features_config.pkl"), "wb") as f:
            pickle.dump(feature_config, f)
        
        print(f"✓ New processors saved to '{ARTIFACTS_PATH}/'")
    else:
        print(f"✓ Processors loaded from '{ARTIFACTS_PATH}/' were used.")

    print(f"✓ Preprocessing complete.")
    print(f"  - Train shape: {X_train_scaled.shape}")
    print(f"  - Test shape: {X_test_scaled.shape}")

if __name__ == "__main__":
    run_preprocessing_pipeline()