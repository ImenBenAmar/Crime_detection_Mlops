import pandas as pd
import numpy as np
import os
import pickle
import re
from sklearn.preprocessing import LabelEncoder, RobustScaler

class CrimeFeatureStore:
    def __init__(self, processors_path="processors"):
        self.processors_path = processors_path
        self.artifacts = {}
        self.is_loaded = False
        
        # Define features that the model actually expects (The Schema)
        self.required_features = [
            'mocodes', 'premis_cd', 'location', 'weapon_used_cd', 'vict_age',
            'day', 'area', 'crm_risk', 'month', 'vict_descent', 'status',
            'weekday', 'hour_bin', 'year', 'vict_sex_f', 'vict_sex_m', 'vict_sex_x'
        ]
        
        self.categorical_cols = [
            'crm_risk', 'mocodes', 'vict_descent', 
            'status', 'status_desc', 'location', 'hour_bin'
        ]

    def load_artifacts(self):
        """Loads Scalers and Encoders from disk."""
        if self.is_loaded: return
        
        try:
            with open(os.path.join(self.processors_path, "feature_label_encoders.pkl"), "rb") as f:
                self.artifacts["feature_encoders"] = pickle.load(f)
            with open(os.path.join(self.processors_path, "robust_scaler.pkl"), "rb") as f:
                self.artifacts["scaler"] = pickle.load(f)
            # Try loading target encoder (preferred)
            if os.path.exists(os.path.join(self.processors_path, "target_label_encoder.pkl")):
                with open(os.path.join(self.processors_path, "target_label_encoder.pkl"), "rb") as f:
                    self.artifacts["target_encoder"] = pickle.load(f)
            
            self.is_loaded = True
            print("✅ Feature Store: Artifacts loaded.")
        except FileNotFoundError:
            print("⚠️ Feature Store: Artifacts not found. Run training first.")

    def categorize_crime(self, crime):
        if not isinstance(crime, str):
            return 'جرائم متنوعة / Miscellaneous Crimes'
        crime = crime.upper()
        if any(x in crime for x in ['CREDIT CARDS', 'EMBEZZLEMENT', 'DEFRAUDING', 'FORGERY', 'IDENTITY']):
            return 'الاحتيال والتزوير / Fraud and Forgery'
        elif any(x in crime for x in ['ASSAULT', 'BATTERY', 'ROBBERY', 'HOMICIDE', 'KIDNAPPING', 'STALKING']):
            return 'العنف والاعتداء / Violence and Assault'
        elif any(x in crime for x in ['VANDALISM', 'ARSON', 'SHOTS FIRED', 'THROWING OBJECT', 'DAMAGE']):
            return 'التخريب والتدمير / Vandalism and Destruction'
        elif any(x in crime for x in ['VEHICLE - STOLEN', 'BURGLARY', 'THEFT', 'BUNCO', 'PICKPOCKET']):
            return 'السرقة والسطو / Theft and Burglary'
        elif any(x in crime for x in ['COURT', 'CONTEMPT', 'VIOLATION', 'TRESPASSING', 'WEAPON', 'FIREARM']):
            return 'المخالفات القانونية والجرائم المتعلقة بالأسلحة / Legal Offences & Weapons'
        elif any(x in crime for x in ['RAPE', 'SEX', 'LEWD', 'PIMPING', 'TRAFFICKING', 'INCEST']):
            return 'الجرائم الجنسية والاتجار / Sexual Crimes & Exploitation'
        return 'جرائم متنوعة / Miscellaneous Crimes'

    def _clean_column_names(self, df):
        """Internal: Standardize names"""
        df.columns = (
            df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('-', '_')
            .str.replace('/', '_')
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
        if "part_1_2" in df.columns:
            df = df.rename(columns={"part_1_2": "crm_risk"})
        return df

    def _engineer_features(self, df):
        """Internal: Create derived features (Time, Date)"""
        # Ensure date format
        if 'date_occ' in df.columns and 'time_occ' in df.columns:
            df['date_occ'] = pd.to_datetime(df['date_occ'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
            
            if df['date_occ'].isnull().any():
                df['date_occ'] = df['date_occ'].fillna(pd.Timestamp("1900-01-01"))

            df['year'] = df['date_occ'].dt.year
            df['month'] = df['date_occ'].dt.month
            df['day'] = df['date_occ'].dt.day
            df['weekday'] = df['date_occ'].dt.weekday
            
            df['hour'] = df['time_occ'] // 100
            df['minute'] = df['time_occ'] % 100 
            
            bins = [0, 6, 12, 18, 24]
            labels = ['Night', 'Morning', 'Afternoon', 'Evening']
            df['hour_bin'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False).astype(str)
        return df

    def _clean_text_and_fill(self, df):
        """Internal: Impute missing values"""
        if 'premis_desc' in df.columns:
            df['premis_desc'] = df['premis_desc'].str.title().str.replace(r'\(.*?\)', '', regex=True).str.strip()
        
        if 'vict_sex' in df.columns:
            df['vict_sex'] = df['vict_sex'].fillna('X').replace({'H': 'X', '-': 'X'})
        if 'vict_descent' in df.columns:
            df['vict_descent'] = df['vict_descent'].fillna('UNKNOWN')

        if 'vict_age' in df.columns:
            df.loc[(df['vict_age'] < 0) | (df['vict_age'] > 100), 'vict_age'] = np.nan
            df['vict_age'] = df['vict_age'].fillna(30) 

        defaults = {'mocodes': '0', 'premis_cd': 0, 'weapon_used_cd': 0, 'status': 'IC'}
        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        
        return df

    def get_online_features(self, input_dict):
        """
        PUBLIC API: Transforms a single dictionary of raw inputs into model-ready vector.
        """
        if not self.is_loaded: self.load_artifacts()

        # 1. To DataFrame
        df = pd.DataFrame([input_dict])
        
        # 2. Transformations
        df = self._clean_column_names(df)
        df = self._engineer_features(df)
        df = self._clean_text_and_fill(df)
        
        # 3. Encoding (One-Hot Manually)
        # We handle this manually because get_dummies might miss columns on a single row
        sex = str(input_dict.get('Vict Sex', 'X')).upper()
        df['vict_sex_f'] = 1 if sex == 'F' else 0
        df['vict_sex_m'] = 1 if sex == 'M' else 0
        df['vict_sex_x'] = 1 if sex not in ['F', 'M'] else 0
        
        # 4. Encoding (Label)
        for col, le in self.artifacts["feature_encoders"].items():
            col_lower = col.lower()
            if col_lower in df.columns:
                val = str(df[col_lower].iloc[0])
                if val in le.classes_:
                    df[col_lower] = le.transform([val])[0]
                else:
                    df[col_lower] = 0 # Handle unknown classes
        
        # 5. Selection & Ordering (Strict)
        final_df = pd.DataFrame()
        for feat in self.required_features:
            if feat in df.columns:
                final_df[feat] = df[feat]
            else:
                final_df[feat] = 0
        
        # 6. Scaling
        X_scaled = self.artifacts["scaler"].transform(final_df)
        
        return X_scaled

    def decode_target(self, pred_idx):
        if "target_encoder" in self.artifacts:
            return self.artifacts["target_encoder"].inverse_transform([pred_idx])[0]
        return str(pred_idx)