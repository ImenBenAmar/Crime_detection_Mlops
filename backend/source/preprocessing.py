import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def categorize_crime(crime):
    """
    Fonction auxiliaire pour regrouper les descriptions de crimes en catégories plus larges.
    """
    if not isinstance(crime, str):
        return 'جرائم متنوعة / Miscellaneous Crimes'
        
    crime = crime.upper()

    # 1️⃣ السرقة والسطو / Theft and Burglary
    if any(x in crime for x in [
        'VEHICLE - STOLEN', 'BURGLARY FROM VEHICLE', 'BIKE - STOLEN',
        'SHOPLIFTING-GRAND THEFT', 'BURGLARY', 'THEFT-GRAND', 'BUNCO, GRAND THEFT',
        'THEFT PLAIN', 'THEFT FROM MOTOR VEHICLE', 'TILL TAP', 'BOAT - STOLEN',
        'DISHONEST EMPLOYEE', 'PURSE SNATCHING', 'PETTY THEFT - AUTO REPAIR',
        'SHOPLIFTING - PETTY THEFT', 'THEFT FROM PERSON', 'BUNCO, PETTY THEFT',
        'THEFT, PERSON', 'THEFT, COIN MACHINE', 'GRAND THEFT / AUTO REPAIR',
        'BIKE - ATTEMPTED STOLEN', 'VEHICLE - ATTEMPT STOLEN',
        'VEHICLE, STOLEN - OTHER', 'PICKPOCKET', 'SHOPLIFTING - ATTEMPT',
        'BUNCO, ATTEMPT', 'PICKPOCKET, ATTEMPT'
    ]):
        return 'السرقة والسطو / Theft and Burglary'

    # 2️⃣ العنف والاعتداء / Violence and Assault
    elif any(x in crime for x in [
        'ASSAULT', 'BATTERY', 'ROBBERY', 'KIDNAPPING', 'CRIMINAL HOMICIDE',
        'MANSLAUGHTER', 'ATTEMPTED ROBBERY', 'INTIMATE PARTNER - SIMPLE ASSAULT',
        'INTIMATE PARTNER - AGGRAVATED ASSAULT', 'OTHER ASSAULT',
        'BATTERY POLICE', 'BATTERY ON A FIREFIGHTER',
        'EXTORTION', 'FALSE IMPRISONMENT', 'STALKING',
        'CHILD', 'CHILD ABUSE', 'CHILD NEGLECT', 'CHILD ANNOYING',
        'CHILD STEALING', 'DISRUPT SCHOOL', 'DRUGS, TO A MINOR',
        'CRM AGNST CHLD',
        'CONTRIBUTING', 'TRAIN WRECKING', 'FAILURE TO DISPERSE', 'BLOCKING DOOR INDUCTION CENTER'
    ]):
        return 'العنف والاعتداء / Violence and Assault'

    # 3️⃣ التخريب والتدمير / Vandalism and Destruction
    elif any(x in crime for x in [
        'VANDALISM', 'ARSON', 'SHOTS FIRED', 'THROWING OBJECT', 'DAMAGE', 'BOMB SCARE','DISTURBING THE PEACE'
    ]):
        return 'التخريب والتدمير / Vandalism and Destruction'

    # 4️⃣ الاحتيال والتزوير / Fraud and Forgery
    elif any(x in crime for x in [
        'CREDIT CARDS', 'EMBEZZLEMENT', 'DEFRAUDING', 'THEFT OF SERVICES',
        'DOCUMENT WORTHLESS', 'GRAND THEFT / INSURANCE FRAUD', 'THEFT OF IDENTITY'
    ]):
        return 'الاحتيال والتزوير / Fraud and Forgery'

    # 5️⃣ المخالفات القانونية والجرائم المتعلقة بالأسلحة / Legal Offences & Weapons
    elif any(x in crime for x in [
        'COURT ORDER', 'VIOLATION OF COURT', 'CONTEMPT', 'FALSE POLICE REPORT',
        'DOCUMENT FORGERY', 'COUNTERFEIT', 'BRIBERY', 'CONSPIRACY', 'THREATENING PHONE CALLS',
        'VIOLATION OF RESTRAINING ORDER', 'VIOLATION OF TEMPORARY RESTRAINING ORDER',
        'TRESPASSING', 'RESISTING ARREST', 'UNAUTHORIZED COMPUTER ACCESS',
        'WEAPON', 'FIREARM', 'BRANDISH', 'DISCHARGE', 'REPLICA FIREARMS', 'FIREARMS RESTRAINING ORDER'
    ]):
        return 'المخالفات القانونية والجرائم المتعلقة بالأسلحة / Legal Offences & Weapons'

    # 6️⃣ الجرائم الجنسية والاتجار / Sexual Crimes & Exploitation
    elif any(x in crime for x in [
        'RAPE', 'SEX', 'INDECENT', 'LEWD', 'SODOMY', 'ORAL COPULATION',
        'SEXUAL PENETRATION', 'CHILD PORNOGRAPHY', 'HUMAN TRAFFICKING',
        'BATTERY WITH SEXUAL CONTACT', 'BEASTIALITY', 'INCEST', 'PEEPING TOM', 'BIGAMY',
        'TRAFFICKING', 'PIMPING', 'PANDERING'
    ]):
        return 'الجرائم الجنسية والاتجار / Sexual Crimes & Exploitation'

    # 8️⃣ جرائم متنوعة / Miscellaneous Crimes
    elif any(x in crime for x in [
        'OTHER MISCELLANEOUS CRIME', 'ANIMAL', 'CRUELTY', 'ILLEGAL DUMPING',
        'LYNCHING', 'INCITING', 'THREAT', 'PROWLER', 'INCITING A RIOT','DRIVING', 'RECKLESS', 'FAILURE TO YIELD', 'DRUNK'
    ]):
        return 'جرائم متنوعة / Miscellaneous Crimes'

    # Default
    return 'جرائم متنوعة / Miscellaneous Crimes'


def preprocess_data(df_raw, save_artifacts=True, artifacts_path="processors"):
    """
    Exécute le pipeline complet de nettoyage et de feature engineering.
    
    Args:
        df_raw (pd.DataFrame): Le dataframe brut chargé depuis le CSV.
        save_artifacts (bool): Si True, sauvegarde les données et encodeurs sur le disque.
        artifacts_path (str): Chemin du dossier où sauvegarder les artefacts.
        
    Returns:
        X (pd.DataFrame): Features processées.
        y (pd.Series): Target encodée.
        encoders (dict): Dictionnaire contenant les LabelEncoders entraînées.
    """
    # Copie pour éviter les warnings de modification
    df = df_raw.copy()

    # --- 1. Nettoyage des noms de colonnes ---
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('/', '_')
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )

    # --- 2. Suppression de colonnes et renommage initial ---
    if "dr_no" in df.columns:
        df.drop("dr_no", axis=1, inplace=True)
    
    df = df.rename(columns={"part_1_2": "crm_categories"})

    # --- 3. Suppression des doublons ---
    df = df.drop_duplicates()

    # --- 4. Conversion des dates et Feature Engineering Temporel ---
    df['date_occ'] = pd.to_datetime(df['date_occ'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    
    df['Year']      = df['date_occ'].dt.year
    df['Month']     = df['date_occ'].dt.month
    df['Day']       = df['date_occ'].dt.day
    df['Hour']      = df['time_occ'] // 100
    df['Minute']    = df['time_occ'] % 100 
    df['Weekday']   = df['date_occ'].dt.weekday
    df['is_weekend']= df['Weekday'].isin([5, 6])

    # --- 5. Gestion des valeurs manquantes (Imputation) ---
    
    # Vict Descent
    df['vict_descent'] = df['vict_descent'].fillna('UNKNOWN')
    df['vict_descent'] = df['vict_descent'].replace({'-': 'UNKNOWN'})

    # Vict Sex
    df['vict_sex'] = df['vict_sex'].fillna('X')
    df['vict_sex'] = df['vict_sex'].replace({'H': 'X', '-': 'X'})

    # Modes (Mocodes, Premis, Status)
    for col in ['mocodes', 'premis_cd', 'premis_desc', 'status']:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

    # Armes
    df['weapon_desc'] = df['weapon_desc'].fillna('NO WEAPON')
    df['weapon_used_cd'] = df['weapon_used_cd'].fillna(0.0)

    # --- 6. Nettoyage de texte (Premis Desc) ---
    if 'premis_desc' in df.columns:
        df['premis_desc'] = df['premis_desc'].str.title()
        df['premis_desc'] = df['premis_desc'].str.replace(r'\(.*?\)', '', regex=True)
        df['premis_desc'] = df['premis_desc'].apply(lambda x: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str(x)))
        df['premis_desc'] = df['premis_desc'].str.strip()
        df['premis_desc'] = df['premis_desc'].str.replace(r'\s+', ' ', regex=True)

    # --- 7. Suppression de colonnes inutiles ou trop vides ---
    cols_to_drop = ['crm_cd_1', 'crm_cd_2', 'crm_cd_3', 'crm_cd_4', 'cross_street']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # --- 8. Encodage ---

    # One-Hot Encoding pour le sexe
    df = pd.get_dummies(df, columns=['vict_sex'], prefix='vict_sex', dtype=int)

    # Label Encoding pour la descendance
    le_descent = LabelEncoder()
    df['Vict_Descent_LE'] = le_descent.fit_transform(df['vict_descent'].astype(str))

    # --- 9. Création de la Target (Crime Class) ---
    df['Crime_Class'] = df['crm_cd_desc'].apply(categorize_crime)

    # Label Encoding de la Target
    le_target = LabelEncoder()
    df['Crime_Class_Enc'] = le_target.fit_transform(df['Crime_Class'])

    # --- 10. Sélection finale des colonnes ---
    cols_to_keep = [
        'Year', 'Month', 'Day', 'Hour', 'Minute', 
        'area', 'rpt_dist_no', 'crm_categories',
        'crm_cd', 'vict_age', 'Vict_Descent_LE',
        'premis_cd', 'weapon_used_cd', 'lat', 'lon'
    ]
    # Ajouter dynamiquement les colonnes issues du One-Hot Encoding
    vict_sex_cols = [col for col in df.columns if col.startswith('vict_sex_')]
    cols_to_keep += vict_sex_cols

    # Séparation X et y
    X = df[cols_to_keep].copy()
    y = df['Crime_Class_Enc']

    # Dictionnaire des encodeurs
    encoders = {
        'le_descent': le_descent,
        'le_target': le_target
    }

    # --- 11. Sauvegarde optionnelle (Code demandé) ---
    if save_artifacts:
        print("\n" + "=" * 80)
        print("SAUVEGARDE POUR L'ÉTAPE DE MODELING")
        print("=" * 80)

        os.makedirs(artifacts_path, exist_ok=True)

        # Identifier les colonnes numériques pour le scaling futur
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # A. Sauvegarder X et y
        data_package = {
            "X": X,
            "y": y,
            "numeric_cols": numeric_cols,
            "categorical_cols": [c for c in X.columns if c not in numeric_cols]
        }
        with open(os.path.join(artifacts_path, "data_for_modeling.pkl"), "wb") as f:
            pickle.dump(data_package, f)
        print("✓ Données X et y sauvegardées (brutes, prêtes pour optimisation)")

        # B. Sauvegarder le LabelEncoder de la cible
        with open(os.path.join(artifacts_path, "label_encoder_target.pkl"), "wb") as f:
            pickle.dump(le_target, f)
        print("✓ LabelEncoder de la cible sauvegardé")

        # C. Sauvegarder le mapping cible (lisible)
        target_mapping = {
            "classes": list(le_target.classes_),
            "mapping": {cls: int(le_target.transform([cls])[0]) for cls in le_target.classes_}
        }
        with open(os.path.join(artifacts_path, "target_mapping.pkl"), "wb") as f:
            pickle.dump(target_mapping, f)
        print("✓ Mapping cible (Dictionnaire) sauvegardé")

        # D. Sauvegarder la config des features
        feature_config = {
            "final_features": list(X.columns),
            "target_name": "Crime_Class_Enc"
        }
        with open(os.path.join(artifacts_path, "features_config.pkl"), "wb") as f:
            pickle.dump(feature_config, f)
        print("✓ Configuration des colonnes sauvegardée")

        print("\n🚀 PRÊT POUR LE MODELING ! Vous pouvez passer au notebook suivant.")

    return X, y, encoders