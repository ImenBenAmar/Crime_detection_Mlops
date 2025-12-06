# frontend/app.py

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, time

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Crime Prediction Dashboard",
    page_icon="👮‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Dictionnaires de Mapping et Constantes
# ==========================================
SEX_MAP = {"Female": "F", "Male": "M", "Unknown": "X"}
DESCENT_MAP = {"White": "W", "Black": "B", "Hispanic": "H", "Asian": "A", "Other": "O"}
STATUS_MAP = {"Investigation Continued": "IC", "Adult Arrest": "AA", "Adult Other": "AO", "Juvenile Arrest": "JA"}
INV_STATUS_MAP = {v: k for k, v in STATUS_MAP.items()}
API_URL = "http://127.0.0.1:7000/predict"

# ==========================================
# Fonction de Prédiction avec Mise en Cache
# ==========================================
# Cette fonction gère l'appel à l'API pour un DataFrame nettoyé.
@st.cache_data
def get_predictions_from_api(df_cleaned):
    list_of_dicts = df_cleaned.to_dict('records')
    predictions = []
    confidences = []
    
    progress_bar = st.progress(0, text="Prédiction en cours...")
    
    for i, payload in enumerate(list_of_dicts):
        try:
            response = requests.post(API_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                result = response.json()
                predictions.append(result['prediction'])
                confidences.append(result.get('confidence'))
            else:
                predictions.append(f"Erreur API ({response.status_code})")
                confidences.append(None)
        except Exception:
            predictions.append("Erreur de Connexion")
            confidences.append(None)
        
        progress_bar.progress((i + 1) / len(list_of_dicts), text=f"Prédiction en cours... {i+1}/{len(list_of_dicts)}")
        
    progress_bar.empty()
    return predictions, confidences

# ==========================================
# Interface Principale
# ==========================================
st.title("👮‍♂️ Dashboard de Prédiction et d'Analyse de la Criminalité")
st.sidebar.title("Configuration")

input_method = st.sidebar.radio(
    "**1. Choisissez votre méthode de saisie**",
    ("Formulaire Manuel (une seule prédiction)", "Téléverser un fichier CSV (prédictions en masse)")
)

# =============================================================================
# --- CAS 1: FORMULAIRE MANUEL (Version Complète)
# =============================================================================
if input_method == "Formulaire Manuel (une seule prédiction)":
    st.sidebar.header("📝 Remplir les Détails de l'Incident")
    st.sidebar.info("Ce formulaire contient tous les champs bruts nécessaires au modèle. Le pré-traitement est géré automatiquement par l'API.")

    with st.sidebar.form(key='crime_form_full'):
        st.subheader("🗓️ Date et Heure")
        col1, col2 = st.columns(2)
        with col1:
            date_occ_val = st.date_input("Date de l'incident", value=datetime(2023, 1, 1))
        with col2:
            time_occ_val_input = st.time_input("Heure de l'incident", value=time(13, 30))
        
        time_occ_val = int(time_occ_val_input.strftime("%H%M"))
        date_occ_str = date_occ_val.strftime("%m/%d/%Y") + " " + time_occ_val_input.strftime("%I:%M:%S %p")

        st.subheader("📍 Lieu et Zone")
        area_val = st.number_input("Zone (AREA)", min_value=1, max_value=21, value=1, step=1, help="Numéro de la zone de police (1-21)")
        rpt_dist_no_val = st.number_input("District de rapport (Rpt Dist No)", value=784, step=1)
        location_val = st.text_input("Adresse (LOCATION)", value="800 W OLYMPIC BLVD")
        lat_val = st.number_input("Latitude (LAT)", value=34.0459, format="%.4f")
        lon_val = st.number_input("Longitude (LON)", value=-118.2623, format="%.4f")
        
        st.subheader("👤 Détails sur la Victime")
        vict_age_val = st.number_input("Âge de la victime (Vict Age)", min_value=0, max_value=120, value=35)
        vict_sex_display = st.selectbox("Sexe de la victime (Vict Sex)", options=list(SEX_MAP.keys()), index=1)
        vict_descent_display = st.selectbox("Origine de la victime (Vict Descent)", options=list(DESCENT_MAP.keys()), index=0)

        st.subheader("📜 Détails sur le Crime")
        part_1_2_val = st.number_input("Type de rapport (Part 1-2)", value=1, step=1, help="Généralement 1 ou 2")
        crm_cd_val = st.number_input("Code du crime (Crm Cd)", value=510, step=1, help="Code numérique spécifique au crime (ex: 510 pour vol de véhicule)")
        mocodes_val = st.text_input("Modus Operandi (Mocodes)", value="0344 1822", help="Codes décrivant la méthode du crime, séparés par des espaces.")
        status_display = st.selectbox("Statut de l'affaire (Status)", options=list(STATUS_MAP.keys()), index=0)

        st.subheader("🏢 Lieux et Armes")
        premis_cd_val = st.number_input("Code des lieux (Premis Cd)", value=101.0, format="%.1f")
        premis_desc_val = st.text_input("Description des lieux (Premis Desc)", value="STREET")
        weapon_used_cd_val = st.number_input("Code de l'arme (Weapon Used Cd)", value=400.0, format="%.1f", help="0 si aucune arme")
        weapon_desc_val = st.text_input("Description de l'arme (Weapon Desc)", value="STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)")
        
        submit_button = st.form_submit_button(label='▶️ Obtenir la Prédiction')

    if submit_button:
        payload = {
            "DATE OCC": date_occ_str, "TIME OCC": time_occ_val, "AREA": area_val,
            "Rpt Dist No": rpt_dist_no_val, "Part 1-2": part_1_2_val, "Crm Cd": crm_cd_val,
            "Mocodes": mocodes_val, "Vict Age": vict_age_val, "Vict Sex": SEX_MAP[vict_sex_display],
            "Vict Descent": DESCENT_MAP[vict_descent_display], "Premis Cd": premis_cd_val,
            "Premis Desc": premis_desc_val, "Weapon Used Cd": weapon_used_cd_val,
            "Weapon Desc": weapon_desc_val, "Status": STATUS_MAP[status_display],
            "LOCATION": location_val, "LAT": lat_val, "LON": lon_val
        }
        df_single = pd.DataFrame([payload])
        predictions, confidences = get_predictions_from_api(df_single)

        if predictions:
            prediction_code = predictions[0]
            confidence = confidences[0] if confidences and confidences[0] is not None else 0
            prediction_display = INV_STATUS_MAP.get(prediction_code, prediction_code)
            
            st.header("Résultat de la Prédiction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Catégorie de Crime Prédite", value=prediction_display)
                st.metric(label="Niveau de Confiance", value=f"{confidence:.2%}")
                if confidence < 0.6: st.warning("Confiance faible. Prédiction à considérer avec prudence.", icon="⚠️")
            with col2:
                map_data = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                st.map(map_data, zoom=13)

# =============================================================================
# --- CAS 2: TÉLÉVERSEMENT CSV
# =============================================================================
elif input_method == "Téléverser un fichier CSV (prédictions en masse)":
    st.sidebar.header("📤 Téléversement")
    uploaded_file = st.sidebar.file_uploader("**2. Choisissez un fichier CSV**", type="csv")

    if uploaded_file is not None:
        try:
            df_original = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            if df_original.columns[0].startswith('Unnamed'): df_original = df_original.iloc[:, 1:]
            
            st.header(f"Analyse du fichier : `{uploaded_file.name}`")
            st.metric(label="Nombre de lignes à traiter", value=len(df_original))
            
            required_columns = {
                "DATE OCC": str, "TIME OCC": int, "AREA": int, "Part 1-2": int, "Crm Cd": int,
                "Vict Age": float, "Vict Sex": str, "Vict Descent": str, "Premis Cd": float,
                "Premis Desc": str, "Weapon Used Cd": float, "Weapon Desc": str, "Status": str,
                "LOCATION": str, "LAT": float, "LON": float, "Rpt Dist No": int, "Mocodes": str
            }
            df_cleaned = pd.DataFrame()
            for col, expected_type in required_columns.items():
                if col in df_original.columns:
                    series = df_original[col]
                    if pd.api.types.is_numeric_dtype(expected_type):
                        series = pd.to_numeric(series, errors='coerce').fillna(0)
                    else:
                        series = series.fillna('')
                    df_cleaned[col] = series.astype(expected_type)
            
            if st.button("🚀 Lancer les Prédictions sur le Fichier", type="primary"):
                predictions, confidences = get_predictions_from_api(df_cleaned)

                if predictions:
                    df_results = df_original.copy()
                    df_results['PREDICTION_CODE'] = predictions
                    df_results['PREDICTION_LABEL'] = pd.Series(predictions).map(INV_STATUS_MAP).fillna(pd.Series(predictions))
                    df_results['CONFIDENCE'] = confidences
                    
                    tab1, tab2, tab3 = st.tabs(["📊 Tableau de Bord", "📍 Carte des Crimes", "📄 Données Complètes"])

                    with tab1:
                        st.subheader("Synthèse des Prédictions")
                        col1, col2 = st.columns(2)
                        with col1:
                            success_count = len([p for p in predictions if 'Erreur' not in str(p)])
                            st.metric("Prédictions Réussies", f"{success_count}/{len(df_original)}")
                            avg_confidence = pd.Series([c for c in confidences if c is not None]).mean()
                            if pd.notna(avg_confidence): st.metric("Confiance Moyenne", f"{avg_confidence:.2%}")
                        with col2:
                            st.subheader("Top 5 des Catégories Prédites")
                            top_5_crimes = df_results['PREDICTION_LABEL'].value_counts().nlargest(5)
                            st.bar_chart(top_5_crimes)

                    with tab2:
                        st.subheader("Carte des Incidents Prédits")
                        if 'LAT' in df_results.columns and 'LON' in df_results.columns:
                            df_map = df_results.rename(columns={'LAT': 'lat', 'LON': 'lon'})
                            st.map(df_map.dropna(subset=['lat', 'lon']), zoom=10)
                        else:
                            st.warning("Colonnes 'LAT' et 'LON' manquantes pour afficher la carte.")

                    with tab3:
                        st.subheader("Résultats Détaillés")
                        st.dataframe(df_results)
                        @st.cache_data
                        def convert_df(df):
                            return df.to_csv(index=False).encode('utf-8')
                        csv_results = convert_df(df_results)
                        st.download_button(
                            label="📥 Télécharger les résultats en CSV",
                            data=csv_results,
                            file_name=f'predictions_{uploaded_file.name}',
                            mime='text/csv',
                        )
        except Exception as e:
            st.error(f"Une erreur est survenue lors du traitement du fichier : {e}", icon="🚨")

# Section "À Propos" dans la barre latérale
st.sidebar.title(" ")
with st.sidebar.expander("ℹ️ À Propos de l'Application"):
    st.info("""
    Cette application est une interface pour un modèle de Machine Learning qui prédit les catégories de crimes.
    - **Backend:** API FastAPI avec MLflow.
    - **Frontend:** Streamlit.
    - **Modèle:** Classifieur entraîné sur des données de crimes.
    - **Auteur:** YomnaJL
    """)