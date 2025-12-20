import os
import sys
from dotenv import load_dotenv
import mlflow

# 1. Charger les variables d'environnement (Tokens)
load_dotenv()

# 2. Configurer les chemins pour importer ton code
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../backend/src')))

# On importe les VRAIES fonctions de ton API
from api import setup_mlflow, download_model_from_registry, REGISTERED_MODEL_NAME

def test_actual_dagshub_registry():
    print("\n" + "="*60)
    print(f"📡 CONNEXION RÉELLE AU REGISTRY DAGSHUB")
    print("="*60)

    # A. Initialisation de la connexion
    print(f"🔗 Tentative de connexion à : {os.getenv('DAGSHUB_REPO_NAME')}...")
    setup_mlflow()
    
    print(f"📍 Tracking URI actuel : {mlflow.get_tracking_uri()}")

    # B. Appel de la fonction de téléchargement réelle
    print(f"\n🔍 Recherche du modèle '{REGISTERED_MODEL_NAME}' dans le Cloud...")
    
    try:
        # On appelle ta fonction qui interroge DagsHub
        model, model_name, processors_path = download_model_from_registry()

        if model:
            print("\n✨ --- RÉSULTATS CLOUD RÉELS ---")
            print(f"✅ MODÈLE TROUVÉ  : {REGISTERED_MODEL_NAME}")
            print(f"✅ VERSION DÉTECTÉE : {model_name.split('_v')[-1]}")
            print(f"✅ NOM COMPLET     : {model_name}")
            print(f"📂 CHEMIN LOCAL DES ARTEFACTS : {processors_path}")
            
            # Vérification physique des fichiers téléchargés
            files = os.listdir(processors_path)
            print(f"📦 FICHIERS RÉCUPÉRÉS : {files}")
            
            print("\n🏆 SUCCÈS : Ton API est parfaitement connectée à DagsHub !")
        else:
            print("\n⚠️ CONNEXION OK mais AUCUN MODÈLE trouvé dans le Registry.")
            print(f"Vérifiez que le nom '{REGISTERED_MODEL_NAME}' est bien écrit sur DagsHub.")

    except Exception as e:
        print(f"\n❌ ERREUR DE CONNEXION : {str(e)}")
        print("Vérifiez votre DAGSHUB_TOKEN et votre connexion internet.")

if __name__ == "__main__":
    test_actual_dagshub_registry()