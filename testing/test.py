import mlflow
import os
from dotenv import load_dotenv

# 1. Charger les variables du fichier .env
# (Assurez-vous que le fichier .env est à la racine du projet ou ajustez le chemin)
load_dotenv() 

# Récupération des variables
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN")

print(f"📂 Chargement de la configuration...")
if not tracking_uri or not username or not password:
    print("❌ ERREUR : Certaines variables sont manquantes dans le .env")
    print(f"   - URI: {tracking_uri}")
    print(f"   - User: {username}")
    print(f"   - Pass: {'******' if password else 'MANQUANT'}")
    exit(1)

# 2. Configuration MLflow
os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = password
mlflow.set_tracking_uri(tracking_uri)

print(f"📡 Connexion à : {tracking_uri}")

# 3. Test de connexion et listage des expériences
client = mlflow.tracking.MlflowClient()

try:
    experiments = client.search_experiments()
    
    if not experiments:
        print("⚠️ Aucune expérience trouvée. C'est vide !")
    else:
        print(f"\n✅ Connexion RÉUSSIE ! Voici les expériences disponibles :")
        print("=" * 60)
        print(f"{'ID':<5} | {'Nom de l\'expérience':<30} | {'État'}")
        print("-" * 60)
        
        for exp in experiments:
            print(f"{exp.experiment_id:<5} | {exp.name:<30} | {exp.lifecycle_stage}")
            
        print("=" * 60)
        print("👉 Utilisez l'un de ces noms exacts dans votre code.")

except Exception as e:
    print(f"\n❌ Échec de la connexion : {e}")
    print("Vérifiez que votre token DagsHub est valide et que l'URL ne contient pas de fautes de frappe.")
