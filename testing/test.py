import mlflow

mlflow_tracking_uri = "https://c669e471815fdf79c8aef414adc9fa1f2946d6f3@dagshub.com/YomnaJL/MLOPS_Project.mlflow"
mlflow.set_tracking_uri(mlflow_tracking_uri)

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("default")  # nom exact de l'expérience sur DagsHub
if exp:
    print("Connexion OK :", exp)
else:
    print("L'expérience 'default' n'existe pas. Vérifie son nom sur DagsHub.")
