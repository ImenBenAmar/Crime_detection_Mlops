#Vous avez ouvert l'interface, c'est bien.
#Mais est-ce que le Backend arrive vraiment à télécharger le modèle depuis DagsHub et à faire une prédiction sans crasher (OOMKilled) ?


import requests
import json

# L'URL locale grâce au port-forward (Backend direct ou via Frontend)
# Ici on teste l'API directement pour voir les logs techniques
URL = "http://127.0.0.1:5000/predict" 

# Note : Si vous voulez tester via le tunnel frontend, l'URL backend n'est pas exposée.
# Il faut faire un 'kubectl port-forward service/backend-service 5000:5000' dans un autre terminal.

data = {
  "DATE OCC": "01/01/2023 12:00:00 PM",
  "TIME OCC": 1200,
  "AREA": 1,
  "Part 1-2": 1,
  "Vict Age": 30,
  "Vict Sex": "M",
  "Vict Descent": "W",
  "Premis Cd": 101.0,
  "Weapon Used Cd": 100.0,
  "Status": "AA",
  "Mocodes": "0100",
  "LOCATION": "POINT(34.0 -118.2)",
  "LAT": 34.0,
  "LON": -118.2
}

try:
    print(f"📡 Envoi de la requête à {URL}...")
    response = requests.post(URL, json=data)
    
    if response.status_code == 200:
        print("✅ SUCCÈS ! Réponse du cluster Kubernetes :")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ ERREUR {response.status_code} :")
        print(response.text)

except Exception as e:
    print(f"❌ Impossible de contacter l'API : {e}")
    print("Astuce : Avez-vous lancé 'kubectl port-forward service/backend-service 5000:5000' ?")