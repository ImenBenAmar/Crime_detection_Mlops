# Étape 1: Utiliser une image Python officielle comme base
FROM python:3.9-slim

# Étape 2: Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3: Copier le fichier des dépendances et les installer
# On copie d'abord ce fichier pour profiter du cache Docker.
# Si requirements.txt ne change pas, Docker n'exécutera pas cette étape à chaque build.
COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

# Étape 4: Copier tout le code source du backend dans le conteneur
COPY . .

# Étape 5: Exposer le port que l'API va utiliser
# Uvicorn tournera sur le port 5000 à l'intérieur du conteneur
EXPOSE 5000

# Étape 6: La commande pour lancer l'application lorsque le conteneur démarre
# On utilise --host 0.0.0.0 pour que l'API soit accessible depuis l'extérieur du conteneur.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]