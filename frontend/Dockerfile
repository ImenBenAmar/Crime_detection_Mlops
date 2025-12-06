# Étape 1: Utiliser une image Python officielle comme base
FROM python:3.9-slim

# Étape 2: Définir le répertoire de travail
WORKDIR /app

# Étape 3: Copier et installer les dépendances
COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

# Étape 4: Copier le code source du frontend
COPY . .

# Étape 5: Exposer le port par défaut de Streamlit
EXPOSE 8501

# Étape 6: La commande pour lancer l'application
# --server.address 0.0.0.0 permet d'accéder à l'interface depuis l'extérieur du conteneur
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]