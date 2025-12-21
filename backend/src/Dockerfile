# Étape 1: Image de base
FROM python:3.9-slim

# Étape 2: Définir le répertoire de travail
WORKDIR /app

# Étape 3: Copier les requirements backend uniquement
COPY requirements-backend.txt .

# Étape 4: Installer les dépendances backend seulement
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-backend.txt

# Étape 5: Copier tout le code source
COPY . .

# Étape 6: Exposer le port de l'API
EXPOSE 5000

# Étape 7: Commande pour lancer l'API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
