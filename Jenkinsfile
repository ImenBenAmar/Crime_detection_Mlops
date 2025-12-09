pipeline {
    agent any

    environment {
        // --- Configuration Docker Hub ---
        // Votre repo unique
        DOCKER_IMAGE_NAME = 'imen835/mlops-crime'
        
        // --- Secrets (Injectés depuis Jenkins Credentials) ---
        // Assurez-vous d'avoir créé ces IDs dans Jenkins > Credentials
        DAGSHUB_TOKEN = credentials('dagshub-token-id')
        DOCKERHUB_CREDS = credentials('dockerhub-id') // User: imen835, Pass: ...
        ARIZE_API_KEY = credentials('arize-api-key-id')
        
        // --- Variables non sensibles ---
        DAGSHUB_USERNAME = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = 'https://dagshub.com/YomnaJL/MLOPS_Project.mlflow'
        ARIZE_SPACE_ID = 'U3BhY2U6MzEyNjA6QzFTdw=='
    }

    stages {
        // 1. Nettoyage et Récupération du code
        stage('Clean & Checkout') {
            steps {
                cleanWs()
                checkout scm
            }
        }

        // 2. Tests Unitaires (Backend)
        stage('Test Backend') {
            steps {
                script {
                    echo "🚀 Lancement des tests dans un environnement isolé..."
                    // On utilise Docker pour lancer les tests (Python 3.9)
                    docker.image('python:3.9-slim').inside {
                        // Installation des dépendances
                        sh 'pip install --no-cache-dir -r backend/src/requirements-backend.txt'
                        sh 'pip install pytest'
                        
                        // Exécution des tests avec injection des secrets en mémoire
                        // (Ils ne seront pas écrits sur le disque)
                        withEnv([
                            "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}",
                            "DAGSHUB_USERNAME=${DAGSHUB_USERNAME}",
                            "DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME}",
                            "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}",
                            "ARIZE_SPACE_ID=${ARIZE_SPACE_ID}",
                            "ARIZE_API_KEY=${ARIZE_API_KEY}"
                        ]) {
                            // On ajoute le dossier src au PYTHONPATH pour les imports
                            sh 'export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src && pytest testing/'
                        }
                    }
                }
            }
        }

        // 3. Construction des Images (Build)
        stage('Build Images') {
            steps {
                script {
                    echo "🏗️ Construction du Backend..."
                    // Tag: backend-BuildNumber (ex: backend-42)
                    sh "docker build -t ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER} -t ${DOCKER_IMAGE_NAME}:backend-latest ./backend/src"
                    
                    echo "🏗️ Construction du Frontend..."
                    // Tag: frontend-BuildNumber (ex: frontend-42)
                    sh "docker build -t ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER} -t ${DOCKER_IMAGE_NAME}:frontend-latest ./frontend"
                }
            }
        }

        // 4. Envoi vers Docker Hub (Push)
        stage('Push to Docker Hub') {
            steps {
                script {
                    echo "🔓 Connexion au registre..."
                    sh "echo $DOCKERHUB_CREDS_PSW | docker login -u $DOCKERHUB_CREDS_USR --password-stdin"
                    
                    echo "⬆️ Push des images..."
                    // Push Backend (Version précise + Latest)
                    sh "docker push ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER}"
                    sh "docker push ${DOCKER_IMAGE_NAME}:backend-latest"
                    
                    // Push Frontend (Version précise + Latest)
                    sh "docker push ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER}"
                    sh "docker push ${DOCKER_IMAGE_NAME}:frontend-latest"
                }
            }
        }
    }

    post {
        always {
            script {
                echo "🧹 Nettoyage des images locales..."
                // On supprime les images de la machine Jenkins pour ne pas saturer le disque
                sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-latest || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-latest || true"
                sh "docker logout"
            }
        }
        success {
            echo "✅ Succès ! Images disponibles sur : https://hub.docker.com/r/imen835/mlops-crime"
        }
        failure {
            echo "❌ Échec du pipeline."
        }
    }
}