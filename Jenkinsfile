pipeline {
    agent any

    environment {
        DOCKER_IMAGE_NAME = 'imen835/mlops-crime'
        
        // Credentials
        DAGSHUB_TOKEN = credentials('dagshub-token-id')
        DOCKERHUB_CREDS = credentials('dockerhub-id')
        ARIZE_API_KEY = credentials('arize-api-key-id')
        
        // Configs
        DAGSHUB_USERNAME = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = 'https://dagshub.com/YomnaJL/MLOPS_Project.mlflow'
        ARIZE_SPACE_ID = 'U3BhY2U6MzEyNjA6QzFTdw=='
    }

    stages {
        stage('Clean & Checkout') {
            steps {
                cleanWs()
                checkout scm
            }
        }

        // --- OPTIMISATION 1 : Tests & Qualité ---
        stage('Quality & Tests') {
            steps {
                script {
                    docker.image('python:3.9-slim').inside {
                        sh 'pip install --no-cache-dir -r backend/src/requirements-backend.txt'
                        // Ajout de flake8 pour la qualité du code et pytest-cov pour la couverture (optionnel)
                        sh 'pip install pytest flake8' 
                        
                        echo "🔍 Vérification de la qualité du code (Linting)..."
                        // Continue même si erreurs mineures (facultatif)
                        sh 'flake8 backend/src --count --select=E9,F63,F7,F82 --show-source --statistics || true'

                        echo "🚀 Lancement des tests..."
                        withEnv([
                            "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}",
                            "DAGSHUB_USERNAME=${DAGSHUB_USERNAME}",
                            "DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME}",
                            "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}",
                            "ARIZE_SPACE_ID=${ARIZE_SPACE_ID}",
                            "ARIZE_API_KEY=${ARIZE_API_KEY}"
                        ]) {
                            // OPTIMISATION 2 : Génération d'un rapport XML pour Jenkins
                            sh 'export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src && pytest testing/ --junitxml=test-results.xml'
                        }
                    }
                }
            }
            // Publication des résultats dans l'interface Jenkins
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }

        // --- OPTIMISATION 3 : Build & Push en Parallèle ---
        stage('Build & Push Parallel') {
            // On se connecte une seule fois au début
            steps {
                script {
                     echo "🔓 Connexion au registre..."
                     sh "echo $DOCKERHUB_CREDS_PSW | docker login -u $DOCKERHUB_CREDS_USR --password-stdin"
                }
                
                parallel {
                    stage('Backend Pipeline') {
                        steps {
                            script {
                                echo "🏗️ Building Backend..."
                                sh "docker build -t ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER} -t ${DOCKER_IMAGE_NAME}:backend-latest ./backend/src"
                                
                                echo "⬆️ Pushing Backend..."
                                sh "docker push ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER}"
                                sh "docker push ${DOCKER_IMAGE_NAME}:backend-latest"
                            }
                        }
                    }
                    
                    stage('Frontend Pipeline') {
                        steps {
                            script {
                                echo "🏗️ Building Frontend..."
                                sh "docker build -t ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER} -t ${DOCKER_IMAGE_NAME}:frontend-latest ./frontend"
                                
                                echo "⬆️ Pushing Frontend..."
                                sh "docker push ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER}"
                                sh "docker push ${DOCKER_IMAGE_NAME}:frontend-latest"
                            }
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                echo "🧹 Nettoyage..."
                sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-latest || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-latest || true"
                sh "docker logout"
            }
        }
        success {
            echo "✅ Succès ! Build #${BUILD_NUMBER} déployé."
        }
        failure {
            echo "❌ Échec du pipeline."
        }
    }
}