pipeline {
    agent any

    // 1. Options pour la maintenance du serveur Jenkins
    options {
        // Ne garde que les 10 derniers builds pour économiser de l'espace
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // Ajoute un timestamp aux logs console
        timestamps()
        // Empêche deux builds de tourner en même temps sur la même branche (évite les conflits)
        disableConcurrentBuilds()
    }

    environment {
        // Nom de base de l'image
        DOCKER_IMAGE_NAME = 'imen835/mlops-crime'
        
        // Récupération du Hash Git court (ex: a1b2c3d) pour la traçabilité MLOps
        GIT_COMMIT_HASH = sh(returnStdout: true, script: "git rev-parse --short HEAD").trim()

        // --- Secrets ---
        DAGSHUB_TOKEN = credentials('daghub-credentials')
        DOCKERHUB_CREDS = credentials('docker-hub-credentials')
        // ARIZE_API_KEY = credentials('arize-api-key-id')
        
        // --- Configs MLOps ---
        DAGSHUB_USERNAME = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = 'https://dagshub.com/YomnaJL/MLOPS_Project.mlflow'
        // ARIZE_SPACE_ID = 'U3BhY2U6MzEyNjA6QzFTdw=='
    }

    stages {
        // Étape 1 : Préparation
        stage('Initialize') {
            steps {
                cleanWs() // Nettoie le workspace avant de commencer
                checkout scm //Récupérer le code du repository
                script {
                    echo "ℹ️ Démarrage du Build #${BUILD_NUMBER} sur le commit ${GIT_COMMIT_HASH}"
                }
            }
        }

        // Étape 2 : CI (Intégration Continue - Qualité & Tests)
        stage('CI: Quality & Tests') {
            steps {
                script {
                    docker.image('python:3.9-slim').inside {
                        
                        echo "📦 Création d'un environnement virtuel (pour éviter les erreurs de permission)..."
                        // 1. Création du venv nommé 'venv'
                        sh 'python -m venv venv'
                        
                        echo "⬇️ Installation des dépendances dans le venv..."
                        // 2. On utilise le pip DU venv (./venv/bin/pip)
                        // Note : J'ai gardé le timeout=1000 pour ta connexion internet
                        sh './venv/bin/pip install --upgrade pip'
                        sh './venv/bin/pip install --default-timeout=1000 --no-cache-dir -r backend/src/requirements-backend.txt'
                        sh './venv/bin/pip install --default-timeout=1000 pytest flake8 pytest-cov' 

                        echo "🔍 Linting..."
                        // 3. On utilise le flake8 DU venv
                        sh './venv/bin/flake8 backend/src --count --select=E9,F63,F7,F82 --show-source --statistics || true'

                        echo "🚀 Tests..."
                        withEnv([
                            "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}",
                            "DAGSHUB_USERNAME=${DAGSHUB_USERNAME}",
                            "DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME}",
                            "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
                        ]) {
                            // 4. On lance pytest via le venv
                            // Important : PYTHONPATH doit pointer sur ton code source
                            sh 'export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src && ./venv/bin/pytest testing/ --junitxml=test-results.xml'
                        }
                    }
                }
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }

        // Étape 3 : Login Docker (Séparé pour éviter l'erreur de syntaxe parallel)
        stage('Docker Login') {
            steps {
                script {
                    echo "🔓 Connexion sécurisée au Docker Hub..."
                    sh "echo $DOCKERHUB_CREDS_PSW | docker login -u $DOCKERHUB_CREDS_USR --password-stdin"
                }
            }
        }

        // Étape 4 : CD (Livraison Continue - Build & Push)
        stage('CD: Build & Push Images') {
            // Le bloc parallel doit être direct ici
            parallel {
                stage('Backend Image') {
                    steps {
                        script {
                            def imageBackend = "${DOCKER_IMAGE_NAME}:backend"
                            echo "🏗️ Construction Backend..."
                            // On tague avec :latest, :BuildNumber, et :GitHash
                            sh "docker build -t ${imageBackend}-${BUILD_NUMBER} -t ${imageBackend}-${GIT_COMMIT_HASH} -t ${imageBackend}-latest ./backend/src"
                            
                            echo "⬆️ Envoi Backend..."
                            sh "docker push ${imageBackend}-${BUILD_NUMBER}"
                            sh "docker push ${imageBackend}-${GIT_COMMIT_HASH}"
                            sh "docker push ${imageBackend}-latest"
                        }
                    }
                }
                
                stage('Frontend Image') {
                    steps {
                        script {
                            def imageFrontend = "${DOCKER_IMAGE_NAME}:frontend"
                            echo "🏗️ Construction Frontend..."
                            sh "docker build -t ${imageFrontend}-${BUILD_NUMBER} -t ${imageFrontend}-${GIT_COMMIT_HASH} -t ${imageFrontend}-latest ./frontend"
                            
                            echo "⬆️ Envoi Frontend..."
                            sh "docker push ${imageFrontend}-${BUILD_NUMBER}"
                            sh "docker push ${imageFrontend}-${GIT_COMMIT_HASH}"
                            sh "docker push ${imageFrontend}-latest"
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                echo "🧹 Nettoyage de l'environnement..."
                // Nettoyage agressif pour éviter de saturer le disque
                sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-${GIT_COMMIT_HASH} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-latest || true"
                
                sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-${GIT_COMMIT_HASH} || true"
                sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-latest || true"
                
                sh "docker logout"
            }
        }
        success {
            echo "✅ Pipeline réussi !"
            echo "🐳 Images disponibles sur DockerHub : ${DOCKER_IMAGE_NAME}"
            echo "🔖 Commit déployé : ${GIT_COMMIT_HASH}"
        }
        failure {
            echo "❌ Le pipeline a échoué. Vérifiez les logs."
        }
    }
}