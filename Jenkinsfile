pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        // ✅ Ton nom d'utilisateur Docker Hub correct
        DOCKER_IMAGE_NAME = 'imen835/mlops-crime'
        GIT_COMMIT_HASH = sh(returnStdout: true, script: "git rev-parse --short HEAD").trim()

        // --- Secrets ---
        DAGSHUB_TOKEN = credentials('daghub-credentials')
        DOCKERHUB_CREDS = credentials('docker-hub-credentials')
        
        // --- Configs MLOps ---
        DAGSHUB_USERNAME = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = 'https://dagshub.com/YomnaJL/MLOPS_Project.mlflow'
    }

    stages {
        stage('Initialize') {
            steps {
                cleanWs()
                checkout scm
                script {
                    echo "ℹ️ Build #${BUILD_NUMBER} - Commit ${GIT_COMMIT_HASH}"
                }
            }
        }

        stage('CI: Quality & Tests') {
            steps {
                script {
                    docker.image('python:3.9-slim').inside {
                        sh 'python -m venv venv'
                        sh './venv/bin/pip install --upgrade pip'
                        sh './venv/bin/pip install --default-timeout=1000 --no-cache-dir -r backend/src/requirements-backend.txt'
                        sh './venv/bin/pip install --default-timeout=1000 pytest flake8 pytest-cov' 
                        
                        // Linting & Tests
                        sh './venv/bin/flake8 backend/src --count --select=E9,F63,F7,F82 --show-source --statistics || true'
                        
                        withEnv([
                            "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}",
                            "DAGSHUB_USERNAME=${DAGSHUB_USERNAME}",
                            "DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME}",
                            "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
                        ]) {
                            sh 'export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src && ./venv/bin/pytest testing/ --junitxml=test-results.xml'
                        }
                    }
                }
            }
            post {
                always { junit 'test-results.xml' }
            }
        }

        stage('Docker Login') {
            steps {
                script {
                    sh "echo $DOCKERHUB_CREDS_PSW | docker login -u $DOCKERHUB_CREDS_USR --password-stdin"
                }
            }
        }

        stage('CD: Build & Push Images') {
            parallel {
                stage('Backend Image') {
                    steps {
                        script {
                            def imageBackend = "${DOCKER_IMAGE_NAME}:backend"
                            // Build
                            sh "docker build -t ${imageBackend}-${BUILD_NUMBER} -t ${imageBackend}-${GIT_COMMIT_HASH} -t ${imageBackend}-latest ./backend/src"
                            // Push
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
                            // Build
                            sh "docker build -t ${imageFrontend}-${BUILD_NUMBER} -t ${imageFrontend}-${GIT_COMMIT_HASH} -t ${imageFrontend}-latest ./frontend"
                            // Push
                            sh "docker push ${imageFrontend}-${BUILD_NUMBER}"
                            sh "docker push ${imageFrontend}-${GIT_COMMIT_HASH}"
                            sh "docker push ${imageFrontend}-latest"
                        }
                    }
                }
            }
        }

        // --- NOUVELLES ÉTAPES POUR LE DÉPLOIEMENT ---

        stage('Update Manifests') {
            steps {
                script {
                    echo "📝 Mise à jour des fichiers Kubernetes..."
                    
                    // Définition des images exactes pour ce build
                    def newBackendImage = "${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER}"
                    def newFrontendImage = "${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER}"
                    
                    // Utilisation de SED pour remplacer les PLACEHOLDERS dans les fichiers YAML
                    // On utilise '|' comme séparateur pour ne pas casser les slashs de l'image
                    sh "sed -i 's|REPLACE_ME_BACKEND_IMAGE|${newBackendImage}|g' k8s/backend-deployment.yaml"
                    sh "sed -i 's|REPLACE_ME_FRONTEND_IMAGE|${newFrontendImage}|g' k8s/frontend-deployment.yaml"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                script {
                    echo "🚀 Déploiement vers Kubernetes..."
                    // Application des fichiers modifiés
                    sh "kubectl apply -f k8s/backend-deployment.yaml"
                    sh "kubectl apply -f k8s/frontend-deployment.yaml"
                }
            }
        }
    }

    post {
        always {
            script {
                echo "🧹 Nettoyage..."
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
            echo "✅ Pipeline et Déploiement réussis !"
        }
        failure {
            echo "❌ Échec du pipeline."
        }
    }
}