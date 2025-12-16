pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
        disableConcurrentBuilds()
        // Timeout global du job pour éviter qu'il ne tourne indéfiniment
        timeout(time: 1, unit: 'HOURS') 
    }

    environment {
        // ✅ Variable statique (OK ici)
        DOCKER_IMAGE_NAME = 'imen835/mlops-crime'
        
        // ❌ GIT_COMMIT_HASH supprimé d'ici car il nécessite le code source d'abord

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
                // ✅ CORRECTION 1 : Configuration Git explicite avec Timeout augmenté
                checkout([$class: 'GitSCM', 
                    branches: [[name: '*/main']], // ou '*/master' selon votre branche
                    doGenerateSubmoduleConfigurations: false, 
                    extensions: [
                        [$class: 'CloneOption', timeout: 60, shallow: false, noTags: false]
                    ], 
                    submoduleCfg: [], 
                    userRemoteConfigs: [[credentialsId: 'github-credentials', url: 'https://github.com/votre-repo/votre-projet.git']] // Mettez l'URL correcte si 'scm' ne marche pas
                ])
                
                script {
                    // ✅ CORRECTION 2 : Calcul du Hash après le checkout
                    env.GIT_COMMIT_HASH = sh(returnStdout: true, script: "git rev-parse --short HEAD").trim()
                    echo "ℹ️ Build #${BUILD_NUMBER} - Commit ${env.GIT_COMMIT_HASH}"
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
                always { 
                    // junit 'test-results.xml' // Commenté si le fichier n'est pas généré en cas d'échec d'install
                    echo "Fin des tests"
                }
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
                            sh "docker build -t ${imageBackend}-${BUILD_NUMBER} -t ${imageBackend}-${GIT_COMMIT_HASH} -t ${imageBackend}-latest ./backend/src"
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
                            sh "docker build -t ${imageFrontend}-${BUILD_NUMBER} -t ${imageFrontend}-${GIT_COMMIT_HASH} -t ${imageFrontend}-latest ./frontend"
                            sh "docker push ${imageFrontend}-${BUILD_NUMBER}"
                            sh "docker push ${imageFrontend}-${GIT_COMMIT_HASH}"
                            sh "docker push ${imageFrontend}-latest"
                        }
                    }
                }
            }
        }

        stage('Update Manifests') {
            steps {
                script {
                    echo "📝 Mise à jour des fichiers Kubernetes..."
                    def newBackendImage = "${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER}"
                    def newFrontendImage = "${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER}"
                    
                    sh "sed -i 's|REPLACE_ME_BACKEND_IMAGE|${newBackendImage}|g' k8s/backend-deployment.yml"
                    sh "sed -i 's|REPLACE_ME_FRONTEND_IMAGE|${newFrontendImage}|g' k8s/frontend-deployment.yml"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                script {
                    echo "🚀 Déploiement vers Kubernetes..."
                    withCredentials([file(credentialsId: 'kubeconfig-secret', variable: 'KUBECONFIG')]) {
                        sh "kubectl apply -f k8s/backend-deployment.yml"
                        sh "kubectl apply -f k8s/frontend-deployment.yml"
                        sh "sleep 5"
                        sh "kubectl get pods" 
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                echo "🧹 Nettoyage..."
                // ✅ CORRECTION 3 : Vérifier si GIT_COMMIT_HASH existe avant de nettoyer
                if (env.GIT_COMMIT_HASH) {
                    sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-${BUILD_NUMBER} || true"
                    sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-${GIT_COMMIT_HASH} || true"
                    sh "docker rmi ${DOCKER_IMAGE_NAME}:backend-latest || true"
                    sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-${BUILD_NUMBER} || true"
                    sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-${GIT_COMMIT_HASH} || true"
                    sh "docker rmi ${DOCKER_IMAGE_NAME}:frontend-latest || true"
                } else {
                    echo "⚠️ Checkout échoué, pas d'images Docker à nettoyer."
                }
                sh "docker logout || true"
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