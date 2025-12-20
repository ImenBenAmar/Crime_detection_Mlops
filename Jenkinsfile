pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '5'))
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 1, unit: 'HOURS') 
    }

    environment {
        DOCKER_IMAGE_NAME = 'imen835/mlops-crime'
        DOCKERHUB_CREDS = credentials('docker-hub-credentials')
        
        DAGSHUB_USERNAME = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.mlflow"
        
        ACTIVATE_VENV = ". venv/bin/activate"
        PYTHON_PATH_CMD = "export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src"
        // Supprime les warnings Git et force le mode sans écran
        GIT_PYTHON_REFRESH = "quiet"
    }

    stages {
        stage('1. Initialize & Docker Login') {
            steps {
                cleanWs()
                checkout scm
                script {
                    env.GIT_COMMIT_HASH = sh(returnStdout: true, script: "git rev-parse --short HEAD").trim()
                    
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                        sh "echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin"
                    }
                }
            }
        }

        stage('2. Pull Data (DVC)') {
            steps {
                script {
                    echo "📥 Récupération des données depuis DagsHub..."
                    def dagshubUrl = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.dvc"
                    
                    withCredentials([usernamePassword(credentialsId: 'daghub-credentials', usernameVariable: 'DW_USER', passwordVariable: 'DW_PASS')]) {
                        docker.image('iterativeai/cml:latest').inside("-u root") {
                            withEnv(['HOME=.']) {
                                sh """
                                dvc remote add -d -f origin ${dagshubUrl}
                                dvc remote modify origin --local auth basic
                                dvc remote modify origin --local user \$DW_USER
                                dvc remote modify origin --local password \$DW_PASS
                                dvc pull -v
                                """
                            }
                        }
                    }
                }
            }
        }

        stage('3. CI: Quality & Tests') {
            steps {
                script {
                    echo "🧪 Setup Environnement & Tests Unitaires..."
                    docker.image('python:3.9-slim').inside("-u root") {
                        withEnv(['HOME=.']) {
                            sh """
                            apt-get update && apt-get install -y libgomp1
                            python -m venv venv
                            ${ACTIVATE_VENV}
                            pip install --upgrade pip
                            pip install --no-cache-dir -r backend/src/requirements-backend.txt
                            pip install --no-cache-dir pytest pytest-mock flake8 evidently
                            ${PYTHON_PATH_CMD}
                            pytest testing/ --junitxml=test-results.xml
                            """
                        }
                    }
                }
            }
            post {
                always {
                    script {
                        if (fileExists('test-results.xml')) { junit 'test-results.xml' }
                    }
                }
            }
        }

        stage('4. Monitoring & Drift Detection') {
            steps {
                script {
                    echo "🔍 Analyse du Data Drift (Evidently)..."
                    docker.image('python:3.9-slim').inside("-u root") {
                        withEnv(['HOME=.']) {
                            sh """
                            apt-get update && apt-get install -y libgomp1
                            ${ACTIVATE_VENV}
                            # Ré-installation rapide si le module est manquant
                            pip install evidently
                            ${PYTHON_PATH_CMD}
                            python monitoring/check_drift.py || touch drift_detected
                            """
                        }
                    }
                }
            }
            post {
                always {
                    script {
                        if (fileExists('drift_report.html')) {
                            archiveArtifacts artifacts: 'drift_report.html'
                        }
                    }
                }
            }
        }

        stage('5. Continuous Training (CT)') {
            when { expression { fileExists 'drift_detected' } }
            steps {
                script {
                    echo "🚨 DRIFT DÉTECTÉ : Lancement du ré-entraînement..."
                    withCredentials([string(credentialsId: 'daghub-credentials', variable: 'TOKEN')]) {
                        docker.image('python:3.9-slim').inside("-u root") {
                            // Injection des credentials pour MLflow (Évite le popup navigateur)
                            withEnv([
                                "HOME=.", 
                                "DAGSHUB_TOKEN=${TOKEN}",
                                "MLFLOW_TRACKING_USERNAME=${DAGSHUB_USERNAME}",
                                "MLFLOW_TRACKING_PASSWORD=${TOKEN}"
                            ]) {
                                sh """
                                apt-get update && apt-get install -y libgomp1
                                ${ACTIVATE_VENV}
                                ${PYTHON_PATH_CMD}
                                python backend/src/trainning.py
                                """
                            }
                        }
                    }
                }
            }
        }

        stage('6. Docker Build & Push (Parallel)') {
            steps {
                script {
                    parallel(
                        "Backend": { 
                            sh "docker build -t ${DOCKER_IMAGE_NAME}:backend-latest ./backend"
                            sh "docker push ${DOCKER_IMAGE_NAME}:backend-latest" 
                        },
                        "Frontend": { 
                            sh "docker build -t ${DOCKER_IMAGE_NAME}:frontend-latest ./frontend"
                            sh "docker push ${DOCKER_IMAGE_NAME}:frontend-latest" 
                        }
                    )
                }
            }
        }

        stage('7. Kubernetes Deploy') {
            steps {
                script {
                    echo "🚀 Mise à jour et déploiement Kubernetes..."
                    def backendImg = "${DOCKER_IMAGE_NAME}:backend-latest"
                    def frontendImg = "${DOCKER_IMAGE_NAME}:frontend-latest"
                    
                    sh "sed -i 's|REPLACE_ME_BACKEND_IMAGE|${backendImg}|g' k8s/backend-deployment.yml"
                    sh "sed -i 's|REPLACE_ME_FRONTEND_IMAGE|${frontendImg}|g' k8s/frontend-deployment.yml"
                    
                    withCredentials([file(credentialsId: 'kubeconfig-secret', variable: 'KUBECONFIG')]) {
                        sh """
                        kubectl --kubeconfig=\$KUBECONFIG apply -f k8s/mlops-config.yml
                        kubectl --kubeconfig=\$KUBECONFIG apply -f k8s/backend-deployment.yml
                        kubectl --kubeconfig=\$KUBECONFIG apply -f k8s/frontend-deployment.yml
                        kubectl --kubeconfig=\$KUBECONFIG rollout restart deployment/backend-deployment
                        """
                    }
                }
            }
        }
    }
    
    post {
        always {
            sh "rm -rf venv drift_detected || true"
            sh "docker logout || true"
        }
        success {
            echo "✨ Pipeline MLOps exécuté avec succès !"
        }
    }
}