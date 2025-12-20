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
        // On définit le credential une seule fois ici
        DAGSHUB_AUTH = credentials('daghub-credentials')
        DOCKERHUB_CREDS = credentials('docker-hub-credentials')
        
        DAGSHUB_USERNAME  = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.mlflow"
        
        ACTIVATE_VENV     = ". venv/bin/activate"
        PYTHON_PATH_CMD   = "export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src"
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
                    echo "📥 Pulling data via DVC (CML image)..."
                    def dagshubUrl = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.dvc"
                    
                    // Utilisation de DAGSHUB_AUTH_USR et DAGSHUB_AUTH_PSW
                    docker.image('iterativeai/cml:latest').inside("-u root") {
                        withEnv(['HOME=.']) {
                            sh """
                            dvc remote add -d -f origin ${dagshubUrl}
                            dvc remote modify origin --local auth basic
                            dvc remote modify origin --local user ${DAGSHUB_AUTH_USR}
                            dvc remote modify origin --local password ${DAGSHUB_AUTH_PSW}
                            dvc pull -v
                            """
                        }
                    }
                }
            }
        }

        stage('3. ML Pipeline (Tests, Monitoring, CT)') {
            steps {
                script {
                    docker.image('python:3.9-slim').inside("-u root") {
                        withEnv(['HOME=.']) {
                            
                            // --- ÉTAPE A: INSTALLATION UNIQUE ---
                            echo "🛠️ Installation des dépendances..."
                            sh """
                            apt-get update && apt-get install -y libgomp1
                            python -m venv venv
                            ${ACTIVATE_VENV}
                            pip install --upgrade pip
                            pip install --no-cache-dir -r backend/src/requirements-backend.txt
                            pip install --no-cache-dir pytest pytest-mock flake8 evidently
                            """

                            // --- ÉTAPE B: TESTS UNITAIRES ---
                            echo "🧪 Exécution des tests..."
                            sh """
                            ${ACTIVATE_VENV}
                            ${PYTHON_PATH_CMD}
                            pytest testing/ --junitxml=test-results.xml
                            """

                            // --- ÉTAPE C: MONITORING DE DRIFT ---
                            echo "🔍 Analyse du Drift..."
                            sh """
                            ${ACTIVATE_VENV}
                            ${PYTHON_PATH_CMD}
                            # On utilise le PSW (le token) pour l'auth MLflow
                            export MLFLOW_TRACKING_USERNAME=${DAGSHUB_AUTH_USR}
                            export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_AUTH_PSW}
                            python monitoring/check_drift.py || touch monitoring/drift_detected
                            """

                            // --- ÉTAPE D: RETRAINING ---
                            if (fileExists('monitoring/drift_detected')) {
                                echo "🚨 DRIFT DÉTECTÉ ! Lancement du ré-entraînement..."
                                sh """
                                ${ACTIVATE_VENV}
                                ${PYTHON_PATH_CMD}
                                export MLFLOW_TRACKING_USERNAME=${DAGSHUB_AUTH_USR}
                                export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_AUTH_PSW}
                                python backend/src/trainning.py
                                """
                            } else {
                                echo "✅ Pas de drift. Retraining ignoré."
                            }
                        }
                    }
                }
            }
            post {
                always {
                    script {
                        if (fileExists('test-results.xml')) { junit 'test-results.xml' }
                        if (fileExists('monitoring/monitoring_drift_report.html')) { 
                            archiveArtifacts artifacts: 'monitoring/monitoring_drift_report.html' 
                        }
                    }
                }
            }
        }

        stage('4. Docker Build & Push (Parallel)') {
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

        stage('5. Kubernetes Deploy') {
            steps {
                script {
                    echo "🚀 Déploiement K8s..."
                    sh "sed -i 's|REPLACE_ME_BACKEND_IMAGE|${DOCKER_IMAGE_NAME}:backend-latest|g' k8s/backend-deployment.yml"
                    sh "sed -i 's|REPLACE_ME_FRONTEND_IMAGE|${DOCKER_IMAGE_NAME}:frontend-latest|g' k8s/frontend-deployment.yml"
                    
                    withCredentials([file(credentialsId: 'kubeconfig-secret', variable: 'KUBECONFIG')]) {
                        sh """
                        kubectl --kubeconfig=\$KUBECONFIG apply -f k8s/config-env.yml
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
            sh "rm -rf venv monitoring/drift_detected || true"
            sh "docker logout || true"
        }
    }
}