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
        DAGSHUB_TOKEN = credentials('daghub-credentials') 
        DOCKERHUB_CREDS = credentials('docker-hub-credentials')
        DAGSHUB_USERNAME = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.mlflow"
        
        ACTIVATE_VENV = ". venv/bin/activate"
        PYTHON_PATH_CMD = "export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src"
    }

    stages {
        stage('1. Initialize & Docker Login') {
            steps {
                cleanWs()
                checkout scm
                script {
                    env.GIT_COMMIT_HASH = sh(returnStdout: true, script: "git rev-parse --short HEAD").trim()
                    
                    // ✅ Docker Login validé au démarrage
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                        sh "echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin"
                    }
                }
            }
        }

        stage('2. CI: Quality & Tests') {
            steps {
                script {
                    echo "🧪 Setup Environment & Unit Tests..."
                    docker.image('python:3.9-slim').inside {
                        withEnv(['HOME=.']) {
                            sh """
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
        stage('3. Pull Data (DVC) - OPTIMIZED') {
            steps {
                script {
                    echo "📥 Nettoyage et Configuration DVC..."
                    // On définit l'URL ici pour plus de clarté
                    def dagshubUrl = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.dvc"
                    
                    withCredentials([usernamePassword(credentialsId: 'daghub-credentials', usernameVariable: 'DW_USER', passwordVariable: 'DW_PASS')]) {
                        docker.image('iterativeai/cml:latest').inside("-u root") {
                            withEnv(['HOME=.']) {
                                sh """
                                # 1. Supprimer toute configuration existante pour repartir à neuf
                                dvc remote remove origin --local || true
                                dvc remote remove origin || true

                                # 2. Ajouter le remote DagsHub de manière explicite
                                dvc remote add -d origin ${dagshubUrl}

                                # 3. Configurer l'authentification (Utiliser \$ pour les variables d'env shell)
                                dvc remote modify origin --local auth basic
                                dvc remote modify origin --local user \$DW_USER
                                dvc remote modify origin --local password \$DW_PASS

                                # 4. Vérification : Afficher la config pour débugger en cas d'échec
                                dvc remote list

                                # 5. Pull des données
                                dvc pull -v
                                """
                            }
                        }
                    }
                }
            }
        }

        stage('4. Monitoring & Drift Detection') {
            steps {
                script {
                    echo "🔍 Analyse du Data Drift..."
                    docker.image('python:3.9-slim').inside {
                        withEnv(['HOME=.']) {
                            sh """
                            ${ACTIVATE_VENV}
                            ${PYTHON_PATH_CMD}
                            python monitoring/check_drift.py || touch drift_detected
                            """
                        }
                    }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'drift_report.html', allowEmptyArchive: true
                }
            }
        }

       stage('5. Continuous Training (CT)') {
            when { expression { fileExists 'drift_detected' } }
            steps {
                script {
                    echo "🚨 DRIFT DÉTECTÉ : Ré-entraînement..."
                    docker.image('python:3.9-slim').inside {
                        withEnv(['HOME=.']) {
                            sh """
                            ${ACTIVATE_VENV}
                            ${PYTHON_PATH_CMD}
                            python backend/src/trainning.py
                            """
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
                    echo "🚀 Déploiement K8s (.yml)..."
                    def newBackend = "${DOCKER_IMAGE_NAME}:backend-latest"
                    def newFrontend = "${DOCKER_IMAGE_NAME}:frontend-latest"
                    
                    sh "sed -i 's|REPLACE_ME_BACKEND_IMAGE|${newBackend}|g' k8s/backend-deployment.yml"
                    sh "sed -i 's|REPLACE_ME_FRONTEND_IMAGE|${newFrontend}|g' k8s/frontend-deployment.yml"
                    
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
    }
}