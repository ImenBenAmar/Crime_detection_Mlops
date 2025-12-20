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
        
        // --- Credentials ---
        DAGSHUB_TOKEN = credentials('daghub-credentials') 
        DOCKERHUB_CREDS = credentials('docker-hub-credentials')
        
        // --- Configs MLOps ---
        DAGSHUB_USERNAME = 'YomnaJL'
        DAGSHUB_REPO_NAME = 'MLOPS_Project'
        MLFLOW_TRACKING_URI = 'https://dagshub.com/YomnaJL/MLOPS_Project.mlflow'
    }

    stages {
        stage('1. Initialize') {
            steps {
                cleanWs()
                checkout scm
                script {
                    env.GIT_COMMIT_HASH = sh(returnStdout: true, script: "git rev-parse --short HEAD").trim()
                }
            }
        }

        stage('2. CI: Tests Unitaires') {
            steps {
                script {
                    echo "🧪 Lancement des tests dans le dossier testing/..."
                    docker.image('python:3.9-slim').inside {
                        sh 'pip install --upgrade pip'
                        sh 'pip install -r backend/requirements-backend.txt'
                        sh 'pip install pytest pytest-mock flake8' 
                        
                        // Ajout du PYTHONPATH pour trouver preprocessing2.py, feature_store.py, etc.
                        sh """
                        export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src
                        pytest testing/ --junitxml=test-results.xml
                        """
                    }
                }
            }
            post {
                always {
                    junit 'test-results.xml' 
                }
            }
        }

        stage('3. Pull Data (DVC)') {
            steps {
                script {
                    echo "📥 Récupération des données via DVC..."
                    withCredentials([usernamePassword(credentialsId: 'dagshub-credentials', usernameVariable: 'DW_USER', passwordVariable: 'DW_PASS')]) {
                        docker.image('iterative/dvc').inside {
                            sh "dvc remote modify origin --local auth basic"
                            sh "dvc remote modify origin --local user $DW_USER"
                            sh "dvc remote modify origin --local password $DW_PASS"
                            sh "dvc pull"
                        }
                    }
                }
            }
        }

        stage('4. Monitoring & Drift Detection') {
            steps {
                script {
                    echo "🔍 Analyse du Data Drift via monitoring/check_drift.py..."
                    docker.image('python:3.9-slim').inside {
                        sh 'pip install -r backend/requirements-backend.txt'
                        sh 'pip install evidently'
                        
                        // On lance TON script de drift
                        // S'il détecte un drift, il crée le fichier 'drift_detected'
                        sh """
                        export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src
                        python monitoring/check_drift.py || touch drift_detected
                        """
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
            // ✅ Correction ici : ajout de 'expression'
            when { 
                expression { fileExists 'drift_detected' } 
            }
            steps {
                script {
                    echo "🚨 DRIFT DÉTECTÉ : Lancement du ré-entraînement via training.py..."
                    docker.image('python:3.9-slim').inside {
                        sh 'pip install -r backend/requirements-backend.txt'
                        sh "export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src && python backend/src/trainning.py"
                    }
                }
            }
        }

        stage('6. Docker Build & Push') {
            steps {
                script {
                    // Login Docker Hub
                    sh "echo \$DOCKERHUB_CREDS_PSW | docker login -u \$DOCKERHUB_CREDS_USR --password-stdin"
                    
                    // Backend
                    sh "docker build -t ${DOCKER_IMAGE_NAME}:backend-${GIT_COMMIT_HASH} -t ${DOCKER_IMAGE_NAME}:backend-latest ./backend"
                    sh "docker push ${DOCKER_IMAGE_NAME}:backend-latest"
                    
                    // Frontend
                    sh "docker build -t ${DOCKER_IMAGE_NAME}:frontend-${GIT_COMMIT_HASH} -t ${DOCKER_IMAGE_NAME}:frontend-latest ./frontend"
                    sh "docker push ${DOCKER_IMAGE_NAME}:frontend-latest"
                }
            }
        }

        stage('7. Kubernetes Deploy') {
            steps {
                script {
                    echo "🚀 Déploiement Kubernetes (fichiers .yml)..."
                    def newBackend = "${DOCKER_IMAGE_NAME}:backend-latest"
                    def newFrontend = "${DOCKER_IMAGE_NAME}:frontend-latest"
                    
                    // Mise à jour des images dans les fichiers .yml
                    sh "sed -i 's|REPLACE_ME_BACKEND_IMAGE|${newBackend}|g' k8s/backend-deployment.yml"
                    sh "sed -i 's|REPLACE_ME_FRONTEND_IMAGE|${newFrontend}|g' k8s/frontend-deployment.yml"
                    
                    withCredentials([file(credentialsId: 'kubeconfig-secret', variable: 'KUBECONFIG')]) {
                        sh "kubectl --kubeconfig=\$KUBECONFIG apply -f k8s/mlops-config.yml"
                        sh "kubectl --kubeconfig=\$KUBECONFIG apply -f k8s/backend-deployment.yml"
                        sh "kubectl --kubeconfig=\$KUBECONFIG apply -f k8s/frontend-deployment.yml"
                        
                        // Force le redémarrage pour charger les nouveaux processeurs/modèles
                        sh "kubectl --kubeconfig=\$KUBECONFIG rollout restart deployment/backend-deployment"
                    }
                }
            }
        }
    }
    
    post {
        always {
            // Nettoyage final du workspace
            sh "rm drift_detected || true"
            sh "docker logout || true"
        }
        success {
            echo "✨ Pipeline MLOps terminé avec succès !"
        }
    }
}