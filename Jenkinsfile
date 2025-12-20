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
        
        // Aide pour les scripts
        ACTIVATE_VENV = ". venv/bin/activate"
        PYTHON_PATH_CMD = "export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src"
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

        stage('2. CI: Quality & Tests') {
            steps {
                script {
                    echo "🧪 Création de l'environnement virtuel et tests..."
                    docker.image('python:3.9-slim').inside {
                        // LE FIX : On définit HOME sur le dossier courant pour que pip puisse écrire
                        withEnv(['HOME=.']) {
                            sh """
                            python -m venv venv
                            ${ACTIVATE_VENV}
                            pip install --upgrade pip
                            pip install -r backend/src/requirements-backend.txt
                            pip install pytest pytest-mock flake8
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
                        if (fileExists('test-results.xml')) {
                            junit 'test-results.xml'
                        }
                    }
                }
            }
        }

        stage('3. Pull Data (DVC)') {
            steps {
                script {
                    echo "📥 Pull des données via DVC..."
                    withCredentials([usernamePassword(credentialsId: 'daghub-credentials', usernameVariable: 'DW_USER', passwordVariable: 'DW_PASS')]) {
                        docker.image('python:3.9-slim').inside {
                            withEnv(['HOME=.']) {
                                sh """
                                ${ACTIVATE_VENV}
                                pip install dvc dvc-s3
                                dvc remote modify origin --local auth basic
                                dvc remote modify origin --local user $DW_USER
                                dvc remote modify origin --local password $DW_PASS
                                dvc pull
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
                    archiveArtifacts artifacts: 'drift_report.html', allowEmptyArchive: true
                }
            }
        }

       stage('5. Continuous Training (CT)') {
            when { 
                expression { fileExists 'drift_detected' } 
            }
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
                    // Login Docker Hub
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', passwordVariable: 'DOCKER_PASS', usernameVariable: 'DOCKER_USER')]) {
                        sh "echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin"
                    }

                    // Exécution en parallèle
                    parallel(
                        "Backend": {
                            sh """
                            docker build -t ${DOCKER_IMAGE_NAME}:backend-${GIT_COMMIT_HASH} -t ${DOCKER_IMAGE_NAME}:backend-latest ./backend
                            docker push ${DOCKER_IMAGE_NAME}:backend-latest
                            """
                        },
                        "Frontend": {
                            sh """
                            docker build -t ${DOCKER_IMAGE_NAME}:frontend-${GIT_COMMIT_HASH} -t ${DOCKER_IMAGE_NAME}:frontend-latest ./frontend
                            docker push ${DOCKER_IMAGE_NAME}:frontend-latest
                            """
                        }
                    )
                }
            }
        }

        stage('7. Kubernetes Deploy') {
            steps {
                script {
                    echo "🚀 Déploiement K8s..."
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