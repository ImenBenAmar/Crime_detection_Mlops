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
    DAGSHUB_USERNAME  = 'YomnaJL'
    DAGSHUB_REPO_NAME = 'MLOPS_Project'
    MLFLOW_TRACKING_URI = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.mlflow"
    
    // Variables pour simplifier les commandes
    PYTHON_PATH_CMD   = "export PYTHONPATH=\$PYTHONPATH:\$(pwd)/backend/src"
    ACTIVATE_VENV     = ". venv/bin/activate"
  }

  stages {

    stage('1. Initialize & Docker Login') {
      steps {
        cleanWs()
        checkout scm
        script {
          withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
            sh "echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin"
          }
        }
      }
    }

    stage('2. Pull Data (DVC)') {
      steps {
        script {
          def dagshubUrl = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.dvc"
          withCredentials([usernamePassword(credentialsId: 'daghub-credentials', usernameVariable: 'DVC_USER', passwordVariable: 'DVC_PASS')]) {
            docker.image('iterativeai/cml:latest').inside("-u root") {
              sh """
                dvc remote add -d -f origin ${dagshubUrl}
                dvc remote modify origin --local auth basic
                dvc remote modify origin --local user ${DVC_USER}
                dvc remote modify origin --local password ${DVC_PASS}
                dvc pull -v
              """
            }
          }
        }
      }
    }

    stage('3. Unit & Integration Tests') {
      steps {
        script {
          docker.image('python:3.9-slim').inside("-u root") {
            sh """
              apt-get update && apt-get install -y libgomp1
              python -m venv venv
              ${ACTIVATE_VENV}
              pip install --upgrade pip
              pip install -r testing/requirements-testing.txt
              ${PYTHON_PATH_CMD}
              pytest testing/ --junitxml=test-results.xml
            """
          }
        }
      }
    }

/* ===================================================== */
    stage('4. Monitoring (Evidently)') {
      steps {
        script {
          docker.image('python:3.9-slim').inside("-u root") {
            withCredentials([usernamePassword(credentialsId: 'daghub-credentials', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
              sh """
                # 1. Installation des dépendances système (Indispensable pour Evidently/Numba)
                apt-get update && apt-get install -y build-essential libgomp1
                
                # 2. Préparation de l'environnement virtuel
                python -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                
                # 3. Installation des requirements (Vérifie bien ton fichier requirements-monitoring.txt)
                pip install -r monitoring/requirements-monitoring.txt
                
                # 4. Configuration des variables d'environnement
                ${PYTHON_PATH_CMD}
                export MLFLOW_TRACKING_USERNAME=${USER}
                export MLFLOW_TRACKING_PASSWORD=${PASS}
                export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
                
                # 5. Exécution du script de monitoring
                # Si le script échoue (drift détecté), on crée le fichier trigger
                python monitoring/check_drift.py || touch monitoring/drift_detected
              """
            }
          }
        }
      }
    }
    
    /* ===================================================== */
    stage('5. Conditional Retraining') {
      when {
        expression { fileExists('monitoring/drift_detected') }
      }
      steps {
        script {
          docker.image('python:3.9-slim').inside("-u root") {
            withCredentials([usernamePassword(credentialsId: 'daghub-credentials', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
              sh """
                apt-get update && apt-get install -y libgomp1
                python -m venv venv
                ${ACTIVATE_VENV}
                pip install --upgrade pip
                pip install -r backend/src/requirements-backend.txt
                pip install -r backend/src/requirements-train.txt
                ${PYTHON_PATH_CMD}
                
                export MLFLOW_TRACKING_USERNAME=${USER}
                export MLFLOW_TRACKING_PASSWORD=${PASS}
                
                python backend/src/trainning.py
              """
            }
          }
        }
      }
    }

    stage('6. Docker Build & Push') {
      steps {
        script {
          parallel(
            Backend: {
              sh "docker build -t ${DOCKER_IMAGE_NAME}:backend-latest ./backend/src"
              sh "docker push ${DOCKER_IMAGE_NAME}:backend-latest"
            },
            Frontend: {
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
          withCredentials([file(credentialsId: 'kubeconfig-secret', variable: 'KUBECONFIG')]) {
            sh """
              sed -i 's|REPLACE_ME_BACKEND_IMAGE|${DOCKER_IMAGE_NAME}:backend-latest|g' k8s/backend-deployment.yml
              sed -i 's|REPLACE_ME_FRONTEND_IMAGE|${DOCKER_IMAGE_NAME}:frontend-latest|g' k8s/frontend-deployment.yml

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
      script {
        if (fileExists('test-results.xml')) { junit 'test-results.xml' }
        // On nettoie les fichiers de trigger pour le prochain build
        sh "rm -rf venv monitoring/drift_detected || true"
      }
    }
  }
}