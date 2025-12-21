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

    /* ===================================================== */
    stage('1. Initialize & Docker Login') {
      steps {
        cleanWs()
        checkout scm

        script {
          env.GIT_COMMIT_HASH = sh(
            returnStdout: true,
            script: "git rev-parse --short HEAD"
          ).trim()

          withCredentials([
            usernamePassword(
              credentialsId: 'docker-hub-credentials',
              usernameVariable: 'DOCKER_USER',
              passwordVariable: 'DOCKER_PASS'
            )
          ]) {
            sh "echo \$DOCKER_PASS | docker login -u \$DOCKER_USER --password-stdin"
          }
        }
      }
    }

    /* ===================================================== */
    stage('2. Pull Data (DVC)') {
      steps {
        script {
          def dagshubUrl = "https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO_NAME}.dvc"

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

    /* ===================================================== */
    stage('3. Tests') {
      steps {
        script {
          docker.image('python:3.9-slim').inside("-u root") {
            withEnv(['HOME=.']) {
              sh """
                apt-get update && apt-get install -y libgomp1
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
    }

    /* ===================================================== */
    stage('4. Monitoring (Evidently)') {
      steps {
        script {
          docker.image('python:3.9-slim').inside("-u root") {
            withEnv(['HOME=.']) {
              sh """
                ${ACTIVATE_VENV}
                pip install --upgrade pip
                pip install "evidently>=0.5.0"

                ${PYTHON_PATH_CMD}
              """

              withEnv([
                "MLFLOW_TRACKING_USERNAME=${DAGSHUB_AUTH_USR}",
                "MLFLOW_TRACKING_PASSWORD=${DAGSHUB_AUTH_PSW}"
              ]) {
                sh "python monitoring/check_drift.py || touch monitoring/drift_detected"
              }
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
            withEnv(['HOME=.']) {

              sh """
                apt-get update && apt-get install -y libgomp1
                python -m venv venv
                ${ACTIVATE_VENV}
                pip install --upgrade pip
                pip install -r backend/src/requirements-backend.txt
                pip install mlflow
                ${PYTHON_PATH_CMD}
              """

              withCredentials([
                usernamePassword(
                  credentialsId: 'daghub-credentials',
                  usernameVariable: 'MLFLOW_TRACKING_USERNAME',
                  passwordVariable: 'MLFLOW_TRACKING_PASSWORD'
                )
              ]) {
                sh "python backend/src/trainning.py"
              }
            }
          }
        }
      }
    }


    /* ===================================================== */
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

    /* ===================================================== */
    stage('7. Kubernetes Deploy') {
      steps {
        script {
          withCredentials([
            file(credentialsId: 'kubeconfig-secret', variable: 'KUBECONFIG')
          ]) {
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
        if (fileExists('test-results.xml')) {
          junit 'test-results.xml'
        }

        if (fileExists('monitoring/monitoring_drift_report.html')) {
          archiveArtifacts artifacts: 'monitoring/monitoring_drift_report.html'
        }

        sh "rm -rf venv monitoring/drift_detected || true"
        sh "docker logout || true"
      }
    }
  }
}
