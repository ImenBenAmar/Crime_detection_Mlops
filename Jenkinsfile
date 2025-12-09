pipeline {
    agent none  // On ne définit pas d'agent global pour pouvoir choisir par étape

    environment {
        IMAGE_TAG = "latest"
    }

    stages {
        // --- ÉTAPE 1 : TESTS (DANS UN CONTENEUR PYTHON) ---
        stage('Run Tests') {
            agent {
                docker { 
                    image 'python:3.9' 
                    // On monte le code actuel dans le conteneur
                    reuseNode true 
                }
            }
            steps {
                // On injecte les secrets DagsHub
                withCredentials([usernamePassword(credentialsId: 'dagshub-credentials', usernameVariable: 'DAGSHUB_USERNAME', passwordVariable: 'DAGSHUB_TOKEN')]) {
                    script {
                        echo "🧪 Lancement des tests dans le conteneur Python..."
                        
                        // Plus besoin de venv ! On est déjà dans un environnement Python isolé.
                        sh '''
                            pip install --upgrade pip
                            pip install -r backend/src/requirements-backend.txt
                            pip install pytest httpx
                            
                            export DAGSHUB_REPO_NAME="MLOPS_Project"
                            export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src
                            
                            python -m pytest testing/preprocessing_test.py
                            python -m pytest testing/test_model_loading.py
                        '''
                    }
                }
            }
        }

        // --- ÉTAPE 2 : BUILD & PUSH (SUR LA MACHINE HÔTE) ---
        stage('Build & Push Docker') {
            agent any // Ici on a besoin de Docker-in-Docker, donc on revient sur l'agent principal
            
            steps {
                script {
                    echo "🐳 Construction des images finales..."
                    
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        sh '''
                            echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin

                            docker build -t $DOCKER_USER/crime-backend:${IMAGE_TAG} -f backend/src/Dockerfile .
                            docker push $DOCKER_USER/crime-backend:${IMAGE_TAG}

                            docker build -t $DOCKER_USER/crime-frontend:${IMAGE_TAG} -f frontend/Dockerfile ./frontend
                            docker push $DOCKER_USER/crime-frontend:${IMAGE_TAG}
                        '''
                    }
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}