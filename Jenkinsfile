pipeline {
    agent any

    environment {
        // Remplacez par votre vrai nom d'utilisateur Docker Hub
        DOCKER_HUB_USER = "YomnaJL" 
        IMAGE_TAG = "latest"
    }

    stages {
        
        // --- ÉTAPE 1 : INSTALLATION ---
        stage('Setup Environment') {
            steps {
                script {
                    echo "🐍 Création de l'environnement virtuel..."
                    // On est sous Linux dans le conteneur, donc 'sh'
                    sh '''
                        python3 -m venv venv
                        . venv/bin/activate
                        pip install --upgrade pip
                        # Installation des requirements
                        pip install -r backend/src/requirements-backend.txt
                        pip install pytest httpx
                    '''
                }
            }
        }

        // --- ÉTAPE 2 : TESTS (CI) ---
        stage('Run Tests') {
            steps {
                // On injecte les secrets DagsHub pour que l'API puisse charger le modèle
                withCredentials([usernamePassword(credentialsId: 'dagshub-credentials', usernameVariable: 'DAGSHUB_USERNAME', passwordVariable: 'DAGSHUB_TOKEN')]) {
                    script {
                        sh '''
                            . venv/bin/activate
                            
                            # On définit les variables d'environnement manquantes pour le test
                            export DAGSHUB_REPO_NAME="MLOPS_Project"
                            export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src
                            
                            echo "🧪 Lancement des tests Preprocessing..."
                            python3 -m pytest testing/preprocessing_test.py
                            
                            echo "🧪 Lancement des tests Model Loading..."
                            python3 -m pytest testing/test_model_loading.py
                        '''
                    }
                }
            }
        }

        // --- ÉTAPE 3 : BUILD & PUSH (CD) ---
        stage('Build & Push Docker') {
            steps {
                script {
                    echo "🐳 Connexion et Push vers Docker Hub..."
                    
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                        sh '''
                            # 1. Login
                            echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin

                            # 2. Backend
                            docker build -t $DOCKER_HUB_USER/crime-backend:${IMAGE_TAG} -f backend/src/Dockerfile .
                            docker push $DOCKER_HUB_USER/crime-backend:${IMAGE_TAG}

                            # 3. Frontend
                            docker build -t $DOCKER_HUB_USER/crime-frontend:${IMAGE_TAG} -f frontend/Dockerfile ./frontend
                            docker push $DOCKER_HUB_USER/crime-frontend:${IMAGE_TAG}
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            // Nettoyage pour ne pas saturer le disque
            sh 'rm -rf venv'
            cleanWs()
        }
    }
}