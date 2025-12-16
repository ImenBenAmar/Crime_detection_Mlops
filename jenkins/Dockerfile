# Dockerfile pour Jenkins avec accès Docker et Kubernetes
FROM jenkins/jenkins:lts

USER root

# Installer Docker CLI, docker-compose et kubectl
RUN apt-get update && \
    apt-get install -y docker.io docker-compose curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    usermod -aG docker jenkins && \
    apt-get clean

# Revenir à l’utilisateur Jenkins
USER jenkins

EXPOSE 8080


