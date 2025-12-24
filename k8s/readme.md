## Diagramme d'architecture Kubernetes (Mermaid)

Colle le bloc ci‑dessous tel quel dans un fichier Markdown compatible Mermaid (par ex. GitHub Markdown avec Mermaid activé, VSCode + extension "Markdown Preview Mermaid Support", ou https://mermaid.live) :

```mermaid
graph LR
  subgraph Cluster["Kubernetes Cluster (minikube) - Node IP: 192.168.49.2"]
    direction TB

    subgraph Deployments["Deployments"]
      direction LR
  BE_DEP["backend-deployment (replicas: 3)"]
  FE_DEP["frontend-deployment (replicas: 3)"]
    end

    subgraph ReplicaSets["ReplicaSets (contrôlés par Deployments)"]
      direction LR
  BE_RS["ReplicaSet backend-deployment-7bcc8b665f"]
  FE_RS["ReplicaSet frontend-deployment-75849d4d7f"]
    end

    subgraph Pods["Pods (exemples réels)"]
      direction LR
      BE_POD1["backend-pod A\nIP: 10.244.1.185\nPort:5000"]
      BE_POD2["backend-pod B\nIP: 10.244.1.189\nPort:5000"]
      BE_POD3["backend-pod C\nIP: 10.244.1.191\nPort:5000"]

      FE_POD1["frontend-pod A\nIP: 10.244.1.186\nPort:8501"]
      FE_POD2["frontend-pod B\nIP: 10.244.1.187\nPort:8501"]
      FE_POD3["frontend-pod C\nIP: 10.244.1.188\nPort:8501"]
    end

    subgraph Services["Services"]
      direction TB
  BACK_SVC["backend-service\nType: ClusterIP\nClusterIP: 10.99.255.133\nPort: 5000"]
  FRONT_SVC["frontend-service\nType: NodePort\nClusterIP: 10.111.51.73\nPort: 8501 -> NodePort: 30001"]
    end

    subgraph Node["Node (minikube)\nExternal IP: 192.168.49.2"]
      direction TB
      NODE["minikube node"]
    end
  end

  %% relations
  BE_DEP --> BE_RS
  FE_DEP --> FE_RS

  BE_RS --> BE_POD1
  BE_RS --> BE_POD2
  BE_RS --> BE_POD3

  FE_RS --> FE_POD1
  FE_RS --> FE_POD2
  FE_RS --> FE_POD3

  BACK_SVC -->|routes traffic to| BE_POD1
  BACK_SVC -->|routes traffic to| BE_POD2
  BACK_SVC -->|routes traffic to| BE_POD3

  FRONT_SVC -->|routes traffic to| FE_POD1
  FRONT_SVC -->|routes traffic to| FE_POD2
  FRONT_SVC -->|routes traffic to| FE_POD3

  FRONT_SVC -->|exposed on NodePort 30001| NODE
  NODE -->|External access| Browser["Browser\nhttp://192.168.49.2:30001"]

  %% Rolling update flow
  subgraph RollingUpdate["RollingUpdate (Deployment strategy)"]
    direction LR
    NewReplicaSet["New ReplicaSet (new image)"] --> ReplacePods["Replace old pods gradually\nmaintain availability"]
  end

  BE_DEP --- RollingUpdate
  FE_DEP --- RollingUpdate
```

Notes rapides :
- Frontend : accessible depuis l'extérieur via Node IP `192.168.49.2` et NodePort `30001` -> `http://192.168.49.2:30001`
- Backend : `ClusterIP` (10.99.255.133) — exposé en interne, utilisable par le frontend via `backend-service:5000` ; pour la démo tu peux faire `kubectl port-forward service/backend-service 5000:5000` et ouvrir `http://127.0.0.1:5000/docs`.

Si tu veux, je peux aussi générer une version simplifiée pour une slide.