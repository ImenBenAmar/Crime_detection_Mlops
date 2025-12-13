#Récupérer toutes les branches distantes (obligatoire la première fois)
git fetch origin

#voir liste des branches 
git branch -a

#se met sur la branch
git checkout modeling
#puller 
git pull

# 1. Se mettre à jour sur main
git checkout main
git pull origin main

# 2. Créer et aller sur une nouvelle branche
 git checkout -b refactor/modeling

# 3. Travailler + committer plusieurs fois
git add .
git commit -m "Description claire"

# 4. Pousser la branche (1ère fois)
git push origin refactor/modeling

# 5. Aller sur GitHub → créer la Pull Request

# 6. Quand la PR est validée et mergée sur GitHub :
git checkout main
git pull origin main

# 7. Supprimer la branche en local
git branch -d feature/nouvelle-fonctionnalite

#1. Supprimer une branche en local (sur ton PC)
    # Revenir sur main d’abord (obligatoire)
    git checkout main

    # Supprimer la branche locale mal orthographiée
    git branch -D feature/modeling        # suppression « douce » (refuse si pas mergée)
