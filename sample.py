import pandas as pd

# Charger le CSV original
df = pd.read_csv("D:\mlops\classe\MLOPS\data\Crime_Data_from_2020_to_Present.csv")

# Obtenir les lignes avec différentes valeurs de 'Crm Cd Desc'
unique_crm_desc = df['Crm Cd Desc'].dropna().unique()

# On prend un maximum de 20 lignes en essayant d'avoir différentes valeurs
sample_list = []
for desc in unique_crm_desc:
    rows = df[df['Crm Cd Desc'] == desc]
    sample_list.append(rows.iloc[0])  # prendre la première ligne pour chaque Crm Cd Desc
    if len(sample_list) >= 150:
        break

# Créer le DataFrame final
sample_df = pd.DataFrame(sample_list)

# Sauvegarder l'échantillon
sample_df.to_csv("crime_sample_150.csv", index=False)

print("Échantillon de 20 lignes créé avec succès : d:/crime_sample_20.csv")
