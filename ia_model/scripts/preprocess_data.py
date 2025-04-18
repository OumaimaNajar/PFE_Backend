import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv('../../data/clean_workorders.csv', sep=';')

# Nettoyage et filtrage des données
df_cleaned = df.dropna()
keywords = ['leak', 'stopped', 'overheating', 'failure', 'broken']
df_pannes = df_cleaned[df_cleaned['Description'].str.contains('|'.join(keywords), case=False, na=False)]
df_pannes.to_csv('Pannes_Data.csv', index=False, sep=';')

# Fonction pour générer les analyses et graphiques par attribut
def analyser_pannes_par_attribut(df, attribut):
    pannes_par_attribut = df[attribut].value_counts()
    
    print(f"\n=== Nombre de pannes par {attribut} ===")
    print(pannes_par_attribut)
    
    plt.figure(figsize=(10, 6))
    pannes_par_attribut.plot(kind='bar', color='skyblue')
    plt.title(f'Pannes par {attribut}', fontsize=14, fontweight='bold')
    plt.xlabel(attribut, fontsize=12)
    plt.ylabel('Nombre de Pannes', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Analyse pour chaque attribut
attributs = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']
for attribut in attributs:
    analyser_pannes_par_attribut(df_pannes, attribut)

# Fréquence des types de pannes (Top 10)
pannes_par_type = df_pannes['Description'].value_counts().head(10)
plt.figure(figsize=(12, 6))
pannes_par_type.plot(kind='barh', color='lightgreen')
plt.title('Top 10 des Types de Pannes', fontsize=14, fontweight='bold')
plt.xlabel('Nombre d\'Occurrences', fontsize=12)
plt.ylabel('Description de la Panne', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualisation de la priorité moyenne
priorite_moyenne = df_pannes['WOPRIORITY'].mean()
priorite_par_location = df_pannes.groupby('LOCATION')['WOPRIORITY'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(priorite_par_location.index, priorite_par_location.values, color='salmon')

# Ajout de la ligne de priorité moyenne
plt.axhline(y=priorite_moyenne, color='red', linestyle='--', 
            linewidth=2, label=f'Priorité Moyenne: {priorite_moyenne:.2f}')

# Ajout des valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom')

plt.title('Priorité Moyenne des Pannes par Emplacement', fontsize=14, fontweight='bold')
plt.xlabel('Emplacement', fontsize=12)
plt.ylabel('Priorité Moyenne', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f"\n=== Priorité moyenne globale des pannes ===")
print(f"{priorite_moyenne:.2f}")