import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data with correct encoding
df = pd.read_csv('data/facteurs_influencent_pannes.csv', sep=';', encoding='latin1')

# Prétraitement : encoder les variables catégorielles
le_type = LabelEncoder()
le_facteur = LabelEncoder()
le_valeur = LabelEncoder()

df['type_panne_enc'] = le_type.fit_transform(df['type_panne'])
df['facteur_enc'] = le_facteur.fit_transform(df['facteur'])
df['valeur_enc'] = le_valeur.fit_transform(df['valeur'])

# Définir X et y (par exemple, prédire le type de panne à partir des facteurs)
X = df[['facteur_enc', 'valeur_enc', 'pourcentage']]
y = df['type_panne_enc']

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prédire et afficher la précision
score = clf.score(X_test, y_test)
print(f"Précision du modèle Random Forest : {score:.2f}")