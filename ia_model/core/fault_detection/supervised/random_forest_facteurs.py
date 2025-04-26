import sys
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def detecter_facteurs_influents(type_panne, chemin_csv="../../../../data/facteurs_influencent_pannes.csv"):
    # Charger les données
    df = pd.read_csv(chemin_csv, sep=';', encoding="latin1")
    # Filtrer sur le type de panne
    df_type = df[df['type_panne'].str.upper() == type_panne.upper()]
    if df_type.empty:
        print(json.dumps({
            "error": f"Aucun facteur trouvé pour le type de panne : {type_panne}"
        }, ensure_ascii=False))
        return

    # On va prédire le facteur à partir des autres colonnes (valeur, pourcentage, description)
    X = df_type[['valeur', 'pourcentage', 'description']].copy()
    y = df_type['facteur']

    # Encodage des variables catégorielles
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    y = LabelEncoder().fit_transform(y.astype(str))

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modèle Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Importance des features
    importances = clf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    # Liste des facteurs influents pour ce type de panne
    facteurs_list = df_type[['facteur', 'valeur', 'pourcentage', 'description']].to_dict(orient='records')

    # Sortie JSON pour intégration Node.js
    result = {
        "feature_importance": importance_df.to_dict(orient='records'),
        "facteurs": facteurs_list
    }
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    # Récupérer le type de panne depuis les arguments de la ligne de commande
    if len(sys.argv) > 1:
        type_panne = sys.argv[1]
    else:
        type_panne = "SURCHAUFFE"  # Valeur par défaut pour test

    detecter_facteurs_influents(type_panne)