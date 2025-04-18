import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
import joblib # type: ignore

# Charger les données prétraitées
data = pd.read_csv('../../data/preprocessed_data.csv')

# Séparer les caractéristiques (X) et la cible (y)
X = data.drop('STATUS', axis=1)
y = data['STATUS']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(model, '../models/random_forest_model.pkl')
print("Modèle sauvegardé dans 'models/random_forest_model.pkl'.")