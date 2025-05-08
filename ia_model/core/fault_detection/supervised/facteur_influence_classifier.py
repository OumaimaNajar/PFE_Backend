import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class FacteurInfluenceClassifier:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'facteurs_influencent_pannes.csv')
            )
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.type_encoder = None
        self.facteur_encoder = None
        self._load_data()
        self._train_model()

    def _load_data(self):
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, sep=';', encoding=encoding)
                print(f"[INFO] CSV chargé avec encodage: {encoding}", file=sys.stderr)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Aucun encodage valide trouvé pour le CSV des facteurs influents.")

    def _prepare_features(self):
        df = self.df.dropna(subset=['type_panne', 'facteur'])
        X = df['type_panne']
        y = df['facteur']

        self.type_encoder = LabelEncoder()
        X_encoded = self.type_encoder.fit_transform(X)

        self.facteur_encoder = LabelEncoder()
        y_encoded = self.facteur_encoder.fit_transform(y)
        return X_encoded.reshape(-1, 1), y_encoded

    def _train_model(self):
        X, y = self._prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        score = self.model.score(X_test, y_test)
        print(f"[INFO] Score de validation : {score:.2f}", file=sys.stderr)

    def predict_factors(self, type_panne):
        # Prédire le facteur principal pour un type de panne donné
        type_panne_enc = self.type_encoder.transform([type_panne])
        pred = self.model.predict(type_panne_enc.reshape(-1, 1))
        facteur = self.facteur_encoder.inverse_transform(pred)[0]
        # On peut aussi retourner tous les facteurs possibles pour ce type de panne depuis le CSV
        facteurs_possibles = self.df[self.df['type_panne'].str.lower() == type_panne.lower()]['facteur'].unique().tolist()
        return {
            "type_panne": type_panne,
            "facteur_principal": facteur,
            "facteurs_possibles": facteurs_possibles
        }

    def save_model(self, path='facteur_influence_model.pkl'):
        joblib.dump({
            'model': self.model,
            'type_encoder': self.type_encoder,
            'facteur_encoder': self.facteur_encoder
        }, path)
        print(f"[INFO] Modèle sauvegardé à {path}", file=sys.stderr)

    def load_model(self, path='facteur_influence_model.pkl'):
        saved = joblib.load(path)
        self.model = saved['model']
        self.type_encoder = saved['type_encoder']
        self.facteur_encoder = saved['facteur_encoder']
        print(f"[INFO] Modèle chargé depuis {path}", file=sys.stderr)

if __name__ == '__main__':
    classifier = FacteurInfluenceClassifier()
    # Exemple d'utilisation
    type_panne_test = "Surchauffe moteur"
    result = classifier.predict_factors(type_panne_test)
    print(result)
    classifier.save_model()