import sys
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib


class FaultTypeClassifier:
    def __init__(self):
        self.df = None
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self._load_data()
        self._train_model()

    def _load_data(self):
        """Charge le CSV et vérifie les colonnes nécessaires"""
        try:
            # Ajuster selon l'organisation de ton projet
            csv_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'pannes_industrielles_organisees.csv')
            )
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(csv_path, sep=';', encoding=encoding)
                    print(f"[INFO] CSV chargé avec encodage: {encoding}", file=sys.stderr)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Aucun encodage valide trouvé.")
        except Exception as e:
            print(f"[ERREUR] Chargement CSV : {e}", file=sys.stderr)
            raise e

    def _prepare_features(self):
        """Prépare les features et labels pour l'entraînement"""
        df = self.df.dropna(subset=['type_panne', 'description'])

        # Combiner les colonnes texte pertinentes
        df['text'] = df['description'].fillna('') + " " + df['machine'].fillna('')

        X = df['text']
        y = df['type_panne']

        # Encodage des labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    def _train_model(self):
        """Entraîne le modèle Random Forest"""
        X, y = self._prepare_features()

        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Pipeline TF-IDF + Random Forest
        self.vectorizer = TfidfVectorizer(max_features=1000)

        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        self.model.fit(X_train, y_train)
        print("[INFO] Modèle entraîné.", file=sys.stderr)

        # Évaluation
        y_pred = self.model.predict(X_test)
        print("\n--- Évaluation ---", file=sys.stderr)
        print("Accuracy:", accuracy_score(y_test, y_pred), file=sys.stderr)
        print("Classification Report:\n", classification_report(y_test, y_pred), file=sys.stderr)
        print("Matrice de Confusion:\n", confusion_matrix(y_test, y_pred), file=sys.stderr)

    def predict_fault_type(self, description, machine=""):
        text = description + " " + machine
        pred_encoded = self.model.predict([text])[0]
        predicted_type = self.label_encoder.inverse_transform([pred_encoded])[0]
        return {
            "etat": predicted_type,

            "details": {
                "confidence": "N/A",
                "risk_level": "Inconnu",
                "probabilities": {"fonctionnel": "0%", "panne": "0%"},
                #"risk_factors": [],
                "fault_diagnosis": None
            },
            "message": "Prédiction réalisée avec succès"
        }

    def save_model(self, path='fault_classifier_model.pkl'):
        """Sauvegarde le modèle entraîné"""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder
        }, path)
        print(f"[INFO] Modèle sauvegardé à {path}", file=sys.stderr)

    def load_model(self, path='fault_classifier_model.pkl'):
        """Charge un modèle existant"""
        saved = joblib.load(path)
        self.model = saved['model']
        self.label_encoder = saved['label_encoder']
        print(f"[INFO] Modèle chargé depuis {path}", file=sys.stderr)


# ----------- TEST ----------
if __name__ == '__main__':
    classifier = FaultTypeClassifier()

    test_description = "La pompe ne fonctionne plus correctement, bruit anormal"
    test_machine = "Pompe condensat"

    predicted = classifier.predict_fault_type(test_description, test_machine)
    print(f"\n[PRÉDICTION] Type de panne : {predicted}", file=sys.stderr)
