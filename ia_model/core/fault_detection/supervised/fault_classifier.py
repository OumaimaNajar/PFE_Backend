import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pymongo import MongoClient
import logging

# Configuration du système de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fault_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FaultClassifier')

class FaultTypeClassifier:
    def __init__(self):
        # Connexion à MongoDB
        try:
            logger.info("Tentative de connexion à MongoDB...")
            self.client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
            # Vérification de la connexion
            self.client.server_info()
            logger.info("Connexion à MongoDB établie avec succès")
            
            self.db = self.client['back-ia']
            logger.info(f"Base de données sélectionnée : 'back-ia'")
            
            # Vérification des collections
            collections = self.db.list_collection_names()
            logger.info(f"Collections disponibles dans la base de données : {collections}")
            
            if 'pannes' in collections:
                self.pannes_collection = self.db['pannes']
                logger.info(f"Collection 'pannes' trouvée avec {self.pannes_collection.count_documents({})} documents")
            else:
                raise ValueError("Collection 'pannes' non trouvée dans la base de données")
                
        except Exception as e:
            logger.error(f"Erreur lors de la connexion à MongoDB : {str(e)}")
            raise ConnectionError(f"Impossible de se connecter à MongoDB : {str(e)}")

        self.df = None
        self.model = None
        self.type_encoder = None
        self.facteur_encoder = None
        self.col_encoders = {}
        self._load_data()
        self._train_model()

    def _load_data(self):
        """Charge les données depuis MongoDB"""
        logger.info("Chargement des données depuis MongoDB...")
        cursor = self.pannes_collection.find({})
        mongo_data = list(cursor)
        
        if not mongo_data:
            raise ValueError("Aucune donnée trouvée dans la collection 'pannes'")
            
        self.df = pd.DataFrame(mongo_data)
        
        # Suppression de l'ID MongoDB s'il existe
        if '_id' in self.df.columns:
            self.df = self.df.drop('_id', axis=1)
            
        logger.info(f"Données chargées depuis MongoDB: {len(self.df)} enregistrements")
        logger.debug(f"Colonnes disponibles: {self.df.columns.tolist()}")

    def _prepare_features(self):
        # Reste du code inchangé pour la préparation des features
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
        # Reste du code inchangé pour l'entraînement du modèle
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
        # Reste du code inchangé pour la prédiction
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
