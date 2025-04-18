import sys
import json
import pickle
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Forcer l'encodage UTF-8 pour stdin, stdout et stderr
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Fonction pour écrire les logs sur stderr
def log(message):
    print(message, file=sys.stderr)

class FaultDetectionModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.label_encoders = {}
        self.features = []
        self.feature_categories = {}

    def load_model(self):
        """Charge le modèle depuis le fichier"""
        try:
            log("Chargement du modèle...")
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            log("Modèle chargé avec succès.")
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.features = model_data['features']
            self.feature_categories = model_data.get('feature_categories', {})
        except Exception as e:
            log(f"Erreur pendant le chargement du modèle : {str(e)}")
            raise

    def preprocess_data(self, df):
        """Prétraitement des données avec gestion des valeurs inconnues"""
        log("\nPrétraitement des données...")
        
        for feature in self.features:
            if feature in self.feature_categories:  # Si c'est une variable catégorielle
                # Ajouter "UNKNOWN" comme classe valide si elle n'existe pas déjà
                if "UNKNOWN" not in self.feature_categories[feature]:
                    self.feature_categories[feature].append("UNKNOWN")
                
                # Remplacer les valeurs inconnues par "UNKNOWN"
                df[feature] = df[feature].apply(
                    lambda x: x if x in self.feature_categories[feature] else "UNKNOWN"
                )
                
                # Encodage
                self.label_encoders[feature].fit(self.feature_categories[feature])
                df[feature] = self.label_encoders[feature].transform(df[feature])
                log(f"{feature} - Classes : {self.label_encoders[feature].classes_}")
        
        X = df[self.features]
        y = df['PANNE']
        
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def predict(input_data):
    """Charge le modèle et effectue une prédiction"""
    try:
        log("Chargement du modèle...")
        with open('c:\\Users\\omaim\\backend_ia\\ia_model\\core\\fault_detection\\supervised\\random_forest_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        log("Modèle chargé avec succès.")
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        features = model_data['features']

        log("Classes des LabelEncoders après chargement :")
        for feature, le in label_encoders.items():
            log(f"{feature}: {le.classes_}")

        log("Prétraitement des données...")
        processed_data = {}
        for feature in features:
            if feature in label_encoders:
                le = label_encoders[feature]
                log(f"Classes pour {feature} : {le.classes_}")  # Log des classes
                value = input_data.get(feature, "UNKNOWN")
                if value not in le.classes_:
                    log(f"Valeur inconnue détectée pour {feature}: {value}")
                    value = "UNKNOWN"  # Remplacer par "UNKNOWN"
                processed_data[feature] = le.transform([value])[0]
            else:
                processed_data[feature] = input_data.get(feature, 0)
        
        X_input = pd.DataFrame([processed_data])
        log(f"Données prétraitées : {X_input}")

        log("Prédiction en cours...")
        prediction = model.predict(X_input)
        proba = model.predict_proba(X_input)[0]
        result = {
            "etat": "Panne détectée" if prediction[0] == 1 else "Aucun problème détecté",
            "details": {
                "confidence": f"{float(proba.max()) * 100:.2f}%",  # Confiance en pourcentage
                "probabilities": {
                    "fonctionnel": f"{float(proba[0]) * 100:.2f}%",
                    "panne": f"{float(proba[1]) * 100:.2f}%"
                }
            },
            "message": "Le modèle Random Forest a analysé les données et a détecté l'état du système."
        }
        log(f"Prédiction terminée : {result}")
        return result
    except Exception as e:
        log(f"Erreur pendant la prédiction : {str(e)}")
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            log(f"Chemin du fichier reçu : {file_path}")
            if not os.path.exists(file_path):
                log(f"Erreur : Le fichier {file_path} n'existe pas.")
                sys.exit(1)
            with open(file_path, 'r') as f:
                input_data = json.load(f)
                log(f"Données chargées depuis le fichier : {input_data}")
        else:
            log("Lecture des données d'entrée...")
            input_data = json.loads(sys.stdin.read())

        log(f"Données reçues : {input_data}")
        result = predict(input_data)
        log(f"Résultat : {result}")
        
        # N'envoyer que le JSON final sur stdout pour être capturé par Node.js
        print(json.dumps(result))
    except Exception as e:
        log(f"Erreur principale : {str(e)}")
        sys.exit(1)