import joblib
import pandas as pd
import json
import sys
import logging
import os
import matplotlib
matplotlib.use('Agg')  # Désactive l'affichage des graphiques

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.features = []
        self.load_model()

    def load_model(self):
        """Charge le modèle et les encodeurs"""
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'random_forest_model.joblib'
            )
            logger.info(f"Chargement du modèle depuis: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Fichier modèle non trouvé: {model_path}")

            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.label_encoders = model_data.get('label_encoders', {})
            self.features = model_data['features']
            logger.info("Modèle chargé avec succès")

        except Exception as e:
            logger.error(f"Erreur de chargement du modèle: {str(e)}")
            raise

    def preprocess_input(self, input_data):
        """Prétraite les données d'entrée"""
        processed = {}
        for feature in self.features:
            value = input_data.get(feature)
            if value is None:
                raise ValueError(f"Feature manquante: {feature}")

            if feature in self.label_encoders:
                try:
                    processed[feature] = self.label_encoders[feature].transform([str(value)])[0]
                except ValueError:
                    # Gestion des nouvelles valeurs
                    classes = list(self.label_encoders[feature].classes_)
                    processed[feature] = len(classes)  # Ou une autre stratégie
                    logger.warning(f"Valeur inconnue pour {feature}: {value}. Utilisation d'une valeur par défaut.")
            
            else:
                processed[feature] = value

        return pd.DataFrame([processed])

    def predict(self, input_data):
        """Effectue la prédiction"""
        try:
            df = self.preprocess_input(input_data)
            prediction = self.model.predict(df)[0]
            proba = self.model.predict_proba(df)[0][1]
            
            return {
                "etat": "panne" if prediction == 1 else "fonctionnel",
                "confidence": float(proba),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Erreur de prédiction: {str(e)}")
            raise

def main():
    try:
        # Initialisation
        predictor = ModelPredictor()
        
        # Lecture des données d'entrée
        if len(sys.argv) < 2:
            raise ValueError("Aucune donnée d'entrée fournie")
        
        input_data = json.loads(sys.argv[1])
        logger.info(f"Données reçues: {input_data}")

        # Prédiction
        result = predictor.predict(input_data)
        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "error": str(e),
            "success": False
        }
        logger.error(f"Erreur: {json.dumps(error_result)}")
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()