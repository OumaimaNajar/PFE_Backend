import sys
import json
import pickle
import os
import pandas as pd

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print("Model loaded successfully")
        print("Model data keys:", model_data.keys())
        return model_data
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Failed to load model: {str(e)}",
            "prediction": {
                "etat": "Inconnu",
                "details": {
                    "confidence": "0%",
                    "risk_level": "Inconnu",
                    "probabilities": {
                        "fonctionnel": "0%",
                        "panne": "0%"
                    },
                    "risk_factors": []
                },
                "message": "Erreur lors du chargement du modèle"
            }
        }))
        raise

def predict():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Input file path required"}))
        return

    input_file = sys.argv[1]
    
    try:
        # Load and validate input data
        print(f"Reading input from: {input_file}")
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        print("Input data:", input_data)
            
        # Load and validate model
        model_data = load_model()
        if not isinstance(model_data, dict) or 'model' not in model_data:
            raise ValueError("Invalid model data structure")
            
        model = model_data['model']
        label_encoders = model_data.get('label_encoders', {})
        print("Available label encoders:", list(label_encoders.keys()))

        # Process input data
        processed_data = {}
        risk_score = 0.2
        risk_factors = []
        
        for feature in ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']:
            value = str(input_data.get(feature, 'UNKNOWN'))
            if feature in label_encoders:
                le = label_encoders[feature]
                if value not in le.classes_:
                    risk_factors.append(f"Unknown {feature}: {value}")
                    risk_score += 0.3
                    value = 'UNKNOWN'
                processed_data[feature] = le.transform([value])[0]
            else:
                if feature == 'WOPRIORITY':
                    try:
                        priority = int(value)
                        risk_score += (4 - priority) * 0.2
                    except ValueError:
                        risk_score += 0.2
                processed_data[feature] = value

        # Check description for keywords
        description = str(input_data.get('description', '')).lower()
        keywords = model_data.get('keywords', [])
        found_keywords = [k for k in keywords if k in description]
        if found_keywords:
            risk_score += min(0.5, len(found_keywords) * 0.1)
            risk_factors.append(f"Keywords found: {', '.join(found_keywords)}")

        # Make prediction using correct method
        X_input = pd.DataFrame([processed_data])
        prediction_proba = model.predict_proba(X_input)[0]
        prediction = model.predict(X_input)[0]

        # Calculate final probabilities
        final_fault_prob = min(0.95, prediction_proba[1] + risk_score)
        final_ok_prob = 1 - final_fault_prob

        result = {
            "success": True,
            "prediction": {
                "etat": "En panne" if final_fault_prob > 0.5 else "Fonctionnel",
                "details": {
                    "confidence": f"{max(final_fault_prob, final_ok_prob) * 100:.2f}%",
                    "risk_level": "Élevé" if final_fault_prob > 0.7 else 
                                "Moyen" if final_fault_prob > 0.3 else "Faible",
                    "probabilities": {
                        "fonctionnel": f"{final_ok_prob * 100:.2f}%",
                        "panne": f"{final_fault_prob * 100:.2f}%"
                    },
                    "risk_factors": risk_factors
                },
                "message": "Analyse complétée avec succès"  # Added message field
            }
        }
        
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "success": False,  # Changed 'false' to False
            "error": str(e),
            "prediction": {
                "etat": "Inconnu",
                "details": {
                    "confidence": "0%",
                    "risk_level": "Inconnu",
                    "probabilities": {
                        "fonctionnel": "0%",
                        "panne": "0%"
                    },
                    "risk_factors": []
                },
                "message": "Une erreur s'est produite lors de l'analyse"
            }
        }))

if __name__ == "__main__":
    predict()