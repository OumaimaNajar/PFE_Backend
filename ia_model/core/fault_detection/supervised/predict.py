import sys
import json
import pickle
import os
import pandas as pd
from fault_classifier import FaultTypeClassifier

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
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
        
        # Load model
        model_data = load_model()
        
        # Process input data for prediction
        processed_data = process_input_data(input_data, model_data)
        
        # Make prediction
        X_input = pd.DataFrame([processed_data])
        prediction_proba = model_data['model'].predict_proba(X_input)[0]
        final_fault_prob = min(0.95, prediction_proba[1] + calculate_risk_score(input_data))
        final_ok_prob = 1 - final_fault_prob
        
        # Determine state
        is_fault = final_fault_prob > 0.5
        etat = "En panne" if is_fault else "Fonctionnel"
        
        # If fault detected, classify it
        fault_diagnosis = None
        if is_fault:
            classifier = FaultTypeClassifier()
            fault_diagnosis = classifier.predict_fault_type(input_data)
        
        result = {
            "success": True,
            "prediction": {
                "etat": etat,
                "details": {
                    "confidence": f"{max(final_fault_prob, final_ok_prob) * 100:.2f}%",
                    "risk_level": get_risk_level(final_fault_prob),
                    "probabilities": {
                        "fonctionnel": f"{final_ok_prob * 100:.2f}%",
                        "panne": f"{final_fault_prob * 100:.2f}%"
                    },
                    "risk_factors": get_risk_factors(input_data),
                    "fault_diagnosis": fault_diagnosis
                },
                "message": "Analyse complétée avec succès"
            }
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "success": False,
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
                    "risk_factors": [],
                    "fault_diagnosis": None
                },
                "message": "Une erreur s'est produite lors de l'analyse"
            }
        }))

def process_input_data(input_data, model_data):
    processed_data = {}
    
    # Process each feature using the label encoders
    for feature in model_data['features']:
        value = input_data.get(feature, 'UNKNOWN')
        if feature in model_data['label_encoders']:
            # Ensure value exists in label encoder classes
            if value not in model_data['label_encoders'][feature].classes_:
                value = 'UNKNOWN'
            processed_data[feature] = model_data['label_encoders'][feature].transform([value])[0]
        else:
            processed_data[feature] = value
            
    return processed_data

def calculate_risk_score(input_data):
    risk_score = 0.0
    
    # Priority-based risk
    priority_risk = {
        '1': 0.3,  # High priority
        '2': 0.2,  # Medium priority
        '3': 0.1   # Low priority
    }
    risk_score += priority_risk.get(str(input_data.get('WOPRIORITY')), 0.15)
    
    # Status-based risk
    status_risk = {
        'OPEN': 0.2,
        'INPRG': 0.15,
        'WAPPR': 0.1,
        'CLOSE': 0.05,
        'COMP': 0.05
    }
    risk_score += status_risk.get(input_data.get('STATUS', '').upper(), 0.1)
    
    return risk_score

def get_risk_factors(input_data):
    risk_factors = []
    
    # Priority-based factors
    if input_data.get('WOPRIORITY') == '1':
        risk_factors.append("High priority work order")
    
    # Status-based factors
    status = input_data.get('STATUS', '').upper()
    if status == 'OPEN':
        risk_factors.append("Open work order")
    elif status == 'INPRG':
        risk_factors.append("Work in progress")
    
    # Location-based factors
    location = input_data.get('LOCATION', '').upper()
    critical_locations = ['SHIPPING', 'RECEIVING', 'PRODUCTION']
    if location in critical_locations:
        risk_factors.append(f"Critical location: {location}")
    
    return risk_factors

def get_risk_level(fault_prob):
    if fault_prob > 0.7:
        return "Élevé"
    elif fault_prob > 0.3:
        return "Moyen"
    return "Faible"

if __name__ == "__main__":
    predict()