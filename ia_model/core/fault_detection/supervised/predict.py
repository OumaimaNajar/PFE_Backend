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

def predict(input_data):
    try:
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
        influencing_factors = []
        if is_fault:
            classifier = FaultTypeClassifier()
            fault_diagnosis = classifier.predict_fault_type(input_data)
            if fault_diagnosis and "type" in fault_diagnosis:
                influencing_factors = get_influencing_factors(fault_diagnosis["type"])

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
                    "fault_diagnosis": fault_diagnosis,
                    "influencing_factors": influencing_factors
                },
                "message": "Analyse complétée avec succès"
            }
        }
        
        return result
        
    except Exception as e:
        return {
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
        }

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

def get_influencing_factors(fault_type):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        csv_path = os.path.join(base_dir, 'data', 'facteurs_influencent_pannes.csv')
        
        if not os.path.exists(csv_path):
            print(f"[ERREUR] Fichier CSV non trouvé : {csv_path}")
            return []
            
        # Lecture du CSV avec gestion des encodages
        try:
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, sep=';', encoding='latin1')
            
        # Normalisation du type de panne (suppression des guillemets et du point)
        fault_type_norm = fault_type.upper().strip().replace('"', '').replace('.', '')
        
        # Recherche des facteurs
        facteurs = df[df['type_panne'].str.upper().str.strip() == fault_type_norm]
        
        if facteurs.empty:
            # Recherche partielle si aucune correspondance exacte
            for type_existant in df['type_panne'].unique():
                if fault_type_norm in type_existant.upper().strip():
                    facteurs = df[df['type_panne'] == type_existant]
                    break
            
        if not facteurs.empty:
            result = []
            for _, row in facteurs.iterrows():
                factor = {
                    "facteur": str(row['facteur']),
                    "valeur": str(row['valeur']),
                    "pourcentage": float(row['pourcentage']),
                    "description": str(row.get('Description', '')),  # Utilisation de 'Description' au lieu de 'description'
                    "action_recommandee": str(row.get('action_recommandee', '')),
                    "action_secondaire": str(row.get('action_secondaire', '')),
                    "code_probleme": str(row.get('code_probleme', '')),
                    "code_defaillance": str(row.get('code_defaillance', ''))
                }
                result.append(factor)
            return result
        
        return []
        
    except Exception as e:
        print(f"[ERREUR] Exception lors de la lecture des facteurs : {str(e)}")
        return [{"error": str(e)}]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Input file path required"}))
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        result = predict(input_data)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)