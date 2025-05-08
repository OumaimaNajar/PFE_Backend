import sys
import json
import pickle
import os
import pandas as pd
from fault_classifier import FaultTypeClassifier
import joblib

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
        print(f"Loading model from: {model_path}", file=sys.stderr)
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load model: {str(e)}", file=sys.stderr)
        raise

def load_facteur_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'facteur_influence_model.pkl')
        print(f"Loading facteur model from: {model_path}", file=sys.stderr)
        return joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load facteur model: {str(e)}", file=sys.stderr)
        return None

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
        
        # If fault detected, classify it and analyze factors
        fault_diagnosis = None
        influencing_factors = []
        if is_fault:
            classifier = FaultTypeClassifier()
            fault_diagnosis = classifier.predict_fault_type(
                input_data.get("Description", ""),
                input_data.get("ASSETNUM", "")
            )
            # Charger le modèle de facteurs influents
            facteur_model = load_facteur_model()
            if facteur_model:
                type_encoder = facteur_model['type_encoder']
                facteur_encoder = facteur_model['facteur_encoder']
                rf_model = facteur_model['model']
                type_panne = fault_diagnosis["etat"]
                type_panne_enc = type_encoder.transform([type_panne])
                pred = rf_model.predict(type_panne_enc.reshape(-1, 1))
                facteur = facteur_encoder.inverse_transform(pred)[0]
                influencing_factors = [{"facteur_principal": facteur}]
            else:
                influencing_factors = []

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
                   # "risk_factors": get_risk_factors(input_data),
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
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'data', 'facteurs_influencent_pannes.csv')
        
        if not os.path.exists(csv_path):
            print(f"[ERREUR] Fichier CSV non trouvé : {csv_path}", file=sys.stderr)
            return []

        # Essayer plusieurs encodages avec latin1 en premier
        try:
            df = pd.read_csv(csv_path, sep=';', encoding='latin1')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
            except Exception as e:
                print(f"[ERREUR] Impossible de lire le fichier CSV: {str(e)}", file=sys.stderr)
                return []
        
        # Liste des encodages à tester avec priorité pour le français
        encodings = ['utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1', 'utf-8']
        
        for encoding in encodings:
            try:
                # Test de lecture avec pandas
                df = pd.read_csv(csv_path, sep=';', encoding=encoding)
                print(f"Encodage réussi avec {encoding}", file=sys.stderr)
                break
            except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
                print(f"Échec avec l'encodage {encoding}: {str(e)}", file=sys.stderr)
                continue
        else:
            print("[ERREUR] Aucun encodage valide trouvé", file=sys.stderr)
            return []
            
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
        print(f"[ERREUR] Exception lors de la lecture des facteurs : {str(e)}", file=sys.stderr)
        return [{"error": str(e)}]

if __name__ == "__main__":
    try:
        # If a file path is provided as argument, read from file. Otherwise, read from stdin.
        if len(sys.argv) > 1:
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(sys.stdin.read())
        result = predict(input_data)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        error_result = {
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
                "message": "Erreur lors de l'exécution du modèle"
            }
        }
        print(json.dumps(error_result, ensure_ascii=False))