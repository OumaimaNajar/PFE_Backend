import sys
import json
import pickle
import os
import pandas as pd
from fault_classifier import FaultTypeClassifier
import joblib
import re

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

def load_factors_csv():
    """Chargement du CSV des facteurs d'influence avec gestion de plusieurs encodages."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'data', 'facteurs_influencent_pannes.csv')
    
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier CSV non trouvé : {csv_path}", file=sys.stderr)
        return None
        
    # Liste des encodages à tester
    encodings = ['latin1', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'utf-8']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, sep=';', encoding=encoding)
            print(f"[INFO] CSV chargé avec succès. Encodage: {encoding}", file=sys.stderr)
            print(f"[DEBUG] Colonnes disponibles: {df.columns.tolist()}", file=sys.stderr)
            
            # Vérifier si les colonnes nécessaires existent
            required_columns = ['type_panne', 'facteur']
            numerical_columns = ['PB', 'FC', 'oil_level', 'downtime']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"[AVERTISSEMENT] Colonnes manquantes dans le CSV: {missing_columns}", file=sys.stderr)
            
            # Ajouter les colonnes numériques si elles n'existent pas
            for col in numerical_columns:
                if col not in df.columns:
                    print(f"[INFO] Ajout de la colonne manquante '{col}' avec valeurs par défaut", file=sys.stderr)
                    df[col] = '0.0'  # Utiliser une chaîne par défaut au lieu d'un nombre
                    
            # Ne pas convertir les colonnes PB et FC en nombres, les garder comme strings
            # Pour les autres colonnes numériques, les convertir en nombres
            for col in ['oil_level', 'downtime']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
        except Exception as e:
            print(f"[ERREUR] Échec de chargement avec encodage {encoding}: {str(e)}", file=sys.stderr)
            continue
    
    print("[ERREUR] Impossible de charger le CSV avec tous les encodages testés", file=sys.stderr)
    return None

def normalize_string(text):
    """Normalise les chaînes pour une comparaison cohérente."""
    if not isinstance(text, str):
        return ""
    # Supprimer les accents, mettre en majuscule, supprimer la ponctuation et les espaces multiples
    text = text.upper().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

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
            # Classifier le type de panne
            classifier = FaultTypeClassifier()
            fault_diagnosis = classifier.predict_fault_type(
                input_data.get("Description", ""),
                input_data.get("ASSETNUM", "")
            )
            
            # Récupérer le type de panne
            type_panne = fault_diagnosis.get("etat", "")
            print(f"[DEBUG] Type de panne détecté: {type_panne}", file=sys.stderr)
            
            if type_panne:
                # Approche 1: Utiliser le modèle de facteurs
                facteur_model = load_facteur_model()
                facteur_principal = ""
                
                if facteur_model:
                    try:
                        type_encoder = facteur_model.get('type_encoder')
                        facteur_encoder = facteur_model.get('facteur_encoder')
                        rf_model = facteur_model.get('model')
                        
                        if type_encoder and facteur_encoder and rf_model:
                            # Vérifier si le type de panne est connu par l'encodeur
                            if type_panne in type_encoder.classes_:
                                type_panne_enc = type_encoder.transform([type_panne])
                                pred = rf_model.predict(type_panne_enc.reshape(-1, 1))
                                facteur_principal = facteur_encoder.inverse_transform(pred)[0]
                                print(f"[INFO] Facteur principal prédit par le modèle: {facteur_principal}", file=sys.stderr)
                            else:
                                print(f"[AVERTISSEMENT] Type de panne '{type_panne}' inconnu du modèle", file=sys.stderr)
                    except Exception as e:
                        print(f"[ERREUR] Échec de prédiction du facteur: {str(e)}", file=sys.stderr)
                
                # Approche 2: Rechercher le type de panne dans le CSV
                df = load_factors_csv()
                if df is not None:
                    # Normaliser pour la comparaison
                    df['type_panne_norm'] = df['type_panne'].apply(normalize_string)
                    type_panne_norm = normalize_string(type_panne)
                    
                    print(f"[DEBUG] Recherche du type de panne normalisé: '{type_panne_norm}'", file=sys.stderr)
                    
                    # Chercher d'abord une correspondance exacte
                    panne_match = df[df['type_panne_norm'] == type_panne_norm]
                    
                    # Si aucune correspondance exacte, essayer une correspondance partielle
                    if panne_match.empty:
                        print(f"[DEBUG] Aucune correspondance exacte, essai de correspondance partielle", file=sys.stderr)
                        for idx, row in df.iterrows():
                            if type_panne_norm in row['type_panne_norm'] or row['type_panne_norm'] in type_panne_norm:
                                panne_match = df.iloc[[idx]]
                                print(f"[DEBUG] Correspondance partielle trouvée: {row['type_panne']}", file=sys.stderr)
                                break
                    
                    if not panne_match.empty:
                        panne_data = panne_match.iloc[0]
                        
                        # Utiliser le facteur du modèle s'il existe, sinon prendre celui du CSV
                        if not facteur_principal:
                            facteur_principal = panne_data.get('facteur', "")
                        
                        # Récupérer toutes les valeurs du CSV
                        influencing_factors = [{
                            "facteur_principal": facteur_principal,
                            "type_panne": type_panne,
                            "PB": str(panne_data.get('PB', '0.0')),
                            "FC": str(panne_data.get('FC', '0.0')),
                            "oil_level": float(panne_data.get('oil_level', 0)),
                            "downtime": float(panne_data.get('downtime', 0)),
                            "recommended_action": str(panne_data.get('recommended_action', '')),
                            "type_lubrification": str(panne_data.get('type_lubrification', '')),
                            "vibration": str(panne_data.get('vibration', '')),
                            "power_alimentation": str(panne_data.get('power_alimentation', '')),
                            "maintenance_frequency": str(panne_data.get('maintenance_frequency', '')),
                            "seniority": int(panne_data.get('seniority', 0)),
                            "valeur": str(panne_data.get('valeur', '')),
                            "pourcentage": float(panne_data.get('pourcentage', 0)),
                            "feature_importance": []
                        }]

                        # Récupérer les importances calculées dynamiquement depuis le modèle
                        if facteur_model and isinstance(facteur_model, dict) and 'model' in facteur_model:
                            model = facteur_model['model']
                            if hasattr(model, 'feature_importances_'):
                                print(f"[DEBUG] Récupération des feature importances du modèle", file=sys.stderr)
                                
                                # Définir les features dans le même ordre que l'entraînement
                                features = [
                                    'downtime',
                                    'oil_level',
                                    'type_lubrification',
                                    'vibration',
                                    'power_alimentation',
                                    'maintenance_frequency',
                                    'seniority'
                                ]
                                
                                # Calculer les pourcentages d'importance en ignorant type_panne, PB, FC
                                importances = model.feature_importances_[3:]  # Ignorer les 3 premières features
                                total_importance = sum(importances)
                                
                                for feature, importance in zip(features, importances):
                                    percentage = (importance / total_importance) * 100
                                    impact = "Tres fort" if percentage > 25 else \
                                            "Fort" if percentage > 20 else \
                                            "Moyen" if percentage > 10 else "Faible"
                                            
                                    description = {
                                        'downtime': "Temps d'arrêt",
                                        'oil_level': "Niveau d'huile",
                                        'type_lubrification': "Type de lubrification utilisé",
                                        'vibration': "Niveau de vibration",
                                        'power_alimentation': "Alimentation électrique",
                                        'maintenance_frequency': "Fréquence de maintenance",
                                        'seniority': "Ancienneté de l'équipement"
                                    }
                                    
                                    influencing_factors[0]["feature_importance"].append({
                                        "feature": feature,
                                        "importance": round(percentage, 2),
                                        "impact": impact,
                                        "contribution": f"{percentage:.1f}%",
                                        "description": description[feature]
                                    })
                                
                                print(f"[INFO] Feature importances récupérées avec succès", file=sys.stderr)
                            else:
                                print(f"[AVERTISSEMENT] Le modèle n'a pas d'attribut feature_importances_", file=sys.stderr)
                        else:
                            print(f"[AVERTISSEMENT] Structure du modèle de facteurs invalide", file=sys.stderr)
                    else:
                        print(f"[AVERTISSEMENT] Aucune correspondance trouvée pour le type de panne: {type_panne}", file=sys.stderr)
                        influencing_factors = [{
                            "facteur_principal": facteur_principal if facteur_principal else "Inconnu",
                            "type_panne": type_panne,
                            "PB": "0.0",  # Utiliser une chaîne par défaut
                            "FC": "0.0",  # Utiliser une chaîne par défaut
                            "oil_level": 0,
                            "downtime": 0,
                            "message": "Type de panne non trouvé dans la base de données"
                        }]
                else:
                    # Cas où le CSV n'a pas pu être chargé
                    influencing_factors = [{
                        "facteur_principal": facteur_principal if facteur_principal else "Inconnu",
                        "type_panne": type_panne,
                        "PB": "0.0",  # Utiliser une chaîne par défaut
                        "FC": "0.0",  # Utiliser une chaîne par défaut
                        "oil_level": 0,
                        "downtime": 0,
                        "message": "Impossible de charger la base de données des facteurs"
                    }]
            else:
                # Cas où aucun type de panne n'a été détecté
                influencing_factors = [{
                    "facteur_principal": "Inconnu",
                    "type_panne": "Non spécifié",
                    "PB": "0.0",  # Utiliser une chaîne par défaut
                    "FC": "0.0",  # Utiliser une chaîne par défaut
                    "oil_level": 0,
                    "downtime": 0,
                    "message": "Type de panne non détecté"
                }]

        # Construire le résultat final
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
                    "fault_diagnosis": {
                        "etat": str(fault_diagnosis.get("etat", "")),
                        "type": str(fault_diagnosis.get("type", "")),
                        "cause": str(fault_diagnosis.get("cause", "")),
                        "solution": str(fault_diagnosis.get("solution", ""))
                    } if fault_diagnosis else None,
                    "influencing_factors": influencing_factors
                },
                "message": "Analysis successfully completed"
            }
        }
        
        # Ajouter les logs pour afficher risk_level et probabilities
        print(f"\n[INFO] Niveau de risque : {result['prediction']['details']['risk_level']}", file=sys.stderr)
        print(f"[INFO] Probabilités :", file=sys.stderr)
        print(f"  - Fonctionnel : {result['prediction']['details']['probabilities']['fonctionnel']}", file=sys.stderr)
        print(f"  - Panne : {result['prediction']['details']['probabilities']['panne']}", file=sys.stderr)
        
        return result
        
    except Exception as e:
        print(f"[ERREUR] Exception dans predict(): {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
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
                    "fault_diagnosis": None,
                    "influencing_factors": [{
                        "facteur_principal": "Erreur",
                        "type_panne": "Inconnu",
                        "PB": "0.0",  # Utiliser une chaîne par défaut
                        "FC": "0.0",  # Utiliser une chaîne par défaut
                        "oil_level": 0,
                        "downtime": 0,
                        "error": str(e)
                    }]
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
                
            try:
                processed_data[feature] = model_data['label_encoders'][feature].transform([value])[0]
            except Exception as e:
                print(f"[ERREUR] Échec de transformation pour {feature}: {str(e)}", file=sys.stderr)
                processed_data[feature] = 0
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

def get_risk_level(fault_prob):
    if fault_prob > 0.7:
        return "Élevé"
    elif fault_prob > 0.3:
        return "Moyen"
    return "Faible"

if __name__ == "__main__":
    try:
        # If a file path is provided as argument, read from file. Otherwise, read from stdin.
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            print(f"[INFO] Lecture du fichier d'entrée: {input_file}", file=sys.stderr)
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        else:
            print("[INFO] Lecture des données depuis stdin", file=sys.stderr)
            input_data = json.loads(sys.stdin.read())
            
        print(f"[DEBUG] Données d'entrée reçues: {json.dumps(input_data, ensure_ascii=False)}", file=sys.stderr)
        
        result = predict(input_data)
        
        # NE PAS essayer de convertir PB et FC en float
        # S'assurer que l'on préserve les valeurs comme des chaînes de caractères
        for factor in result.get('prediction', {}).get('details', {}).get('influencing_factors', []):
            # Vérifier que PB et FC sont bien des strings
            if 'PB' in factor and not isinstance(factor['PB'], str):
                factor['PB'] = str(factor['PB'])
            if 'FC' in factor and not isinstance(factor['FC'], str):
                factor['FC'] = str(factor['FC'])
                
            # Conversion des autres valeurs numériques pour la sérialisation
            for key in ['oil_level', 'downtime']:
                if key in factor:
                    try:
                        factor[key] = float(factor[key])
                    except (ValueError, TypeError):
                        factor[key] = 0
        
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        print(f"[ERREUR] Exception principale: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        
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
                    "fault_diagnosis": None,
                    "influencing_factors": [{
                        "facteur_principal": "Erreur",
                        "type_panne": "Inconnu",
                        "PB": "0.0",  # Utiliser une chaîne par défaut
                        "FC": "0.0",  # Utiliser une chaîne par défaut
                        "oil_level": 0,
                        "downtime": 0,
                        "error": str(e)
                    }]
                },
                "message": "Erreur lors de l'exécution du modèle"
            }
        }