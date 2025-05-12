import sys
import json
import pickle
import os
import pandas as pd
from fault_classifier import FaultTypeClassifier
import joblib
import re

# Ajouter en haut du fichier
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
        print(f"Loading model from: {model_path}", file=sys.stderr)
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load model: {str(e)}", file=sys.stderr)
        raise

@lru_cache(maxsize=1)
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
    encodings = ['latin1', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'utf-8', 'utf-16', 'ascii', 'windows-1250', 'windows-1252']
    
    # Tentative de détection automatique de l'encodage
    try:
        import chardet
        with open(csv_path, 'rb') as f:
            result = chardet.detect(f.read())
        detected_encoding = result['encoding']
        if detected_encoding and detected_encoding not in encodings:
            encodings.insert(0, detected_encoding)
            print(f"[INFO] Encodage détecté automatiquement: {detected_encoding}", file=sys.stderr)
    except ImportError:
        print("[INFO] Module chardet non disponible, utilisation de la liste d'encodages prédéfinie", file=sys.stderr)
    
    # Reste du code inchangé
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

def predict(input_data, preloaded_model=None):
    try:
        print("[DEBUG] Début de la prédiction avec les données:", file=sys.stderr)
        print(f"[DEBUG] {json.dumps(input_data, ensure_ascii=False)}", file=sys.stderr)
        
        # Load model if not provided
        model_data = preloaded_model if preloaded_model is not None else load_model()
        print(f"[DEBUG] Modèle chargé avec succès. Features: {model_data['features']}", file=sys.stderr)
        
        # Process input data for prediction
        processed_data = process_input_data(input_data, model_data)
        
        # Make prediction
        X_input = pd.DataFrame([processed_data])
        prediction_proba = model_data['model'].predict_proba(X_input)[0]
        
        # Analyse de la description pour détecter les situations critiques
        desc = input_data.get('Description', '').lower()
        
        # Détection des mots-clés critiques
        critical_keywords = [
            'arrêt d\'urgence', 'critique', 'emergency', 'critical', 'urgent', 'urgente', 
            'majeure', 'severe', 'grave', 'graves', 'sérieux', 'sérieuse', 'important', 'significatif',
            'dangereux', 'dangereuse', 'risque', 'prioritaire', 'immédiat', 'immédiate'
        ]
        technical_issues = [
            'surchauffe', 'overheating', 'fuite', 'leak', 'dysfonctionnement', 'malfunction',
            'hydraulique', 'pression', 'huile', 'oil', 'rupture', 'cassé', 'cassée', 'brisé', 'brisée',
            'bloqué', 'bloquée', 'vibration', 'bruit', 'noise', 'odeur', 'smell'
        ]
        critical_count = sum(1 for kw in critical_keywords if kw in desc)
        issues_count = sum(1 for issue in technical_issues if issue in desc)
        
        # Ajustement de la probabilité pour les situations critiques
        if critical_count > 0 or issues_count >= 2:
            # Augmenter la probabilité de panne en fonction de la gravité détectée
            severity_factor = min(0.98, 0.5 + (critical_count * 0.2) + (issues_count * 0.15))
            final_fault_prob = max(prediction_proba[1], severity_factor)
            print(f"[INFO] Situation critique détectée! Probabilité ajustée: {final_fault_prob:.2f}", file=sys.stderr)
        else:
            final_fault_prob = min(0.95, prediction_proba[1])
            
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
            if str(value) not in map(str, model_data['label_encoders'][feature].classes_):
                print(f"[AVERTISSEMENT] Valeur '{value}' non trouvée pour la feature '{feature}', utilisation de 'UNKNOWN'", file=sys.stderr)
                value = 'UNKNOWN'
                
            try:
                processed_data[feature] = model_data['label_encoders'][feature].transform([value])[0]
            except Exception as e:
                print(f"[ERREUR] Échec de transformation pour {feature}: {str(e)}", file=sys.stderr)
                # Utiliser 0 comme valeur par défaut pour les erreurs d'encodage
                processed_data[feature] = 0
        else:
            # Pour les features numériques, s'assurer qu'elles sont converties en nombres
            if feature in ['oil_level', 'downtime', 'vibration', 'seniority']:
                try:
                    processed_data[feature] = float(value) if value != 'UNKNOWN' else 0.0
                except (ValueError, TypeError):
                    print(f"[AVERTISSEMENT] Conversion en nombre échouée pour {feature}, utilisation de 0", file=sys.stderr)
                    processed_data[feature] = 0.0
            else:
                processed_data[feature] = value
    
    # Vérifier que toutes les features requises sont présentes
    missing_features = [f for f in model_data['features'] if f not in processed_data]
    if missing_features:
        print(f"[AVERTISSEMENT] Features manquantes: {missing_features}", file=sys.stderr)
        # Ajouter des valeurs par défaut pour les features manquantes
        for feature in missing_features:
            if feature in ['oil_level', 'downtime', 'vibration', 'seniority']:
                processed_data[feature] = 0.0
            else:
                processed_data[feature] = 0
    
    # Vérifier que toutes les valeurs sont numériques avant de retourner
    for feature, value in processed_data.items():
        if not isinstance(value, (int, float)):
            print(f"[AVERTISSEMENT] Conversion forcée de {feature} en nombre", file=sys.stderr)
            try:
                processed_data[feature] = float(value)
            except (ValueError, TypeError):
                processed_data[feature] = 0.0
    
    print(f"[DEBUG] Données prétraitées: {processed_data}", file=sys.stderr)
    return processed_data

# Supprimer cette fonction entière
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
    if fault_prob > 0.4:  # Seuil abaissé pour mieux détecter les situations critiques
        return "Critique"
    elif fault_prob > 0.2:  # Seuil abaissé pour le niveau élevé
        return "Eleve"
    elif fault_prob > 0.1:  # Seuil pour les risques moyens
        return "Moyen"
    return "Faible"

def monitor_model_performance(prediction_result, actual_outcome=None):
    """
    Surveille les performances du modèle en production.
    
    Args:
        prediction_result (dict): Résultat de la prédiction
        actual_outcome (str, optional): Résultat réel (si disponible)
        
    Returns:
        None
    """
    try:
        # Créer un répertoire pour stocker les données de surveillance
        monitoring_dir = os.path.join(os.path.dirname(__file__), 'monitoring')
        os.makedirs(monitoring_dir, exist_ok=True)
        
        # Fichier de surveillance
        monitoring_file = os.path.join(monitoring_dir, f'model_monitoring_{datetime.now().strftime("%Y%m%d")}.csv')
        
        # Préparer les données à enregistrer
        monitoring_data = {
            'timestamp': datetime.now().isoformat(),
            'predicted_state': prediction_result.get('prediction', {}).get('etat', 'Inconnu'),
            'confidence': prediction_result.get('prediction', {}).get('details', {}).get('confidence', '0%'),
            'actual_outcome': actual_outcome if actual_outcome is not None else 'Unknown'
        }
        
        # Créer un DataFrame
        monitoring_df = pd.DataFrame([monitoring_data])
        
        # Ajouter au fichier CSV (créer s'il n'existe pas)
        if os.path.exists(monitoring_file):
            monitoring_df.to_csv(monitoring_file, mode='a', header=False, index=False)
        else:
            monitoring_df.to_csv(monitoring_file, index=False)
            
        logger.info(f"Données de surveillance enregistrées: {monitoring_data}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la surveillance du modèle: {str(e)}")

def explain_prediction(input_data):
    """
    Explique la prédiction en utilisant les valeurs SHAP.
    
    Args:
        input_data (dict): Données d'entrée pour la prédiction
        
    Returns:
        dict: Explication de la prédiction avec les contributions des features
    """
    try:
        # Importer SHAP (nécessite d'installer le package)
        import shap
        
        # Charger le modèle
        model_data = load_model()
        model = model_data['model']
        
        # Prétraiter les données
        processed_data = process_input_data(input_data, model_data)
        X_input = pd.DataFrame([processed_data])
        
        # Créer l'explainer SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        
        # Obtenir les noms des features
        feature_names = X_input.columns.tolist()
        
        # Créer l'explication
        explanation = []
        for i, feature in enumerate(feature_names):
            # Pour un modèle de classification binaire, shap_values est une liste de deux arrays
            # Le premier pour la classe 0, le second pour la classe 1
            contribution_class_1 = shap_values[1][0][i]
            explanation.append({
                "feature": feature,
                "contribution": float(contribution_class_1),
                "impact": "Positif" if contribution_class_1 > 0 else "Négatif",
                "magnitude": abs(float(contribution_class_1))
            })
        
        # Trier par magnitude décroissante
        explanation.sort(key=lambda x: x["magnitude"], reverse=True)
        
        # Faire une prédiction normale
        prediction_result = predict(input_data)
        
        # Ajouter l'explication à la prédiction
        prediction_result["explanation"] = explanation
        
        return prediction_result
        
    except ImportError:
        print("[AVERTISSEMENT] Le package shap n'est pas installé. L'explication n'est pas disponible.", file=sys.stderr)
        # Faire une prédiction normale sans explication
        return predict(input_data)
    except Exception as e:
        print(f"[ERREUR] Échec de l'explication: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {
            "success": False,
            "error": str(e),
            "message": "Impossible de générer l'explication"
        }
# Ajouter en haut du fichier
import logging
from datetime import datetime

# Configuration du logger
def setup_logger():
    logger = logging.getLogger('predict')
    logger.setLevel(logging.INFO)
    
    # Créer un gestionnaire de fichier qui écrit les logs dans un fichier
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'predict_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Créer un gestionnaire de console qui écrit les logs sur stderr
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    
    # Créer un formateur et l'ajouter aux gestionnaires
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Ajouter les gestionnaires au logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialiser le logger
logger = setup_logger()

# À la fin du fichier, ajoutez ce code pour le traitement des entrées depuis stdin
if __name__ == "__main__":
    try:
        # Lire les données d'entrée depuis stdin ou un fichier
        if len(sys.argv) > 1:
            # Si un fichier est spécifié en argument
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        else:
            # Sinon, lire depuis stdin
            input_data = json.load(sys.stdin)
        
        # Effectuer la prédiction
        result = predict(input_data)
        
        # Écrire le résultat sur stdout au format JSON
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "message": "Une erreur s'est produite lors du traitement"
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)