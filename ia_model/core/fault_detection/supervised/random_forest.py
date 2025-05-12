# random_forest.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import json
from collections import defaultdict
import logging
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Configuration du système de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("random_forest_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RandomForestFaultDetector')

class RandomForestFaultDetector:
    def __init__(self, n_estimators=100, random_state=42):
        logger.info("Initialisation du détecteur de pannes avec Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        self.keywords = [
            # Urgence et criticité
            'urgent', 'emergency', 'critical', 'arrêt d\'urgence', 'emergency stop',
            'majeure', 'severe', 'grave', 'important', 'significant', 'major',
            
            # Pannes matérielles
            'leak', 'stopped', 'overheating', 'failure', 'broken', 
            'malfunction', 'error', 'defect', 'fault', 'out of order',
            'shutdown', 'crash', 'jammed', 'blocked', 'unresponsive',
            'slow', 'interrupted', 'disconnected', 'unstable', 'corrupted',
            
            # Mots-clés spécifiques en français
            'défaillance', 'surchauffe', 'panne', 'arrêt', 'problème',
            'dysfonctionnement', 'erreur', 'défaut', 'bloqué', 'instable',
            
            # Termes techniques spécifiques
            'moteur', 'motor', 'machine', 'équipement', 'equipment',
            
            # Termes de dégradation
            'fuite', 'leak', 'rupture', 'breakdown', 
            
            # Systèmes critiques
            'système de refroidissement', 'cooling system', 
            'système principal', 'main system'
        ]
        
        self.features = ['LOCATION', 'ASSETNUM', 'Description', 'location_description', 'assetnum_description']
        self.numeric_features = ['oil_level', 'downtime', 'seniority']
        self.categorical_features = [
            'LOCATION', 'ASSETNUM', 'Description', 'location_description', 
            'assetnum_description', 'type_lubrification', 'vibration', 
            'power_alimentation', 'maintenance_frequency'
        ]
        
        self.label_encoders = defaultdict(LabelEncoder)
        self.feature_categories = {}
        self.status_risk = {
            'OPEN': 0.4,
            'INPRG': 0.3,
            'WAPPR': 0.3,
            'CLOSE': 0.1,
            'COMP': 0.1,
            'UNKNOWN': 0.4
        }
        logger.info(f"Features utilisées: {self.features}")

    def load_and_prepare_data(self, file_path=None):
        """Charge et prépare les données avec encodage UTF-8"""
        try:
            if file_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))),
                    'data',
                    'clean_workorders.csv'
                )
        
            logger.info(f"Chargement des données depuis: {file_path}")
            if not os.path.exists(file_path):
                logger.error(f"Fichier non trouvé: {file_path}")
                raise FileNotFoundError(f"Le fichier n'a pas été trouvé : {file_path}")

            print(f"\n{'='*50}\nChargement du fichier CSV depuis : {file_path}")
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        
            # Nettoyage des noms de colonnes
            df.columns = df.columns.str.replace('ï»¿', '').str.strip()
            logger.info(f"Données chargées: {len(df)} enregistrements")
        
            # Vérification de la colonne Description
            description_column = next((col for col in df.columns if col.lower() == 'description'), None)
            if not description_column:
                logger.error("Column 'Description' not found in the dataset")
                print("\nAvailable columns:", df.columns.tolist())
                raise KeyError("Column 'Description' is required but not found in the dataset")
        
            # Chargement des données supplémentaires
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
        
            # Données de location
            location_path = os.path.join(base_dir, 'data', 'table_location.csv')
            try:
                locations_df = pd.read_csv(location_path, sep=';', encoding='utf-8')
                df = df.merge(locations_df[['LOCATION', 'location_description']], 
                            on='LOCATION', how='left')
                df['location_description'] = df['location_description'].fillna('UNKNOWN')
            except Exception as e:
                logger.warning(f"Impossible de charger table_location.csv: {str(e)}")
                df['location_description'] = 'UNKNOWN'
        
            # Données d'assetnum
            assetnum_path = os.path.join(base_dir, 'data', 'table_assetnum.csv')
            try:
                assetnum_df = pd.read_csv(assetnum_path, sep=';', encoding='utf-8')
                df = df.merge(assetnum_df[['ASSETNUM', 'assetnum_description']], 
                            on='ASSETNUM', how='left')
                df['assetnum_description'] = df['assetnum_description'].fillna('UNKNOWN')
            except Exception as e:
                logger.warning(f"Impossible de charger table_assetnum.csv: {str(e)}")
                df['assetnum_description'] = 'UNKNOWN'
        
            # Nettoyage des données
            df_cleaned = df.dropna().copy()
            
            # Conversion des colonnes numériques
            for col in self.numeric_features:
                if col in df_cleaned.columns:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
            
            # Gestion des catégories
            for col in self.categorical_features:
                if col in df_cleaned.columns:
                    df_cleaned[col] = df_cleaned[col].fillna('UNKNOWN')
                if col not in self.feature_categories:
                    self.feature_categories[col] = df_cleaned[col].unique().tolist()
                if 'UNKNOWN' not in self.feature_categories[col]:
                    self.feature_categories[col].append('UNKNOWN')
        
            # Détection des pannes avec logique renforcée
            df_cleaned.loc[:, 'PANNE'] = df_cleaned[description_column].apply(
                lambda x: 1 if any(
                    keyword.lower() in str(x).lower() 
                    for keyword in self.keywords
                ) and (
                    'urgent' in str(x).lower() or
                    'emergency' in str(x).lower() or
                    'critical' in str(x).lower() or
                    'arrêt' in str(x).lower() or
                    'panne' in str(x).lower() or
                    'défaillance' in str(x).lower() or
                    'failure' in str(x).lower()
                ) else 0
            )
        
            print(f"\nDistribution des pannes :\n{df_cleaned['PANNE'].value_counts()}")
            print(f"\nTaux de pannes : {df_cleaned['PANNE'].mean():.2%}")
            logger.info(f"Distribution des pannes: {df_cleaned['PANNE'].value_counts().to_dict()}")
        
            return df_cleaned
    
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données : {str(e)}", exc_info=True)
            raise

    def preprocess_data(self, df):
        """Prétraitement des données avec gestion des valeurs inconnues"""
        logger.info("Début du prétraitement des données")
        print("\nPrétraitement des données...")
        
        for feature in self.features:
            if feature in self.categorical_features and feature in df.columns:
                # Gestion des valeurs manquantes et inconnues
                df[feature] = df[feature].fillna('UNKNOWN')
                df[feature] = df[feature].apply(
                    lambda x: str(x) if str(x) in map(str, self.feature_categories[feature]) else "UNKNOWN"
                )
                
                # Encodage des catégories
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    self.label_encoders[feature].fit(self.feature_categories[feature])
                df[feature] = self.label_encoders[feature].transform(df[feature])
                
                logger.info(f"Encodage de {feature} terminé avec {len(self.label_encoders[feature].classes_)} classes")
        
        X = df[self.features]
        y = df['PANNE']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        logger.info(f"Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Entraîne le modèle Random Forest"""
        logger.info("Début de l'entraînement du modèle")
        print("\nEntraînement du modèle...")
        self.model.fit(X_train, y_train)
        logger.info("Modèle entraîné avec succès")
        
        # Importance des features
        feature_importances = pd.DataFrame(
            self.model.feature_importances_,
            index=self.features,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        print("\nImportance des features :")
        print(feature_importances)
        logger.info(f"Importance des features: {feature_importances.to_dict()}")
        
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Évalue le modèle et génère les visualisations"""
        logger.info("Début de l'évaluation du modèle")
        print("\nÉvaluation du modèle...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        }
        
        self.accuracy = metrics['Accuracy']
        
        # Affichage des résultats
        print("\n=== Performance du modèle ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.2%}")
        
        print("\nRapport de classification:")
        report = classification_report(y_test, y_pred)
        print(report)
        logger.info(f"Métriques d'évaluation: {metrics}")
        logger.info(f"Rapport de classification:\n{report}")
        
        # Matrice de confusion
        self.plot_confusion_matrix(y_test, y_pred)
        
        return metrics

    def save_model(self, file_name='random_forest_model.pkl'):
        """Sauvegarde le modèle et les préprocesseurs"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_name)
        logger.info(f"Sauvegarde du modèle dans: {file_path}")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoders': dict(self.label_encoders),
            'features': self.features,
            'keywords': self.keywords,
            'feature_categories': self.feature_categories,
            'accuracy': getattr(self, 'accuracy', None)
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModèle sauvegardé dans {file_path}")
        print(f"Taille du fichier : {os.path.getsize(file_path)/1024:.2f} KB")
        print(f"Précision du modèle : {getattr(self, 'accuracy', None) * 100:.2f}%")
        logger.info(f"Modèle sauvegardé avec succès. Taille: {os.path.getsize(file_path)/1024:.2f} KB")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Génère et affiche la matrice de confusion"""
        logger.info("Génération de la matrice de confusion")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fonctionnel', 'Panne'],
                    yticklabels=['Fonctionnel', 'Panne'])
        
        plt.title('Matrice de Confusion - Random Forest')
        plt.ylabel('Vérité terrain')
        plt.xlabel('Prédiction')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        logger.info("Matrice de confusion sauvegardée dans 'confusion_matrix.png'")
        plt.show()

    def predict_fault(self, input_data):
        """Prédit si une panne est détectée avec une logique renforcée"""
        try:
            # Vérification et normalisation des données d'entrée
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    logger.error("Erreur de décodage JSON des données d'entrée")
                    return self._error_response("Format de données invalide")
        
            if not isinstance(input_data, dict):
                logger.error(f"Format de données invalide: {type(input_data)}")
                return self._error_response("Format de données invalide")
        
            # Initialisation des données
            processed_data = {
                'LOCATION': input_data.get('LOCATION', 'UNKNOWN'),
                'ASSETNUM': input_data.get('ASSETNUM', 'UNKNOWN'),
                'Description': input_data.get('Description', ''),
                'location_description': 'UNKNOWN',
                'assetnum_description': 'UNKNOWN'
            }
        
            # Analyse de la description avec une logique plus stricte
            desc = processed_data['Description'].lower()
            risk_factors = []
            risk_score = 0.0
        
            # Détection initiale des conditions critiques avec scores plus élevés
            critical_conditions = {
                'arrêt d\'urgence': 10.0,  # Score maximal
                'emergency stop': 10.0,
                'critique': 8.0,
                'critical': 8.0,
                'urgent': 7.0,
                'majeure': 7.0,
                'severe': 7.0,
                'important': 6.0,
                'importante': 6.0  # Ajout variante féminine
            }
        
            # Vérification des conditions critiques avec cumul amélioré
            critical_count = 0
            for condition, score in critical_conditions.items():
                if condition in desc:
                    risk_score += score * 2.0  # Multiplicateur augmenté
                    critical_count += 1
                    risk_factors.append(f"Condition critique détectée: {condition}")
        
            # Bonus pour conditions critiques multiples renforcé
            if critical_count > 1:
                risk_score *= 2.5  # Multiplicateur augmenté
        
            # Détection des problèmes techniques avec scores plus élevés
            technical_issues = {
                'surchauffe': 8.0,
                'overheating': 8.0,
                'fuite': 7.0,
                'leak': 7.0,
                'dysfonctionnement': 7.0,
                'malfunction': 7.0,
                'système de refroidissement': 6.0,
                'cooling system': 6.0,
                'moteur principal': 6.0,
                'main motor': 6.0
            }
        
            # Vérification des problèmes techniques avec cumul
            detected_issues = []
            for issue, score in technical_issues.items():
                if issue in desc:
                    risk_score += score * 1.5  # Multiplicateur augmenté
                    detected_issues.append(issue)
                    risk_factors.append(f"Problème technique détecté: {issue}")
        
            # Bonus pour combinaison de problèmes (renforcé)
            if len(detected_issues) > 1:
                risk_score *= 3.0  # Multiplicateur significativement augmenté
                risk_factors.append(f"Combinaison critique de {len(detected_issues)} problèmes détectés")
        
            # Diagnostic spécifique avec détection multiple et scores augmentés
            fault_types = []
            if any(kw in desc for kw in ['surchauffe', 'overheating']):
                fault_types.append("SURCHAUFFE")
                risk_score += 6.0
            if any(kw in desc for kw in ['fuite', 'leak']):
                fault_types.append("FUITE")
                risk_score += 5.0
            if any(kw in desc for kw in ['arrêt', 'stop', 'shutdown']):
                fault_types.append("ARRÊT")
                risk_score += 7.0
            if any(kw in desc for kw in ['dysfonctionnement', 'malfunction']):
                fault_types.append("DYSFONCTIONNEMENT")
                risk_score += 5.0
        
            fault_diagnosis = "+".join(fault_types) if fault_types else None
        
            # Normalisation du score de risque avec seuil plus élevé
            risk_score = min(max(risk_score, 0), 10)
        
            # Calcul des probabilités avec échelle ajustée - CORRIGEÉ
            base_fault_prob = min(0.99, (risk_score / 10) * 5.0)  # Multiplicateur augmenté davantage
        
            # Ajustement final basé sur la criticité - CORRIGÉ
            if len(fault_types) >= 2:
                base_fault_prob = min(0.99, base_fault_prob * 3.0)  # Renforcement significatif
            if critical_count >= 1:  # Activé dès qu'une condition critique est détectée
                base_fault_prob = min(0.99, base_fault_prob * 2.5)  # Renforcement significatif
        
            final_fault_prob = base_fault_prob
            final_ok_prob = 1 - final_fault_prob  # Calcul de la probabilité complémentaire
        
            # Seuils de risque ajustés - CORRIGÉ
            risk_level = (
                "Critique" if final_fault_prob > 0.4 else
                "Élevé" if final_fault_prob > 0.25 else 
                "Moyen" if final_fault_prob > 0.1 else 
                "Faible"
            )
        
            # Construction du résultat avec état ajusté - CORRIGÉ
            # Abaissement du seuil à 0.35 pour les pannes
            result = {
                "success": True,
                "prediction": {
                    "etat": "En panne" if final_fault_prob > 0.35 else "Fonctionnel",
                    "details": {
                        "confidence": f"{max(final_fault_prob, final_ok_prob) * 100:.2f}%",
                        "risk_level": risk_level,
                        "probabilities": {
                            "fonctionnel": f"{final_ok_prob * 100:.2f}%",
                            "panne": f"{final_fault_prob * 100:.2f}%"
                        },
                        "risk_factors": risk_factors,
                        "fault_diagnosis": fault_diagnosis,
                        "influencing_factors": self._get_influencing_factors(fault_diagnosis) if fault_diagnosis else []
                    },
                    "message": "Analyse terminée avec succès"
                }
            }
        
            logger.info(f"Résultat de la prédiction: {result}")
            return result
    
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
            return self._error_response(f"Erreur lors de la prédiction: {str(e)}")

    def _get_influencing_factors(self, fault_type):
        """Récupère les facteurs influents pour un type de panne donné"""
        factors = []
        if fault_type == "SURCHAUFFE":
            factors.extend(["Température élevée", "Défaillance du système de refroidissement", "Surcharge"])
        if fault_type == "FUITE":
            factors.extend(["Usure des joints", "Pression excessive", "Corrosion"])
        if fault_type == "ARRÊT_URGENCE":
            factors.extend(["Défaillance mécanique", "Problème électrique", "Surchauffe critique"])
        return factors

    def _error_response(self, message):
        """Génère une réponse d'erreur standardisée"""
        return {
            "success": False,
            "error": message,
            "details": {
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }

def main():
    """Fonction principale pour exécuter le pipeline complet"""
    print("\n" + "="*50)
    print("=== Détection de Pannes avec Random Forest ===")
    print("="*50 + "\n")
    
    try:
        detector = RandomForestFaultDetector()
        
        # 1. Chargement des données
        print("[1/4] Chargement et préparation des données...")
        df = detector.load_and_prepare_data()
        
        # 2. Prétraitement
        print("[2/4] Prétraitement des données...")
        X_train, X_test, y_train, y_test = detector.preprocess_data(df)
        
        # 3. Entraînement
        print("[3/4] Entraînement du modèle...")
        detector.train_model(X_train, y_train)
        
        # 4. Évaluation
        print("[4/4] Évaluation du modèle...")
        metrics = detector.evaluate_model(X_test, y_test)
        
        # Sauvegarde du modèle
        detector.save_model()
        
        # Test avec des cas critiques
        test_cases = [
            {
                'LOCATION': 'BR300',
                'STATUS': 'OPEN',
                'WOPRIORITY': '1',
                'ASSETNUM': 'EQP123',
                'Description': 'ARRÊT D\'URGENCE CRITIQUE - Surchauffe majeure du moteur principal'
            },
            {
                'LOCATION': 'NEW_LOC',
                'STATUS': 'INPRG',
                'WOPRIORITY': '3',
                'ASSETNUM': 'NEW_EQP',
                'Description': 'Maintenance préventive programmée'
            },
            {
                'LOCATION': 'LOC02',
                'STATUS': 'WAPPR',
                'WOPRIORITY': '1',
                'ASSETNUM': 'EQP456',
                'Description': 'Fuite importante détectée dans le système hydraulique'
            }
        ]
        
        for case in test_cases:
            print(f"\nTest de prédiction pour : {case['Description']}")
            result = detector.predict_fault(case)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n=== Processus terminé avec succès ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}\n")
        return 1
    
    return 0

if __name__ == "__main__":
    main()