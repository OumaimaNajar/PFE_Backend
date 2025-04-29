import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class RandomForestFaultDetector:
    def __init__(self, n_estimators=100, random_state=42):
        """Initialise le détecteur de pannes avec Random Forest"""
        # Modèle pour la détection de panne (fonctionnel vs en panne)
        self.detection_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced',
            min_samples_leaf=2  # Ensure at least 2 samples per leaf
        )
        
        # Modèle pour la classification du type de panne
        self.classification_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced',
            min_samples_leaf=1  # Allow single sample leaves for rare faults
        )
        
        # Modèle pour l'analyse des facteurs influençant les pannes
        self.factors_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        
        # Préprocesseurs
        self.label_encoders = defaultdict(LabelEncoder)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Paramètres
        self.features = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']
        self.text_features = ['Description']
        self.fault_keywords = {}
        self.feature_categories = {}
        self.fault_types = {}
        self.factors = {}
        
        # État d'entraînement
        self.detection_trained = False
        self.classification_trained = False
        self.factors_trained = False

    def load_data(self):
        """Charge les données depuis les fichiers CSV"""
        # Chemins absolus des fichiers
        workorders_path = r"../../../../data/clean_workorders.csv"
        fault_types_path = r"../../../../data/pannes_industrielles_organisees.csv"
        factors_path = r"../../../../data/facteurs_influencent_pannes.csv"
        
        # Afficher les chemins absolus pour le débogage
        print(f"Chemin absolu des ordres de travail: {os.path.abspath(workorders_path)}")
        print(f"Chemin absolu des types de pannes: {os.path.abspath(fault_types_path)}")
        print(f"Chemin absolu des facteurs: {os.path.abspath(factors_path)}")
        
        # Vérifier si les fichiers existent
        if not os.path.exists(workorders_path):
            raise FileNotFoundError(f"Fichier non trouvé: {workorders_path}")
        if not os.path.exists(fault_types_path):
            raise FileNotFoundError(f"Fichier non trouvé: {fault_types_path}")
        if not os.path.exists(factors_path):
            raise FileNotFoundError(f"Fichier non trouvé: {factors_path}")
        
        # Chargement des données
        print(f"\n{'='*50}")
        print("Chargement des données...")
        
        workorders_df = pd.read_csv(workorders_path, sep=';')
        fault_types_df = pd.read_csv(fault_types_path, sep=';')
        factors_df = pd.read_csv(factors_path, sep=';')
        
        # Afficher un aperçu des données
        print("\nAperçu des ordres de travail:")
        print(workorders_df.head(2))
        print("\nColonnes des ordres de travail:", workorders_df.columns.tolist())
        
        print("\nAperçu des types de pannes:")
        print(fault_types_df.head(2))
        print("\nColonnes des types de pannes:", fault_types_df.columns.tolist())
        
        print("\nAperçu des facteurs influençant:")
        print(factors_df.head(2))
        print("\nColonnes des facteurs influençant:", factors_df.columns.tolist())
        
        print(f"\nDonnées chargées avec succès:")
        print(f"- Ordres de travail: {len(workorders_df)} entrées")
        print(f"- Types de pannes: {len(fault_types_df)} entrées")
        print(f"- Facteurs influençant: {len(factors_df)} entrées")
        
        return workorders_df, fault_types_df, factors_df

    def preprocess_workorders(self, df):
        """Prétraite les données des ordres de travail pour la détection de pannes"""
        print("\nPrétraitement des ordres de travail...")
        
        # Nettoyage des données
        df_cleaned = df.dropna(subset=self.features + self.text_features).copy()
        
        # Détection des pannes (à adapter selon vos données)
        # Si la colonne 'PANNE' existe déjà, l'utiliser, sinon la créer
        if 'PANNE' not in df_cleaned.columns:
            # Mots-clés pour détecter les pannes dans la description
            keywords = [
                'panne', 'défaillance', 'problème', 'dysfonctionnement', 'arrêt',
                'brisé', 'cassé', 'fuite', 'surchauffe', 'erreur', 'défaut'
            ]
            df_cleaned['PANNE'] = df_cleaned['Description'].str.contains(
                '|'.join(keywords), case=False, na=False
            ).astype(int)
        
        # Analyse des catégories pour chaque feature
        for feature in self.features:
            if df_cleaned[feature].dtype == 'object':
                self.feature_categories[feature] = df_cleaned[feature].unique().tolist()
                self.feature_categories[feature].append('UNKNOWN')  # Pour les valeurs inconnues
                
                # Encodage des valeurs
                self.label_encoders[feature].fit(self.feature_categories[feature])
                df_cleaned[feature] = df_cleaned[feature].apply(
                    lambda x: x if x in self.feature_categories[feature] else "UNKNOWN"
                )
                df_cleaned[feature] = self.label_encoders[feature].transform(df_cleaned[feature])
        
        print(f"Distribution des pannes: {df_cleaned['PANNE'].value_counts().to_dict()}")
        
        return df_cleaned

    def preprocess_fault_types(self, df, workorders_df):
        """Prétraite les données des types de pannes pour la classification"""
        print("\nPrétraitement des types de pannes...")
        
        # Afficher les colonnes disponibles pour le débogage
        print(f"Colonnes disponibles dans le fichier des types de pannes: {df.columns.tolist()}")
        
        # Renommer la colonne fault_type en type_panne pour correspondre au reste du code
        if 'fault_type' in df.columns and 'type_panne' not in df.columns:
            df = df.rename(columns={'fault_type': 'type_panne'})
            print("Colonne 'fault_type' renommée en 'type_panne'")
        
        # Vérifier si les colonnes nécessaires existent
        required_columns = ['type_panne', 'machine', 'gravite']  # Updated required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le fichier des types de pannes: {missing_columns}")
        
        # Nettoyage des données
        df_cleaned = df.dropna(subset=['type_panne']).copy()
        
        # Encodage du type de panne
        self.label_encoders['type_panne'].fit(df_cleaned['type_panne'].unique())
        
        # Create description from available columns
        df_cleaned['description'] = df_cleaned.apply(
            lambda row: f"{row['type_panne']} - Machine: {row['machine']}, Gravité: {row['gravite']}", 
            axis=1
        )
        
        # Extraction des mots-clés par type de panne
        for fault_type in df_cleaned['type_panne'].unique():
            # Fix: Use type_panne instead of 'description'
            descriptions = df_cleaned[df_cleaned['type_panne'] == fault_type]['description']
            words = ' '.join(descriptions).lower().split()
            # Sélectionner les mots significatifs (plus de 4 caractères)
            keywords = [word for word in words if len(word) > 4]
            # Prendre les 5 mots les plus fréquents
            word_counts = pd.Series(keywords).value_counts()
            self.fault_keywords[fault_type] = word_counts.head(5).index.tolist() if not word_counts.empty else []
        
        # Vectorisation des descriptions
        self.vectorizer.fit(df_cleaned['description'])
        
        print(f"Types de pannes identifiés: {len(df_cleaned['type_panne'].unique())}")
        
        return df_cleaned

    def normalize_fault_type(self, fault_type):
        """Normalise les noms des types de pannes"""
        # Mapping des types de pannes avec plus de variations
        mapping = {
            'FUITE': ['fuite hydraulique', 'fuite_hydraulique', 'FUITE_HYDRAULIQUE', 'Fuite'],
            'SURCHAUFFE': ['surchauffe moteur', 'surchauffe_moteur', 'SURCHAUFFE_MOTEUR', 'Surchauffe'],
            'PANNE_ELECTRIQUE': ['panne electrique', 'panne_electrique', 'PANNE_ELECTRIQUE'],
            'USURE': ['usure mecanique', 'usure_mecanique', 'USURE_MECANIQUE', 'Usure'],
            'BLOCAGE': ['probleme mecanique', 'blocage_mecanique', 'BLOCAGE_MECANIQUE', 'Probleme mecanique'],
            'DEFAILLANCE': ['defaillance capteur', 'defaillance_capteur', 'DEFAILLANCE_CAPTEUR', 'Defaillance'],
            'DEFAUT_LOGICIEL': ['defaut logiciel', 'defaut_logiciel', 'DEFAUT_LOGICIEL'],
            'PANNE_PNEUMATIQUE': ['panne pneumatique', 'panne_pneumatique', 'PANNE_PNEUMATIQUE']
        }
        
        # Normaliser l'entrée
        fault_type = str(fault_type).strip()
        fault_type_lower = fault_type.lower()
        
        # Rechercher une correspondance dans le mapping
        for key, values in mapping.items():
            if any(value.lower() == fault_type_lower or 
                  fault_type_lower in value.lower() or 
                  value.lower() in fault_type_lower 
                  for value in values):
                return key
        
        # Si aucune correspondance n'est trouvée, retourner une version normalisée
        return fault_type.upper().replace(' ', '_')

    def preprocess_factors(self, df):
        """Prétraite les données des facteurs influençant les pannes"""
        print("\nPrétraitement des facteurs influençant...")
        
        # Afficher les colonnes disponibles pour le débogage
        print(f"Colonnes disponibles dans le fichier des facteurs: {df.columns.tolist()}")
        
        # Renommer la colonne TYPE_PANNE en type_panne pour correspondre au reste du code
        if 'TYPE_PANNE' in df.columns and 'type_panne' not in df.columns:
            df = df.rename(columns={'TYPE_PANNE': 'type_panne'})
            print("Colonne 'TYPE_PANNE' renommée en 'type_panne'")
        
        # Renommer la colonne FACTEUR en facteur si nécessaire
        if 'FACTEUR' in df.columns and 'facteur' not in df.columns:
            df = df.rename(columns={'FACTEUR': 'facteur'})
            print("Colonne 'FACTEUR' renommée en 'facteur'")
        
        # Vérifier si les colonnes nécessaires existent
        required_columns = ['type_panne', 'facteur']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le fichier des facteurs: {missing_columns}")
        
        # Nettoyage des données
        df_cleaned = df.dropna(subset=['type_panne', 'facteur']).copy()
        
        # Normaliser les types de pannes AVANT l'encodage
        df_cleaned['type_panne'] = df_cleaned['type_panne'].apply(self.normalize_fault_type)
        
        # Encodage des facteurs
        self.label_encoders['facteur'].fit(df_cleaned['facteur'].unique())
        
        # Création d'un dictionnaire des facteurs par type de panne
        for fault_type in df_cleaned['type_panne'].unique():
            factors = df_cleaned[df_cleaned['type_panne'] == fault_type]
            self.factors[fault_type] = factors['facteur'].tolist()
        
        print(f"Facteurs identifiés: {len(df_cleaned['facteur'].unique())}")
        
        return df_cleaned

    def train(self):
        """Entraîne les trois modèles (détection, classification, facteurs)"""
        try:
            # Chargement des données
            workorders_df, fault_types_df, factors_df = self.load_data()
            
            # Prétraitement des données
            workorders_processed = self.preprocess_workorders(workorders_df)
            fault_types_processed = self.preprocess_fault_types(fault_types_df, workorders_df)
            factors_processed = self.preprocess_factors(factors_df)
            
            # 1. Entraînement du modèle de détection de panne
            print("\n" + "="*50)
            print("Entraînement du modèle de détection de panne...")
            
            X_detect = workorders_processed[self.features]
            y_detect = workorders_processed['PANNE']
            
            X_train_detect, X_test_detect, y_train_detect, y_test_detect = train_test_split(
                X_detect, y_detect, test_size=0.3, random_state=42, stratify=y_detect
            )
            
            self.detection_model.fit(X_train_detect, y_train_detect)
            self.detection_trained = True
            
            # Évaluation du modèle de détection
            y_pred_detect = self.detection_model.predict(X_test_detect)
            print("\nRésultats du modèle de détection:")
            print(classification_report(y_test_detect, y_pred_detect))
            
            # 2. Entraînement du modèle de classification des types de pannes
            print("\n" + "="*50)
            print("Entraînement du modèle de classification des types de pannes...")
            
            # Vectorisation des descriptions
            X_text = self.vectorizer.transform(fault_types_processed['description'])
            y_class = self.label_encoders['type_panne'].transform(fault_types_processed['type_panne'])
            
            # Check class distribution before splitting
            unique_classes, class_counts = np.unique(y_class, return_counts=True)
            min_samples = np.min(class_counts)
            
            if min_samples < 2:
                print(f"Warning: Some fault types have less than 2 samples (minimum: {min_samples})")
                # Use regular train_test_split without stratification
                X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
                    X_text, y_class, test_size=0.3, random_state=42
                )
            else:
                # Use stratified split if we have enough samples per class
                X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
                    X_text, y_class, test_size=0.3, random_state=42, stratify=y_class
                )
            
            self.classification_model.fit(X_train_class, y_train_class)
            self.classification_trained = True
            
            # Évaluation du modèle de classification
            y_pred_class = self.classification_model.predict(X_test_class)
            print("\nRésultats du modèle de classification:")
            print(classification_report(y_test_class, y_pred_class))
            
            # 3. Entraînement du modèle d'analyse des facteurs
            print("\n" + "="*50)
            print("Entraînement du modèle d'analyse des facteurs...")
            
            # Préparation des données pour l'analyse des facteurs
            # Group similar factors to reduce the number of classes
            # Correction du mapping des facteurs (suppression de la virgule)
            factor_mapping = {
                'MAINTENANCE': ['maintenance', 'entretien', 'reparation'],
                'ENVIRONNEMENT': ['environnement', 'temperature', 'humidite', 'climat'],
                'OPERATION': ['operation', 'utilisation', 'manipulation'],
                'QUALITE': ['qualite', 'defaut', 'materiau']
            }  # Suppression de la virgule ici
            
            # Correction du traitement des facteurs filtrés
            factors_filtered = pd.DataFrame({
                'facteur': factors_processed['facteur'],
                'type_panne': factors_processed['type_panne']
            })
            
            # S'assurer que le LabelEncoder est entraîné sur toutes les étiquettes possibles
            all_fault_types = set(factors_filtered['type_panne'].unique()) | set(fault_types_processed['type_panne'].unique())
            self.label_encoders['type_panne'].fit(list(all_fault_types))
            
            # Normaliser et encoder les types de pannes
            factors_filtered['type_panne'] = factors_filtered['type_panne'].apply(self.normalize_fault_type)
            factors_filtered['type_panne_encoded'] = self.label_encoders['type_panne'].transform(factors_filtered['type_panne'])
            
            # Application du groupement des facteurs
            factors_filtered['facteur_groupe'] = factors_filtered['facteur'].apply(
                lambda x: next(
                    (k for k, v in factor_mapping.items() if any(term in str(x).lower() for term in v)),
                    'AUTRE'
                )
            )
            
            # Encodage des groupes de facteurs
            self.label_encoders['facteur_groupe'].fit(factors_filtered['facteur_groupe'].unique())
            factors_filtered['facteur_groupe_encoded'] = self.label_encoders['facteur_groupe'].transform(
                factors_filtered['facteur_groupe']
            )
            
            # Préparation des features
            X_factors = factors_filtered[['type_panne_encoded']]
            y_factors = factors_filtered['facteur_groupe_encoded']
            
            # Entraînement avec gestion des erreurs
            if len(np.unique(y_factors)) < 2:
                print("ATTENTION: Pas assez de classes distinctes pour l'entraînement du modèle des facteurs")
                self.factors_trained = False
            else:
                X_train_factors, X_test_factors, y_train_factors, y_test_factors = train_test_split(
                    X_factors, y_factors, test_size=0.3, random_state=42
                )
                
                self.factors_model.fit(X_train_factors, y_train_factors)
                self.factors_trained = True
                
                # Évaluation avec gestion des avertissements
                y_pred_factors = self.factors_model.predict(X_test_factors)
                print("\nRésultats du modèle d'analyse des facteurs:")
                print(classification_report(y_test_factors, y_pred_factors, zero_division=0))
            
            return {
                'detection': {
                    'accuracy': accuracy_score(y_test_detect, y_pred_detect),
                    'f1': f1_score(y_test_detect, y_pred_detect, average='weighted')
                },
                'classification': {
                    'accuracy': accuracy_score(y_test_class, y_pred_class),
                    'f1': f1_score(y_test_class, y_pred_class, average='weighted')
                },
                'factors': {
                    'accuracy': accuracy_score(y_test_factors, y_pred_factors) if self.factors_trained else 0.0,
                    'f1': f1_score(y_test_factors, y_pred_factors, average='weighted', zero_division=0) if self.factors_trained else 0.0
                }
            }
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement : {str(e)}")
            raise

    def predict(self, input_data):
        """Effectue une prédiction complète pour un équipement"""
        if not self.detection_trained:
            return {"error": "Le modèle de détection n'a pas été entraîné"}
        
        try:
            # Prétraitement des données d'entrée
            processed_data = {}
            
            # Traitement des features catégorielles
            for feature in self.features:
                if feature in input_data:
                    value = str(input_data[feature])
                    if feature in self.label_encoders:
                        if value in self.feature_categories[feature]:
                            processed_data[feature] = self.label_encoders[feature].transform([value])[0]
                        else:
                            processed_data[feature] = self.label_encoders[feature].transform(['UNKNOWN'])[0]
                else:
                    processed_data[feature] = self.label_encoders[feature].transform(['UNKNOWN'])[0]
            
            # 1. Détection de panne
            X_detect = np.array([list(processed_data.values())])
            is_fault = self.detection_model.predict(X_detect)[0]
            fault_proba = self.detection_model.predict_proba(X_detect)[0][1]
            
            result = {
                "success": True,
                "prediction": {
                    "etat": "En panne" if is_fault == 1 else "Fonctionnel",
                    "confidence": f"{max(fault_proba, 1-fault_proba) * 100:.2f}%"
                }
            }
            
            # Si c'est une panne, classifier le type et analyser les facteurs
            if is_fault == 1 and self.classification_trained:
                # 2. Classification du type de panne
                description = input_data.get('Description', '')
                X_text = self.vectorizer.transform([description])
                
                fault_type_idx = self.classification_model.predict(X_text)[0]
                fault_type = self.label_encoders['type_panne'].inverse_transform([fault_type_idx])[0]
                
                result["prediction"]["type_panne"] = fault_type
                
                # 3. Analyse des facteurs influençant
                if self.factors_trained:
                    # Préparation des données pour l'analyse des facteurs
                    X_factors = np.array([[fault_type_idx]])
                    
                    factor_idx = self.factors_model.predict(X_factors)[0]
                    factor = self.label_encoders['facteur'].inverse_transform([factor_idx])[0]
                    
                    result["prediction"]["facteurs_influencants"] = [factor]
                    
                    # Ajouter d'autres facteurs connus pour ce type de panne
                    if fault_type in self.factors:
                        additional_factors = self.factors[fault_type]
                        if factor in additional_factors:
                            additional_factors.remove(factor)
                        result["prediction"]["facteurs_influencants"].extend(additional_factors[:2])
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": {"error_type": str(type(e).__name__)}
            }

    def save_model(self, file_path=None):
        """Sauvegarde le modèle entraîné"""
        if file_path is None:
            # Sauvegarder dans le même répertoire que ce script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, 'random_forest_model.pkl')
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Données du modèle à sauvegarder
        model_data = {
            'detection_model': self.detection_model,
            'classification_model': self.classification_model,
            'factors_model': self.factors_model,
            'label_encoders': dict(self.label_encoders),
            'vectorizer': self.vectorizer,
            'features': self.features,
            'text_features': self.text_features,
            'fault_keywords': self.fault_keywords,
            'feature_categories': self.feature_categories,
            'fault_types': self.fault_types,
            'factors': self.factors
        }
        
        # Sauvegarder avec pickle au lieu de joblib
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModèle sauvegardé dans: {file_path}")
        print(f"Taille du fichier: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
        
        return file_path

    @classmethod
    def load_model(cls, file_path):
        """Charge un modèle préalablement entraîné"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier modèle non trouvé: {file_path}")
        
        # Charger avec pickle au lieu de joblib
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Créer une nouvelle instance
        model = cls()
        
        # Restaurer les attributs
        model.detection_model = model_data['detection_model']
        model.classification_model = model_data['classification_model']
        model.factors_model = model_data['factors_model']
        model.label_encoders = defaultdict(LabelEncoder)
        
        # Restaurer les encodeurs d'étiquettes
        for key, encoder in model_data['label_encoders'].items():
            model.label_encoders[key] = encoder
        
        model.vectorizer = model_data['vectorizer']
        model.features = model_data['features']
        model.text_features = model_data['text_features']
        model.fault_keywords = model_data['fault_keywords']
        model.feature_categories = model_data['feature_categories']
        model.fault_types = model_data['fault_types']
        model.factors = model_data['factors']
        
        # Marquer les modèles comme entraînés
        model.detection_trained = True
        model.classification_trained = True
        model.factors_trained = True
        
        print(f"Modèle chargé depuis: {file_path}")
        
        return model

# Fonction principale pour exécuter l'entraînement
def main():
    """Fonction principale pour exécuter l'entraînement du modèle"""
    print("\n" + "="*50)
    print("=== Détection et Classification de Pannes avec Random Forest ===")
    print("="*50 + "\n")
    
    try:
        # Initialisation du détecteur
        detector = RandomForestFaultDetector()
        
        # Entraînement des modèles
        results = detector.train()
        
        # Sauvegarde du modèle
        model_path = detector.save_model()
        
        # Affichage des résultats
        print("\nRésumé des performances:")
        for model_name, metrics in results.items():
            print(f"- Modèle {model_name}:")
            for metric_name, value in metrics.items():
                print(f"  - {metric_name}: {value:.4f}")
        
        print(f"\nModèle sauvegardé dans: {model_path}")
        
    except Exception as e:
        print(f"Erreur lors de l'exécution: {str(e)}")

# Exécution du script si lancé directement
if __name__ == "__main__":
    main()


    import requests
import json

# Sample data to send to the API
data = {
    "LOCATION": "Zone1",
    "STATUS": "Urgent",
    "WOPRIORITY": "High",
    "ASSETNUM": "MACH001",
    "Description": "Machine en panne avec fuite hydraulique et surchauffe"
}

# API endpoint
api_url = "http://localhost:3000/api/predict"

try:
    # Send POST request to the API
    response = requests.post(api_url, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Print the prediction result
        print("API Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: API returned status code {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error connecting to the API: {str(e)}")
except json.JSONDecodeError as e:
    print(f"Error decoding API response: {str(e)}")
    print(f"Raw response: {response.text}")


# Séparation du code de test API dans un fichier différent
if __name__ == '__main__':
    # Initialisation et entraînement du modèle
    detector = RandomForestFaultDetector()
    results = detector.train()
    
    # Affichage des résultats
    print("\nRésultats finaux:")
    for model_name, metrics in results.items():
        print(f"\nModèle {model_name}:")
        for metric_name, value in metrics.items():
            print(f"- {metric_name}: {value:.4f}")