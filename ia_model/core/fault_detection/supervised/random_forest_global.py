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

    #Initialise trois modèles Random Forest distincts
    #detection_model : Pour détecter si un équipement est en panne
    #classification_model : Pour classer les pannes en fonction de leur type
    #factors_model : Pour identifier les facteurs qui influencent les pannes
    
    def __init__(self, n_estimators=100, random_state=42):
        """Initialise le détecteur de pannes avec Random Forest"""
        # Modèle pour la détection de panne (fonctionnel vs en panne)
        self.detection_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced', # pour gérer le déséquilibre de classes
            min_samples_leaf=2  # pour éviter le surapprentissage
        )
        
        # Modèle pour la classification du type de panne
        self.classification_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced',
            min_samples_leaf=1  # pour permettre la classification de pannes rares
        )
        
        # Modèle pour l'analyse des facteurs influençant les pannes
        self.factors_model = RandomForestClassifier( # Pour déterminer les facteurs contribuant aux pannes
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        
        # Préprocesseurs : utilisés pour les features catégorielles et textuelle

         # LabelEncoder pour les variables catégorielles
        self.label_encoders = defaultdict(LabelEncoder)
        # TfidfVectorizer pour les descriptions
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Paramètres
        # Les features utilisées pour la détection de pannes
        self.features = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']
        self.text_features = ['Description'] #Les features utilisées pour la détection de pannes
        self.fault_keywords = {} # Dictionnaire des mots-clés par type de panne
        self.feature_categories = {} 
        self.fault_types = {} # Dictionnaire des types de pannes
        self.factors = {} 
        
        # État d'entraînement des modèles
        self.detection_trained = False 
        self.classification_trained = False #
        self.factors_trained = False


    # Charge les données depuis trois fichiers CSV
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
        
        # Chargement des données
        workorders_df = pd.read_csv(workorders_path, sep=';')
        fault_types_df = pd.read_csv(fault_types_path, sep=';')
        factors_df = pd.read_csv(factors_path, sep=';')
        
        # Afficher un aperçu des données
        # Afficher les premières lignes des DataFrames
        # Afficher les colonnes des DataFrames
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

    # Prétraitement des données des wo pour la détection de pannes
    def preprocess_workorders(self, df):
        """Prétraite les données des workorder pour la détection de pannes"""
        print("\nPrétraitement des ordres de travail...")
        
        # Nettoyage des données
        df_cleaned = df.dropna(subset=self.features + self.text_features).copy()
        
        # Détection des pannes 
        # Si la colonne 'PANNE' existe déjà, l'utiliser, sinon la créer
        if 'PANNE' not in df_cleaned.columns:
            # Mots-clés pour détecter les pannes dans la description
            keywords = [
                'panne', 'défaillance', 'problème', 'dysfonctionnement', 'arrêt',
                'brisé', 'cassé', 'fuite', 'surchauffe', 'erreur', 'défaut'
            ]
            # Création de la colonne PANNE
            # Si la colonne 'Description' existe, utilise-la, sinon utilise 'Description'
            df_cleaned['PANNE'] = df_cleaned['Description'].str.contains(
                '|'.join(keywords), case=False, na=False
            ).astype(int)
        
        # Analyse des catégories pour chaque feature
        for feature in self.features:
            # Si la colonne est catégorielle
            if df_cleaned[feature].dtype == 'object':
                #
                self.feature_categories[feature] = df_cleaned[feature].unique().tolist()
                # Ajouter une catégorie 'UNKNOWN' pour les valeurs inconnues
                self.feature_categories[feature].append('UNKNOWN')  # Pour les valeurs inconnues
                
                # Encodage des valeurs
                self.label_encoders[feature].fit(self.feature_categories[feature])
                # Transformation des valeurs
                df_cleaned[feature] = df_cleaned[feature].apply(
                    lambda x: x if x in self.feature_categories[feature] else "UNKNOWN"
                )
                df_cleaned[feature] = self.label_encoders[feature].transform(df_cleaned[feature])
        
        print(f"Distribution des pannes: {df_cleaned['PANNE'].value_counts().to_dict()}")
        # Afficher les classes après l'encodage
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
            # Afficher les colonnes manquantes pour le débogage
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

    # Fonction pour normaliser les noms des types de pannes
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
        # Supprimer les espaces et convertir en minuscules
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

    # Prétraitement des données des facteurs influençant les pannes
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

    # Entraîne les trois modèles (détection, classification, facteurs)
    # Fonction d'entraînement
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

            # Séparation des features et de la target
            X_detect = workorders_processed[self.features]
            y_detect = workorders_processed['PANNE']
            
            # Vérification de la distribution des classes
            class_dist = pd.Series(y_detect).value_counts()
            print("\nDistribution des classes avant équilibrage:")
            print(class_dist)
            
            # Équilibrage des classes
            # Vérification de la distribution des classes
            # Si le nombre de classes est inférieur à 2 ou si la classe minoritaire est inférieure à 10%
            # Appliquer un rééchantillonnage
            if len(class_dist) < 2 or class_dist.min() < 10:
                print("\nATTENTION: Déséquilibre important des classes ou données insuffisantes!")
                print("Considérez l'utilisation de techniques de rééchantillonnage.")
            
            # Ajustement des poids des classes
            class_weights = dict(zip(
                class_dist.index,
                [len(y_detect)/(len(class_dist)*x) for x in class_dist]
            ))
            
            # Création du modèle
            print("\nCréation du modèle...")
            print(f"Poids des classes: {class_weights}")

            # Création du modèle avec les poids ajustés
            self.detection_model = RandomForestClassifier(
                n_estimators=100,
                class_weight=class_weights,  # Utilisation des poids calculés
                random_state=42
            )
            
            # Entraînement du modèle
            X_train_detect, X_test_detect, y_train_detect, y_test_detect = train_test_split(
                X_detect, y_detect, test_size=0.3, random_state=42, stratify=y_detect
            )
            
            # Entraînement du modèle
            self.detection_model.fit(X_train_detect, y_train_detect)
            self.detection_trained = True
            
            # Évaluation du modèle de détection
            y_pred_detect = self.detection_model.predict(X_test_detect)
            print("\nRésultats du modèle de détection:")
            print(classification_report(y_test_detect, y_pred_detect))
            
            # Visualisation de l'importance des features pour le modèle de détection
            plt.figure(figsize=(10, 6))
            importances = self.detection_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title('Importance des Features - Modèle de Détection')
            plt.bar(range(X_detect.shape[1]), importances[indices])
            plt.xticks(range(X_detect.shape[1]), [X_detect.columns[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('detection_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()  # Ajout de plt.show()
            plt.close()
            
            # Matrice de confusion pour le modèle de détection
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test_detect, y_pred_detect)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Panne', 'Panne'],
                       yticklabels=['Non-Panne', 'Panne'])
            plt.title('Matrice de Confusion - Modèle de Détection')
            plt.ylabel('Vraies étiquettes')
            plt.xlabel('Prédictions')
            plt.tight_layout()
            plt.savefig('detection_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()  # Ajout de plt.show()
            plt.close()
            
            # 2. Entraînement du modèle de classification des types de pannes
            print("\n" + "="*50)
            print("Entraînement du modèle de classification des types de pannes...")
            
            # Vectorisation des descriptions
            X_text = self.vectorizer.transform(fault_types_processed['description'])
            y_class = self.label_encoders['type_panne'].transform(fault_types_processed['type_panne'])
            
            # Check class distribution before splitting
            unique_classes, class_counts = np.unique(y_class, return_counts=True)
            min_samples = np.min(class_counts)
            
            # Check if we have enough samples per class
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
            
            # Création du modèle
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
            }
            
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
                # Entraînement
                X_train_factors, X_test_factors, y_train_factors, y_test_factors = train_test_split(
                    X_factors, y_factors, test_size=0.3, random_state=42
                )

                # Entraînement du modèle d'analyse des facteurs
                self.factors_model.fit(X_train_factors, y_train_factors)
                self.factors_trained = True
                
                # Évaluation avec gestion des avertissements
                y_pred_factors = self.factors_model.predict(X_test_factors)
                print("\nRésultats du modèle d'analyse des facteurs:")
                print(classification_report(y_test_factors, y_pred_factors, zero_division=0))
            
            # Retour des scores
            return {
                # Ajout du score de F1 pour le modèle de détection
                'detection': {
                    'accuracy': accuracy_score(y_test_detect, y_pred_detect),
                    'f1': f1_score(y_test_detect, y_pred_detect, average='weighted')
                },
                # Ajout du score de F1 pour le modèle de classification
                'classification': {
                    'accuracy': accuracy_score(y_test_class, y_pred_class),
                    'f1': f1_score(y_test_class, y_pred_class, average='weighted')
                },
                # Ajout du score de F1 pour le modèle d'analyse des facteurs
                'factors': {
                    'accuracy': accuracy_score(y_test_factors, y_pred_factors) if self.factors_trained else 0.0,
                    'f1': f1_score(y_test_factors, y_pred_factors, average='weighted', zero_division=0) if self.factors_trained else 0.0
                }
            }
        
        # Gestion des erreurs
        except Exception as e:
            print(f"Erreur lors de l'entraînement : {str(e)}")
            raise

    # Fonction de prédiction :Effectue une prédiction complète pour un équipement
    # Détecte si l'équipement est en panne
    # Si en panne, prédit le type de panne
    # Si le type de panne est connu, prédit les facteurs influençants
    def predict(self, input_data):
        """Effectue une prédiction complète pour un équipement"""
        # Vérification de l'entraînement du modèle de détection
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
            
            # Construction du résultat
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
                # Ajout du type de panne au résultat
                result["prediction"]["type_panne"] = fault_type
                
                # 3. Analyse des facteurs influençant
                if self.factors_trained:
                    # Préparation des données pour l'analyse des facteurs
                    X_factors = np.array([[fault_type_idx]])
                    
                    factor_idx = self.factors_model.predict(X_factors)[0]
                    factor = self.label_encoders['facteur'].inverse_transform([factor_idx])[0]
                    # Ajout des facteurs au résultat
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

    # Fonction de sauvegarde du modèle
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

    # Fonction de chargement du modèle
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
        
        # Restaurer les autres attributs
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