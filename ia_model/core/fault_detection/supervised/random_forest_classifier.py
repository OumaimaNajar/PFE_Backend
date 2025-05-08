# fault_type_classifier.py
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import logging

# Configuration du système de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fault_type_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FaultTypeClassifier')

class FaultTypeClassifier:
    """
    Classificateur de types de pannes basé sur Random Forest.
    Ce modèle est conçu pour être utilisé après qu'une panne a été détectée
    par le modèle de détection de pannes principal.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """Initialise le classificateur de types de pannes"""
        logger.info("Initialisation du classificateur de types de pannes")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        self.features = ['machine', 'gravite', 'duree_heures', 'site']
        self.target = 'type_panne'
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path=None):
        """Charge les données depuis le fichier CSV des pannes industrielles"""
        if file_path is None:
            # Utiliser un chemin absolu basé sur l'emplacement du script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))), 
                                    'data', 'pannes_industrielles_organisees.csv')
        
        logger.info(f"Chargement des données depuis: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Fichier non trouvé: {file_path}")
            raise FileNotFoundError(f"Le fichier n'a pas été trouvé : {file_path}")
        
        print(f"\n{'='*50}\nChargement du fichier CSV depuis : {file_path}")
        
        # Essayer différents encodages pour gérer les caractères spéciaux
        encodings = ['utf-8', 'latin-1', 'ISO-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=';', encoding=encoding)
                logger.info(f"Données chargées avec l'encodage {encoding}: {len(df)} enregistrements")
                break
            except Exception as e:
                logger.warning(f"Échec du chargement avec l'encodage {encoding}: {str(e)}")
        
        if df is None:
            logger.error("Impossible de charger le fichier avec les encodages disponibles")
            raise ValueError("Impossible de charger le fichier CSV des pannes")
        
        # Vérifier les colonnes requises
        required_columns = ['type_panne', 'machine', 'description', 'gravite']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Colonnes manquantes dans le CSV: {missing_columns}")
            raise ValueError(f"Colonnes requises manquantes dans le CSV: {missing_columns}")
        
        # Nettoyage des données
        df_cleaned = df.dropna(subset=[self.target]).copy()
        
        # Ajouter des colonnes si elles n'existent pas
        for feature in self.features:
            if feature not in df_cleaned.columns:
                logger.warning(f"Colonne {feature} manquante, ajout d'une colonne par défaut")
                if feature == 'duree_heures':
                    df_cleaned[feature] = 1.0  # Valeur par défaut pour la durée
                elif feature == 'gravite':
                    df_cleaned[feature] = 'Moyenne'  # Valeur par défaut pour la gravité
                else:
                    df_cleaned[feature] = 'UNKNOWN'  # Valeur par défaut pour les autres
        
        # Afficher des informations sur les données
        print(f"\nDistribution des types de pannes :")
        type_counts = df_cleaned[self.target].value_counts()
        print(type_counts)
        
        logger.info(f"Après nettoyage: {len(df_cleaned)} enregistrements")
        logger.info(f"Types de pannes uniques: {len(type_counts)}")
        
        return df_cleaned
    
    def preprocess_data(self, df):
        """Prétraite les données pour l'entraînement"""
        logger.info("Prétraitement des données")
        print("\nPrétraitement des données...")
        
        # Encodage des variables catégorielles
        for feature in self.features:
            if df[feature].dtype == 'object':
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                print(f"{feature} - Classes : {self.label_encoders[feature].classes_}")
                logger.info(f"Encodage de {feature} terminé avec {len(self.label_encoders[feature].classes_)} classes")
        
        # Encodage de la variable cible
        self.label_encoders[self.target] = LabelEncoder()
        y = self.label_encoders[self.target].fit_transform(df[self.target].astype(str))
        logger.info(f"Encodage de {self.target} terminé avec {len(self.label_encoders[self.target].classes_)} classes")
        
        # Sélection des features
        X = df[self.features].values
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        logger.info(f"Données divisées: {len(X_train)} échantillons d'entraînement, {len(X_test)} échantillons de test")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Entraîne le modèle avec optimisation des hyperparamètres"""
        logger.info("Début de l'entraînement du modèle")
        print("\nEntraînement du modèle de classification des types de pannes...")
        
        # Définition de la grille de paramètres pour GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Optimisation des hyperparamètres
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Récupération du meilleur modèle
        self.model = grid_search.best_estimator_
        
        print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
        print(f"Meilleur score F1: {grid_search.best_score_:.2%}")
        
        logger.info(f"Modèle entraîné avec succès. Meilleurs paramètres: {grid_search.best_params_}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Évalue le modèle et génère des visualisations"""
        logger.info("Évaluation du modèle")
        print("\nÉvaluation du modèle...")
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nPrécision: {accuracy:.2%}")
        print(f"Score F1 pondéré: {f1:.2%}")
        
        # Rapport de classification détaillé
        print("\nRapport de classification:")
        class_names = self.label_encoders[self.target].classes_
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            zero_division=0
        )
        print(report)
        
        # Matrice de confusion
        self._plot_confusion_matrix(y_test, y_pred, class_names)
        
        # Importance des features
        self._plot_feature_importance()
        
        logger.info(f"Évaluation terminée. Précision: {accuracy:.2%}, F1: {f1:.2%}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Génère et affiche la matrice de confusion"""
        logger.info("Génération de la matrice de confusion")
        
        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Si trop de classes, limiter l'affichage
        if len(class_names) > 10:
            # Trouver les classes les plus fréquentes
            class_counts = np.sum(cm, axis=1)
            top_indices = np.argsort(class_counts)[-10:]  # Top 10 classes
            
            # Filtrer la matrice de confusion et les noms de classes
            cm = cm[top_indices][:, top_indices]
            class_names = class_names[top_indices]
            
            print("Note: Affichage limité aux 10 types de pannes les plus fréquents")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Matrice de Confusion - Types de Pannes')
        plt.ylabel('Vérité terrain')
        plt.xlabel('Prédiction')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix_fault_types.png')
        logger.info("Matrice de confusion sauvegardée dans 'confusion_matrix_fault_types.png'")
        plt.show()
    
    def _plot_feature_importance(self):
        """Affiche l'importance des features"""
        logger.info("Génération du graphique d'importance des features")
        
        # Récupération de l'importance des features
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Importance des Features')
        plt.bar(range(len(self.features)), importances[indices], align='center')
        plt.xticks(range(len(self.features)), [self.features[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance_fault_types.png')
        logger.info("Graphique d'importance des features sauvegardé dans 'feature_importance_fault_types.png'")
        plt.show()
    
    def save_model(self, file_name='fault_type_classifier_model.pkl'):
        """Sauvegarde le modèle et les préprocesseurs"""
        # Obtenir le chemin absolu du répertoire contenant ce script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_name)
        logger.info(f"Sauvegarde du modèle dans: {file_path}")
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Sauvegarder le modèle avec tous les préprocesseurs
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'target': self.target,
            'scaler': self.scaler
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModèle sauvegardé dans {file_path}")
        print(f"Taille du fichier : {os.path.getsize(file_path)/1024:.2f} KB")
        logger.info(f"Modèle sauvegardé avec succès. Taille: {os.path.getsize(file_path)/1024:.2f} KB")
    
    def predict_fault_type(self, input_data):
        """
        Prédit le type de panne à partir des données d'entrée.
        
        Args:
            input_data (dict): Dictionnaire contenant les données d'entrée
                               (machine, gravite, duree_heures, site)
        
        Returns:
            dict: Résultat de la prédiction avec le type de panne et la confiance
        """
        try:
            logger.info(f"Prédiction pour les données: {input_data}")
            
            # Vérifier que toutes les features nécessaires sont présentes
            for feature in self.features:
                if feature not in input_data:
                    logger.warning(f"Feature manquante: {feature}, utilisation d'une valeur par défaut")
                    if feature == 'duree_heures':
                        input_data[feature] = 1.0
                    elif feature == 'gravite':
                        input_data[feature] = 'Moyenne'
                    else:
                        input_data[feature] = 'UNKNOWN'
            
            # Prétraitement des données d'entrée
            processed_data = {}
            
            for feature in self.features:
                if feature in self.label_encoders:
                    le = self.label_encoders[feature]
                    value = str(input_data.get(feature, 'UNKNOWN'))
                    
                    # Gérer les valeurs inconnues
                    if value not in le.classes_:
                        logger.warning(f"Valeur inconnue pour {feature}: {value}, utilisation de 'UNKNOWN'")
                        value = 'UNKNOWN'
                        if 'UNKNOWN' not in le.classes_:
                            # Si 'UNKNOWN' n'est pas dans les classes, utiliser la première classe
                            value = le.classes_[0]
                    
                    processed_data[feature] = le.transform([value])[0]
                else:
                    # Pour les features numériques
                    processed_data[feature] = float(input_data.get(feature, 0))
            
            # Créer un tableau numpy pour la prédiction
            X_input = np.array([list(processed_data.values())])
            
            # Normalisation
            X_input_scaled = self.scaler.transform(X_input)
            
            # Prédiction
            prediction_idx = self.model.predict(X_input_scaled)[0]
            probabilities = self.model.predict_proba(X_input_scaled)[0]
            
            # Récupérer le type de panne et la confiance
            fault_type = self.label_encoders[self.target].inverse_transform([prediction_idx])[0]
            confidence = probabilities[prediction_idx]
            
            # Récupérer les probabilités pour tous les types de pannes
            all_types_proba = {}
            for i, prob in enumerate(probabilities):
                type_name = self.label_encoders[self.target].inverse_transform([i])[0]
                all_types_proba[type_name] = f"{prob:.2%}"
            
            # Trier les probabilités par ordre décroissant
            all_types_proba = dict(sorted(all_types_proba.items(), key=lambda x: float(x[1].strip('%'))/100, reverse=True))
            
            # Résultat
            result = {
                "success": True,
                "prediction": {
                    "type_panne": fault_type,
                    "confidence": f"{confidence:.2%}",
                    "all_probabilities": all_types_proba
                }
            }
            
            logger.info(f"Type de panne prédit: {fault_type} avec confiance {confidence:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "details": {"error_type": str(type(e).__name__)}
            }

def main():
    """Fonction principale pour exécuter le pipeline complet"""
    print("\n" + "="*50)
    print("=== Classification des Types de Pannes avec Random Forest ===")
    print("="*50 + "\n")
    
    try:
        # Initialisation du classificateur
        classifier = FaultTypeClassifier()
        
        # 1. Chargement des données
        print("[1/4] Chargement et préparation des données...")
        df = classifier.load_data()
        
        # 2. Prétraitement
        print("[2/4] Prétraitement des données...")
        X_train, X_test, y_train, y_test = classifier.preprocess_data(df)
        
        # 3. Entraînement
        print("[3/4] Entraînement du modèle...")
        classifier.train_model(X_train, y_train)
        
        # 4. Évaluation
        print("[4/4] Évaluation du modèle...")
        metrics = classifier.evaluate_model(X_test, y_test)
        
        # Sauvegarde du modèle
        classifier.save_model()
        
        # Exemple de prédiction
        print("\n[5/5] Test de prédiction...")
        test_cases = [
            {
                'machine': 'POMPE_HYDRAULIQUE',
                'gravite': 'Élevée',
                'duree_heures': 4.5,
                'site': 'USINE_A'
            },
            {
                'machine': 'MOTEUR_ELECTRIQUE',
                'gravite': 'Critique',
                'duree_heures': 2.0,
                'site': 'USINE_B'
            },
            {
                'machine': 'COMPRESSEUR',
                'gravite': 'Moyenne',
                'duree_heures': 1.5,
                'site': 'USINE_C'
            }
        ]
        
        for case in test_cases:
            print(f"\nPrédiction pour : {case}")
            result = classifier.predict_fault_type(case)
            print(f"Type de panne prédit: {result['prediction']['type_panne']}")
            print(f"Confiance: {result['prediction']['confidence']}")
            print("Top 3 probabilités:")
            top_3 = list(result['prediction']['all_probabilities'].items())[:3]
            for type_name, prob in top_3:
                print(f"  - {type_name}: {prob}")
        
        print("\n=== Processus terminé avec succès ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}\n")
        logger.error(f"Erreur dans le processus principal: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    main()