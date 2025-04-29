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


# Add to imports at the top
import seaborn as sns
from sklearn.metrics import roc_auc_score

class RandomForestFaultDetector:
    def __init__(self, n_estimators=100, random_state=42):
        """Initialise le détecteur de pannes avec Random Forest"""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'
        )
        self.keywords = [
            'leak', 'stopped', 'overheating', 'failure', 'broken', 
            'malfunction', 'error', 'defect', 'fault', 'out of order'
        ]
        self.features = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']
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

    def load_and_prepare_data(self, file_path='../../../../data/clean_workorders.csv'):
        """Charge et prépare les données"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier n'a pas été trouvé : {file_path}")
        
        print(f"\n{'='*50}\nChargement du fichier CSV depuis : {file_path}")
        df = pd.read_csv(file_path, sep=';')
        df_cleaned = df.dropna().copy()
        
        # Détection des pannes
        df_cleaned.loc[:, 'PANNE'] = df_cleaned['Description'].str.contains(
            '|'.join(self.keywords), case=False, na=False
        ).astype(int)
        
        # Analyse des catégories pour chaque feature
        for feature in self.features:
            if df_cleaned[feature].dtype == 'object':
                self.feature_categories[feature] = df_cleaned[feature].unique().tolist()
                self.feature_categories[feature].append('UNKNOWN')  # Ajout de la catégorie pour valeurs inconnues
        
        print(f"\nDistribution des pannes :\n{df_cleaned['PANNE'].value_counts()}")
        print(f"\nTaux de pannes : {df_cleaned['PANNE'].mean():.2%}")
        
        return df_cleaned

    def preprocess_data(self, df):
        """Prétraitement des données avec gestion des valeurs inconnues"""
        print("\nPrétraitement des données...")
        
        for feature in self.features:
            if feature in self.feature_categories:  # Si c'est une variable catégorielle
                # Ajouter "UNKNOWN" comme classe valide si elle n'existe pas déjà
                if "UNKNOWN" not in self.feature_categories[feature]:
                    self.feature_categories[feature].append("UNKNOWN")
                
                # Remplacer les valeurs inconnues par "UNKNOWN"
                df[feature] = df[feature].apply(
                    lambda x: x if x in self.feature_categories[feature] else "UNKNOWN"
                )
                
                # Encodage
                self.label_encoders[feature].fit(self.feature_categories[feature])
                df[feature] = self.label_encoders[feature].transform(df[feature])
                print(f"{feature} - Classes : {self.label_encoders[feature].classes_}")
        
        X = df[self.features]
        y = df['PANNE']
        
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    def train_model(self, X_train, y_train):
        """Entraîne le modèle Random Forest avec validation croisée intégrée"""
        print("\nEntraînement du modèle...")
        self.model.fit(X_train, y_train)
        
        # Importance des features
        feature_importances = pd.DataFrame(
            self.model.feature_importances_,
            index=self.features,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        print("\nImportance des features :")
        print(feature_importances)
        
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Évalue le modèle et génère les visualisations"""
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
        
        # Affichage des résultats
        print("\n=== Performance du modèle ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.2%}")
        
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred))
        
        # Matrice de confusion - Correction de y_true en y_test
        self.plot_confusion_matrix(y_test, y_pred)
        
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """Génère et affiche la matrice de confusion améliorée"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Affichage avec seaborn pour plus de clarté
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fonctionnel', 'Panne'],
                    yticklabels=['Fonctionnel', 'Panne'])
        
        plt.title('Matrice de Confusion - Random Forest')
        plt.ylabel('Vérité terrain')
        plt.xlabel('Prédiction')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

    def save_model(self, file_name='random_forest_model.pkl'):
        """Sauvegarde le modèle et les préprocesseurs dans le même répertoire que le script"""
        # Obtenir le chemin absolu du répertoire contenant ce script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_name)
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Sauvegarder le modèle
        model_data = {
            'model': self.model,
            'label_encoders': dict(self.label_encoders),
            'features': self.features,
            'keywords': self.keywords,
            'feature_categories': self.feature_categories
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModèle sauvegardé dans {file_path}")
        print(f"Taille du fichier : {os.path.getsize(file_path)/1024:.2f} KB")

    def predict_fault(self, input_data):
        try:
            processed_data = {}
            risk_factors = []
            risk_score = 0.2  # Base risk score
            
            for feature in self.features:
                if feature in self.label_encoders:
                    le = self.label_encoders[feature]
                    value = str(input_data.get(feature, 'UNKNOWN')).upper()
                    
                    # Risk assessment for unknown values
                    if value not in le.classes_:
                        risk_factors.append(f"Unknown {feature}: {value}")
                        risk_score += 0.3
                        value = 'UNKNOWN'
                    
                    processed_data[feature] = le.transform([value])[0]
                    
                    # Feature-specific risk assessment
                    if feature == 'STATUS':
                        status_risk = self.status_risk.get(value, 0.4)
                        risk_score += status_risk
                        if status_risk > 0.3:
                            risk_factors.append(f"Status risk: {value}")
                    elif feature == 'WOPRIORITY':
                        try:
                            priority = int(value)
                            if priority <= 2:
                                risk_score += 0.3
                                risk_factors.append("High priority work order")
                        except ValueError:
                            risk_score += 0.2
            
            X_input = pd.DataFrame([processed_data])
            base_proba = self.model.predict_proba(X_input)[0]
            
            final_fault_prob = min(0.95, base_proba[1] + risk_score)
            final_ok_prob = 1 - final_fault_prob
            
            return {
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
                    }
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": {"error_type": str(type(e).__name__)}
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
        
        # Exemple de prédiction
        print("\n[5/5] Test de prédiction...")
        test_cases = [
            {'LOCATION': 'BR300', 'STATUS': 'CLOSED', 'WOPRIORITY': '2', 'ASSETNUM': 'EQP123'},
            {'LOCATION': 'NEW_LOC', 'STATUS': 'OPEN', 'WOPRIORITY': '1', 'ASSETNUM': 'NEW_EQP'},
            {'LOCATION': 'LOC02', 'STATUS': 'IN PROGRESS', 'WOPRIORITY': '3', 'ASSETNUM': 'EQP456'}
        ]
        
        for case in test_cases:
            print(f"\nPrédiction pour : {case}")
            result = detector.predict_fault(case)
            print(json.dumps(result, indent=2))
        
        print("\n=== Processus terminé avec succès ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}\n")
        return 1
    
    return 0


if __name__ == "__main__":
    main()


import pandas as pd

def afficher_facteurs_influents(type_panne, chemin_csv="c:/Users/omaim/backend_ia/data/facteurs_influencent_pannes.csv"):
    """
    Affiche les facteurs influençant le type de panne détecté.
    :param type_panne: (str) Le type de panne détecté (ex: 'SURCHAUFFE')
    :param chemin_csv: (str) Chemin vers le fichier CSV des facteurs
    """
    try:
        df = pd.read_csv(chemin_csv, sep=';')
        facteurs = df[df['type_panne'].str.upper() == type_panne.upper()]
        if facteurs.empty:
            print(f"Aucun facteur trouvé pour le type de panne : {type_panne}")
        else:
            print(f"Facteurs influençant la panne '{type_panne}':")
            for _, row in facteurs.iterrows():
                print(f"- {row['facteur']} : {row['valeur']} ({row['pourcentage']}%) -> {row['description']}")
    except Exception as e:
        print(f"Erreur lors de la lecture des facteurs : {e}")

# Exemple d'utilisation après la classification :
# type_panne_detecte = "SURCHAUFFE"
# afficher_facteurs_influents(type_panne_detecte)