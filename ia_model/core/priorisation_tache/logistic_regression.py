# maximo_criticality_classifier_final.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                            ConfusionMatrixDisplay, roc_curve, auc, 
                            precision_recall_curve, average_precision_score)
from sklearn.pipeline import make_pipeline
import pyodbc
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import warnings

class FinalMaximoClassifier:
    """
    Classifieur final avec gestion complète des erreurs et visualisations
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = ['ASSETNUM', 'WOPRIORITY', 'LOCATION', 'STATUS']
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"logistic_regression_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configuration des styles et avertissements
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 12
        warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
        warnings.filterwarnings('ignore', category=FutureWarning)

    def connect_to_db(self):
        """Connexion à la base de données Maximo"""
        try:
            conn = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=maxgps.smartech-tn.com;'
                'DATABASE=demo7613;'
                'UID=smguest;'
                'PWD=smguest;'
                'Encrypt=no;'
                'Connection Timeout=30;'
            )
            return conn
        except Exception as e:
            raise ConnectionError(f"Échec de la connexion: {str(e)}")

    def load_data(self):
        """Charge les données depuis la table workorder"""
        print("Chargement des données depuis IBM Maximo...")
        try:
            with self.connect_to_db() as conn:
                query = """
                SELECT 
                    WORKORDERID,
                    ASSETNUM,
                    WOPRIORITY,
                    LOCATION,
                    STATUS,
                    FAILURECODE
                FROM 
                    workorder
                WHERE 
                    STATUS IN ('COMP', 'CLOSE', 'CAN')
                    AND WOPRIORITY IS NOT NULL
                """
                df = pd.read_sql(query, conn)
                
                if df.empty:
                    raise ValueError("Aucune donnée valide trouvée")
                
                # Nettoyage des données
                df['WOPRIORITY'] = pd.to_numeric(df['WOPRIORITY'], errors='coerce')
                df = df.dropna(subset=['WOPRIORITY'])
                
                # Définition de la criticité
                df['is_critical'] = (
                    (df['WOPRIORITY'] <= 2) | 
                    (df['FAILURECODE'].notna())
                ).astype(int)
                
                print("\nDistribution des classes:")
                print(df['is_critical'].value_counts(normalize=True))
                
                return df
                
        except Exception as e:
            raise ValueError(f"Erreur de chargement: {str(e)}")

    def preprocess_data(self, df):
        """Prétraitement robuste des données"""
        print("\nPrétraitement des données...")
        
        # Copie des données pour éviter les modifications inattendues
        df_processed = df.copy()
        
        # Encodage des variables catégorielles avec gestion des valeurs manquantes
        categorical_cols = ['ASSETNUM', 'LOCATION', 'STATUS']
        for col in categorical_cols:
            # Ajout d'une catégorie 'UNKNOWN' pour les nouvelles valeurs
            unique_values = df_processed[col].astype(str).unique()
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(np.append(unique_values, 'UNKNOWN'))
            
            # Transformation des données
            df_processed[col] = df_processed[col].fillna('UNKNOWN').astype(str)
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in self.label_encoders[col].classes_ else 'UNKNOWN')
            df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Sélection des features
        X = df_processed[self.feature_names]
        y = df_processed['is_critical']
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, df_processed

    def train_model(self, X, y):
        """Entraîne et évalue le modèle avec visualisations"""
        print("\nEntraînement du modèle...")
        
        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Modèle de régression logistique
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        )
        
        # Entraînement
        self.model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métriques
        print("\nRapport de classification détaillé:")
        print(classification_report(y_test, y_pred, digits=4))
        
        # Visualisations
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_roc_curve(y_test, y_proba)
        self._plot_precision_recall_curve(y_test, y_proba)
        self._plot_feature_importance()
        self._plot_decision_boundary(X_train, y_train)
        self._plot_probability_distribution(y_proba, y_test)
        
        return self.model

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Matrice de confusion améliorée"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Critique', 'Critique'],
                   yticklabels=['Non-Critique', 'Critique'])
        plt.title('Matrice de Confusion', pad=20)
        plt.xlabel('Prédictions')
        plt.ylabel('Vérités Terrain')
        plt.savefig(f"{self.results_dir}/confusion_matrix.png", bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_roc_curve(self, y_true, y_proba):
        """Courbe ROC avec AUC"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Courbe ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC', pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.results_dir}/roc_curve.png", bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_precision_recall_curve(self, y_true, y_proba):
        """Courbe Precision-Recall"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Courbe Precision-Recall', pad=20)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.results_dir}/precision_recall_curve.png", bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_feature_importance(self):
        """Importance des caractéristiques"""
        if hasattr(self.model.named_steps['logisticregression'], 'coef_'):
            importance = np.abs(self.model.named_steps['logisticregression'].coef_[0])
            feature_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure()
            # Correction du warning FutureWarning
            sns.barplot(x='Importance', y='Feature', data=feature_imp, 
                        hue='Feature', palette='viridis', legend=False)
            plt.title('Importance des Caractéristiques', pad=20)
            
            # Ajout des valeurs sur les barres
            for index, value in enumerate(feature_imp['Importance']):
                plt.text(value, index, f'{value:.3f}', va='center')
            
            plt.savefig(f"{self.results_dir}/feature_importance.png", bbox_inches='tight', dpi=300)
            plt.close()

    def _plot_decision_boundary(self, X, y):
        """Frontière de décision avec PCA"""
        try:
            # Réduction de dimension
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Modèle pour la visualisation
            viz_model = LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            viz_model.fit(X_pca, y)
            
            # Figure
            plt.figure(figsize=(10, 8))
            DecisionBoundaryDisplay.from_estimator(
                viz_model,
                X_pca,
                cmap=plt.cm.RdBu,
                response_method="predict_proba",
                alpha=0.8,
                eps=0.5,
            )
            
            # Points de données
            scatter = plt.scatter(
                X_pca[:, 0], X_pca[:, 1], 
                c=y, edgecolors='k', 
                cmap=plt.cm.RdBu_r, 
                s=40
            )
            
            plt.title('Frontière de Décision (Projection PCA)', pad=20)
            plt.xlabel('Composante Principale 1')
            plt.ylabel('Composante Principale 2')
            plt.legend(*scatter.legend_elements(), title="Classes")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.results_dir}/decision_boundary.png", bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"\nAvertissement: Impossible de tracer la frontière de décision - {str(e)}")

    def _plot_probability_distribution(self, y_proba, y_true):
        """Distribution des probabilités prédites"""
        plt.figure()
        for label in [0, 1]:
            sns.kdeplot(y_proba[y_true == label], label=f'Classe {label}')
        plt.title('Distribution des Probabilités Prédites', pad=20)
        plt.xlabel('Probabilité Prédite')
        plt.ylabel('Densité')
        plt.legend(title='Vraie Classe')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.results_dir}/probability_distribution.png", bbox_inches='tight', dpi=300)
        plt.close()

    def predict_criticality(self, new_data):
        """Prédiction robuste avec gestion des nouvelles catégories"""
        print("\nPrédiction de criticité...")
        
        try:
            # Vérification des colonnes requises
            missing_cols = [col for col in self.feature_names if col not in new_data.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes: {missing_cols}")
            
            # Prétraitement
            new_data_processed = new_data.copy()
            for col in ['ASSETNUM', 'LOCATION', 'STATUS']:
                if col in new_data_processed.columns:
                    # Transformation sécurisée avec gestion des nouvelles catégories
                    new_data_processed[col] = new_data_processed[col].fillna('UNKNOWN').astype(str)
                    new_data_processed[col] = new_data_processed[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else 'UNKNOWN')
                    new_data_processed[col] = self.label_encoders[col].transform(new_data_processed[col])
                else:
                    new_data_processed[col] = 0  # Valeur par défaut
            
            # Remplissage des autres valeurs manquantes
            for feat in self.feature_names:
                if feat in new_data_processed.columns:
                    new_data_processed[feat] = new_data_processed[feat].fillna(0)
            
            # Prédiction
            X_new = new_data_processed[self.feature_names]
            predictions = self.model.predict(X_new)
            probabilities = self.model.predict_proba(X_new)[:, 1]
            
            # Formatage des résultats
            results = pd.DataFrame({
                'is_critical_pred': predictions,
                'criticality_prob': probabilities
            })
            
            return pd.concat([new_data, results], axis=1)
            
        except Exception as e:
            raise ValueError(f"Erreur de prédiction: {str(e)}")

    def save_model(self):
        """Sauvegarde du modèle complet"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'timestamp': self.timestamp
        }
        
        filename = f"{self.results_dir}/maximo_criticality_model.joblib"
        joblib.dump(model_data, filename)
        print(f"\nModèle sauvegardé sous: {filename}")

    def run_pipeline(self):
        """Exécution complète du pipeline"""
        try:
            # 1. Chargement des données
            df = self.load_data()
            
            # 2. Prétraitement
            X, y, df_processed = self.preprocess_data(df)
            
            # 3. Entraînement et évaluation
            self.train_model(X, y)
            
            # 4. Exemple de prédiction
            if not df_processed.empty:
                sample_data = df_processed.sample(min(5, len(df_processed)), random_state=42)
                sample_data = sample_data.drop(columns=['is_critical'], errors='ignore')
                predictions = self.predict_criticality(sample_data)
                
                print("\nExemple de prédictions:")
                print(predictions[['WORKORDERID', 'WOPRIORITY', 'is_critical_pred', 'criticality_prob']])
            
            # 5. Sauvegarde
            self.save_model()
            
            print("\nPipeline exécuté avec succès!")
            print(f"Visualisations sauvegardées dans: {os.path.abspath(self.results_dir)}")
            
        except Exception as e:
            print(f"\nERREUR: {str(e)}")

if __name__ == "__main__":
    print("=== CLASSIFICATION DES PANNES CRITIQUES ===")
    print("=== IBM Maximo - Régression Logistique Améliorée ===\n")
    
    classifier = FinalMaximoClassifier()
    classifier.run_pipeline()