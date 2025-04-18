# svm.py
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.metrics import (classification_report, confusion_matrix,  # type: ignore
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.decomposition import PCA # type: ignore
import joblib # type: ignore
import pickle  # Ajout de pickle pour la sérialisation
from pathlib import Path
from datetime import datetime

class SVMFaultDetector:
    def __init__(self):
        """Initialise le détecteur de pannes avec SVM"""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
        ])
        
        self.keywords = [
            'leak', 'failure', 'broken', 'error', 'defect', 
            'fault', 'repair', 'replace', 'damage', 'problem'
        ]
        
        self.features = ['WOPRIORITY', 'LOCATION', 'STATUS', 'ASSETNUM']
        self.label_encoders = {}
        self.result_dir = Path("svm_results")
        self.result_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self, file_path='../../../data/clean_workorders.csv'):
        """Charge les données depuis le fichier CSV"""
        print("\n[1/4] Chargement des données...")
        try:
            df = pd.read_csv(file_path, sep=';')
            required_cols = self.features + ['Description', 'FAILURECODE']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes: {missing_cols}")
            
            df_cleaned = df.dropna(subset=self.features + ['Description']).copy()
            mask = (
                df_cleaned['Description'].str.contains('|'.join(self.keywords), case=False, na=False) |
                df_cleaned['FAILURECODE'].notna()
            )
            df_cleaned.loc[:, 'PANNE'] = mask.astype(int)
            
            print(f"\nDistribution des classes:\n{df_cleaned['PANNE'].value_counts(normalize=True)}")
            return df_cleaned
            
        except Exception as e:
            raise ValueError(f"Erreur de chargement: {str(e)}")

    def preprocess_data(self, df):
        """Prétraitement des données"""
        print("\n[2/4] Prétraitement des données...")
        
        for feature in self.features:
            if df[feature].dtype == 'object':
                self.label_encoders[feature] = LabelEncoder()
                df.loc[:, feature] = self.label_encoders[feature].fit_transform(
                    df[feature].astype(str))
        
        if 'WOPRIORITY' in df.columns:
            df.loc[:, 'WOPRIORITY'] = pd.to_numeric(df['WOPRIORITY'], errors='coerce')
            median_val = df['WOPRIORITY'].median()
            df.loc[:, 'WOPRIORITY'] = df['WOPRIORITY'].fillna(median_val)
        
        return train_test_split(
            df[self.features], 
            df['PANNE'], 
            test_size=0.3, 
            random_state=42, 
            stratify=df['PANNE']
        )

    def train_model(self, X_train, y_train):
        """Entraîne le modèle SVM"""
        print("\n[3/4] Entraînement du modèle SVM...")
        
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.1, 1],
            'classifier__kernel': ['rbf', 'linear']
        }
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
        print(f"Meilleur score F1: {grid_search.best_score_:.2%}")

    def evaluate_and_plot(self, X_test, y_test):
        """Évalue et affiche les résultats avec matrice de confusion et frontière de décision"""
        print("\n[4/4] Évaluation du modèle...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
        
        # Création et affichage des figures
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_decision_boundary(X_test, y_test)
        
        # Affichage des résultats
        print("\n=== Performances ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.2%}")
        
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred, target_names=['Fonctionnel', 'Panne']))
        
        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Génère, affiche et sauvegarde la matrice de confusion"""
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fonctionnel', 'Panne'])
        disp.plot(cmap='Blues', ax=ax)
        plt.title('Matrice de Confusion - SVM', fontsize=14, pad=20)
        
        # Sauvegarde de la figure
        filename = f"confusion_matrix_{self.timestamp}.png"
        filepath = self.result_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nMatrice de confusion sauvegardée sous: {filepath}")
        
        # Affichage
        plt.show()
        plt.close(fig)

    def _plot_decision_boundary(self, X, y):
        """Visualise et sauvegarde la frontière de décision"""
        fig = plt.figure(figsize=(10, 8))
        
        # Réduction de dimension avec PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Création de la grille
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        # Prédiction sur la grille
        Z = self.model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        
        # Tracé
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                            cmap=plt.cm.coolwarm, edgecolors='k', s=50)
        
        # Légende et titres
        plt.xlabel('Première Composante Principale', fontsize=12)
        plt.ylabel('Deuxième Composante Principale', fontsize=12)
        plt.title('Frontière de Décision du SVM (Projection PCA)', fontsize=14, pad=20)
        
        # Légende
        legend_labels = ['Fonctionnel', 'Panne']
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=plt.cm.coolwarm(0.), markersize=10),
                  plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=plt.cm.coolwarm(1.), markersize=10)]
        plt.legend(handles, legend_labels, title="Classes")
        
        plt.grid(alpha=0.2)
        
        # Sauvegarde de la figure
        filename = f"decision_boundary_{self.timestamp}.png"
        filepath = self.result_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Frontière de décision sauvegardée sous: {filepath}")
        
        # Affichage
        plt.show()
        plt.close(fig)

    def predict_fault(self, new_data):
        """Prédit la probabilité de panne"""
        if not hasattr(self.model, 'predict_proba'):
            raise RuntimeError("Modèle non entraîné")
        
        if isinstance(new_data, dict):
            df = pd.DataFrame([new_data])
        else:
            df = pd.DataFrame(new_data)
        
        for feature in self.features:
            if feature in self.label_encoders:
                if feature not in df.columns:
                    raise ValueError(f"Colonne manquante: {feature}")
                df.loc[:, feature] = self.label_encoders[feature].transform(
                    df[feature].astype(str))
        
        missing_cols = [col for col in self.features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
        
        return float(self.model.predict_proba(df[self.features])[:, 1][0])

    def save_model(self, filename='svm_maximo_detector.pkl'):
        """Sauvegarde le modèle au format .pkl"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'keywords': self.keywords
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModèle sauvegardé sous {filename}")

def main():
    """Point d'entrée principal"""
    print("=== DÉTECTION DE PANNES IBM MAXIMO ===")
    print("=== Algorithme SVM ===")
    
    try:
        detector = SVMFaultDetector()
        
        # 1. Chargement des données
        df = detector.load_data()
        print("\nAperçu des données:")
        print(df[['Description', 'WOPRIORITY', 'LOCATION', 'STATUS', 'PANNE']].head())
        
        # 2. Prétraitement
        X_train, X_test, y_train, y_test = detector.preprocess_data(df)
        
        # 3. Entraînement
        detector.train_model(X_train, y_train)
        
        # 4. Évaluation et visualisation
        metrics = detector.evaluate_and_plot(X_test, y_test)
        
        # Sauvegarde du modèle au format .pkl
        detector.save_model()
        
        # Exemple de prédiction
        sample_data = {
            'WOPRIORITY': 2,
            'LOCATION': 'BR300',
            'STATUS': 'CLOSE',
            'ASSETNUM': '1001'
        }
        proba = detector.predict_fault(sample_data)
        print(f"\nExemple de prédiction - Probabilité de panne: {proba:.1%}")
        
        print("\n=== PROCESSUS TERMINÉ AVEC SUCCÈS ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()