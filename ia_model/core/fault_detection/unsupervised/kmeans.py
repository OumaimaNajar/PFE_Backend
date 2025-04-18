# kmeans_anomaly_detector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import joblib

class KMeansAnomalyDetector:
    def __init__(self, n_clusters=2, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.label_encoders = {}
        self.features = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']
        
    def load_data(self, file_path='../../../../data/clean_workorders.csv'):
        """Charge et prépare les données"""
        df = pd.read_csv(file_path, sep=';')
        return df.dropna()
    
    def preprocess_data(self, df):
        """Prétraitement des données"""
        # Encodage des caractéristiques catégorielles
        for feature in self.features:
            if df[feature].dtype == 'object':
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(
                    df[feature].astype(str))
        
        X = df[self.features]
        return self.scaler.fit_transform(X)
    
    def detect_anomalies(self, X, percentile=95):
        """Détection des anomalies avec K-Means"""
        # Entraînement du modèle
        clusters = self.model.fit_predict(X)
        
        # Calcul des distances aux centroïdes
        distances = self.model.transform(X)
        min_distances = np.min(distances, axis=1)
        
        # Détection des anomalies (points les plus éloignés)
        threshold = np.percentile(min_distances, percentile)
        anomalies = (min_distances > threshold).astype(int)
        
        return anomalies, min_distances
    
    def visualize_results(self, X, anomalies, true_labels=None):
        """Visualisation des résultats en 2D avec PCA"""
        # Réduction de dimension
        X_pca = self.pca.fit_transform(X)
        
        # Création de la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Visualisation des anomalies détectées
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=anomalies, cmap='coolwarm')
        ax1.set_title('Anomalies détectées par K-Means')
        ax1.set_xlabel('Composante Principale 1')
        ax1.set_ylabel('Composante Principale 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Comparaison avec les vraies étiquettes si disponibles
        if true_labels is not None:
            scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='coolwarm')
            ax2.set_title('Vraies étiquettes de panne')
            ax2.set_xlabel('Composante Principale 1')
            ax2.set_ylabel('Composante Principale 2')
            plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig('kmeans_anomaly_detection.png')
        plt.show()
    
    def evaluate(self, anomalies, true_labels):
        """Évaluation des performances"""
        print("\n=== Performance de la détection d'anomalies ===")
        print(classification_report(true_labels, anomalies))
        
        cm = confusion_matrix(true_labels, anomalies)
        print("\nMatrice de confusion:")
        print(cm)
        
        return cm
    
    def save_model(self, file_path='kmeans_model.joblib'):
        """Sauvegarde du modèle et des préprocesseurs"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'label_encoders': self.label_encoders,
            'features': self.features
        }, file_path)
        print(f"Modèle K-Means sauvegardé dans {file_path}")

def main():
    """Fonction principale pour exécuter le pipeline complet"""
    print("=== Détection d'Anomalies avec K-Means ===")
    
    try:
        # Initialisation
        detector = KMeansAnomalyDetector()
        
        # 1. Chargement des données
        print("\n[1/4] Chargement des données...")
        df = detector.load_data()
        
        # Création des vraies étiquettes (si disponible)
        keywords = ['leak', 'stopped', 'overheating', 'failure', 'broken',
                   'malfunction', 'error', 'defect', 'fault', 'out of order']
        true_labels = df['Description'].str.contains('|'.join(keywords), 
                                                   case=False, na=False).astype(int)
        
        # 2. Prétraitement
        print("[2/4] Prétraitement des données...")
        X_scaled = detector.preprocess_data(df)
        
        # 3. Détection des anomalies
        print("[3/4] Détection des anomalies...")
        anomalies, scores = detector.detect_anomalies(X_scaled)
        
        # 4. Visualisation et évaluation
        print("[4/4] Évaluation des résultats...")
        detector.visualize_results(X_scaled, anomalies, true_labels)
        
        if true_labels is not None:
            detector.evaluate(anomalies, true_labels)
        
        # Sauvegarde
        detector.save_model()
        
        print("\n=== Processus terminé avec succès ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()