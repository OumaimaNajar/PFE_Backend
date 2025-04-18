# diagnostic_pannes_knn_location_final.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import joblib
import os
from datetime import datetime
import pyodbc
import warnings
from sklearn.exceptions import UndefinedMetricWarning

class DiagnosticKNN:
    """
    Système de diagnostic des pannes basé sur la localisation avec KNN
    Version finale avec gestion robuste des données
    """
    
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.feature_names = ['LOCATION', 'WOPRIORITY', 'STATUS', 'ASSETNUM']  # Retiré LOCATION_FAULT_RATE
        self.label_encoders = {}
        self.fault_descriptions = {0: "Non-Panne", 1: "Panne"}
        self.fault_recommendations = {
            0: "Aucune action requise",
            1: "Maintenance nécessaire - Vérifier l'historique à cette localisation"
        }
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.results_dir = "knn_location_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def connect_to_database(self):
        """Connexion à la base de données SQL Server"""
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

    def load_data_from_db(self):
        """Charge les données depuis la table workorder"""
        print("\n[1/4] Chargement des données depuis la base de données...")
        conn = None
        try:
            conn = self.connect_to_database()
            
            # Requête simplifiée et robuste
            query = """
            SELECT 
                LOCATION,
                WOPRIORITY, 
                STATUS, 
                ASSETNUM,
                FAILURECODE
            FROM 
                workorder
            WHERE 
                LOCATION IS NOT NULL
                AND WOPRIORITY IS NOT NULL
                AND STATUS IS NOT NULL
                AND ASSETNUM IS NOT NULL
            """
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_sql(query, conn)
            
            if df.empty:
                raise ValueError("Aucune donnée valide trouvée dans la table workorder")
                
            # Création de la variable cible
            df['PANNE'] = df['FAILURECODE'].notna().astype(int)
            
            # Encodage des variables catégorielles
            print("\nEncodage des caractéristiques...")
            for col in ['LOCATION', 'STATUS', 'ASSETNUM']:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            
            # Préparation des données finales
            X = df[self.feature_names].values
            y = df['PANNE'].values
            
            print(f"\nDonnées préparées avec succès:")
            print(f"- Enregistrements: {len(df)}")
            print(f"- Pannes: {sum(y)} | Non-pannes: {len(y)-sum(y)}")
            print(f"- Localisations uniques: {len(df['LOCATION'].unique())}")
            
            return X, y, df
            
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement: {str(e)}")
        finally:
            if conn:
                conn.close()

    def train_and_evaluate(self):
        """Entraîne et évalue le modèle KNN"""
        X, y, self.df = self.load_data_from_db()
        
        if len(np.unique(y)) < 2:
            print("\nAttention: Données insuffisantes pour l'entraînement")
            return self
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Entraînement
        print("\n[2/4] Entraînement du modèle KNN...")
        self.model.fit(X_train_scaled, self.y_train)
        
        # Évaluation
        print("\n[3/4] Évaluation du modèle...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Métriques
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            print("\nRapport de classification:")
            print(classification_report(
                self.y_test, y_pred,
                target_names=['Non-Panne', 'Panne'],
                zero_division=0
            ))
        
        # Visualisations
        print("\n[4/4] Génération des visualisations...")
        self._generate_visualizations(X_test_scaled, y_pred)
        
        if len(self.X_test) > 0:
            self._plot_similar_cases(self.X_test[0])
        
        return self

    def _generate_visualizations(self, X_test_scaled, y_pred):
        """Génère les visualisations standard"""
        plt.close('all')
        figures = []
        
        # 1. Matrice de confusion
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Panne', 'Panne'])
        disp.plot(ax=ax1, cmap='Blues', values_format='d')
        ax1.set_title("Matrice de Confusion")
        figures.append(fig1)
        
        # 2. Importance des caractéristiques
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        importance = np.mean(np.abs(X_test_scaled), axis=0)
        indices = np.argsort(importance)[::-1]
        ax2.barh(np.array(self.feature_names)[indices], importance[indices],
                color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Importance relative')
        ax2.set_title('Importance des Caractéristiques')
        figures.append(fig2)
        
        # 3. Répartition des pannes par localisation (top 10)
        if hasattr(self, 'df'):
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            top_locs = self.df.groupby('LOCATION')['PANNE'].sum().nlargest(10)
            top_locs.plot(kind='bar', ax=ax3, color='salmon')
            ax3.set_title("Top 10 Localisations avec le plus de pannes")
            ax3.set_xlabel("Localisation (encodée)")
            ax3.set_ylabel("Nombre de pannes")
            figures.append(fig3)
        
        # Sauvegarde
        for i, fig in enumerate(figures):
            filename = f"{self.results_dir}/{['conf_matrix', 'feat_importance', 'top_locations'][i]}_{self.timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée: {filename}")
        
        plt.show()
        plt.close('all')

    def _plot_similar_cases(self, new_case):
        """Visualisation des cas similaires"""
        new_case_scaled = self.scaler.transform([new_case])
        distances, indices = self.model.kneighbors(new_case_scaled)
        
        # Projection PCA
        pca = PCA(n_components=2)
        X_all = np.vstack([self.X_train, new_case])
        X_pca = pca.fit_transform(self.scaler.transform(X_all))
        
        plt.figure(figsize=(12, 8))
        
        # Points d'entraînement
        plt.scatter(X_pca[:-1, 0], X_pca[:-1, 1],
                   c=self.y_train, alpha=0.3, cmap='coolwarm')
        
        # Points similaires
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                   s=100, edgecolor='black', facecolor='none')
        
        # Nouveau cas
        plt.scatter(X_pca[-1, 0], X_pca[-1, 1],
                   s=200, marker='*', c='gold', edgecolor='black')
        
        plt.title("Cas similaires (Projection PCA)")
        plt.xlabel("Composante Principale 1")
        plt.ylabel("Composante Principale 2")
        plt.grid(True, alpha=0.3)
        
        # Affichage des informations
        print("\nCas similaires trouvés:")
        for i, idx in enumerate(indices[0], 1):
            loc = self.label_encoders['LOCATION'].inverse_transform([self.X_train[idx, 0]])[0]
            print(f"{i}. Localisation: {loc} | Panne: {self.y_train[idx]}")
        
        filename = f"{self.results_dir}/similar_cases_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def predict_fault(self, location, wopriority=3, status=1, assetnum=1):
        """Prédit une panne pour une localisation donnée"""
        try:
            # Encodage
            encoded = [
                self.label_encoders['LOCATION'].transform([str(location)])[0],
                wopriority,
                status,
                self.label_encoders['ASSETNUM'].transform([str(assetnum)])[0]
            ]
            
            # Prédiction
            X_new = self.scaler.transform([encoded])
            fault_class = self.model.predict(X_new)[0]
            proba = self.model.predict_proba(X_new)[0][fault_class]
            
            return {
                'location': location,
                'prediction': self.fault_descriptions[fault_class],
                'confidence': float(proba),
                'recommendation': self.fault_recommendations[fault_class]
            }
            
        except Exception as e:
            raise ValueError(f"Erreur de prédiction: {str(e)}")

    def save_model(self, filename=None):
        """Sauvegarde le modèle"""
        if filename is None:
            filename = f"{self.results_dir}/knn_model_{self.timestamp}.joblib"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'timestamp': self.timestamp
        }
        
        joblib.dump(model_data, filename)
        print(f"\nModèle sauvegardé sous: {filename}")

def main():
    print("=== DIAGNOSTIC DE PANNES PAR LOCALISATION ===")
    print("=== KNN - Version Finale ===")
    
    try:
        # Initialisation
        knn = DiagnosticKNN(n_neighbors=3)
        knn.train_and_evaluate()
        
        # Sauvegarde
        knn.save_model()
        
        # Exemple de prédiction
        if hasattr(knn, 'label_encoders'):
            sample_loc = knn.label_encoders['LOCATION'].inverse_transform([0])[0]
            print(f"\nExemple pour la localisation: {sample_loc}")
            result = knn.predict_fault(location=sample_loc)
            print("\nRésultat:")
            print(f"- Prédiction: {result['prediction']}")
            print(f"- Confiance: {result['confidence']:.1%}")
            print(f"- Recommandation: {result['recommendation']}")
        
        print("\n=== PROCESSUS TERMINÉ AVEC SUCCÈS ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}")

if __name__ == "__main__":
    main()