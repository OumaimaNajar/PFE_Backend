# decision_tree_maximo.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
from pathlib import Path
from datetime import datetime
import pyodbc
import warnings

class MaximoFaultDiagnoser:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=3,
            criterion='entropy',
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        self.keywords = [
            'leak', 'failure', 'broken', 'error', 'defect',
            'fault', 'repair', 'replace', 'damage', 'problem',
            'investigate', 'rebuild', 'worn', 'lubricate', 'clean'
        ]
        self.features = ['WOPRIORITY', 'LOCATION', 'STATUS', 'ASSETNUM']
        self.label_encoders = {}
        self.result_dir = Path("descision_tree_results")
        self.result_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def connect_to_database(self):
        """Méthode de connexion à la base de données"""
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
        """Charge les données depuis la base de données"""
        print("\n[1/4] Chargement des données depuis la base de données...")
        conn = None
        try:
            conn = self.connect_to_database()
            
            # Requête principale adaptée
            query = """
            SELECT 
                WOPRIORITY, 
                LOCATION, 
                STATUS, 
                ASSETNUM,
                DESCRIPTION,
                FAILURECODE
            FROM 
                workorder
            WHERE 
                DESCRIPTION IS NOT NULL
            """
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_sql(query, conn)
            
            if df.empty:
                raise ValueError("La requête n'a retourné aucun résultat")
                
            # Nettoyage et préparation des données
            df = df.dropna(subset=self.features + ['DESCRIPTION'])
            
            # Création de la variable cible
            mask = (
                df['DESCRIPTION'].str.contains('|'.join(self.keywords), case=False, na=False) |
                df['FAILURECODE'].notna()
            )
            df['PANNE'] = mask.astype(int)
            
            print(f"Données chargées avec succès. {len(df)} enregistrements.")
            print(f"Distribution des classes:\n{df['PANNE'].value_counts(normalize=True)}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement: {str(e)}")
        finally:
            if conn:
                conn.close()

    def preprocess_data(self, df):
        """Prétraitement des données"""
        print("\n[2/4] Prétraitement des données...")
        
        # Encodage des variables catégorielles
        for feature in ['LOCATION', 'STATUS', 'ASSETNUM']:
            if feature in df.columns:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
        
        # Normalisation des priorités
        if 'WOPRIORITY' in df.columns:
            df['WOPRIORITY'] = pd.to_numeric(df['WOPRIORITY'], errors='coerce')
            df['WOPRIORITY'] = df['WOPRIORITY'].fillna(df['WOPRIORITY'].median())
            df['WOPRIORITY'] = (df['WOPRIORITY'] - df['WOPRIORITY'].min()) / \
                              (df['WOPRIORITY'].max() - df['WOPRIORITY'].min())
        
        return df

    def train_and_evaluate(self, df):
        """Entraîne et évalue le modèle"""
        print("\n[3/4] Entraînement du modèle...")
        
        # Séparation des données
        X = df[self.features]
        y = df['PANNE']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Entraînement
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred, target_names=['Non-Panne', 'Panne'], digits=4))
        
        # Matrice de confusion
        self._plot_confusion_matrix(y_test, y_pred)
        
        # Visualisation de l'arbre
        self._plot_decision_tree()
        
        # Importance des caractéristiques
        self._plot_feature_importance(X_train.columns)
        
        return self.model

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Affiche et sauvegarde la matrice de confusion"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Panne', 'Panne'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Matrice de Confusion', fontsize=14)
        
        filename = f"confusion_matrix_{self.timestamp}.png"
        plt.savefig(self.result_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()  # Affiche la figure
        plt.close()

    def _plot_decision_tree(self):
        """Visualise et sauvegarde l'arbre de décision"""
        plt.figure(figsize=(24, 12))
        plot_tree(
            self.model,
            feature_names=self.features,
            class_names=['Non-Panne', 'Panne'],
            filled=True,
            rounded=True,
            fontsize=10,
            proportion=True,
            impurity=False
        )
        plt.title("Arbre de Décision pour le Diagnostic des Pannes Maximo", fontsize=16)
        plt.tight_layout()
        
        filename = f"decision_tree_{self.timestamp}.png"
        plt.savefig(self.result_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()  # Affiche la figure
        plt.close()

    def _plot_feature_importance(self, feature_names):
        """Affiche l'importance des caractéristiques"""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Importance des Caractéristiques", fontsize=14)
        bars = plt.bar(range(len(feature_names)), importance[indices], align='center')
        plt.bar_label(bars, fmt='%.3f')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Caractéristiques')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        filename = f"feature_importance_{self.timestamp}.png"
        plt.savefig(self.result_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()  # Affiche la figure
        plt.close()

    def save_model(self, filename=None):
        """Sauvegarde le modèle et les artefacts"""
        if filename is None:
            filename = f"maximo_tree_model_{self.timestamp}.joblib"
            
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'keywords': self.keywords,
            'timestamp': self.timestamp
        }
        joblib.dump(model_data, self.result_dir / filename)
        print(f"\nModèle et artefacts sauvegardés sous: {self.result_dir / filename}")

def main():
    print("=== DIAGNOSTIC DE PANNES IBM MAXIMO ===")
    print("=== Algorithme Arbre de Décision ===")
    
    try:
        diagnoser = MaximoFaultDiagnoser()
        
        # 1. Chargement des données
        df = diagnoser.load_data_from_db()
        
        # 2. Prétraitement
        df_processed = diagnoser.preprocess_data(df)
        
        # 3. Entraînement et évaluation
        diagnoser.train_and_evaluate(df_processed)
        
        # 4. Sauvegarde du modèle
        diagnoser.save_model()
        
        print("\n=== PROCESSUS TERMINÉ AVEC SUCCÈS ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}")

if __name__ == "__main__":
    main()