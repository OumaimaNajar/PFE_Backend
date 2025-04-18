# dnn_fault_detector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, 
                           recall_score, f1_score)
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from pathlib import Path
import joblib

class DNNFaultDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.features = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']
        self.keywords = ['leak', 'stopped', 'overheating', 'failure', 'broken',
                        'malfunction', 'error', 'defect', 'fault', 'out of order']
        self.history = None

    def load_data(self, file_path='../../../../data/clean_workorders.csv'):
        """Charge et prépare les données"""
        df = pd.read_csv(file_path, sep=';')
        df_cleaned = df.dropna()
        
        # Création de la variable cible
        df_cleaned['PANNE'] = df_cleaned['Description'].str.contains(
            '|'.join(self.keywords), case=False, na=False
        ).astype(int)
        
        return df_cleaned

    def preprocess_data(self, df):
        """Prétraitement des données"""
        # Encodage des caractéristiques catégorielles
        for feature in self.features:
            if df[feature].dtype == 'object':
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(
                    df[feature].astype(str)
                )
        
        X = df[self.features]
        y = df['PANNE']
        
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    def build_model(self, input_shape):
        """Construction du modèle DNN"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Entraînement du modèle"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history

    def evaluate(self, X_test, y_test):
        """Évaluation du modèle et affichage des visualisations"""
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n=== Performance du modèle DNN ===")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Création d'une figure globale avec les deux sous-figures
        plt.figure(figsize=(18, 6))
        
        # Sous-figure 1: Matrice de confusion
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        classes = ['Fonctionnel', 'Panne']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.title('Matrice de Confusion - DNN')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        # Sous-figure 2: Historique d'entraînement
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Historique d\'entraînement')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('dnn_results.png')
        plt.show()
        
        return accuracy, precision, recall, f1

    def save_model(self, file_path='dnn_model'):
        """Sauvegarde du modèle et des préprocesseurs"""
        # Sauvegarde du modèle Keras
        self.model.save(f"{file_path}.keras")
        
        # Sauvegarde des préprocesseurs
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'features': self.features,
            'keywords': self.keywords
        }, f"{file_path}_preprocessors.joblib")
        
        print(f"\nModèle DNN sauvegardé dans {file_path}.keras")
        print(f"Préprocesseurs sauvegardés dans {file_path}_preprocessors.joblib")

def main():
    """Fonction principale pour exécuter le pipeline complet"""
    print("=== Détection de Pannes avec DNN ===")
    
    try:
        # Initialisation
        detector = DNNFaultDetector()
        
        # 1. Chargement des données
        print("\n[1/4] Chargement des données...")
        df = detector.load_data()
        
        # 2. Prétraitement
        print("[2/4] Prétraitement des données...")
        X_train, X_test, y_train, y_test = detector.preprocess_data(df)
        
        # 3. Construction et entraînement
        print("[3/4] Construction et entraînement du modèle DNN...")
        detector.model = detector.build_model(X_train.shape[1])
        detector.train(X_train, y_train, X_test, y_test, epochs=30)
        
        # 4. Évaluation et visualisation
        print("[4/4] Évaluation du modèle...")
        detector.evaluate(X_test, y_test)
        
        # Sauvegarde
        detector.save_model()
        
        print("\n=== Processus terminé avec succès ===")
        
    except Exception as e:
        print(f"\n=== ERREUR ===\n{str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()