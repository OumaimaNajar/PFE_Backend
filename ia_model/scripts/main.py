import pandas as pd
from sklearn.model_selection import train_test_split
from core.fault_detection.supervised.random_forest import RandomForestFaultDetector

from core.fault_detection.supervised.dnn import DNNFaultDetector
from core.fault_detection.unsupervised.kmeans import KMeansAnomalyDetector

def load_and_preprocess_data():
    """Charge et prépare les données"""
    df = pd.read_csv('/data/clean_workorders.csv', sep=';')
    df_cleaned = df.dropna()
    
    # Création de la variable cible
    keywords = ['leak', 'stopped', 'overheating', 'failure', 'broken']
    df_cleaned['PANNE'] = df_cleaned['Description'].str.contains('|'.join(keywords), case=False).astype(int)
    
    return df_cleaned

def main():
    # 1. Chargement des données
    df = load_and_preprocess_data()
    
    # 2. Préparation pour Random Forest (supervisé)
    X = df[['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']]
    y = df['PANNE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Initialisation et entraînement des modèles
    rf_detector = RandomForestFaultDetector()
 
    
    print("Entraînement du Random Forest...")
    rf_detector.train(X_train, y_train)
    
    print("Entraînement de l'Isolation Forest...")
  
    # 4. Évaluation
    print("\nÉvaluation Random Forest:")
    rf_predictions = rf_detector.predict(X_test)
    
    print("\nDétection d'anomalies:")
 
    # 5. Sauvegarde
    rf_detector.save_model('../models/rf_model.joblib')
   
    print("\nModèles sauvegardés avec succès!")

if __name__ == "__main__":
    main()