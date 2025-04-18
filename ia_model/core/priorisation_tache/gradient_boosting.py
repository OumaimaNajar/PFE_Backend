import pandas as pd # type: ignore
import numpy as np # type: ignore
import pyodbc # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
import xgboost as xgb # type: ignore
import lightgbm as lgb # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
import joblib # type: ignore
import warnings

warnings.filterwarnings('ignore')

class MaximoPriorityPredictor:
    def __init__(self):
        self.conn = None
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
    
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
            self.conn = conn
            return conn
        except Exception as e:
            raise ConnectionError(f"Échec de la connexion: {str(e)}")
    
    def fetch_data(self):
        """Récupération des données depuis la table WORKORDER"""
        if not self.conn:
            self.connect_to_db()
        
        query = """
        SELECT WORKORDERID, Description, ASSETNUM, FAILURECODE, WOPRIORITY, LOCATION, STATUS 
        FROM WORKORDER 
        WHERE STATUS IN ('CAN', 'CLOSE') AND WOPRIORITY IS NOT NULL
        """
        
        try:
            df = pd.read_sql(query, self.conn)
            return df
        except Exception as e:
            raise Exception(f"Erreur lors de la récupération des données: {str(e)}")
    
    def preprocess_data(self, df):
        """Prétraitement des données"""
        # Suppression des doublons
        df = df.drop_duplicates(subset=['WORKORDERID'])
        
        # Filtrage des priorités non nulles
        df = df[df['WOPRIORITY'].notna()]
        
        # Conversion de la priorité en entier (classification)
        df['WOPRIORITY'] = df['WOPRIORITY'].astype(int)
        
        # Encodage des variables catégorielles
        categorical_cols = ['ASSETNUM', 'FAILURECODE', 'LOCATION', 'STATUS']
        for col in categorical_cols:
            df[col] = df[col].fillna('UNKNOWN')
        
        # Séparation des features et de la target
        X = df[['Description', 'ASSETNUM', 'FAILURECODE', 'LOCATION', 'STATUS']]
        y = df['WOPRIORITY']
        
        return X, y
    
    def create_preprocessor(self):
        """Création du préprocesseur pour les données"""
        # Transformation pour le texte des descriptions
        text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))
        ])
        
        # Transformation pour les features catégorielles
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Préprocesseur complet
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, 'Description'),
                ('cat', categorical_transformer, ['ASSETNUM', 'FAILURECODE', 'LOCATION', 'STATUS'])
            ])
        
        return preprocessor
    
    def train_model(self, model_type='xgboost'):
        """Entraînement du modèle"""
        # Récupération des données
        df = self.fetch_data()
        X, y = self.preprocess_data(df)
        
        # Création du préprocesseur
        self.preprocessor = self.create_preprocessor()
        
        # Séparation train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Choix du modèle
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(y.unique()),
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # LightGBM
            model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=len(y.unique()),
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        # Création du pipeline complet
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        # Entraînement
        pipeline.fit(X_train, y_train)
        
        # Évaluation
        y_pred = pipeline.predict(X_test)
        print("Rapport de classification:\n", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        
        # Sauvegarde du modèle
        self.model = pipeline
        joblib.dump(pipeline, 'maximo_priority_model.pkl')
        
        return pipeline
    
    def plot_feature_importance(self):
        """Visualisation de l'importance des features"""
        if not self.model:
            raise Exception("Modèle non entraîné. Veuillez d'abord entraîner le modèle.")
        
        # Récupération du modèle final
        if isinstance(self.model.named_steps['classifier'], xgb.XGBClassifier):
            model = self.model.named_steps['classifier']
            
            # Pour XGBoost
            fig, ax = plt.subplots(figsize=(10, 8))
            xgb.plot_importance(model, ax=ax)
            plt.title("Importance des features - XGBoost")
            plt.show()
            
        elif isinstance(self.model.named_steps['classifier'], lgb.LGBMClassifier):
            model = self.model.named_steps['classifier']
            
            # Pour LightGBM
            lgb.plot_importance(model, figsize=(10, 8))
            plt.title("Importance des features - LightGBM")
            plt.show()
        
        # Pour le préprocesseur textuel (si TF-IDF)
        if 'text' in self.model.named_steps['preprocessor'].named_transformers_:
            tfidf = self.model.named_steps['preprocessor'].named_transformers_['text'].named_steps['tfidf']
            feature_names = tfidf.get_feature_names_out()
            
            if isinstance(self.model.named_steps['classifier'], xgb.XGBClassifier):
                importance = self.model.named_steps['classifier'].feature_importances_
            else:
                importance = self.model.named_steps['classifier'].feature_importances_
            
            # On prend seulement les features textuelles
            text_feature_importance = importance[:len(feature_names)]
            
            # Création d'un DataFrame pour visualisation
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': text_feature_importance
            }).sort_values('importance', ascending=False).head(20)
            
            # Visualisation
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=df_importance)
            plt.title("Top 20 des mots les plus importants dans les descriptions")
            plt.show()
    
    def plot_confusion_matrix(self):
        """Visualisation de la matrice de confusion"""
        if not self.model:
            raise Exception("Modèle non entraîné. Veuillez d'abord entraîner le modèle.")
        
        # Récupération des données pour évaluation
        df = self.fetch_data()
        X, y = self.preprocess_data(df)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Prédiction
        y_pred = self.model.predict(X_test)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Visualisation
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(y.unique()), 
                    yticklabels=sorted(y.unique()))
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.title('Matrice de confusion')
        plt.show()
    
    def predict_new_workorder(self, new_data):
        """Prédiction pour de nouvelles workorders"""
        if not self.model:
            raise Exception("Modèle non entraîné. Veuillez d'abord entraîner le modèle.")
        
        # Conversion en DataFrame si nécessaire
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        
        # Prédiction
        predictions = self.model.predict(new_data)
        probabilities = self.model.predict_proba(new_data)
        
        # Formatage des résultats
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'WORKORDERID': new_data.iloc[i]['WORKORDERID'] if 'WORKORDERID' in new_data.columns else f"NEW_{i}",
                'Predicted_Priority': int(pred),
                'Probabilities': {cls: float(prob) for cls, prob in enumerate(probabilities[i], start=1)}
            }
            results.append(result)
        
        return results


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation
    predictor = MaximoPriorityPredictor()
    
    # Entraînement du modèle (choisir 'xgboost' ou 'lightgbm')
    print("Entraînement du modèle XGBoost...")
    predictor.train_model(model_type='xgboost')
    
    # Visualisations
    print("\nGénération des visualisations...")
    predictor.plot_feature_importance()
    predictor.plot_confusion_matrix()
    
    # Exemple de prédiction pour de nouvelles workorders
    print("\nPrédiction pour de nouvelles workorders...")
    new_workorders = pd.DataFrame({
        'WORKORDERID': [999999, 999998],
        'Description': [
            "Urgent: Motor failure causing production halt",
            "Routine inspection of conveyor system"
        ],
        'ASSETNUM': ['1001', '12600'],
        'FAILURECODE': ['MOTOR', None],
        'LOCATION': ['BR300', 'SHIPPING'],
        'STATUS': ['CAN', 'CAN']
    })
    
    predictions = predictor.predict_new_workorder(new_workorders)
    for pred in predictions:
        print(f"\nWorkorder {pred['WORKORDERID']}:")
        print(f"Priorité prédite: {pred['Predicted_Priority']}")
        print("Probabilités par classe:")
        for priority, prob in pred['Probabilities'].items():
            print(f"  Priority {priority}: {prob:.2f}")