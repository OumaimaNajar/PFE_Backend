import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class FacteurInfluenceClassifier:
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'facteurs_influencent_pannes.csv')
            )
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.type_encoder = None
        self.facteur_encoder = None
        self.col_encoders = {}
        self._load_data()
        self._train_model()

    def _load_data(self):
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, sep=';', encoding=encoding)
                print(f"[INFO] CSV chargé avec encodage: {encoding}", file=sys.stderr)
                # Ajout des logs pour vérifier le contenu
                print(f"[DEBUG] Colonnes disponibles: {self.df.columns.tolist()}", file=sys.stderr)
                print(f"[DEBUG] Premières lignes:\n{self.df.head()}", file=sys.stderr)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Aucun encodage valide trouvé pour le CSV des facteurs influents.")


    def _prepare_features(self):
        df = self.df.dropna(subset=['type_panne', 'facteur'])
        
        # Vérifier si les colonnes existent avant de les sélectionner
        available_columns = self.df.columns.tolist()
        required_columns = ['type_panne', 'facteur', 'PB', 'FC', 'oil_level', 'downtime']
        
        for col in required_columns:
            if col not in available_columns:
                print(f"[WARNING] Colonne '{col}' non trouvée dans le CSV. Une colonne vide sera créée.", file=sys.stderr)
                self.df[col] = 0
        
        # Ajout des colonnes pour les caractéristiques
        selected_features = [
            'type_panne',
            'PB',
            'FC',
            'oil_level',
            'downtime'
        ]
        
        # Encodage des colonnes catégorielles
        for col in selected_features:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.col_encoders[col] = le
        
        X = df[selected_features]
        y = df['facteur']
    
        self.type_encoder = LabelEncoder()
        X['type_panne'] = self.type_encoder.fit_transform(X['type_panne'].astype(str))
        
        self.facteur_encoder = LabelEncoder()
        y_encoded = self.facteur_encoder.fit_transform(y.astype(str))
        return X, y_encoded



    def predict_factors(self, type_panne):
        """
        Prédit les facteurs d'influence pour un type de panne donné
        """
        try:
            # Log pour vérifier le type de panne recherché
            print(f"[DEBUG] Recherche du type de panne: {type_panne}", file=sys.stderr)
            
            # Récupération des données depuis le CSV
            panne_match = self.df[self.df['type_panne'].str.lower() == type_panne.lower()]
            
            # Log pour vérifier les correspondances trouvées
            print(f"[DEBUG] Nombre de correspondances trouvées: {len(panne_match)}", file=sys.stderr)
            
            if not panne_match.empty:
                panne_data = panne_match.iloc[0]
                # Log des valeurs récupérées
                print(f"[DEBUG] Valeurs récupérées - PB: {panne_data.get('PB', 0)}, FC: {panne_data.get('FC', 0)}, oil_level: {panne_data.get('oil_level', 0)}, downtime: {panne_data.get('downtime', 0)}", file=sys.stderr)
                
                # S'assurer que les données sont dans le bon format pour la prédiction
                try:
                    type_panne_enc = self.type_encoder.transform([type_panne])[0]
                    input_data = pd.DataFrame({
                        'type_panne': [type_panne_enc],
                        'PB': [float(panne_data.get('PB', 0))],
                        'FC': [float(panne_data.get('FC', 0))],
                        'oil_level': [float(panne_data.get('oil_level', 0))],
                        'downtime': [float(panne_data.get('downtime', 0))]
                    })
                    pred = self.model.predict(input_data)[0]
                    facteur = self.facteur_encoder.inverse_transform([pred])[0]
                except Exception as e:
                    print(f"[ERROR] Erreur lors de la prédiction du modèle: {str(e)}", file=sys.stderr)
                    facteur = panne_data.get('facteur', "Inconnu")
                
                # Récupérer tous les facteurs possibles pour ce type de panne
                facteurs_possibles = panne_match['facteur'].unique().tolist()
                
                return {
                    "type_panne": type_panne,
                    "facteur_principal": facteur,
                    "facteurs_possibles": facteurs_possibles,
                    "PB": float(panne_data.get('PB', 0)),
                    "FC": float(panne_data.get('FC', 0)), 
                    "oil_level": float(panne_data.get('oil_level', 0)),
                    "downtime": float(panne_data.get('downtime', 0))
                }
            else:
                print(f"[WARNING] Aucune correspondance trouvée pour le type de panne: {type_panne}", file=sys.stderr)
                # Utiliser le modèle pour prédire le facteur à partir du type de panne seul
                try:
                    type_panne_enc = self.type_encoder.transform([type_panne])[0]
                    input_data = pd.DataFrame({
                        'type_panne': [type_panne_enc],
                        'PB': [0],
                        'FC': [0],
                        'oil_level': [0],
                        'downtime': [0]
                    })
                    pred = self.model.predict(input_data)[0]
                    facteur = self.facteur_encoder.inverse_transform([pred])[0]
                    
                    return {
                        "type_panne": type_panne,
                        "facteur_principal": facteur,
                        "facteurs_possibles": [facteur],
                        "PB": 0,
                        "FC": 0,
                        "oil_level": 0,
                        "downtime": 0
                    }
                except Exception as e:
                    print(f"[ERROR] Erreur lors de la prédiction du modèle: {str(e)}", file=sys.stderr)
                    return {
                        "type_panne": type_panne,
                        "facteur_principal": "Inconnu",
                        "facteurs_possibles": [],
                        "PB": 0,
                        "FC": 0,
                        "oil_level": 0,
                        "downtime": 0,
                        "error": f"Aucune donnée trouvée pour le type de panne '{type_panne}'"
                    }
                
        except Exception as e:
            print(f"[ERREUR] Prédiction échouée : {str(e)}", file=sys.stderr)
            return {
                "type_panne": type_panne,
                "facteur_principal": "Inconnu",
                "facteurs_possibles": [],
                "PB": 0,
                "FC": 0,
                "oil_level": 0,
                "downtime": 0,
                "error": str(e)
            }

    def _train_model(self):
        X, y = self._prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        score = self.model.score(X_test, y_test)
        print(f"[INFO] Score de validation : {score:.2f}", file=sys.stderr)

    # def predict_factors(self, type_panne):
    #     try:
    #         type_panne_enc = self.type_encoder.transform([type_panne])[0]
            
    #         # Récupérer les valeurs réelles depuis le CSV
    #         panne_data = self.df[self.df['type_panne'].str.lower() == type_panne.lower()].iloc[0]
            
    #         return {
    #             "type_panne": type_panne,
    #             "facteur_principal": self.facteur_encoder.inverse_transform(
    #                 self.model.predict([[type_panne_enc, 
    #                                    panne_data.get('PB', 0),
    #                                    panne_data.get('FC', 0),
    #                                    panne_data.get('oil_level', 0),
    #                                    panne_data.get('downtime', 0)]])[0]
    #             ),
    #             "PB": panne_data.get('PB', 0),
    #             "FC": panne_data.get('FC', 0),
    #             "oil_level": panne_data.get('oil_level', 0),
    #             "downtime": panne_data.get('downtime', 0)
    #         }
    #     except Exception as e:
    #         print(f"[ERREUR] Prédiction échouée : {str(e)}", file=sys.stderr)
    #         return {"error": str(e)}

    # def _train_model(self):
    #     X, y = self._prepare_features()
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #     self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    #     self.model.fit(X_train, y_train)

    #     score = self.model.score(X_test, y_test)
    #     print(f"[INFO] Score de validation : {score:.2f}", file=sys.stderr)

    # def predict_factors(self, type_panne):
    #     # Prédire le facteur principal pour un type de panne donné
    #     type_panne_enc = self.type_encoder.transform([type_panne])
    #     pred = self.model.predict(type_panne_enc.reshape(-1, 1))
    #     facteur = self.facteur_encoder.inverse_transform(pred)[0]
    #     # On peut aussi retourner tous les facteurs possibles pour ce type de panne depuis le CSV
    #     facteurs_possibles = self.df[self.df['type_panne'].str.lower() == type_panne.lower()]['facteur'].unique().tolist()
    #     return {
    #         "type_panne": type_panne,
    #         "facteur_principal": facteur,
    #         "facteurs_possibles": facteurs_possibles
    #     }

    def save_model(self, path='facteur_influence_model.pkl'):
        """
        Sauvegarde le modèle entraîné
        """
        model_data = {
            'model': self.model,
            'type_encoder': self.type_encoder,
            'facteur_encoder': self.facteur_encoder,
            'col_encoders': self.col_encoders
        }
        joblib.dump(model_data, path)
        print(f"[INFO] Modèle sauvegardé à {path}", file=sys.stderr)



    def load_model(self, path='facteur_influence_model.pkl'):
        """
        Charge un modèle préalablement sauvegardé
        """
        try:
            saved = joblib.load(path)
            self.model = saved['model']
            self.type_encoder = saved['type_encoder']
            self.facteur_encoder = saved['facteur_encoder']
            
            if 'col_encoders' in saved:
                self.col_encoders = saved['col_encoders']
                
            print(f"[INFO] Modèle chargé depuis {path}", file=sys.stderr)
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le modèle : {str(e)}", file=sys.stderr)
            raise


if __name__ == '__main__':
    classifier = FacteurInfluenceClassifier()
    # Exemple d'utilisation
    type_panne_test = "Surchauffe moteur"
    result = classifier.predict_factors(type_panne_test)
    print(result)
    classifier.save_model()