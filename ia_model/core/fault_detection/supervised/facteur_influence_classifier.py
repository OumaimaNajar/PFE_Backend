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
        
        # Normalize type_panne to uppercase
        df['type_panne'] = df['type_panne'].str.upper()
        
        # Vérifier si les colonnes existent avant de les sélectionner
        available_columns = self.df.columns.tolist()
        required_columns = [
            'type_panne', 
            'facteur', 
            'PB', 
            'FC', 
            'oil_level', 
            'downtime',
            'type_lubrification',
            'vibration',
            'power_alimentation',
            'maintenance_frequency',
            'seniority'
        ]
        
        for col in required_columns:
            if col not in available_columns:
                print(f"[WARNING] Colonne '{col}' non trouvée dans le CSV. Une colonne vide sera créée.", file=sys.stderr)
                # Initialiser les valeurs par défaut selon le type de colonne
                if col in ['FC', 'type_lubrification', 'vibration', 'power_alimentation', 'maintenance_frequency']:
                    self.df[col] = ''
                else:
                    self.df[col] = 0
        
        # Ajout des colonnes pour les caractéristiques
        selected_features = [
            'type_panne',
            'PB',
            'FC',
            'oil_level',
            'downtime',
            'type_lubrification',
            'vibration',
            'power_alimentation',
            'maintenance_frequency',
            'seniority'
        ]
        
        # Encodage des colonnes catégorielles
        for col in selected_features:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str).str.upper())
                self.col_encoders[col] = le
        
        X = df[selected_features]
        y = df['facteur']
    
        self.type_encoder = LabelEncoder()
        X['type_panne'] = self.type_encoder.fit_transform(X['type_panne'].astype(str))
        
        self.facteur_encoder = LabelEncoder()
        y_encoded = self.facteur_encoder.fit_transform(y.astype(str))
        return X, y_encoded



    def predict_factors(self, type_panne):
        try:
            print(f"[DEBUG] Recherche du type de panne: {type_panne}", file=sys.stderr)
            
            type_panne_normalized = type_panne.upper()
            print(f"[DEBUG] Type de panne normalisé: {type_panne_normalized}", file=sys.stderr)
            
            panne_match = self.df[self.df['type_panne'].str.upper() == type_panne_normalized]
            
            if not panne_match.empty:
                panne_data = panne_match.iloc[0]
                
                # Préparer les données pour la prédiction
                type_panne_enc = self.type_encoder.transform([type_panne_normalized])[0]
                
                # Log des valeurs brutes avant traitement
                print(f"\n[INFO] Données complètes de la panne:", file=sys.stderr)
                print(f"Type de panne: {panne_data['type_panne']}", file=sys.stderr)
                print(f"Action recommandée: {panne_data['recommended_action']}", file=sys.stderr)
                print(f"Type de lubrification: {panne_data['type_lubrification']}", file=sys.stderr)
                print(f"Vibration: {panne_data['vibration']}", file=sys.stderr)
                print(f"Alimentation: {panne_data['power_alimentation']}", file=sys.stderr)
                print(f"Fréquence de maintenance: {panne_data['maintenance_frequency']}", file=sys.stderr)
                print(f"Ancienneté: {panne_data['seniority']}", file=sys.stderr)
                print(f"PB: {panne_data.get('PB')}", file=sys.stderr)
                print(f"FC: {panne_data.get('FC')}", file=sys.stderr)
                
                input_features = {
                    'type_panne': [type_panne_enc],
                    'PB': [str(panne_data.get('PB', 'UNKNOWN'))],
                    'FC': [str(panne_data.get('FC', 'UNKNOWN'))],
                    'oil_level': [float(panne_data.get('oil_level', 0))],
                    'downtime': [float(panne_data.get('downtime', 0))]
                }
                
                # Prédiction
                input_data = pd.DataFrame(input_features)
                pred = self.model.predict(input_data)[0]
                facteur = self.facteur_encoder.inverse_transform([pred])[0]
                
                # Utiliser directement les importances pré-calculées et triées
                result = {
                    "type_panne": type_panne,
                    "facteur_principal": facteur,
                    "facteurs_possibles": panne_match['facteur'].unique().tolist(),
                    "PB": str(panne_data.get('PB', 'UNKNOWN')),
                    "FC": str(panne_data.get('FC', 'UNKNOWN')),
                    "oil_level": float(panne_data.get('oil_level', 0)),
                    "downtime": float(panne_data.get('downtime', 0)),
                    "recommended_action": str(panne_data.get('recommended_action', '')),
                    "type_lubrification": str(panne_data.get('type_lubrification', '')),
                    "vibration": str(panne_data.get('vibration', '')),
                    "power_alimentation": str(panne_data.get('power_alimentation', '')),
                    "maintenance_frequency": str(panne_data.get('maintenance_frequency', '')),
                    "seniority": int(panne_data.get('seniority', 0)),
                    "feature_importance": sorted(self.feature_importances_, key=lambda x: x['importance'], reverse=True)
                }
                
                return result
            
            else:
                print(f"[WARNING] Aucune correspondance trouvée pour le type de panne: {type_panne}", file=sys.stderr)
                # Utiliser le modèle pour prédire le facteur à partir du type de panne seul
                try:
                    type_panne_enc = self.type_encoder.transform([type_panne])[0]
                    input_data = pd.DataFrame({
                        'type_panne': [type_panne_enc],
                        'PB': ['UNKNOWN'],
                        'FC': ['UNKNOWN'],
                        'oil_level': [0],
                        'downtime': [0],
                        'type_lubrification': [''],
                        'vibration': [''],
                        'power_alimentation': [''],
                        'maintenance_frequency': [''],
                        'seniority': [0]
                    })
                    pred = self.model.predict(input_data)[0]
                    facteur = self.facteur_encoder.inverse_transform([pred])[0]
                    
                    # Ajouter l'importance des features même en cas de prédiction sans correspondance
                    feature_impacts = []
                    for feature_info in self.feature_importances_:
                        feature_impacts.append({
                            "feature": feature_info["feature"],
                            "importance": feature_info["importance"],
                            "impact": feature_info["impact"],
                            "contribution": feature_info["contribution"]
                        })
                    
                    # Trier les impacts
                    feature_impacts.sort(key=lambda x: x['importance'], reverse=True)
                    
                    return {
                        "type_panne": type_panne,
                        "facteur_principal": facteur,
                        "facteurs_possibles": [facteur],
                        "PB": "UNKNOWN",
                        "FC": "UNKNOWN",
                        "oil_level": 0,
                        "downtime": 0,
                        "recommended_action": "",
                        "type_lubrification": "",
                        "vibration": "",
                        "power_alimentation": "",
                        "maintenance_frequency": "",
                        "seniority": 0,
                        "feature_importance": feature_impacts
                    }
                except Exception as e:
                    print(f"[ERROR] Erreur lors de la prédiction du modèle: {str(e)}", file=sys.stderr)
                    return {
                        "type_panne": type_panne,
                        "facteur_principal": "Inconnu",
                        "facteurs_possibles": [],
                        "PB": "UNKNOWN",
                        "FC": "UNKNOWN",
                        "oil_level": 0,
                        "downtime": 0,
                        "recommended_action": "",
                        "type_lubrification": "",
                        "vibration": "",
                        "power_alimentation": "",
                        "maintenance_frequency": "",
                        "seniority": 0,
                        "feature_importance": []  # Liste vide en cas d'erreur
                    }

        except Exception as e:
            print(f"[ERREUR] Prédiction échouée : {str(e)}", file=sys.stderr)
            return {
                "type_panne": type_panne,
                "facteur_principal": "Inconnu",
                "facteurs_possibles": [],
                "PB": "UNKNOWN",
                "FC": "UNKNOWN",
                "oil_level": 0,
                "downtime": 0,
                "recommended_action": "",
                "type_lubrification": "",
                "vibration": "",
                "power_alimentation": "",
                "maintenance_frequency": "",
                "seniority": 0
            }

    def _train_model(self):
        X, y = self._prepare_features()
        
        # Vérification de la distribution des classes
        print("\n[INFO] Distribution des classes:", file=sys.stderr)
        unique_counts = pd.Series(y).value_counts()
        for label, count in unique_counts.items():
            facteur = self.facteur_encoder.inverse_transform([label])[0]
            print(f"[INFO] {facteur}: {count} échantillons", file=sys.stderr)
        
        # Filtrer les classes avec trop peu d'échantillons
        min_samples = 2
        valid_classes = unique_counts[unique_counts >= min_samples].index
        mask = pd.Series(y).isin(valid_classes)
        
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        print(f"\n[INFO] Après filtrage (minimum {min_samples} échantillons par classe):", file=sys.stderr)
        print(f"[INFO] Nombre d'échantillons: {len(X_filtered)}", file=sys.stderr)
        print(f"[INFO] Nombre de classes: {len(valid_classes)}", file=sys.stderr)
        
        # Division des données avec stratification si possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered,
                test_size=0.2,
                random_state=42,
                stratify=y_filtered
            )
        except ValueError as e:
            print(f"[WARNING] Impossible d'utiliser la stratification: {str(e)}", file=sys.stderr)
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered,
                test_size=0.2,
                random_state=42
            )
#
        # Amélioration du modèle avec plus de paramètres
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        # Entraînement du modèle
        self.model.fit(X_train, y_train)
       
       
        # Calculer les importances des caractéristiques
        # Analyse détaillée des caractéristiques
        feature_importance = self.model.feature_importances_
        # selection des features
        features = [
            'downtime',
            'oil_level',
            'type_lubrification',
            'vibration',
            'power_alimentation',
            'maintenance_frequency',
            'seniority'
        ]
        
        print("\n[INFO] Importances des features calculees par le modele:", file=sys.stderr)
        # Calculer les pourcentages d'importance
        total_importance = sum(feature_importance[3:])  # Ignorer les 3 premières features
        feature_importances = {}
        
        # Convertir les importances en pourcentages et les stocker
        for feature, importance in zip(features, feature_importance[3:]):  # Commencer à partir de la 4ème feature
            # Calculer le pourcentage d'importance
            # Stocker les pourcentages d'importance
            percentage = (importance / total_importance) * 100
            feature_importances[feature] = round(percentage, 2)
            print(f"[INFO] {feature}: {percentage:.2f}%", file=sys.stderr)
            
        # Créer la liste des importances avec les impacts
        self.feature_importances_ = []
        # Ajouter les importances avec les impacts
        for feature, importance in feature_importances.items():
            # Categorisation de l'impact en fonction de l'importance
            impact = "Très fort" if importance > 25 else \
                    "Fort" if importance > 20 else \
                    "Moyen" if importance > 10 else "Faible"
            
            description = {
                'downtime': "Temps d'arrêt",
                'oil_level': "Niveau d'huile",
                'type_lubrification': "Type de lubrification utilisé",
                'vibration': "Niveau de vibration",
                'power_alimentation': "Alimentation électrique",
                'maintenance_frequency': "Fréquence de maintenance",
                'seniority': "Ancienneté de l'équipement"
            }
            
            self.feature_importances_.append({
                "feature": feature,
                "importance": importance,
                "impact": impact,
                "contribution": f"{importance:.1f}%",
                "description": description[feature]
            })
        
        # Trier par importance décroissante
        self.feature_importances_ = sorted(self.feature_importances_, 
                                         key=lambda x: x['importance'], 
                                         reverse=True)

    def predict_factors(self, type_panne):
        try:
            print(f"[DEBUG] Recherche du type de panne: {type_panne}", file=sys.stderr)
            
            type_panne_normalized = type_panne.upper()
            print(f"[DEBUG] Type de panne normalisé: {type_panne_normalized}", file=sys.stderr)
            
            panne_match = self.df[self.df['type_panne'].str.upper() == type_panne_normalized]
            
            if not panne_match.empty:
                panne_data = panne_match.iloc[0]
                
                # Préparer les données pour la prédiction
                type_panne_enc = self.type_encoder.transform([type_panne_normalized])[0]
                
                # Log des valeurs brutes avant traitement
                print(f"\n[INFO] Données complètes de la panne:", file=sys.stderr)
                print(f"Type de panne: {panne_data['type_panne']}", file=sys.stderr)
                print(f"Action recommandée: {panne_data['recommended_action']}", file=sys.stderr)
                print(f"Type de lubrification: {panne_data['type_lubrification']}", file=sys.stderr)
                print(f"Vibration: {panne_data['vibration']}", file=sys.stderr)
                print(f"Alimentation: {panne_data['power_alimentation']}", file=sys.stderr)
                print(f"Fréquence de maintenance: {panne_data['maintenance_frequency']}", file=sys.stderr)
                print(f"Ancienneté: {panne_data['seniority']}", file=sys.stderr)
                print(f"PB: {panne_data.get('PB')}", file=sys.stderr)
                print(f"FC: {panne_data.get('FC')}", file=sys.stderr)
                
                input_features = {
                    'type_panne': [type_panne_enc],
                    'PB': [str(panne_data.get('PB', 'UNKNOWN'))],
                    'FC': [str(panne_data.get('FC', 'UNKNOWN'))],
                    'oil_level': [float(panne_data.get('oil_level', 0))],
                    'downtime': [float(panne_data.get('downtime', 0))]
                }
                
                # Prédiction
                input_data = pd.DataFrame(input_features)
                pred = self.model.predict(input_data)[0]
                facteur = self.facteur_encoder.inverse_transform([pred])[0]
                
                # Utiliser directement les importances pré-calculées
                result = {
                    "type_panne": type_panne,
                    "facteur_principal": facteur,
                    "facteurs_possibles": panne_match['facteur'].unique().tolist(),
                    "PB": str(panne_data.get('PB', 'UNKNOWN')),
                    "FC": str(panne_data.get('FC', 'UNKNOWN')),
                    "oil_level": float(panne_data.get('oil_level', 0)),
                    "downtime": float(panne_data.get('downtime', 0)),
                    "recommended_action": str(panne_data.get('recommended_action', '')),
                    "type_lubrification": str(panne_data.get('type_lubrification', '')),
                    "vibration": str(panne_data.get('vibration', '')),
                    "power_alimentation": str(panne_data.get('power_alimentation', '')),
                    "maintenance_frequency": str(panne_data.get('maintenance_frequency', '')),
                    "seniority": int(panne_data.get('seniority', 0)),
                    "feature_importance": sorted(self.feature_importances_, key=lambda x: x['importance'], reverse=True)
                }
                
                return result
            
            else:
                print(f"[WARNING] Aucune correspondance trouvée pour le type de panne: {type_panne}", file=sys.stderr)
                # Utiliser le modèle pour prédire le facteur à partir du type de panne seul
                try:
                    type_panne_enc = self.type_encoder.transform([type_panne])[0]
                    input_data = pd.DataFrame({
                        'type_panne': [type_panne_enc],
                        'PB': ['UNKNOWN'],
                        'FC': ['UNKNOWN'],
                        'oil_level': [0],
                        'downtime': [0],
                        'type_lubrification': [''],
                        'vibration': [''],
                        'power_alimentation': [''],
                        'maintenance_frequency': [''],
                        'seniority': [0]
                    })
                    pred = self.model.predict(input_data)[0]
                    facteur = self.facteur_encoder.inverse_transform([pred])[0]
                    
                    # Ajouter l'importance des features même en cas de prédiction sans correspondance
                    feature_impacts = []
                    for feature, importance in self.feature_importances_.items():
                        impact = "Très fort" if importance > 0.3 else \
                                "Fort" if importance > 0.2 else \
                                "Moyen" if importance > 0.1 else "Faible"
                        feature_impacts.append({
                            "feature": feature,
                            "importance": round(importance * 100, 2),
                            "impact": impact,
                            "contribution": f"{importance * 100:.1f}%"
                        })
                    
                    # Trier les impacts
                    feature_impacts.sort(key=lambda x: x['importance'], reverse=True)
                    
                    return {
                        "type_panne": type_panne,
                        "facteur_principal": facteur,
                        "facteurs_possibles": [facteur],
                        "PB": "UNKNOWN",
                        "FC": "UNKNOWN",
                        "oil_level": 0,
                        "downtime": 0,
                        "recommended_action": "",
                        "type_lubrification": "",
                        "vibration": "",
                        "power_alimentation": "",
                        "maintenance_frequency": "",
                        "seniority": 0,
                        "feature_importance": feature_impacts
                    }
                except Exception as e:
                    print(f"[ERROR] Erreur lors de la prédiction du modèle: {str(e)}", file=sys.stderr)
                    return {
                        "type_panne": type_panne,
                        "facteur_principal": "Inconnu",
                        "facteurs_possibles": [],
                        "PB": "UNKNOWN",
                        "FC": "UNKNOWN",
                        "oil_level": 0,
                        "downtime": 0,
                        "recommended_action": "",
                        "type_lubrification": "",
                        "vibration": "",
                        "power_alimentation": "",
                        "maintenance_frequency": "",
                        "seniority": 0,
                        "feature_importance": []  # Liste vide en cas d'erreur
                    }

        except Exception as e:
            print(f"[ERREUR] Prédiction échouée : {str(e)}", file=sys.stderr)
            return {
                "type_panne": type_panne,
                "facteur_principal": "Inconnu",
                "facteurs_possibles": [],
                "PB": "UNKNOWN",
                "FC": "UNKNOWN",
                "oil_level": 0,
                "downtime": 0,
                "recommended_action": "",
                "type_lubrification": "",
                "vibration": "",
                "power_alimentation": "",
                "maintenance_frequency": "",
                "seniority": 0
            }

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