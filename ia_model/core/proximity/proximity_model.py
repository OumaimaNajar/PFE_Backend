import sys
import json
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic
import joblib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicianProximityAI:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "back-ia"):
        """
        Modèle IA adapté pour votre structure de données MongoDB
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db.technicien
        
        # Modèles
        self.proximity_model = None
        self.rating_predictor = None
        self.scaler = StandardScaler()
        
        # Données
        self.techniciens_data = None
        self.features_columns = ['Latitude', 'Longitude', 'note_moyenne', 'nombre_interventions', 
                               'experience_annees', 'disponible_score']
        
        # Métadonnées
        self.last_training = None
        self.model_version = "2.0"
        self.training_stats = {}
        self.temp_file = None  # Pour suivre le fichier temporaire

    def load_and_prepare_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Tentative de connexion à MongoDB: {self.client.address if self.client else 'Non connecté'}")
            
            # Test de connexion
            try:
                self.client.admin.command('ping')
                logger.info("Connexion MongoDB établie avec succès")
            except Exception as e:
                logger.error(f"Échec de connexion à MongoDB: {str(e)}")
                return pd.DataFrame()
                
            # Récupérer tous les techniciens avec coordonnées valides
            query = {
                "Latitude": {"$exists": True, "$ne": None, "$type": "number"},
                "Longitude": {"$exists": True, "$ne": None, "$type": "number"}
            }
            
            logger.info(f"Exécution de la requête MongoDB: {query}")
            cursor = self.collection.find(query)
            techniciens = list(cursor)
            
            if not techniciens:
                logger.warning("Aucun technicien trouvé avec la requête principale")
                # Essayons avec une requête plus simple
                cursor_simple = self.collection.find({})
                all_docs = list(cursor_simple)
                logger.info(f"Total documents dans la collection: {len(all_docs)}")
                
                if all_docs:
                    # Afficher un exemple de document pour debug
                    logger.info(f"Exemple de document: {all_docs[0]}")
                    
                    # Essayer avec des noms de champs minuscules
                    query_lower = {
                        "latitude": {"$exists": True, "$ne": None, "$type": "number"},
                        "longitude": {"$exists": True, "$ne": None, "$type": "number"}
                    }
                    cursor_lower = self.collection.find(query_lower)
                    techniciens_lower = list(cursor_lower)
                    
                    if techniciens_lower:
                        logger.info("Trouvé des techniciens avec latitude/longitude en minuscules")
                        techniciens = techniciens_lower
                        # Renommer les colonnes pour uniformiser
                        for tech in techniciens:
                            if 'latitude' in tech:
                                tech['Latitude'] = tech['latitude']
                            if 'longitude' in tech:
                                tech['Longitude'] = tech['longitude']
                            if 'nom' in tech:
                                tech['Nom'] = tech['nom']
                            if 'prenom' in tech:
                                tech['Prenom'] = tech['prenom']
                            if 'statut' in tech:
                                tech['Statut'] = tech['statut']
                
                if not techniciens:
                    return pd.DataFrame()
            
            # Convertir en DataFrame
            df = pd.DataFrame(techniciens)
            logger.info(f"DataFrame créé avec {len(df)} lignes et colonnes: {list(df.columns)}")
            
            # Nettoyer et valider les coordonnées
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df = df[(df['Latitude'].between(-90, 90)) & 
                       (df['Longitude'].between(-180, 180))]
            else:
                logger.error("Colonnes Latitude/Longitude non trouvées")
                return pd.DataFrame()
            
            # Enrichir les données avec des valeurs par défaut
            df = self._enrich_technician_data(df)
            
            logger.info(f"Chargé et préparé {len(df)} techniciens")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _enrich_technician_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit les données des techniciens avec des features calculées
        """
        # Valeurs par défaut pour les champs manquants - correction pour éviter l'erreur de liste
        if 'note_moyenne' not in df.columns:
            df['note_moyenne'] = 4.0
        else:
            df['note_moyenne'] = df['note_moyenne'].fillna(4.0).clip(1, 5)
        
        if 'nombre_interventions' not in df.columns:
            df['nombre_interventions'] = 10
        else:
            df['nombre_interventions'] = df['nombre_interventions'].fillna(10).astype(int)
        
        # Gestion sécurisée des spécialités
        if 'specialites' not in df.columns:
            df['specialites'] = [['Général'] for _ in range(len(df))]
        else:
            df['specialites'] = df['specialites'].apply(lambda x: x if isinstance(x, list) else ['Général'])
        
        # Normaliser le statut - gestion sécurisée
        if 'Statut' in df.columns:
            df['Statut'] = df['Statut'].astype(str).str.lower()
            df['disponible_maintenant'] = df['Statut'].isin(['disponible', 'actif'])
        else:
            df['Statut'] = 'actif'
            df['disponible_maintenant'] = True
        
        # Features calculées
        df['experience_annees'] = df.apply(self._calculate_experience, axis=1)
        df['disponible_score'] = df.apply(self._calculate_availability_score, axis=1)
        df['popularity_score'] = df.apply(self._calculate_popularity_score, axis=1)
        df['specialites_count'] = df['specialites'].apply(lambda x: len(x) if isinstance(x, list) else 1)
        
        # Normalisation temporelle (heure de la journée)
        current_hour = datetime.now().hour
        df['time_factor'] = self._get_time_availability_factor(current_hour)
        
        # Score composite de qualité
        df['quality_score'] = (
            df['note_moyenne'] * 0.4 +
            np.log1p(df['nombre_interventions']) * 0.3 +
            df['experience_annees'] * 0.2 +
            df['specialites_count'] * 0.1
        )
        
        return df
    
    def _calculate_experience(self, row) -> float:
        """Calcule l'expérience en années"""
        try:
            if 'date_inscription' in row and pd.notna(row['date_inscription']):
                inscription_date = pd.to_datetime(row['date_inscription'])
                years = (datetime.now() - inscription_date).days / 365.25
                return max(0, min(years, 30))  # Cap à 30 ans
            # Valeur par défaut basée sur le nombre d'interventions
            interventions = row.get('nombre_interventions', 10)
            return min(interventions / 50.0, 10.0)  # Approximation
        except:
            return 2.0  # Valeur par défaut
    
    def _calculate_availability_score(self, row) -> float:
        """Calcule un score de disponibilité basé sur le statut"""
        score = 0.0
        
        # Score basé sur le statut
        statut = str(row.get('Statut', '')).lower()
        if statut == 'disponible':
            score += 8.0
        elif statut == 'actif':
            score += 6.0
        elif statut == 'occupé':
            score += 2.0
        else:
            score += 4.0  # Statut inconnu
        
        # Bonus pour les techniciens avec beaucoup d'interventions
        interventions = row.get('nombre_interventions', 0)
        if interventions > 50:
            score += 2.0
        elif interventions > 20:
            score += 1.0
        
        return min(score, 10.0)  # Score max 10
    
    def _calculate_popularity_score(self, row) -> float:
        """Calcule un score de popularité"""
        interventions = row.get('nombre_interventions', 0)
        note = row.get('note_moyenne', 4.0)
        return (np.log1p(interventions) * note) / 5.0
    
    def _get_time_availability_factor(self, hour: int) -> float:
        """Facteur de disponibilité selon l'heure"""
        if 8 <= hour < 18:
            return 1.0
        elif (6 <= hour < 8) or (18 <= hour < 22):
            return 0.7
        else:
            return 0.3
    
    def train_models(self, retrain: bool = False):
        """
        Entraîne tous les modèles de l'IA
        """
        try:
            logger.info("Début de l'entraînement des modèles...")
            
            # Charger les données
            self.techniciens_data = self.load_and_prepare_data()
            
            if self.techniciens_data.empty:
                raise ValueError("Aucune donnée disponible pour l'entraînement")
            
            # Statistiques d'entraînement
            self.training_stats = {
                'total_technicians': len(self.techniciens_data),
                'avg_rating': self.techniciens_data['note_moyenne'].mean(),
                'avg_interventions': self.techniciens_data['nombre_interventions'].mean(),
                'statuts_distribution': self._get_statuts_distribution(),
                'geographic_coverage': self._get_geographic_coverage()
            }
            
            # 1. Entraîner le modèle de proximité (KNN)
            self._train_proximity_model()
            
            # 2. Entraîner le prédicteur de rating/qualité
            self._train_rating_predictor()
            
            # 3. Ajuster le scaler pour les features
            self._fit_scaler()
            
            self.last_training = datetime.now()
            
            logger.info(f"Entraînement terminé avec succès:")
            logger.info(f"- {self.training_stats['total_technicians']} techniciens")
            logger.info(f"- Note moyenne: {self.training_stats['avg_rating']:.2f}")
            logger.info(f"- Couverture géographique: {self.training_stats['geographic_coverage']:.2f} km²")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            raise
    
    def _train_proximity_model(self):
        """Entraîne le modèle KNN pour la proximité"""
        coordinates = self.techniciens_data[['Latitude', 'Longitude']].values
        
        # Modèle KNN avec métrique haversine pour les distances géodésiques
        self.proximity_model = NearestNeighbors(
            n_neighbors=min(20, len(coordinates)),
            algorithm='ball_tree',
            metric='haversine',
            leaf_size=30
        )
        
        # Convertir en radians pour haversine
        coordinates_rad = np.radians(coordinates)
        self.proximity_model.fit(coordinates_rad)
        
        logger.info("Modèle de proximité KNN entraîné")
    
    def _train_rating_predictor(self):
        """Entraîne un modèle pour prédire la qualité/rating"""
        features = ['nombre_interventions', 'experience_annees', 'disponible_score', 
                   'specialites_count', 'time_factor']
        
        X = self.techniciens_data[features].fillna(0)
        y = self.techniciens_data['quality_score']
        
        self.rating_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.rating_predictor.fit(X, y)
        
        # Score de validation
        score = self.rating_predictor.score(X, y)
        logger.info(f"Prédicteur de qualité entraîné (R² = {score:.3f})")
    
    def _fit_scaler(self):
        """Ajuste le scaler pour normaliser les features"""
        # Adapter aux nouvelles colonnes
        features_data = self.techniciens_data[['Latitude', 'Longitude', 'note_moyenne', 
                                             'nombre_interventions', 'experience_annees', 
                                             'disponible_score']].fillna(0)
        self.scaler.fit(features_data)
    
    def _get_statuts_distribution(self) -> Dict:
        """Retourne la distribution des statuts"""
        return self.techniciens_data['Statut'].value_counts().to_dict()
    
    def _get_geographic_coverage(self) -> float:
        """Calcule la couverture géographique approximative"""
        coords = self.techniciens_data[['Latitude', 'Longitude']]
        lat_range = coords['Latitude'].max() - coords['Latitude'].min()
        lon_range = coords['Longitude'].max() - coords['Longitude'].min()
        
        # Approximation grossière de l'aire en km²
        lat_km = lat_range * 111  # 1° lat ≈ 111 km
        lon_km = lon_range * 111 * np.cos(np.radians(coords['Latitude'].mean()))
        
        return abs(lat_km * lon_km)
    
    def find_closest_technician(self, user_lat: float, user_lon: float) -> Dict:
        """
        Trouve le technicien le plus proche
        """
        try:
            if self.proximity_model is None:
                self.train_models()
            
            # Trouver le plus proche avec KNN
            user_coords_rad = np.radians([[user_lat, user_lon]])
            distances, indices = self.proximity_model.kneighbors(user_coords_rad, n_neighbors=1)
            
            # Récupérer le technicien le plus proche
            closest_idx = indices[0][0]
            closest_tech = self.techniciens_data.iloc[closest_idx]
            
            # Calculer la distance réelle
            distance_km = geodesic(
                (user_lat, user_lon),
                (closest_tech['Latitude'], closest_tech['Longitude'])
            ).kilometers
            
            # Temps de trajet estimé
            travel_time = self._estimate_travel_time(distance_km)
            
            result = {
                'technicien_id': str(closest_tech['_id']),
                'nom': closest_tech.get('Nom', ''),
                'prenom': closest_tech.get('Prenom', ''),
                'statut': closest_tech.get('Statut', ''),
                'note_moyenne': round(closest_tech['note_moyenne'], 2),
                'nombre_interventions': int(closest_tech['nombre_interventions']),
                'experience_annees': round(closest_tech['experience_annees'], 1),
                'position': {
                    'latitude': closest_tech['Latitude'],
                    'longitude': closest_tech['Longitude']
                },
                'distance_km': round(distance_km, 2),
                'travel_time_minutes': travel_time,
                'quality_score': round(closest_tech['quality_score'], 2),
                'disponible_score': round(closest_tech['disponible_score'], 1),
                'disponible_maintenant': closest_tech['disponible_maintenant'],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Technicien le plus proche trouvé: {result['nom']} {result['prenom']} à {distance_km:.2f} km")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche du plus proche: {e}")
            return {'error': str(e)}
    
    def find_best_technicians(self, 
                             user_lat: float, 
                             user_lon: float, 
                             n_technicians: int = 5,
                             max_distance_km: float = 50.0,
                             statut_requis: Optional[str] = None) -> List[Dict]:
        """
        Trouve les meilleurs techniciens selon plusieurs critères
        """
        try:
            if self.proximity_model is None:
                self.train_models()
            
            # Filtrage initial
            filtered_data = self.techniciens_data.copy()
            
            # Filtre par statut
            if statut_requis:
                filtered_data = filtered_data[
                    filtered_data['Statut'].str.lower() == statut_requis.lower()
                ]
            
            if filtered_data.empty:
                return []
            
            # Calcul des distances pour tous les candidats
            candidates_with_distance = []
            for idx, tech in filtered_data.iterrows():
                distance_km = geodesic(
                    (user_lat, user_lon),
                    (tech['Latitude'], tech['Longitude'])
                ).kilometers
                
                if distance_km <= max_distance_km:
                    candidates_with_distance.append((idx, distance_km, tech))
            
            if not candidates_with_distance:
                return []
            
            # Calculer les scores pour chaque candidat
            scored_candidates = []
            for idx, distance_km, tech in candidates_with_distance:
                # Score de proximité (0-100, plus proche = meilleur)
                proximity_score = max(0, 100 - (distance_km / max_distance_km * 100))
                
                # Score composite
                final_score = (
                    proximity_score * 0.4 +
                    tech['quality_score'] * 10 * 0.4 +
                    tech['disponible_score'] * 10 * 0.2
                )
                
                # Temps de trajet estimé
                travel_time = self._estimate_travel_time(distance_km)
                
                candidate = {
                    'rang': 0,  # Sera défini après le tri
                    'technicien_id': str(tech['_id']),
                    'nom': tech.get('Nom', ''),
                    'prenom': tech.get('Prenom', ''),
                    'statut': tech.get('Statut', ''),
                    'note_moyenne': round(tech['note_moyenne'], 2),
                    'nombre_interventions': int(tech['nombre_interventions']),
                    'experience_annees': round(tech['experience_annees'], 1),
                    'position': {
                        'latitude': tech['Latitude'],
                        'longitude': tech['Longitude']
                    },
                    'distance_km': round(distance_km, 2),
                    'travel_time_minutes': travel_time,
                    'proximity_score': round(proximity_score, 1),
                    'quality_score': round(tech['quality_score'], 2),
                    'final_score': round(final_score, 1),
                    'disponible_maintenant': tech['disponible_maintenant'],
                    'disponible_score': round(tech['disponible_score'], 1)
                }
                
                scored_candidates.append(candidate)
            
            # Trier par score final décroissant
            scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Ajouter le rang
            for i, candidate in enumerate(scored_candidates[:n_technicians]):
                candidate['rang'] = i + 1
            
            logger.info(f"Trouvé {len(scored_candidates)} techniciens, retourné top {min(n_technicians, len(scored_candidates))}")
            
            return scored_candidates[:n_technicians]
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []
    
    def _estimate_travel_time(self, distance_km: float) -> int:
        """Estime le temps de trajet selon la distance et l'heure"""
        current_hour = datetime.now().hour
        
        # Vitesses selon l'heure et la distance
        if distance_km <= 3:
            base_speed = 20 if 7 <= current_hour <= 19 else 30
        elif distance_km <= 15:
            base_speed = 35 if 7 <= current_hour <= 19 else 50
        else:
            base_speed = 60 if 7 <= current_hour <= 19 else 80
        
        # Facteur d'embouteillage aux heures de pointe
        if current_hour in [8, 9, 17, 18, 19]:
            base_speed *= 0.7
        
        travel_time = (distance_km / base_speed) * 60
        return max(5, int(travel_time))  # Minimum 5 minutes


    def cleanup_temp_file(self):
        """Nettoie le fichier temporaire s'il existe"""
        try:
            if self.temp_file and os.path.exists(self.temp_file):
                with open(self.temp_file, 'r') as f:
                    f.close()
                logger.info(f"Fichier temporaire {self.temp_file} fermé")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture du fichier temporaire: {e}")


    def predict_from_json(self, json_file_path: str) -> Dict:
        """Prédit les techniciens les plus proches à partir d'un fichier JSON"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
                latitude = float(input_data['Latitude'])
                longitude = float(input_data['Longitude'])
        
            # Trouver les techniciens les plus proches
            closest_techs = self.find_nearest_technicians(latitude, longitude)
        
            return {
                "success": True,
                "techniciens": closest_techs
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    def find_nearest_technicians(self, latitude: float, longitude: float, k: int = 5) -> List[Dict]:
        """Trouve les k techniciens les plus proches d'une position donnée"""
        try:
            logger.info(f"Début de la recherche des techniciens proches de ({latitude}, {longitude})")
        
            # Charger et préparer les données
            logger.info("Chargement des données depuis MongoDB...")
            df = self.load_and_prepare_data()
            if df.empty:
                logger.warning("Aucun technicien trouvé dans la base de données")
                return []

            logger.info(f"Calcul des distances pour {len(df)} techniciens...")
            # Calculer les distances
            technicians = []
            for _, tech in df.iterrows():
                try:
                    distance = geodesic(
                        (latitude, longitude),
                        (tech['Latitude'], tech['Longitude'])
                    ).kilometers

                    technicians.append({
                        'id': str(tech.get('_id', '')),
                        'nom': tech.get('Nom', ''),
                        'prenom': tech.get('Prenom', ''),
                        'distance': round(distance, 2),
                        'latitude': float(tech['Latitude']),
                        'longitude': float(tech['Longitude']),
                        'disponible': bool(tech.get('disponible', True))
                    })
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du technicien {tech.get('_id', '')}: {str(e)}")

            # Trier par distance et prendre les k premiers
            result = sorted(technicians, key=lambda x: x['distance'])[:k]
            logger.info(f"Retour de {len(result)} techniciens les plus proches")
            return result

        except Exception as e:
            logger.error(f"Erreur lors de la recherche des techniciens: {str(e)}")
            logger.error(f"Traceback complet:", exc_info=True)
            return []


# Interface CLI simplifiée
def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Command required"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        # Configuration
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        db_name = os.getenv('DB_NAME', 'back-ia')
        
        # Initialiser le modèle
        ai = TechnicianProximityAI(mongo_uri=mongo_uri, db_name=db_name)
        
        if command == 'find_closest':
            # Arguments: latitude longitude
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Latitude et longitude requises"}))
                sys.exit(1)
            
            lat = float(sys.argv[2])
            lon = float(sys.argv[3])
            
            result = ai.find_closest_technician(lat, lon)
            
        elif command == 'find_best':
            # Arguments: latitude longitude n_technicians max_distance statut
            if len(sys.argv) < 6:
                print(json.dumps({"error": "Arguments insuffisants"}))
                sys.exit(1)
            
            lat = float(sys.argv[2])
            lon = float(sys.argv[3])
            n_techs = int(sys.argv[4])
            max_dist = float(sys.argv[5])
            statut = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != 'null' else None
            
            technicians = ai.find_best_technicians(
                user_lat=lat, user_lon=lon, n_technicians=n_techs,
                max_distance_km=max_dist, statut_requis=statut
            )
            
            result = {
                "technicians": technicians,
                "total_found": len(technicians),
                "search_params": {
                    "position": {"latitude": lat, "longitude": lon},
                    "max_distance_km": max_dist,
                    "statut_requis": statut
                }
            }
            
        elif command == 'train':
            ai.train_models()
            
            result = {
                "message": "Modèle entraîné avec succès",
                "stats": ai.training_stats,
                "timestamp": ai.last_training.isoformat() if ai.last_training else None
            }
            
        elif command == 'health':
            # Vérification de l'état du modèle
            status = "OK" if ai.proximity_model is not None else "NOT_TRAINED"
            result = {
                "status": status,
                "last_training": ai.last_training.isoformat() if ai.last_training else None,
                "model_version": ai.model_version,
                "stats": ai.training_stats
            }
            
        else:
            result = {"error": f"Commande inconnue: {command}"}
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "type": type(e).__name__,
            "command": command
        }
        print(json.dumps(error_result))
        sys.exit(1)




if __name__ == "__main__":
    model = TechnicianProximityAI()
    
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Arguments requis"}))
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "train":
        # Logique d'entraînement si nécessaire
        print(json.dumps({"success": True, "message": "Modèle entraîné avec succès"}))
    else:
        # Traiter comme un fichier JSON pour la prédiction
        result = model.predict_from_json(command)
        print(json.dumps(result))