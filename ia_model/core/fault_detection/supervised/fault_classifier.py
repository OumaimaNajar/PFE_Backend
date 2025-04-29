import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class FaultTypeClassifier:
    def __init__(self):
        self.fault_types = {}
        self.df = None
        self.load_fault_types_from_csv()
        
    def load_fault_types_from_csv(self):
        """Charge les types de pannes depuis le fichier CSV"""
        try:
            # Chemin relatif vers le fichier CSV
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 
                                   'data', 'pannes_industrielles_organisees.csv')
            
            # Chargement des données
            self.df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
            
            # Vérifier les colonnes du CSV et ajouter des colonnes manquantes si nécessaire
            required_columns = ['type_panne', 'machine', 'description', 'gravite', 
                               'action_recommandee', 'action_secondaire', 'code_probleme', 'code_defaillance']
            
            for col in required_columns:
                if col not in self.df.columns:
                    print(f"Colonne manquante dans le CSV: {col}. Ajout d'une colonne vide.")
                    self.df[col] = None
            
            # Création d'un dictionnaire de types de pannes basé sur les données CSV
            fault_types_dict = {}
            
            # Regrouper par type de panne
            for type_panne, group in self.df.groupby('type_panne'):
                if pd.isna(type_panne):
                    continue
                    
                # Extraire les machines associées à ce type de panne
                machines = group['machine'].unique().tolist()
                
                # Extraire les mots-clés à partir du type de panne
                keywords = [word.lower() for word in type_panne.split() if len(word) > 3]
                
                # Ajouter des mots-clés supplémentaires basés sur la description
                for desc in group['description'].dropna().unique():
                    keywords.extend([word.lower() for word in str(desc).split() if len(word) > 3])
                
                # Ajouter des mots-clés spécifiques aux pompes si le type contient "pompe"
                if 'pump' in type_panne.lower() or 'pompe' in type_panne.lower():
                    keywords.extend(['condensate', 'return', 'pump', 'quarterly', 'service'])
                
                # Déterminer le niveau de confiance en fonction de la gravité moyenne
                gravite_map = {'Faible': 0.1, 'Moyenne': 0.2, 'Élevée': 0.3, 'Critique': 0.4}
                gravites = group['gravite'].map(lambda x: gravite_map.get(x, 0.2) if not pd.isna(x) else 0.2)
                confidence_boost = gravites.mean()
                
                # Créer l'entrée pour ce type de panne
                fault_types_dict[type_panne.upper()] = {
                    'keywords': list(set(keywords)),  # Éliminer les doublons
                    'locations': [m.upper() for m in machines if not pd.isna(m)],
                    'confidence_boost': confidence_boost,
                    'actions': self._extract_actions_from_group(group),
                    'codes': self._extract_codes_from_group(group)
                }
            
            self.fault_types = fault_types_dict
            
            # Si aucun type de panne n'a été chargé, lever une exception
            if not self.fault_types:
                raise ValueError("Aucun type de panne n'a été chargé depuis le fichier CSV")
                
        except Exception as e:
            print(f"Erreur lors du chargement du fichier CSV: {str(e)}")
            # Initialiser avec un dictionnaire vide au lieu de valeurs par défaut
            self.fault_types = {}
            print("Aucun type de panne n'a été chargé suite à une erreur")
    
    def _extract_actions_from_group(self, group):
        """Extrait les actions recommandées d'un groupe de pannes"""
        actions = []
        
        # Extraire les actions recommandées
        for action in group['action_recommandee'].dropna().unique():
            actions.append(action)
            
        # Extraire les actions secondaires
        for action in group['action_secondaire'].dropna().unique():
            actions.append(action)
            
        # Éliminer les doublons et limiter à 3 actions maximum
        return list(dict.fromkeys(actions))[:3]
    
    def _extract_codes_from_group(self, group):
        """Extrait les codes Maximo d'un groupe de pannes"""
        # Vérifier si des codes sont disponibles
        if 'code_probleme' in group.columns and 'code_defaillance' in group.columns:
            problem_codes = group['code_probleme'].dropna().unique()
            failure_codes = group['code_defaillance'].dropna().unique()
            
            if len(problem_codes) > 0 and len(failure_codes) > 0:
                return {
                    'problem': problem_codes[0],
                    'failure': failure_codes[0]
                }
        
        # Si aucun code n'est trouvé, générer des codes basés sur le type de panne
        type_panne = group['type_panne'].iloc[0].upper() if not group['type_panne'].empty else ""
        
        if 'MECANIQUE' in type_panne or 'USURE' in type_panne:
            return {'problem': 'PB_MEC', 'failure': 'MECH_FAIL'}
        elif 'ELECTRIQUE' in type_panne:
            return {'problem': 'PB_ELE', 'failure': 'ELEC_FAIL'}
        elif 'HYDRAULIQUE' in type_panne or 'FUITE' in type_panne:
            return {'problem': 'PB_HYD', 'failure': 'HYDR_FAIL'}
        elif 'POMPE' in type_panne or 'PUMP' in type_panne:
            return {'problem': 'PB_PUMP', 'failure': 'PUMP_FAIL'}
        else:
            return {'problem': 'PB_GEN', 'failure': 'GEN_FAIL'}
    
    def predict_fault_type(self, input_data):
        # Vérifier que les types de pannes ont été chargés
        if not self.fault_types:
            # Lever une exception au lieu d'initialiser avec des valeurs par défaut
            raise ValueError("Aucun type de panne n'a été chargé. Impossible de prédire le type de panne.")
            
        # Utiliser à la fois description et Description (avec majuscule)
        description_lower = str(input_data.get('description', '')).lower()
        description_upper = str(input_data.get('Description', '')).lower()
        
        # Combiner les deux descriptions
        description = description_lower + " " + description_upper
        
        location = str(input_data.get('LOCATION', '')).upper()
        
        # Ajouter des informations sur l'équipement
        asset = str(input_data.get('ASSETNUM', '')).upper()
        description += " " + asset
        
        # Si la description est toujours vide après ces tentatives, utiliser une valeur par défaut
        if description.strip() == "":
            # Utiliser l'emplacement et le numéro d'actif pour enrichir la description par défaut
            description = f"maintenance équipement industriel {location} {asset}"
            
            # Ajouter des mots-clés spécifiques si certains modèles sont reconnus
            if "430" in asset or "430" in location:
                description += " pump condensate return service"
        
        # Initialiser les scores avec une valeur minimale pour éviter les résultats nuls
        scores = {}
        for fault_type in self.fault_types:
            scores[fault_type] = 0.1
        
        # Calculer les scores pour chaque type de panne
        for fault_type, config in self.fault_types.items():
            # Keyword matching
            matched_keywords = [k for k in config['keywords'] if k in description]
            if matched_keywords:
                scores[fault_type] += len(matched_keywords) * 0.1
            
            # Location matching
            if location in config['locations']:
                scores[fault_type] += config['confidence_boost']
            elif any(loc in location for loc in config['locations']):
                # Correspondance partielle
                scores[fault_type] += config['confidence_boost'] * 0.5
            
            # Vérifier si l'équipement contient des mots-clés liés à ce type de panne
            if any(keyword in asset.lower() for keyword in config['keywords']):
                scores[fault_type] += 0.15
        
        # Trouver le type de panne le plus probable
        if not scores or max(scores.values()) <= 0.1:
            # Si aucun score significatif n'est trouvé, lever une exception
            raise ValueError("Impossible de déterminer le type de panne avec les données fournies.")
        else:
            # Get the most likely fault type
            predicted_type = max(scores.items(), key=lambda x: x[1])[0]
            confidence = min(0.95, scores[predicted_type] + 0.4)
        
        # Ajout du log ici
        print(f"[LOG] Type de panne prédit : {predicted_type} avec confiance {confidence * 100:.2f}%")
        print(f"[LOG] Description utilisée : {description}")
        print(f"[LOG] Localisation : {location}")
        print(f"[LOG] Asset : {asset}")
        print(f"[LOG] Scores calculés : {scores}")

        # Appel de la méthode pour obtenir les facteurs influençant la panne détectée
        influencing_factors = self.get_influencing_factors(predicted_type)

        return {
            "type": predicted_type,
            "confidence": f"{confidence * 100:.2f}%",
            "maximo_codes": self.get_maximo_codes(predicted_type),
            "matched_patterns": {
                "keywords": [k for k in self.fault_types[predicted_type]['keywords'] 
                           if k in description],
                "location_match": location in self.fault_types[predicted_type]['locations'],
                "description_utilisee": description
            },
            "influencing_factors": influencing_factors  # Add the influencing factors to the output
        }
        
    def get_suggested_actions(self, fault_type):
        """Génère des actions suggérées basées sur le type de panne"""
        # Vérifier si des actions sont déjà stockées pour ce type de panne
        if fault_type in self.fault_types and 'actions' in self.fault_types[fault_type]:
            actions = self.fault_types[fault_type]['actions']
            if actions and len(actions) > 0:
                return actions
        
        # Si aucune action n'est stockée, rechercher dans le CSV
        try:
            # Vérifier si le DataFrame est déjà chargé
            if self.df is None:
                # Chemin relatif vers le fichier CSV
                csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 
                                      'data', 'pannes_industrielles_organisees.csv')
                
                # Vérifier si le fichier existe
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
                    
                # Charger les données
                self.df = pd.read_csv(csv_path)
            
            # Filtrer les actions pour le type de panne spécifié
            actions = []
            
            # Rechercher des correspondances exactes ou partielles
            for index, row in self.df.iterrows():
                type_panne_csv = str(row.get('type_panne', '')).upper()
                
                # Vérifier si le type de panne correspond
                if pd.notna(type_panne_csv) and (type_panne_csv == fault_type or 
                                               type_panne_csv in fault_type or 
                                               fault_type in type_panne_csv):
                    
                    # Extraire les actions recommandées
                    if pd.notna(row.get('action_recommandee')):
                        actions.append(row['action_recommandee'])
                    
                    # Extraire les actions secondaires si disponibles
                    if pd.notna(row.get('action_secondaire')):
                        actions.append(row['action_secondaire'])
            
            # Si aucune action n'a été trouvée, rechercher par mots-clés
            if not actions:
                for index, row in self.df.iterrows():
                    type_panne_csv = str(row.get('type_panne', '')).lower()
                    
                    # Extraire les mots-clés du type de panne
                    keywords = [word for word in fault_type.lower().split() if len(word) > 3]
                    
                    # Vérifier si des mots-clés correspondent
                    if any(keyword in type_panne_csv for keyword in keywords):
                        if pd.notna(row.get('action_recommandee')):
                            actions.append(row['action_recommandee'])
                        
                        if pd.notna(row.get('action_secondaire')):
                            actions.append(row['action_secondaire'])
            
            # Éliminer les doublons et limiter à 3 actions maximum
            actions = list(dict.fromkeys(actions))[:3]
            
            # Si des actions ont été trouvées, les retourner
            if actions:
                return actions
                
            # Si aucune action n'est trouvée, retourner un message générique
            return ["Action non spécifiée dans les données"]
                
        except Exception as e:
            print(f"Erreur lors de la génération des actions suggérées: {str(e)}")
            return ["Erreur lors de la récupération des actions"]
    
    def get_maximo_codes(self, fault_type):
        """Récupère les codes Maximo pour le type de panne"""
        # Vérifier si des codes sont déjà stockés pour ce type de panne
        if fault_type in self.fault_types and 'codes' in self.fault_types[fault_type]:
            codes = self.fault_types[fault_type]['codes']
            if codes and 'problem' in codes and 'failure' in codes:
                return codes
        
        # Si aucun code n'est stocké, rechercher dans le CSV
        try:
            # Vérifier si le DataFrame est déjà chargé
            if self.df is None:
                # Chemin relatif vers le fichier CSV
                csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 
                                      'data', 'pannes_industrielles_organisees.csv')
                
                # Charger les données
                self.df = pd.read_csv(csv_path)
            
            # Rechercher des correspondances pour le type de panne
            for index, row in self.df.iterrows():
                type_panne_csv = str(row.get('type_panne', '')).upper()
                
                # Vérifier si le type de panne correspond
                if pd.notna(type_panne_csv) and (type_panne_csv == fault_type or 
                                               type_panne_csv in fault_type or 
                                               fault_type in type_panne_csv):
                    
                    # Extraire les codes si disponibles
                    problem_code = row.get('code_probleme', 'PB_GEN') if pd.notna(row.get('code_probleme')) else 'PB_GEN'
                    failure_code = row.get('code_defaillance', 'GEN_FAIL') if pd.notna(row.get('code_defaillance')) else 'GEN_FAIL'
                    
                    return {
                        'problem': problem_code,
                        'failure': failure_code
                    }
            
            # Si aucune correspondance n'est trouvée, rechercher par mots-clés
            for index, row in self.df.iterrows():
                type_panne_csv = str(row.get('type_panne', '')).lower()
                
                # Extraire les mots-clés du type de panne
                keywords = [word for word in fault_type.lower().split() if len(word) > 3]
                
                # Vérifier si des mots-clés correspondent
                if any(keyword in type_panne_csv for keyword in keywords):
                    problem_code = row.get('code_probleme', 'PB_GEN') if pd.notna(row.get('code_probleme')) else 'PB_GEN'
                    failure_code = row.get('code_defaillance', 'GEN_FAIL') if pd.notna(row.get('code_defaillance')) else 'GEN_FAIL'
                    
                    return {
                        'problem': problem_code,
                        'failure': failure_code
                    }
            
            # Si aucun code n'est trouvé, retourner des codes génériques
            return {
                'problem': 'PB_GEN',
                'failure': 'GEN_FAIL'
            }
                
        except Exception as e:
            print(f"Erreur lors de la récupération des codes Maximo: {str(e)}")
            return {
                'problem': 'PB_GEN',
                'failure': 'GEN_FAIL'
            }

    def get_influencing_factors(self, fault_type):
        """Retourne les facteurs influençant ce type de panne à partir du CSV facteurs_influencent_pannes.csv"""
        try:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                                   'data', 'facteurs_influencent_pannes.csv')
            if not os.path.exists(csv_path):
                print(f"Fichier des facteurs non trouvé: {csv_path}")
                return []

            print(f"[LOG] Tentative de lecture du fichier CSV des facteurs: {csv_path}")
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
            print("[LOG] Lecture réussie du fichier CSV des facteurs")

            # Vérifier et normaliser les noms de colonnes
            df.columns = [col.strip().upper() for col in df.columns]
            print(f"[LOG] Colonnes disponibles dans le CSV: {df.columns.tolist()}")

            # Vérifier la présence des colonnes requises
            required_columns = ['TYPE_PANNE', 'FACTEUR', 'CATEGORIE', 'IMPACT', 'DESCRIPTION']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"[LOG] Colonnes manquantes dans le CSV: {', '.join(missing_columns)}")
                return []

            # Normalisation pour la comparaison
            def normalize(s):
                import unicodedata
                return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn').upper().strip()

            # Recherche plus flexible des facteurs
            fault_type_norm = normalize(fault_type)
            
            # Recherche par correspondance exacte d'abord
            facteurs = df[df['TYPE_PANNE'].apply(lambda x: normalize(str(x)) == fault_type_norm)]
            
            # Si aucun résultat, essayer une correspondance partielle
            if facteurs.empty:
                facteurs = df[df['TYPE_PANNE'].apply(lambda x: any(
                    normalize(kw) in normalize(str(x)) 
                    for kw in fault_type_norm.split()
                    if len(kw) > 3
                ))]

            if not facteurs.empty:
                print(f"[LOG] {len(facteurs)} facteurs trouvés pour le type de panne: {fault_type}")
                result = []
                for _, row in facteurs.iterrows():
                    factor = {}
                    for col in ['FACTEUR', 'CATEGORIE', 'IMPACT', 'DESCRIPTION']:
                        try:
                            value = str(row[col]).strip() if pd.notna(row[col]) else ""
                            factor[col.lower()] = value
                        except Exception as e:
                            print(f"[ERREUR] Erreur lors de l'accès à la colonne {col}: {str(e)}")
                            factor[col.lower()] = ""
                    result.append(factor)
                return result

            print(f"[LOG] Aucun facteur trouvé pour le type de panne: {fault_type}")
            return []

        except Exception as e:
            print(f"[ERREUR] Exception lors de la lecture des facteurs : {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback complet : {traceback.format_exc()}")
            return []

    def calculate_accuracy(self, test_data=None, num_samples=50, min_description_length=20):
        """
        Calcule et affiche la précision du modèle sur un ensemble de données de test.
        
        Args:
            test_data (list, optional): Liste de dictionnaires contenant les données de test.
                Si None, les données seront extraites du CSV.
            num_samples (int, optional): Nombre d'échantillons à utiliser si test_data est None.
            min_description_length (int, optional): Longueur minimale des descriptions à considérer.
            
        Returns:
            dict: Dictionnaire contenant les métriques de performance
        """
        try:
            # Si aucune donnée de test n'est fournie, extraire du CSV
            if test_data is None:
                # Charger le CSV
                csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 
                                      'data', 'pannes_industrielles_organisees.csv')
                
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
                    
                df = pd.read_csv(csv_path)
                
                # Filtrer les entrées avec des descriptions suffisantes
                df = df.dropna(subset=['description', 'type_panne'])
                df = df[df['description'].str.len() >= min_description_length]
                
                if len(df) == 0:
                    raise ValueError("Aucune donnée ne correspond aux critères de filtrage.")
                
                # Sélectionner des échantillons
                samples = df.sample(min(num_samples, len(df)))
                
                # Préparer les données de test
                test_data = []
                for _, row in samples.iterrows():
                    test_data.append({
                        'LOCATION': row.get('site', 'UNKNOWN'),
                        'ASSETNUM': row.get('machine', 'UNKNOWN'),
                        'description': row.get('description', ''),
                        'STATUS': row.get('gravite', 'NORMAL'),
                        'actual_type': row.get('type_panne', '').upper()
                    })
            
            # Effectuer les prédictions
            results = []
            y_true = []
            y_pred = []
            
            for item in test_data:
                actual_type = item.get('actual_type', '')
                if not actual_type:
                    continue
                    
                try:
                    prediction = self.predict_fault_type(item)
                    predicted_type = prediction['type']
                    
                    match = actual_type == predicted_type
                    results.append({
                        'actual_type': actual_type,
                        'predicted_type': predicted_type,
                        'match': match,
                        'confidence': prediction['confidence']
                    })
                    
                    y_true.append(actual_type)
                    y_pred.append(predicted_type)
                except Exception as e:
                    print(f"Erreur lors de la prédiction: {str(e)}")
            
            # Calculer les métriques
            if not results:
                raise ValueError("Aucune prédiction réussie pour calculer la précision.")
                
            accuracy = sum(1 for r in results if r['match']) / len(results)
            
            # Afficher les résultats
            print(f"\n=== Performance du modèle de classification de pannes ===")
            print(f"Précision globale: {accuracy * 100:.2f}% sur {len(results)} échantillons")
            
            # Générer un rapport de classification détaillé si possible
            try:
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                print("\nRapport de classification détaillé:")
                print(report_df)
                
                # Analyser l'impact de la confiance sur la précision
                confidence_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
                print("\nPrécision par niveau de confiance:")
                for i in range(len(confidence_bins)-1):
                    lower = confidence_bins[i]
                    upper = confidence_bins[i+1]
                    bin_results = [r for r in results if lower <= float(r['confidence'].rstrip('%'))/100 < upper]
                    if bin_results:
                        bin_accuracy = sum(1 for r in bin_results if r['match']) / len(bin_results)
                        print(f"  Confiance {lower:.2f}-{upper:.2f}: {bin_accuracy*100:.2f}% ({len(bin_results)} échantillons)")
                
                # Afficher la matrice de confusion
                print("\nMatrice de confusion:")
                cm = confusion_matrix(y_true, y_pred)
                print(cm)
                
                # Analyser les erreurs les plus fréquentes
                print("\nErreurs les plus fréquentes:")
                errors = [(actual, pred) for actual, pred in zip(y_true, y_pred) if actual != pred]
                error_counts = {}
                for actual, pred in errors:
                    key = f"{actual} -> {pred}"
                    error_counts[key] = error_counts.get(key, 0) + 1
                
                for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {error}: {count} occurrences")
                
            except Exception as e:
                print(f"Impossible de générer le rapport détaillé: {str(e)}")
            
            # Retourner les métriques
            return {
                'accuracy': accuracy,
                'sample_count': len(results),
                'results': results,
                'classification_report': report if 'report' in locals() else None,
                'confusion_matrix': cm.tolist() if 'cm' in locals() else None,
                'common_errors': error_counts if 'error_counts' in locals() else None
            }
            
        except Exception as e:
            print(f"Erreur lors du calcul de la précision: {str(e)}")
            return {
                'accuracy': 0,
                'sample_count': 0,
                'error': str(e)
            }