import sys
import json
import os
import logging
import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime  # Ajout de l'importation du module datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MaximoFaultDiagnoser')


class MaximoFaultDiagnoser:
    """
    A class to diagnose equipment faults using decision trees and influence factors.
    
    This class analyzes equipment faults based on input features and predefined
    influence factors stored in a CSV file.
    """
    
    def __init__(self, factors_file_path=None):
        """
        Initialize the fault diagnoser with features and factors.
        
        Args:
            factors_file_path (str, optional): Path to the CSV file with influence factors.
                If not provided, will attempt to locate it in a default location.
        """
        self.features = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM']
        self.model = DecisionTreeClassifier(max_depth=4, random_state=42)
        
        # Load influence factors from CSV
        if factors_file_path is None:
            factors_file_path = self._get_default_factors_path()
        
        # Ensure the correct file path is used
        self.factors_df = self._load_factors_from_csv(factors_file_path)
    
    def _get_default_factors_path(self):
        """Get the default path for the factors CSV file."""
        
        return os.path.join(
            os.path.dirname(__file__),
            '../../../../data/facteurs_influences_large.csv'  # Chemin mis à jour vers le nouveau fichier
        )
    
    def _load_factors_from_csv(self, file_path):
        if not os.path.exists(file_path):
            logger.warning(f"Fichier des facteurs non trouvé : {file_path}")
            return pd.DataFrame(columns=['type_panne', 'categorie_facteur', 'facteur', 'valeur', 'pourcentage', 'description', 'priorite'])
                
            encodings = ['utf-8', 'latin1', 'ISO-8859-1']
            for encoding in encodings:
                try:
                    logger.info(f"Tentative de chargement du fichier avec l'encodage {encoding} et le délimiteur ';'")
                    df = pd.read_csv(file_path, encoding=encoding, sep=';')
                    
                    # Vérification des colonnes requises
                    required_columns = ['type_panne', 'facteur', 'valeur', 'pourcentage', 'description']
                    if all(col in df.columns for col in required_columns):
                        logger.info(f"Chargement réussi de {len(df)} facteurs d'influence")
                        return df
                    else:
                        logger.warning(f"Colonnes manquantes dans le CSV. Trouvées : {df.columns}")
                except Exception as e:
                    logger.error(f"Échec du chargement avec l'encodage {encoding}: {str(e)}")
                    
            return pd.DataFrame(columns=['type_panne', 'categorie_facteur', 'facteur', 'valeur', 'pourcentage', 'description', 'priorite'])
    
    def get_factors_for_fault_type(self, fault_type):
        """
        Obtenir les facteurs d'influence pour un type de panne spécifique.
        
        Args:
            fault_type (str): Le type de panne pour lequel obtenir les facteurs
            
        Returns:
            dict: Dictionnaire des facteurs organisés par catégorie
        """
        if self.factors_df is None or self.factors_df.empty:
            logger.warning("Aucun facteur d'influence disponible")
            return {}
                
        # Filtrer les facteurs pour ce type de panne
        fault_factors = self.factors_df[self.factors_df['type_panne'] == fault_type]
        
        if fault_factors.empty:
            # Recherche élargie - correspondance partielle
            logger.info(f"Pas de correspondance exacte pour '{fault_type}', recherche élargie")
            fault_factors = self.factors_df[
                self.factors_df['type_panne'].str.contains(fault_type, case=False, na=False)
            ]
            
            if fault_factors.empty:
                # Recherche par mots-clés
                keywords = fault_type.split()
                for keyword in keywords:
                    if len(keyword) >= 4:  # Éviter les mots trop courts
                        temp_factors = self.factors_df[
                            self.factors_df['type_panne'].str.contains(keyword, case=False, na=False)
                        ]
                        if not temp_factors.empty:
                            fault_factors = temp_factors
                            logger.info(f"Correspondance trouvée avec le mot-clé: {keyword}")
                            break
            
            if fault_factors.empty:
                logger.warning(f"Aucun facteur trouvé pour le type de panne: {fault_type}")
                return self._get_generic_factors()  # Utiliser les facteurs génériques directement
                
            logger.info(f"Trouvé {len(fault_factors)} facteurs avec la recherche élargie")
        
        # Organiser les facteurs par catégorie
        factors_dict = {}
        
        # Utiliser la colonne categorie_facteur si elle existe
        if 'categorie_facteur' in fault_factors.columns:
            # Créer dynamiquement les catégories à partir des données
            categories = fault_factors['categorie_facteur'].unique()
            for cat in categories:
                factors_dict[cat] = {}
            
            for _, row in fault_factors.iterrows():
                categorie = row['categorie_facteur']
                factors_dict[categorie][row['facteur']] = {
                    'valeur': row['valeur'],
                    'pourcentage': float(row['pourcentage']),
                    'description': row['description'] if pd.notna(row['description']) else '',
                    'priorite': row['priorite'] if 'priorite' in row and pd.notna(row['priorite']) else 'Moyenne'
                }
        else:
            # Utiliser les catégories prédéfinies comme avant
            factors_dict = {
                'Temporel': {},
                'Gravité': {},
                'Machine': {},
                'Site': {},
                'Action': {}
            }
            
            for _, row in fault_factors.iterrows():
                categorie = self._get_factor_category(row['facteur'])
                factors_dict[categorie][row['facteur']] = {
                    'valeur': row['valeur'],
                    'pourcentage': float(row['pourcentage']),
                    'description': row['description'] if pd.notna(row['description']) else ''
                }
        
        # Si aucun facteur n'a été trouvé dans aucune catégorie, utiliser les facteurs génériques
        if all(not factors for factors in factors_dict.values()):
            return self._get_generic_factors()
                
        return factors_dict
    
    def _get_factor_category(self, factor):
        """
        Déterminer la catégorie d'un facteur.
        """
        if 'durée' in factor.lower():
            return 'Temporel'
        elif 'gravité' in factor.lower():
            return 'Gravité'
        elif 'machine' in factor.lower():
            return 'Machine'
        elif 'site' in factor.lower():
            return 'Site'
        elif 'action' in factor.lower():
            return 'Action'
        return 'Autre'

    def analyze_fault_factors(self, input_data, prediction_result):
        """
        Analyze fault factors based on input data and prediction results.
        """
        try:
            # Extract prediction details
            fault_probability = self._extract_fault_probability(prediction_result)
            risk_level = prediction_result['details']['risk_level']
            fault_type = prediction_result['details']['fault_diagnosis']['type']
            
            logger.info(f"Analyzing factors for fault type: {fault_type}")
            
            # Get influence factors for this fault type
            fault_factors = self.get_factors_for_fault_type(fault_type)
            
            # Use generic factors if no specific ones are found
            if not fault_factors:
                logger.info("Using generic factors as no specific factors were found")
                fault_factors = self._get_generic_factors()
            
            # Extract feature importance from factors
            feature_importance = self._calculate_feature_importance(fault_factors)
            
            # Analyze risk patterns based on prediction results
            risk_patterns = self._analyze_risk_patterns(input_data, feature_importance, fault_factors)
            
            # Extract top influencing factors
            top_factors = self._extract_top_factors(fault_type)
            
            # Compile analysis results
            analysis_result = {
                "risk_patterns": risk_patterns,
                "importance": feature_importance,
                "recommendations": {
                    "high_priority": [factor for factor in top_factors if factor['pourcentage'] >= 70],
                    "medium_priority": [factor for factor in top_factors if 30 <= factor['pourcentage'] < 70],
                    "low_priority": [factor for factor in top_factors if factor['pourcentage'] < 30]
                },
                "fault_probability": fault_probability,
                "risk_level": risk_level,
                "fault_type": fault_type,
                "factors_details": fault_factors
            }
        
            # Update prediction result with factor analysis
            if isinstance(prediction_result, dict):
                if 'details' not in prediction_result:
                    prediction_result['details'] = {}
                
                # Ensure the factor analysis is properly structured in the prediction result
                prediction_result['details']['factor_analysis'] = {
                    "importance": feature_importance,
                    "risk_patterns": risk_patterns,
                    "recommendations": analysis_result['recommendations'],
                    "fault_probability": fault_probability,
                    "risk_level": risk_level,
                    "factors_details": fault_factors
                }
        
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing fault factors: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _extract_fault_probability(self, prediction_result):
        """Extract and normalize fault probability from prediction result."""
        prob_str = prediction_result['details']['probabilities']['panne']
        return float(prob_str.rstrip('%')) / 100
    
    def _get_generic_factors(self):
        """Get generic factors when specific ones aren't available."""
        return {
            'LOCATION': {
                'pourcentage': 70, 
                'description': 'Emplacement critique'
            },
            'WOPRIORITY': {
                'pourcentage': 85, 
                'description': 'Priorité élevée'
            },
            'STATUS': {
                'pourcentage': 50, 
                'description': 'Statut à risque'
            },
            'ASSETNUM': {
                'pourcentage': 60, 
                'description': 'Équipement sensible'
            }
        }
    
    def _calculate_feature_importance(self, fault_factors):
        """Calculate importance for each feature based on available factors."""
        feature_importance = {}
        
        for feature in self.features:
            # Find matching factors for this feature
            matching_factors = [f for f in fault_factors.keys() if feature.lower() in f.lower()]
            
            if matching_factors:
                # Use first matching factor
                feature_importance[feature] = fault_factors[matching_factors[0]]['pourcentage'] / 100
            else:
                # Use average importance if no matching factor
                if fault_factors:
                    avg_importance = sum([f['pourcentage'] for f in fault_factors.values()]) / len(fault_factors) / 100
                    feature_importance[feature] = avg_importance
                else:
                    # Use neutral value if no factors available
                    feature_importance[feature] = 0.5
                    
        return feature_importance
    
    def _analyze_risk_patterns(self, input_data, feature_importance, fault_factors):
        """Analyze risk patterns based on input data and feature importance."""
        risk_patterns = {
            "high_risk": [],
            "medium_risk": [],
            "low_risk": []
        }
        
        for feature, value in input_data.items():
            if feature in self.features:
                # Determine risk level based on factor importance
                importance = feature_importance.get(feature, 0.5)
                
                if importance > 0.7:
                    risk_category = "high_risk"
                elif importance > 0.4:
                    risk_category = "medium_risk"
                else:
                    risk_category = "low_risk"
                
                # Get specific message from factors if available
                message = f"Le facteur {feature} avec la valeur {value} contribue au risque"
                matching_factors = [f for f in fault_factors.keys() if feature.lower() in f.lower()]
                if matching_factors:
                    message = fault_factors[matching_factors[0]]['description']
                
                risk_patterns[risk_category].append({
                    "feature": feature,
                    "importance": importance,
                    "current_value": value,
                    "message": message
                })
                
        return risk_patterns
    
    def _extract_top_factors(self, fault_type):
        """Extract top influencing factors from CSV data."""
        top_factors = []
        if not self.factors_df.empty:
            top_factors_df = self.factors_df[
                self.factors_df['type_panne'] == fault_type
            ].sort_values('pourcentage', ascending=False).head(3)
            
            for _, row in top_factors_df.iterrows():
                top_factors.append({
                    "facteur": row['facteur'],
                    "valeur": row['valeur'],
                    "pourcentage": row['pourcentage'],
                    "description": row['description']
                })
                
        return top_factors


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze equipment fault data using a decision tree model."
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for analyze command with JSON string
    analyze_parser = subparsers.add_parser('analyze', help='Analyze input data from a JSON string')
    analyze_parser.add_argument('json_data', help='Input data in JSON format')
    analyze_parser.add_argument('--factors-file', help='Path to the CSV file with influence factors')
    analyze_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    # Parser for analyze-file command
    file_parser = subparsers.add_parser('analyze-file', help='Analyze input data from a JSON file')
    file_parser.add_argument('file_path', help='Path to the JSON input file')
    file_parser.add_argument('--factors-file', help='Path to the CSV file with influence factors')
    file_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    return parser.parse_args()


def main():
    """Main function to execute the fault diagnosis from command line."""
    try:
        # For backwards compatibility with the original command structure
        if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
            # Handle legacy command format
            if sys.argv[1] == '--analyze':
                if len(sys.argv) < 3:
                    print(json.dumps({"error": "Missing input data"}))
                    sys.exit(1)
                try:
                    input_data = json.loads(sys.argv[2])
                except json.JSONDecodeError as e:
                    print(json.dumps({"error": f"JSON decoding error: {str(e)}", "status": "failed"}))
                    sys.exit(1)
                factors_file = None
            elif sys.argv[1] == '--analyze-file':
                if len(sys.argv) < 3:
                    print(json.dumps({"error": "Missing file path"}))
                    sys.exit(1)
                try:
                    with open(sys.argv[2], 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                except Exception as e:
                    print(json.dumps({"error": f"Error reading file: {str(e)}", "status": "failed"}))
                    sys.exit(1)
                factors_file = None
            else:
                print(json.dumps({"error": f"Unknown command: {sys.argv[1]}"}))
                sys.exit(1)
        else:
            # Use argparse for new command format
            args = parse_arguments()
            
            # Check if command is provided
            if not hasattr(args, 'command') or not args.command:
                print(json.dumps({"error": "No command specified. Use 'analyze' or 'analyze-file'"}))
                sys.exit(1)
                
            # Set log level if verbose flag is present
            if hasattr(args, 'verbose') and args.verbose:
                logger.setLevel(logging.DEBUG)
            
            if args.command == 'analyze':
                try:
                    input_data = json.loads(args.json_data)
                except json.JSONDecodeError as e:
                    print(json.dumps({"error": f"JSON decoding error: {str(e)}", "status": "failed"}))
                    sys.exit(1)
                factors_file = args.factors_file if hasattr(args, 'factors_file') else None
            elif args.command == 'analyze-file':
                try:
                    with open(args.file_path, 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                except Exception as e:
                    print(json.dumps({"error": f"Error reading file: {str(e)}", "status": "failed"}))
                    sys.exit(1)
                factors_file = args.factors_file if hasattr(args, 'factors_file') else None
            else:
                print(json.dumps({"error": f"Unknown command: {args.command}"}))
                sys.exit(1)
            
        if not input_data:
            print(json.dumps({"error": "No valid input data"}))
            sys.exit(1)
            
        # Get prediction result from the environment
        prediction_result = {
            "details": {
                "probabilities": {
                    "panne": input_data.get("fault_probability", "50%")
                },
                "risk_level": input_data.get("risk_level", "Moyen"),
                "fault_diagnosis": {
                    "type": input_data.get("fault_type", "DÉFAILLANCE MÉCANIQUE")
                }
            }
        }

        # Perform analysis
        diagnoser = MaximoFaultDiagnoser(factors_file)
        factor_analysis = diagnoser.analyze_fault_factors(input_data, prediction_result)
        
        # Mise à jour du résultat de prédiction avec l'analyse des facteurs
        if isinstance(prediction_result, dict) and 'details' in prediction_result:
            prediction_result['details']['factor_analysis'] = factor_analysis
        
        # Output result as JSON with UTF-8 encoding
        print(json.dumps(prediction_result, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Unhandled error in main: {str(e)}", exc_info=True)
        print(json.dumps({"error": str(e), "status": "failed"}))


if __name__ == "__main__":
    main()