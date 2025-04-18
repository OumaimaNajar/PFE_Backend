import sys
import json
import pickle
import pandas as pd
import os

# Forcer l'encodage UTF-8 pour stdin, stdout et stderr
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Fonction pour écrire les logs sur stderr
def log(message):
    print(message, file=sys.stderr)

def predict(input_data):
    """Charge le modèle SVM et effectue une prédiction"""
    try:
        log("Chargement du modèle SVM...")
        with open('c:\\Users\\omaim\\backend_ia\\ia_model\\core\\diagnosis\\svm_maximo_detector.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        log("Modèle SVM chargé avec succès.")
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        features = model_data['features']

        log("Prétraitement des données...")
        processed_data = {}
        for feature in features:
            if feature in label_encoders:
                le = label_encoders[feature]
                value = input_data.get(feature, "UNKNOWN")
                if value not in le.classes_:
                    log(f"Valeur inconnue détectée pour {feature}: {value}")
                    value = "UNKNOWN"
                processed_data[feature] = le.transform([value])[0]
            else:
                processed_data[feature] = input_data.get(feature, 0)
        
        X_input = pd.DataFrame([processed_data])
        log(f"Données prétraitées : {X_input}")

        log("Prédiction en cours...")
        prediction = model.predict(X_input)
        proba = model.predict_proba(X_input)[0]
        result = {
            "prediction": int(prediction[0]),
            "probabilities": {
                "fonctionnel": f"{proba[0] * 100:.2f}%",
                "panne": f"{proba[1] * 100:.2f}%"
            },
            "message": "Le modèle SVM a analysé les données et a détecté l'état du système."
        }
        log(f"Prédiction terminée : {result}")
        return result
    except Exception as e:
        log(f"Erreur pendant la prédiction : {str(e)}")
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    try:
        log("Début de l'exécution du script svm_predict.py")
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            try:
                input_data = json.loads(arg)
                log(f"Données reçues directement : {input_data}")
            except json.JSONDecodeError:
                log(f"Chemin du fichier reçu : {arg}")
                if not os.path.exists(arg):
                    log(f"Erreur : Le fichier {arg} n'existe pas.")
                    sys.exit(1)
                with open(arg, 'r') as f:
                    input_data = json.load(f)
                    log(f"Données chargées depuis le fichier : {input_data}")
        else:
            log("Lecture des données d'entrée...")
            input_data = json.loads(sys.stdin.read())

        log(f"Données reçues : {input_data}")
        result = predict(input_data)
        log(f"Résultat : {result}")
        
        # N'envoyer que le JSON final sur stdout pour être capturé par Node.js
        print(json.dumps(result))
    except Exception as e:
        log(f"Erreur principale : {str(e)}")
        sys.exit(1)