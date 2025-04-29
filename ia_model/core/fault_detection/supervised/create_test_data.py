import pandas as pd
import os

# Créer le répertoire data s'il n'existe pas
data_dir = "../../../../data"
os.makedirs(data_dir, exist_ok=True)

# Données de test pour la classification
classification_data = [
    {"description": "Fuite hydraulique détectée dans le système principal", "fault_type": "FUITE"},
    {"description": "Fuite d'huile au niveau du joint", "fault_type": "FUITE"},
    {"description": "Liquide hydraulique qui s'échappe du réservoir", "fault_type": "FUITE"},
    {"description": "Équipement qui surchauffe et fait un bruit anormal", "fault_type": "SURCHAUFFE"},
    {"description": "Température élevée dans le moteur", "fault_type": "SURCHAUFFE"},
    {"description": "Composant électronique très chaud au toucher", "fault_type": "SURCHAUFFE"},
    {"description": "Moteur arrêté de fonctionner après une surtension", "fault_type": "PANNE_ELECTRIQUE"},
    {"description": "Court-circuit dans le panneau de contrôle", "fault_type": "PANNE_ELECTRIQUE"},
    {"description": "Fusible grillé dans l'armoire électrique", "fault_type": "PANNE_ELECTRIQUE"},
    {"description": "Pièce mécanique bloquée et ne tourne plus", "fault_type": "BLOCAGE"},
    {"description": "Engrenage coincé et ne peut pas tourner", "fault_type": "BLOCAGE"},
    {"description": "Vanne bloquée en position fermée", "fault_type": "BLOCAGE"},
    {"description": "Usure importante des roulements", "fault_type": "USURE"},
    {"description": "Pièce métallique usée et nécessitant remplacement", "fault_type": "USURE"},
    {"description": "Courroie usée et effilochée", "fault_type": "USURE"}
]

# Créer le DataFrame et sauvegarder en CSV
df_classification = pd.DataFrame(classification_data)
classification_file = os.path.join(data_dir, "pannes_industrielles_organisees.csv")
df_classification.to_csv(classification_file, sep=';', index=False)

print(f"Fichier de données de classification créé : {classification_file}")

# Créer un fichier de facteurs influençants
facteurs_data = [
    {"TYPE_PANNE": "FUITE", "CATEGORIE": "MAINTENANCE", "FACTEUR": "Joints usés", "IMPACT": "Élevé", "DESCRIPTION": "Les joints d'étanchéité qui n'ont pas été remplacés régulièrement"},
    {"TYPE_PANNE": "FUITE", "CATEGORIE": "ENVIRONNEMENT", "FACTEUR": "Température élevée", "IMPACT": "Moyen", "DESCRIPTION": "Environnement de travail avec des températures élevées qui détériorent les joints"},
    {"TYPE_PANNE": "FUITE", "CATEGORIE": "UTILISATION", "FACTEUR": "Surpression", "IMPACT": "Élevé", "DESCRIPTION": "Utilisation de l'équipement au-delà des pressions recommandées"},
    
    {"TYPE_PANNE": "SURCHAUFFE", "CATEGORIE": "MAINTENANCE", "FACTEUR": "Ventilation obstruée", "IMPACT": "Élevé", "DESCRIPTION": "Grilles de ventilation ou radiateurs obstrués par la poussière"},
    {"TYPE_PANNE": "SURCHAUFFE", "CATEGORIE": "ENVIRONNEMENT", "FACTEUR": "Température ambiante", "IMPACT": "Moyen", "DESCRIPTION": "Température ambiante trop élevée dans l'atelier"},
    {"TYPE_PANNE": "SURCHAUFFE", "CATEGORIE": "UTILISATION", "FACTEUR": "Surcharge", "IMPACT": "Élevé", "DESCRIPTION": "Utilisation de l'équipement au-delà de sa capacité nominale"},
    
    {"TYPE_PANNE": "PANNE_ELECTRIQUE", "CATEGORIE": "MAINTENANCE", "FACTEUR": "Connexions desserrées", "IMPACT": "Élevé", "DESCRIPTION": "Connexions électriques qui se sont desserrées avec le temps"},
    {"TYPE_PANNE": "PANNE_ELECTRIQUE", "CATEGORIE": "ENVIRONNEMENT", "FACTEUR": "Humidité", "IMPACT": "Élevé", "DESCRIPTION": "Présence d'humidité dans l'environnement électrique"},
    {"TYPE_PANNE": "PANNE_ELECTRIQUE", "CATEGORIE": "UTILISATION", "FACTEUR": "Fluctuations électriques", "IMPACT": "Moyen", "DESCRIPTION": "Fluctuations importantes dans l'alimentation électrique"},
    
    {"TYPE_PANNE": "BLOCAGE", "CATEGORIE": "MAINTENANCE", "FACTEUR": "Lubrification insuffisante", "IMPACT": "Élevé", "DESCRIPTION": "Manque de lubrification des pièces mobiles"},
    {"TYPE_PANNE": "BLOCAGE", "CATEGORIE": "ENVIRONNEMENT", "FACTEUR": "Contamination", "IMPACT": "Moyen", "DESCRIPTION": "Présence de contaminants comme la poussière ou des débris"},
    {"TYPE_PANNE": "BLOCAGE", "CATEGORIE": "UTILISATION", "FACTEUR": "Mauvais alignement", "IMPACT": "Élevé", "DESCRIPTION": "Mauvais alignement des composants mécaniques"},
    
    {"TYPE_PANNE": "USURE", "CATEGORIE": "MAINTENANCE", "FACTEUR": "Remplacement tardif", "IMPACT": "Moyen", "DESCRIPTION": "Pièces d'usure non remplacées selon le calendrier recommandé"},
    {"TYPE_PANNE": "USURE", "CATEGORIE": "ENVIRONNEMENT", "FACTEUR": "Abrasifs", "IMPACT": "Élevé", "DESCRIPTION": "Présence de particules abrasives dans l'environnement"},
    {"TYPE_PANNE": "USURE", "CATEGORIE": "UTILISATION", "FACTEUR": "Utilisation intensive", "IMPACT": "Moyen", "DESCRIPTION": "Utilisation de l'équipement au-delà des cycles recommandés"},
    
    {"TYPE_PANNE": "GENERAL", "CATEGORIE": "MAINTENANCE", "FACTEUR": "Maintenance préventive", "IMPACT": "Élevé", "DESCRIPTION": "Absence ou retard dans la maintenance préventive planifiée"},
    {"TYPE_PANNE": "GENERAL", "CATEGORIE": "FORMATION", "FACTEUR": "Formation opérateur", "IMPACT": "Moyen", "DESCRIPTION": "Manque de formation des opérateurs sur l'utilisation correcte"},
    {"TYPE_PANNE": "GENERAL", "CATEGORIE": "QUALITE", "FACTEUR": "Qualité des pièces", "IMPACT": "Élevé", "DESCRIPTION": "Utilisation de pièces de rechange de qualité inférieure"}
]

# Créer le DataFrame et sauvegarder en CSV
df_facteurs = pd.DataFrame(facteurs_data)
facteurs_file = os.path.join(data_dir, "facteurs_influencent_pannes.csv")
df_facteurs.to_csv(facteurs_file, sep=';', index=False)

print(f"Fichier de facteurs influençants créé : {facteurs_file}")