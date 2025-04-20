const handleRunModel = async (workOrder) => {
    try {
        const inputData = {
            LOCATION: workOrder.location || 'UNKNOWN',
            STATUS: workOrder.status || 'UNKNOWN',
            WOPRIORITY: parseInt(workOrder.wopriority) || 5,
            ASSETNUM: workOrder.assetnum || 'UNKNOWN',
            description: workOrder.description || ''  // Changed UNKNOWN to empty string
        };

        console.log('Sending data:', inputData);

        const response = await fetch('http://192.168.1.120:3000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data: inputData }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Received result:', result);

        if (result.success && result.prediction) {
            const { etat, details } = result.prediction;
            let alertMessage = `État : ${etat}\n\n`;
            alertMessage += `Confiance : ${details.confidence}\n`;
            alertMessage += `Niveau de risque : ${details.risk_level}\n\n`;
            alertMessage += `Probabilités :\n`;
            alertMessage += `- Fonctionnel : ${details.probabilities.fonctionnel}\n`;
            alertMessage += `- Panne : ${details.probabilities.panne}\n\n`;

            // Updated to match the new fault_diagnosis structure
            if (details.fault_diagnosis) {
                const fd = details.fault_diagnosis;
                alertMessage += `Diagnostic de la panne :\n`;
                alertMessage += `- Type : ${fd.type}\n`;
                alertMessage += `- Confiance : ${fd.confidence}\n\n`;
                
                alertMessage += `Actions suggérées :\n`;
                fd.suggested_actions.forEach(action => {
                    alertMessage += `- ${action}\n`;
                });
                
                alertMessage += `\nCodes Maximo :\n`;
                alertMessage += `- Problem Code : ${fd.maximo_codes.problem}\n`;
                alertMessage += `- Failure Code : ${fd.maximo_codes.failure}\n`;
                
                alertMessage += `\nMotifs détectés :\n`;
                if (fd.matched_patterns.keywords.length > 0) {
                    alertMessage += `- Mots-clés : ${fd.matched_patterns.keywords.join(', ')}\n`;
                }
                alertMessage += `- Correspondance location : ${fd.matched_patterns.location_match ? 'Oui' : 'Non'}\n`;
            }

            Alert.alert(
                "Résultat de la Prédiction",
                alertMessage,
                [{ text: "OK" }]
            );
        } else {
            Alert.alert('Erreur', result.error || 'Réponse invalide du serveur');
        }
    } catch (error) {
        console.error('Erreur détaillée:', error);
        Alert.alert(
            'Erreur de Connexion',
            'Impossible de se connecter au serveur. Vérifiez votre connexion réseau.',
            [{ text: "OK" }]
        );
    }
};