const express = require('express');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');

const router = express.Router();

// Route pour /predict for random forest
router.post('/predict', async (req, res) => {
    let tempFilePath = null;
    let pyshell = null;
    
    try {
        // Remove the req.setTimeout as it's not needed
        console.log('Received request data:', req.body);
        
        if (!req.body || !req.body.data) {
            return res.status(400).json({ 
                success: false, 
                error: "Missing required data fields" 
            });
        }

        // Validate required fields
        const requiredFields = ['LOCATION', 'STATUS', 'WOPRIORITY', 'ASSETNUM'];
        const missingFields = requiredFields.filter(field => !req.body.data[field]);
        if (missingFields.length > 0) {
            return res.status(400).json({
                success: false,
                error: `Missing required fields: ${missingFields.join(', ')}`
            });
        }

        // Add description if not present
        const inputData = {
            ...req.body.data,
            description: req.body.data.description || ''
        };

        // Write complete data including description
        const supervisedDir = path.join(__dirname, '..', 'ia_model', 'core', 'fault_detection', 'supervised');
        tempFilePath = path.join(supervisedDir, `temp_input_${Date.now()}.json`);
        fs.writeFileSync(tempFilePath, JSON.stringify(inputData));
        console.log(`Fichier temporaire créé : ${tempFilePath}`);

        // Configurez les options pour exécuter le script Python
        const options = {
            mode: 'text',
            pythonPath: 'C:\\Users\\omaim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
            pythonOptions: ['-u', '-X', 'utf8'],
            scriptPath: supervisedDir,
            args: [tempFilePath],
            timeout: 30000, // Increased timeout to 30 seconds
            stderrParser: line => console.error(`[Python stderr]: ${line}`),
            terminalOptions: { windowsHide: true }
        };

        pyshell = new PythonShell('predict.py', options);
        
        let output = [];
        let hasError = false;
        let errorMessage = '';
        
        pyshell.on('message', function(message) {
            console.log(`[Python stdout]: ${message}`);
            output.push(message);
        });
        
        pyshell.on('stderr', function(stderr) {
            console.error(`[Python stderr]: ${stderr}`);
            hasError = true;
            errorMessage = stderr;
        });
        
        pyshell.on('error', function(err) {
            hasError = true;
            errorMessage = err.message;
            console.error(`[Python error]: ${err.message}`);
        });
        
        // Add timeout promise
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Timeout')), 30000);
        });

        // Wait for either completion or timeout
        await Promise.race([
            new Promise((resolve) => pyshell.on('close', resolve)),
            timeoutPromise
        ]);
        
        // Nettoyez le fichier temporaire
        try {
            if (tempFilePath && fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
                console.log(`Fichier temporaire supprimé : ${tempFilePath}`);
            }
        } catch (unlinkErr) {
            console.error(`Erreur lors de la suppression du fichier temporaire : ${unlinkErr.message}`);
        }
        
        if (hasError) {
            return res.status(500).json({
                success: false,
                error: "Erreur lors de l'exécution du modèle",
                details: errorMessage
            });
        }
        
        console.log("Résultats bruts retournés par PythonShell :", output);
        
        // Trouver la dernière ligne qui contient du JSON valide
        // Remove the first response sending
        let prediction = null;
        if (Array.isArray(output) && output.length > 0) {
            for (let i = output.length - 1; i >= 0; i--) {
                try {
                    prediction = JSON.parse(output[i]);
                    console.log(`JSON valide trouvé à la ligne ${i}:`, prediction);
                    break;
                } catch (e) {
                    console.log(`Échec de parsing JSON à la ligne ${i}: ${output[i].substring(0, 100)}...`);
                }
            }
        }
        
        if (prediction) {
            const formattedResponse = {
                success: true,
                prediction: {
                    ...prediction.prediction,
                    timestamp: new Date().toISOString(),
                    input_data: inputData
                }
            };
            
            // If fault is detected, analyze factors
            if (prediction.prediction.etat === "En panne") {
                try {
                    const classifierOptions = {
                        mode: 'text',
                        pythonPath: 'C:\\Users\\omaim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
                        pythonOptions: ['-u', '-X', 'utf8'],
                        scriptPath: path.join(__dirname, '..', 'ia_model', 'core', 'fault_detection', 'supervised'),
                        args: [prediction.prediction.type], 
                        terminalOptions: { windowsHide: true }
                    };

                    const classifierResult = await new Promise((resolve, reject) => {
                        const pyshellClassifier = new PythonShell('fault_classifier.py', classifierOptions);
                        let output = [];
                        pyshellClassifier.on('message', (message) => {
                            console.log('[Facteurs Influents]:', message);
                            output.push(message);
                        });
                        pyshellClassifier.on('error', (err) => {
                            console.error('[Facteurs Influents Error]:', err);
                            reject(err);
                        });
                        pyshellClassifier.on('close', () => resolve(output));
                    });

                    if (classifierResult && classifierResult.length > 0) {
                        const facteurs = JSON.parse(classifierResult[classifierResult.length - 1]);
                        formattedResponse.prediction.facteurs_influents = facteurs;
                        
                        // Ajout des colonnes si disponibles
                        if (facteurs && Array.isArray(facteurs)) {
                            formattedResponse.prediction.facteurs_columns = Object.keys(facteurs[0] || {});
                        } else {
                            formattedResponse.prediction.facteurs_columns = [];
                        }
                    }
                } catch (classifierError) {
                    console.error("Erreur fault_classifier:", classifierError);
                    formattedResponse.prediction.facteurs_influents = [];
                    formattedResponse.prediction.facteurs_columns = [];
                }
            }

            // Envoi de la réponse finale
            return res.json(formattedResponse);
        } else {
            return res.status(500).json({
                success: false,
                error: "Impossible de traiter la réponse du modèle",
                rawOutput: output
            });
        }
        
    } catch (error) {
        console.error("Detailed error:", error);
        
        // Cleanup pyshell if it exists
        if (pyshell) {
            try {
                pyshell.terminate();
            } catch (termError) {
                console.error("Error terminating Python shell:", termError);
            }
        }

        // Cleanup temp file if it exists
        if (tempFilePath && fs.existsSync(tempFilePath)) {
            try {
                fs.unlinkSync(tempFilePath);
            } catch (unlinkError) {
                console.error("Error deleting temp file:", unlinkError);
            }
        }

        let errorMessage = "Erreur lors du traitement de la requête";
        if (error.message === 'Timeout') {
            errorMessage = "Le serveur met trop de temps à répondre";
        } else if (error.message.includes('ENOENT')) {
            errorMessage = "Erreur de configuration Python";
        }

        return res.status(500).json({
            success: false,
            error: errorMessage,
            details: error.message || String(error)
        });
    }
});



module.exports = router;