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
        console.log('Received request data:', req.body);
        
        if (!req.body || !req.body.data) {
            return res.status(400).json({ 
                success: false, 
                error: "Missing required data fields" 
            });
        }

        const requiredFields = ['LOCATION', 'ASSETNUM', 'Description'];  // Added Description
        const missingFields = requiredFields.filter(field => !req.body.data[field]);
        if (missingFields.length > 0) {
            return res.status(400).json({
                success: false,
                error: `Missing required fields: ${missingFields.join(', ')}`
            });
        }

        // Only send 'Description' (capital D) to Python, avoid duplicates
        const inputData = {
            ...req.body.data,
            Description: req.body.data.Description || req.body.data.description || ''
        };

        const supervisedDir = path.join(__dirname, '..', 'ia_model', 'core', 'fault_detection', 'supervised');
        tempFilePath = path.join(supervisedDir, `temp_input_${Date.now()}.json`);
        
        // Encodage UTF-8 pour le fichier temporaire
        const jsonData = JSON.stringify(inputData, null, 2);
        const buffer = Buffer.from(jsonData, 'utf8');
        fs.writeFileSync(tempFilePath, buffer);
        
        console.log(`Fichier temporaire créé : ${tempFilePath}`);

        const options = {
            mode: 'text',
            pythonPath: 'C:\\Users\\omaim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
            pythonOptions: ['-u'],
            scriptPath: supervisedDir,
            args: [tempFilePath],
            timeout: 60000,
            encoding: 'latin1',
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
            // Ne pas setter hasError ici, juste logguer
        });
        
        pyshell.on('error', function(err) {
            hasError = true;
            errorMessage = err.message;
            console.error(`[Python error]: ${err.message}`);
        });
        
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Timeout')), 60000);
        });

        await Promise.race([
            new Promise((resolve) => pyshell.on('close', resolve)),
            timeoutPromise
        ]);
        
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
                details: errorMessage,
                input_data: inputData // Ajout des données d'entrée pour le debugging
            });
        }
        
        console.log("Résultats bruts retournés par PythonShell :", output);
        
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
            
            try {
                const accuracyOptions = {
                    mode: 'text',
                    pythonPath: 'C:\\Users\\omaim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
                    pythonOptions: ['-u', '-X', 'utf8'],
                    scriptPath: path.join(__dirname, '..', 'ia_model', 'core', 'fault_detection', 'supervised'),
                    terminalOptions: { windowsHide: true }
                };

                const accuracyResult = await new Promise((resolve, reject) => {
                    const pyshellAccuracy = new PythonShell('test_accuracy.py', accuracyOptions);
                    let output = [];
                    
                    pyshellAccuracy.on('message', (message) => {
                        console.log('[Accuracy Metrics]:', message);
                        output.push(message);
                    });
                    
                    pyshellAccuracy.on('error', (err) => {
                        console.error('[Accuracy Error]:', err);
                        reject(err);
                    });
                    
                    pyshellAccuracy.on('close', () => resolve(output));
                });

                if (accuracyResult && accuracyResult.length > 0) {
                    formattedResponse.prediction.accuracy_metrics = accuracyResult;
                }
            } catch (accuracyError) {
                console.error("Erreur lors du calcul de l'accuracy:", accuracyError);
            }

            if (prediction.prediction.etat === "En panne") {
                try {
                    // Définir classifierOptions avant de l'utiliser
                    const classifierOptions = {
                        mode: 'text',
                        pythonPath: 'C:\\Users\\omaim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
                        pythonOptions: ['-u', '-X', 'utf8'],
                        scriptPath: path.join(__dirname, '..', 'ia_model', 'core', 'fault_detection', 'supervised'),
                        args: [prediction.prediction.details?.fault_diagnosis?.etat || ''], 
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
                        try {
                            const facteurs = JSON.parse(classifierResult[classifierResult.length - 1]);
                            formattedResponse.prediction.facteurs_influents = facteurs;
                            
                            // S'assurer d'ajouter tous les facteurs d'influence à la réponse
                            if (formattedResponse.prediction.details && 
                                formattedResponse.prediction.details.influencing_factors) {
                                formattedResponse.prediction.details.influencing_factors.forEach(factor => {
                                    // S'assurer que PB et FC sont bien présents dans la réponse
                                    console.log(`Facteur avec PB: ${factor.PB}, FC: ${factor.FC}`);
                                });
                            }
                        } catch (parseError) {
                            console.error("Erreur lors du parsing des facteurs influents:", parseError);
                        }
                    }
                } catch (classifierError) {
                    console.error("Erreur lors de l'analyse des facteurs influents:", classifierError);
                }
            }

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
        
        if (pyshell) {
            try {
                pyshell.terminate();
            } catch (termError) {
                console.error("Error terminating Python shell:", termError);
            }
        }

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