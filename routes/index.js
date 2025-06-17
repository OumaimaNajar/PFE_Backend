const express = require('express');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');
const { promisify } = require('util');
const setTimeoutPromise = promisify(setTimeout);

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
            pythonPath: 'c:\\Users\\omaim\\backend_ia\\.venv\\Scripts\\python.exe',
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
            // Only set error if the message contains 'ERROR' or 'CRITICAL'
            if (stderr && (stderr.includes('ERROR') || stderr.includes('CRITICAL'))) {
                hasError = true;
                errorMessage = stderr;
            }
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
                    pythonPath: 'c:\\Users\\omaim\\backend_ia\\.venv\\Scripts\\python.exe',
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
                        pythonPath: 'c:\\Users\\omaim\\backend_ia\\.venv\\Scripts\\python.exe',
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

// Route pour la proximité des techniciens
router.post('/proximity', async (req, res) => {
    let tempFilePath = null;
    let pyshell = null;
    
    try {
        console.log('Received proximity request data:', req.body);
                                                                                                                                                                                                                            
        if (!req.body || !req.body.data) {
            return res.status(400).json({ 
                success: false, 
                error: "Missing required data fields" 
            });
        }

        const requiredFields = ['Latitude', 'Longitude'];
        const missingFields = requiredFields.filter(field => !req.body.data[field]);
        if (missingFields.length > 0) {
            return res.status(400).json({
                success: false,
                error: `Missing required fields: ${missingFields.join(', ')}`
            });
        }

        const proximityDir = path.join(__dirname, '..', 'ia_model', 'core', 'proximity');
        tempFilePath = path.join(proximityDir, `temp_input_${Date.now()}.json`);
        
        const jsonData = JSON.stringify(req.body.data, null, 2);
        const buffer = Buffer.from(jsonData, 'utf8');
        fs.writeFileSync(tempFilePath, buffer);

        console.log('Created temporary file:', tempFilePath);

        const options = {
            mode: 'text',
            pythonPath: 'c:\\Users\\omaim\\backend_ia\\.venv\\Scripts\\python.exe',
            pythonOptions: ['-u'],
            scriptPath: proximityDir,
            args: [tempFilePath],
            encoding: 'utf8',
            terminalOptions: { windowsHide: true }
        };

        console.log('Starting Python process with options:', options);

        pyshell = new PythonShell('proximity_model.py', options);
        
        let output = [];
        let hasError = false;
        let errorMessage = '';
        
        pyshell.on('message', function(message) {
            console.log(`[Python stdout]: ${message}`);
            output.push(message);
        });

        pyshell.on('stderr', function(stderr) {
            // Only log the stderr output without marking it as an error
            console.log(`[Python log]: ${stderr}`);
        });

        // Modify the error handling to only respond with error for actual errors
        pyshell.on('error', function(err) {
            console.error(`[Python error]: ${err.message}`);
            hasError = true;
            errorMessage = err.message;
        });

        // Ajouter un timeout de 30 secondes
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Timeout')), 30000);
        });

        await Promise.race([
            new Promise((resolve) => pyshell.on('close', resolve)),
            timeoutPromise
        ]);

        pyshell.on('close', () => {
            try {
                if (tempFilePath && fs.existsSync(tempFilePath)) {
                    fs.unlinkSync(tempFilePath);
                }
                
                if (output.length > 0) {
                    const result = JSON.parse(output[output.length - 1]);
                    if (result.success && result.techniciens && result.techniciens.length > 0) {
                        return res.json({
                            success: true,
                            technicians: result.techniciens,
                            message: "Techniciens trouvés"
                        });
                    } else {
                        return res.json({
                            success: false,
                            message: "Aucun technicien trouvé à proximité",
                            details: result
                        });
                    }
                } else {
                    return res.json({
                        success: false,
                        message: "Aucune réponse du modèle"
                    });
                }
            } catch (e) {
                return res.status(500).json({
                    success: false,
                    message: "Erreur de traitement",
                    error: e.message
                });
            }
        });

        if (hasError) {
            throw new Error(`Erreur Python: ${errorMessage}`);
        }
        
        if (tempFilePath && fs.existsSync(tempFilePath)) {
            fs.unlinkSync(tempFilePath);
            console.log('Temporary file deleted:', tempFilePath);
        }
        
        if (!output.length) {
            throw new Error('Aucune donnée reçue du modèle Python');
        }

        if (hasError) {
            return res.status(500).json({
                success: false,
                error: "Erreur lors de l'exécution du modèle",
                details: errorMessage
            });
        } else if (output.length > 0) {
            try {
                const result = JSON.parse(output[output.length - 1]);
                return res.json({
                    success: true,
                    data: result
                });
            } catch (e) {
                return res.status(500).json({
                    success: false,
                    error: "Invalid response format from Python model"
                });
            }
        }

        let result;
        try {
            result = JSON.parse(output[output.length - 1]);
        } catch (parseError) {
            console.error('Error parsing Python output:', output);
            throw new Error('Invalid JSON output from Python');
        }

        return res.json({
            success: true,
            techniciens: result.techniciens
        });
        
    } catch (error) {
        console.error("Detailed error:", error);
        
        if (pyshell) {
            try {
                pyshell.terminate();
                console.log('Python process terminated');
            } catch (termError) {
                console.error("Error terminating Python shell:", termError);
            }
        }

        if (tempFilePath && fs.existsSync(tempFilePath)) {
            try {
                fs.unlinkSync(tempFilePath);
                console.log('Temporary file deleted after error');
            } catch (unlinkError) {
                console.error("Error deleting temp file:", unlinkError);
            }
        }

        let clientError = "Erreur lors du traitement de la requête";
        let statusCode = 500;

        if (error.message === 'Timeout') {
            clientError = "Le serveur met trop de temps à répondre";
        } else if (error.message.includes('ENOENT')) {
            clientError = "Erreur de configuration Python";
        } else if (error.message.includes('MongoDB')) {
            clientError = "Erreur de connexion à la base de données";
            statusCode = 503;
        }

        return res.status(statusCode).json({
            success: false,
            error: clientError,
            details: error.message
        });
    }
});

module.exports = router;

async function deleteFileWithRetry(filePath, maxRetries = 5, delayMs = 1000) {
    if (!fs.existsSync(filePath)) {
        console.log(`Le fichier ${filePath} n'existe pas, aucune suppression nécessaire`);
        return;
    }

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            // Attendre un peu que le processus Python se termine complètement
            if (attempt === 1) {
                await setTimeoutPromise(500);
            }

            fs.unlinkSync(filePath);
            console.log(`Fichier ${filePath} supprimé avec succès`);
            return;
        } catch (err) {
            if (err.code === 'ENOENT') {
                console.log(`Le fichier ${filePath} n'existe plus`);
                return;
            } else if (err.code === 'EBUSY' && attempt < maxRetries) {
                const waitTime = delayMs * attempt; // Délai progressif
                console.log(`Fichier occupé, nouvelle tentative dans ${waitTime}ms (${attempt}/${maxRetries})`);
                await setTimeoutPromise(waitTime);
            } else if (attempt === maxRetries) {
                console.error(`Impossible de supprimer le fichier après ${maxRetries} tentatives:`, err);
                throw err;
            } else {
                throw err;
            }
        }
    }
}

// Utilisation de la fonction
const tempFilePath = 'C:\\Users\\omaim\\backend_ia\\ia_model\\core\\proximity\\temp_input_1750113078889.json';
deleteFileWithRetry(tempFilePath);