const express = require('express');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');

const router = express.Router();

// Route pour /predict for random forest
router.post('/predict', async (req, res) => {
    let tempFilePath = null;
    
    try {
        // Vérifiez que les données sont présentes dans la requête
        if (!req.body || !req.body.data) {
            return res.status(400).json({ success: false, error: "Données manquantes dans la requête." });
        }

        // Définissez le chemin du fichier temporaire dans le dossier supervised
        const supervisedDir = path.join(__dirname, '..', 'ia_model', 'core', 'fault_detection', 'supervised');
        tempFilePath = path.join(supervisedDir, `temp_input_${Date.now()}.json`);
        fs.writeFileSync(tempFilePath, JSON.stringify(req.body.data));
        console.log(`Fichier temporaire créé : ${tempFilePath}`);

        // Configurez les options pour exécuter le script Python
        const options = {
            mode: 'text',
            pythonPath: 'C:\\Users\\omaim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
            pythonOptions: ['-u', '-X', 'utf8'],  // Mode non bufferisé important!
            scriptPath: supervisedDir,
            args: [tempFilePath],
            timeout: 30000, // Timeout de 30 secondes
            stderrParser: line => console.error(`[Python stderr]: ${line}`)
        };
        console.log(`Chemin Python utilisé : ${options.pythonPath}`);
        console.log(`Options pour PythonShell :`, options);

        // Utilisez une implémentation personnalisée pour plus de contrôle
        const pyshell = new PythonShell('predict.py', options);
        
        let output = [];
        let hasError = false;
        let errorMessage = '';
        
        pyshell.on('message', function(message) {
            console.log(`[Python stdout]: ${message}`);
            output.push(message);
        });
        
        pyshell.on('stderr', function(stderr) {
            console.error(`[Python stderr]: ${stderr}`);
        });
        
        pyshell.on('error', function(err) {
            hasError = true;
            errorMessage = err.message;
            console.error(`[Python error]: ${err.message}`);
        });
        
        // Attendez que le script se termine
        await new Promise((resolve) => {
            pyshell.on('close', resolve);
        });
        
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
        let prediction = null;
        if (Array.isArray(output) && output.length > 0) {
            for (let i = output.length - 1; i >= 0; i--) {
                try {
                    prediction = JSON.parse(output[i]);
                    console.log(`JSON valide trouvé à la ligne ${i}:`, prediction);
                    break;  // Sortir de la boucle si parsing réussi
                } catch (e) {
                    console.log(`Échec de parsing JSON à la ligne ${i}: ${output[i].substring(0, 100)}...`);
                }
            }
        }
        
        if (prediction) {
            res.json({ success: true, prediction });
        } else {
            res.status(500).json({
                success: false,
                error: "Impossible de traiter la réponse du modèle",
                rawOutput: output
            });
        }
        
    } catch (error) {
        // En cas d'erreur, essayez de nettoyer le fichier temporaire
        try {
            if (tempFilePath && fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
                console.log(`Fichier temporaire supprimé en cas d'erreur : ${tempFilePath}`);
            }
        } catch (unlinkErr) {
            console.error(`Erreur lors de la suppression du fichier temporaire : ${unlinkErr.message}`);
        }
        
        console.error("Erreur dans la route /api/predict :", error);
        res.status(500).json({
            success: false,
            error: "Erreur interne du serveur.",
            details: error.message || String(error)
        });
    }
});


// Route pour /api/svm-predict
// router.post('/svm-predict', async (req, res) => {
//     try {
//         console.log("Requête reçue pour /api/svm-predict");
//         const inputData = req.body.data;

//         // Définir les champs requis pour le modèle SVM
//         const requiredFields = ['WOPRIORITY', 'LOCATION', 'STATUS', 'ASSETNUM'];

//         // Valider les données d'entrée
//         const missingFields = requiredFields.filter((field) => !(field in inputData));
//         if (missingFields.length > 0) {
//             console.error(`Champs manquants : ${missingFields.join(', ')}`);
//             return res.status(400).json({
//                 success: false,
//                 error: `Champs manquants : ${missingFields.join(', ')}`,
//             });
//         }

//         console.log("Données d'entrée validées :", inputData);

//         const options = {
//             mode: 'text',
//             pythonPath: 'C:\\Users\\omaim\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
//             scriptPath: path.join(__dirname, '../ia_model/core/diagnosis'),
//             args: [JSON.stringify(inputData)],
//         };

//         console.log("Options pour PythonShell :", options);

//         PythonShell.run('svm_predict.py', options, (err, results) => {
//             if (err) {
//                 console.error('Erreur lors de l\'exécution du script Python :', err);
//                 return res.status(500).json({ success: false, error: 'Erreur lors de l\'exécution du modèle.' });
//             }

//             console.log("Résultats bruts retournés par PythonShell :", results);

//             if (!results || results.length === 0) {
//                 console.error('Aucun résultat retourné par le script Python.');
//                 return res.status(500).json({ success: false, error: 'Aucun résultat retourné par le modèle.' });
//             }

//             try {
//                 const prediction = JSON.parse(results[0]);
//                 console.log("Prédiction retournée :", prediction);
//                 res.json({ success: true, prediction });
//             } catch (parseError) {
//                 console.error('Erreur lors du parsing des résultats Python :', parseError);
//                 res.status(500).json({ success: false, error: 'Erreur lors du traitement des résultats.' });
//             }
//         });
//     } catch (error) {
//         console.error('Erreur interne du serveur :', error);
//         res.status(500).json({ success: false, error: 'Erreur interne du serveur.' });
//     }
// });

module.exports = router;