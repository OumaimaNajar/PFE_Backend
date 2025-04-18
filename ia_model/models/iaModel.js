const tf = require('@tensorflow/tfjs');

async function loadModel() {
    const model = await tf.loadLayersModel('file://./models/mon_modele/model.json');
    console.log('Modèle chargé.');
    return model;
}

async function predict(inputData) {
    const model = await loadModel();
    const tensor = tf.tensor2d([inputData]); // Convertir les données en tenseur
    const prediction = model.predict(tensor);
    const result = await prediction.argMax(1).data(); // Récupérer la classe prédite
    return result[0]; // Retourner l'index de la classe
}

module.exports = { predict };