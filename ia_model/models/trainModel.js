const tf = require('@tensorflow/tfjs-node');
console.log('TensorFlow.js chargé avec succès !');
const { preprocessData } = require('./preprocessData');

// Importer la fonction trainModel et les données

const data = require('../data/pannes.json');


async function trainModel(data) {
    // Prétraitement des données
    const { featureTensor, labelTensor } = preprocessData(data);

    // Création du modèle
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [featureTensor.shape[1]] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // 3 classes
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // Entraînement du modèle
    await model.fit(featureTensor, labelTensor, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2
    });

    // Sauvegarde du modèle
    await model.save('file://./models/mon_modele');
    console.log('Modèle entraîné et sauvegardé.');


    // Entraîner le modèle
async function main() {
    try {
        console.log('Début de l\'entraînement du modèle...');
        await trainModel(data);
        console.log('Entraînement terminé et modèle sauvegardé.');
    } catch (error) {
        console.error('Erreur lors de l\'entraînement du modèle :', error);
    }
}

}

module.exports = { trainModel };