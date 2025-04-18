const tf = require('@tensorflow/tfjs-node');

function preprocessData(data) {
    // Encodage des caractéristiques
    const features = data.map(d => [
        d.PRIORITY ? parseInt(d.PRIORITY) : 0, // PRIORITY (remplace NULL par 0)
        d.LOCATION === 'BR300' ? 0 : 1,        // LOCATION (BR300 → 0, SHIPPING → 1)
    ]);

    // Encodage des labels (STATUS)
    const labels = data.map(d => {
        switch (d.STATUS) {
            case 'CLOSE': return 0;
            case 'CAN': return 1;
            default: throw new Error('Statut inconnu');
        }
    });

    // Conversion en tenseurs TensorFlow
    const featureTensor = tf.tensor2d(features);
    const labelTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), 2); // 2 classes (CLOSE, CAN)

    return { featureTensor, labelTensor };
}

module.exports = { preprocessData };