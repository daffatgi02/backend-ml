const express = require('express');
const fileUpload = require('express-fileupload');
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');

const app = express();
const PORT = 3000;

app.use(fileUpload());

const modelPath = './goat_model.tflite'; // ganti dengan path model Anda
const classLabels = ['Healthy Goat', 'Pinkeye Goat'];

const interpretOutput = (output) => {
  if (output[0][0] > 0.5) {
    return 'Healthy Goat';
  } else {
    return 'Pinkeye Goat';
  }
};

const preprocessImage = async (imagePath) => {
  const image = await loadImage(imagePath);
  const canvas = createCanvas(150, 150);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, 150, 150);

  const imageData = ctx.getImageData(0, 0, 150, 150);
  const { data } = imageData;

  // Preprocess the image data
  const processedData = Array.from(data).map((value) => value / 255.0);
  const input = tf.tensor(processedData, [1, 150, 150, 3], 'float32');

  return input;
};

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

app.post('/', async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.render('./index.html', { prediction: '' });
  }

  const image = req.files.image;
  const imageFilePath = './images/' + image.name;

  image.mv(imageFilePath, async (err) => {
    if (err) {
      console.error(err);
      return res.status(500).send(err);
    }

    try {
      const input = await preprocessImage(imageFilePath);

      const model = await tf.lite.loadModel(modelPath);
      const output = model.predict(input);
      const predictions = output.dataSync();

      const prediction = interpretOutput(predictions);

      res.render('./index.html', { prediction });
    } catch (err) {
      console.error(err);
      res.status(500).send(err);
    } finally {
      // Remove the uploaded image file
      fs.unlinkSync(imageFilePath);
    }
  });
});

app.listen('0.0.0.0',PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
