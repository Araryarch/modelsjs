import * as tf from "@tensorflow/tfjs";

// Data penjualan untuk beberapa hari
import { dataMinuman } from "./dataset/data.js";

// data yang mau diprediksi
import { prediksi } from "./predicts/data.js";

// Utility functions for data preprocessing
function normalize(data, min, max) {
  return data.map(value => (value - min) / (max - min));
}

function calculateStats(data) {
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const std = Math.sqrt(
    data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length
  );
  return { mean, std };
}

// Advanced feature engineering
function prepareFeatures(data) {
  const allPenjualan = data.flatMap(item => item.penjualan);
  const allRatings = data.map(item => item.rating);
  const allUlasan = data.map(item => item.ulasan);
  const allStok = data.map(item => item.stok);
  const allHarga = data.map(item => item.harga);

  const stats = {
    penjualan: {
      min: Math.min(...allPenjualan),
      max: Math.max(...allPenjualan)
    },
    rating: { min: Math.min(...allRatings), max: Math.max(...allRatings) },
    ulasan: { min: Math.min(...allUlasan), max: Math.max(...allUlasan) },
    stok: { min: Math.min(...allStok), max: Math.max(...allStok) },
    harga: { min: Math.min(...allHarga), max: Math.max(...allHarga) }
  };

  const features = [];
  const labels = [];

  data.forEach(item => {
    const penjualanStats = calculateStats(item.penjualan);
    const trendPenjualan =
      (item.penjualan[item.penjualan.length - 1] - item.penjualan[0]) /
      item.penjualan[0];
    const pricePerRating = item.harga / item.rating;
    const engagementScore =
      item.ulasan / item.rating * (item.stok < 30 ? 1.2 : 1.0);

    const kategori = calculateCategory(item.penjualan);

    features.push([
      normalize(
        [penjualanStats.mean],
        stats.penjualan.min,
        stats.penjualan.max
      )[0],
      penjualanStats.std / stats.penjualan.max,
      normalize([item.rating], stats.rating.min, stats.rating.max)[0],
      normalize([item.ulasan], stats.ulasan.min, stats.ulasan.max)[0],
      normalize([item.stok], stats.stok.min, stats.stok.max)[0],
      normalize([item.harga], stats.harga.min, stats.harga.max)[0],
      trendPenjualan,
      pricePerRating / stats.harga.max,
      engagementScore / 100,
      ...kategori
    ]);

    labels.push(kategori); // One-hot encoded
  });

  return { features, labels, stats };
}

// Function to calculate category for a product based on its sales trend
function calculateCategory(penjualan) {
  const avgPenjualan = penjualan.reduce((a, b) => a + b, 0) / penjualan.length;

  if (penjualan[penjualan.length - 1] > avgPenjualan) {
    return [0, 0, 1]; // Banyak diminati
  } else if (penjualan[penjualan.length - 1] === avgPenjualan) {
    return [0, 1, 0]; // Sedang diminati
  } else {
    return [1, 0, 0]; // Kurang diminati
  }
}

// Prepare features and labels
const { features: X, labels: Y, stats } = prepareFeatures(dataMinuman);

// Convert to tensors
const xs = tf.tensor2d(X);
const ys = tf.tensor2d(Y);

// Create enhanced model architecture
const model = tf.sequential();

model.add(tf.layers.batchNormalization({ inputShape: [12] })); // inputShape = 12 karena ada 12 fitur

model.add(
  tf.layers.dense({
    units: 128,
    activation: "swish",
    kernelInitializer: "glorotNormal",
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  })
);

model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.batchNormalization());

model.add(
  tf.layers.dense({
    units: 64,
    activation: "swish",
    kernelInitializer: "glorotNormal",
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  })
);
model.add(tf.layers.dropout({ rate: 0.4 }));
model.add(tf.layers.batchNormalization());

model.add(
  tf.layers.dense({
    units: 3,
    activation: "softmax",
    kernelInitializer: "glorotNormal",
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  })
);

const learningRate = 0.001;
const optimizer = tf.train.adam(learningRate);

model.compile({
  optimizer: optimizer,
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"]
});

async function trainModel() {
  const earlyStopping = tf.callbacks.earlyStopping({
    monitor: "loss",
    patience: 20,
    minDelta: 0.0001
  });

  const totalEpochs = 1000;
  const loopCount = 5; // Menentukan berapa kali loop untuk meningkatkan akurasi

  for (let i = 0; i < loopCount; i++) {
    console.log(`Training Loop ${i + 1} of ${loopCount}`);

    await model.fit(xs, ys, {
      epochs: totalEpochs / loopCount, // Membagi epoch untuk setiap iterasi loop
      batchSize: 4,
      validationSplit: 0.2,
      callbacks: [earlyStopping],
      shuffle: true,
      verbose: 1
    });

    console.log(`Training Loop ${i + 1} Completed`);
  }

  console.log("Training completed");

  const testFeatures = prediksi.map(item => {
    const penjualanStats = calculateStats(item.penjualan);
    const trendPenjualan =
      (item.penjualan[item.penjualan.length - 1] - item.penjualan[0]) /
      item.penjualan[0];
    const pricePerRating = item.harga / item.rating;
    const engagementScore =
      item.ulasan / item.rating * (item.stok < 30 ? 1.2 : 1.0);

    const kategori = calculateCategory(item.penjualan);

    return [
      normalize(
        [penjualanStats.mean],
        stats.penjualan.min,
        stats.penjualan.max
      )[0],
      penjualanStats.std / stats.penjualan.max,
      normalize([item.rating], stats.rating.min, stats.rating.max)[0],
      normalize([item.ulasan], stats.ulasan.min, stats.ulasan.max)[0],
      normalize([item.stok], stats.stok.min, stats.stok.max)[0],
      normalize([item.harga], stats.harga.min, stats.harga.max)[0],
      trendPenjualan,
      pricePerRating / stats.harga.max,
      engagementScore / 100,
      ...kategori
    ];
  });

  const testTensor = tf.tensor2d(testFeatures);
  const predictions = await model.predict(testTensor).array();

  const kategori = ["Kurang diminati", "Sedang diminati", "Banyak diminati"];

  prediksi.forEach((item, index) => {
    const prediksi = predictions[index];
    const kategoriIndex = prediksi.indexOf(Math.max(...prediksi));
    const confidence = prediksi[kategoriIndex] * 100;

    console.log(`\nAnalisis untuk ${item.nama}:`);
    console.log(`Kategori: ${kategori[kategoriIndex]}`);
    console.log(`Confidence: ${confidence.toFixed(2)}%`);
    console.log("Probabilitas per kategori:");
    kategori.forEach((kat, idx) => {
      console.log(`- ${kat}: ${(prediksi[idx] * 100).toFixed(2)}%`);
    });
  });
}

trainModel();
