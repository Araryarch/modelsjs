import * as tf from "@tensorflow/tfjs";

// Data penjualan untuk beberapa hari
import { dataMinuman } from "./dataset/data.js";
import { Y } from "./dataset/label.js";

// data
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

  const features = data.map(item => {
    const penjualanStats = calculateStats(item.penjualan);
    const trendPenjualan =
      (item.penjualan[item.penjualan.length - 1] - item.penjualan[0]) /
      item.penjualan[0];
    const pricePerRating = item.harga / item.rating;
    const engagementScore =
      item.ulasan / item.rating * (item.stok < 30 ? 1.2 : 1.0);

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
      engagementScore / 100
    ];
  });

  return { features, stats };
}

// Prepare features
const { features: X, stats } = prepareFeatures(dataMinuman);

// Convert to tensors
const xs = tf.tensor2d(X);
const ys = tf.tensor2d(Y);

// Create enhanced model architecture
const model = tf.sequential();

model.add(tf.layers.batchNormalization({ inputShape: [9] }));

model.add(
  tf.layers.dense({
    units: 128, // Menambah jumlah unit
    activation: "swish", // Aktivasi yang lebih canggih
    kernelInitializer: "glorotNormal",
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  })
);

model.add(tf.layers.dropout({ rate: 0.5 })); // Menambah dropout untuk menghindari overfitting
model.add(tf.layers.batchNormalization());

model.add(
  tf.layers.dense({
    units: 64, // Menambah lapisan lebih banyak
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

  await model.fit(xs, ys, {
    epochs: 300,
    batchSize: 4,
    validationSplit: 0.2,
    callbacks: [earlyStopping],
    shuffle: true,
    verbose: 1
  });

  console.log("Training completed");

  const testFeatures = prediksi.map(item => {
    const penjualanStats = calculateStats(item.penjualan);
    const trendPenjualan =
      (item.penjualan[item.penjualan.length - 1] - item.penjualan[0]) /
      item.penjualan[0];
    const pricePerRating = item.harga / item.rating;
    const engagementScore =
      item.ulasan / item.rating * (item.stok < 30 ? 1.2 : 1.0);

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
      engagementScore / 100
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
