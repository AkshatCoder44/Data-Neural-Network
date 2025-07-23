function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  let s = sigmoid(x);
  return s * (1 - s);
}

let weight = Math.random();
let bias = Math.random();

const data = [
  { input: 0.0, target: 0.0 },
  { input: 0.2, target: 0.0 },
  { input: 0.4, target: 0.0 },
  { input: 0.6, target: 1.0 },
  { input: 0.8, target: 1.0 },
  { input: 1.0, target: 1.0 },
];

const data0 = [0.1, 0.3, 0.5, 0.7, 0.9];

const lr = 0.1;
const epochs = 100000;

function forward(x) {
  const z = weight * x + bias;
  return sigmoid(z);
}

for (let epoch = 0; epoch < epochs; epoch++) {
  for (let i = 0; i < data.length; i++) {
    const { input, target } = data[i];
    const z = weight * input + bias;
    const output = sigmoid(z);
    const error = output - target;
    const dOutput = error * sigmoidDerivative(z);

    weight -= lr * dOutput * input;
    bias -= lr * dOutput;
  }
}

console.log("Testing the trained neuron:");
for (const input of data0) {
  const prediction = forward(input);
  console.log(`Input: ${input.toFixed(1)}, Predicted: ${prediction.toFixed(4)}`);
}

console.log(`\nFinal weight: ${weight.toFixed(4)}`);
console.log(`Final bias: ${bias.toFixed(4)}`);
