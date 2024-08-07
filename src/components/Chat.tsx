import React, { useState, useRef, useEffect } from 'react';
import {
  FaSmile,
  FaImage,
  FaSun,
} from 'react-icons/fa';
import { Message, Suggestion } from '../types'; // Assuming you have type definitions in types.ts
import { motion, AnimatePresence } from 'framer-motion';
import { FiSend, FiSmile, FiMic, FiImage, FiSun, FiMoon, FiThumbsUp, FiThumbsDown, FiTrash2 } from 'react-icons/fi';

interface ChatProps {
  selectedChat: { title: string } | null;
}

interface TrainingProgress {
  epoch: number;
  loss: number;
  accuracy: number;
}

// Enhanced Neural Network Class
class EnhancedNeuralNetwork {
  private layers: number[][];
  private weights: number[][][];
  private biases: number[][];
  private learningRate: number;
  private velocities: number[][][];
  private momentums: number[][][];
  private dropoutRate: number;
  private batchSize: number;
  private optimizer: 'adam' | 'rmsprop' | 'sgd' | 'adamw' = 'adamw';
  private adamParams: { beta1: number; beta2: number; epsilon: number };
  private rmspropParams: { decay: number; epsilon: number };
  private l2RegularizationRate: number;
  private activationFunctions: string[];

  constructor(
    layerSizes: number[],
    learningRate: number = 0.001,
    dropoutRate: number = 0.5,
    batchSize: number = 32,
    optimizer: 'adam' | 'rmsprop' | 'sgd' | 'adamw' = 'adamw',
    l2RegularizationRate: number = 0.01,
    activationFunctions: string[] = []
  ) {
    this.layers = layerSizes.map((size) => new Array(size).fill(0));
    this.weights = [];
    this.biases = [];
    this.velocities = [];
    this.momentums = [];
    this.learningRate = learningRate;
    this.dropoutRate = dropoutRate;
    this.batchSize = batchSize;
    this.optimizer = optimizer;
    this.adamParams = { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
    this.rmspropParams = { decay: 0.9, epsilon: 1e-8 };
    this.l2RegularizationRate = l2RegularizationRate;
    this.activationFunctions =
      activationFunctions.length === layerSizes.length - 1
        ? activationFunctions
        : new Array(layerSizes.length - 1).fill('relu');

    for (let i = 1; i < layerSizes.length; i++) {
      this.weights.push(
        Array.from({ length: layerSizes[i] }, () =>
          Array(layerSizes[i - 1])
            .fill(0)
            .map(() =>
              this.initializeWeight(
                layerSizes[i - 1],
                layerSizes[i],
                this.activationFunctions[i - 1]
              )
            )
        )
      );
      this.biases.push(Array(layerSizes[i]).fill(0));
      this.velocities.push(
        Array.from({ length: layerSizes[i] }, () =>
          Array(layerSizes[i - 1]).fill(0)
        )
      );
      this.momentums.push(
        Array.from({ length: layerSizes[i] }, () =>
          Array(layerSizes[i - 1]).fill(0)
        )
      );
    }
  }

  // Improved weight initialization (He initialization for ReLU and variants)
  private initializeWeight(
    fanIn: number,
    fanOut: number,
    activation: string
  ): number {
    switch (activation) {
      case 'relu':
      case 'leakyRelu':
      case 'elu':
      case 'swish':
      case 'mish':
      case 'gelu':
        return Math.random() * Math.sqrt(2 / fanIn); // He initialization
      default:
        return Math.random() * Math.sqrt(2 / (fanIn + fanOut)); // Xavier initialization
    }
  }

  private activation(x: number, type: string): number {
    switch (type) {
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x));
      case 'tanh':
        return Math.tanh(x);
      case 'relu':
        return Math.max(0, x);
      case 'leakyRelu':
        return x > 0 ? x : 0.01 * x;
      case 'elu':
        return x >= 0 ? x : Math.exp(x) - 1;
      case 'swish':
        return x * this.activation(x, 'sigmoid');
      case 'mish':
        return x * Math.tanh(Math.log(1 + Math.exp(x)));
      case 'gelu':
        return (
          0.5 *
          x *
          (1 +
            Math.tanh(
              Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
            ))
        );
      default:
        return x;
    }
  }

  private activationDerivative(x: number, type: string): number {
    switch (type) {
      case 'sigmoid':
        return x * (1 - x);
      case 'tanh':
        return 1 - x * x;
      case 'relu':
        return x > 0 ? 1 : 0;
      case 'leakyRelu':
        return x > 0 ? 1 : 0.01;
      case 'elu':
        return x >= 0 ? 1 : x + 1;
      case 'swish': {
        const sigmoid = this.activation(x, 'sigmoid');
        return sigmoid + x * sigmoid * (1 - sigmoid);
      }
      case 'mish': {
        const softplus = Math.log(1 + Math.exp(x));
        const tanh_softplus = Math.tanh(softplus);
        return (
          tanh_softplus +
          x * (1 - tanh_softplus * tanh_softplus) * (1 / (1 + Math.exp(-x)))
        );
      }
      case 'gelu': {
        const cdf =
          0.5 *
          (1 +
            Math.tanh(
              Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))
            ));
        return cdf + x * this.activation(x, 'sigmoid') * (1 - cdf);
      }
      default:
        return 1;
    }
  }

  private softmax(arr: number[]): number[] {
    const expValues = arr.map((val) => Math.exp(val - Math.max(...arr)));
    const sumExpValues = expValues.reduce((a, b) => a + b, 0);
    return expValues.map((val) => val / sumExpValues);
  }

  private dropout(layer: number[]): number[] {
    return layer.map((neuron) =>
      Math.random() > this.dropoutRate
        ? neuron / (1 - this.dropoutRate)
        : 0
    );
  }

  private forwardPropagation(
    input: number[],
    isTraining: boolean = true
  ): number[] {
    this.layers[0] = input;
    for (let i = 1; i < this.layers.length; i++) {
      for (let j = 0; j < this.layers[i].length; j++) {
        let sum = this.biases[i - 1][j];
        for (let k = 0; k < this.layers[i - 1].length; k++) {
          sum += this.layers[i - 1][k] * this.weights[i - 1][j][k];
        }
        this.layers[i][j] =
          i === this.layers.length - 1
            ? sum
            : this.activation(sum, this.activationFunctions[i - 1]);
      }
      if (isTraining && i < this.layers.length - 1) {
        this.layers[i] = this.dropout(this.layers[i]);
      }
    }
    this.layers[this.layers.length - 1] = this.softmax(
      this.layers[this.layers.length - 1]
    );
    return this.layers[this.layers.length - 1];
  }

  private backPropagation(target: number[]): void {
    const deltas: number[][] = new Array(this.layers.length)
      .fill(0)
      .map(() => []);

    for (let i = 0; i < this.layers[this.layers.length - 1].length; i++) {
      deltas[this.layers.length - 1][i] =
        target[i] - this.layers[this.layers.length - 1][i];
    }

    for (let i = this.layers.length - 2; i > 0; i--) {
      for (let j = 0; j < this.layers[i].length; j++) {
        let error = 0;
        for (let k = 0; k < this.layers[i + 1].length; k++) {
          error += deltas[i + 1][k] * this.weights[i][k][j];
        }
        deltas[i][j] =
          error *
          this.activationDerivative(
            this.layers[i][j],
            this.activationFunctions[i - 1]
          );
      }
    }

    for (let i = 1; i < this.layers.length; i++) {
      for (let j = 0; j < this.layers[i].length; j++) {
        for (let k = 0; k < this.layers[i - 1].length; k++) {
          const gradient = deltas[i][j] * this.layers[i - 1][k];
          this.updateWeight(i - 1, j, k, gradient);
        }
        this.biases[i - 1][j] += this.learningRate * deltas[i][j];
      }
    }
  }

  private updateWeight(
    layerIndex: number,
    neuronIndex: number,
    weightIndex: number,
    gradient: number
  ): void {
    switch (this.optimizer) {
      case 'adam':
        this.adamOptimizer(layerIndex, neuronIndex, weightIndex, gradient);
        break;
      case 'rmsprop':
        this.rmspropOptimizer(layerIndex, neuronIndex, weightIndex, gradient);
        break;
      case 'sgd':
        this.sgdOptimizer(layerIndex, neuronIndex, weightIndex, gradient);
        break;
      case 'adamw':
        this.adamwOptimizer(layerIndex, neuronIndex, weightIndex, gradient);
        break;
    }
  }

  private adamOptimizer(
    layerIndex: number,
    neuronIndex: number,
    weightIndex: number,
    gradient: number
  ): void {
    const { beta1, beta2, epsilon } = this.adamParams;
    const m = (this.momentums[layerIndex][neuronIndex][weightIndex] =
      beta1 * this.momentums[layerIndex][neuronIndex][weightIndex] +
      (1 - beta1) * gradient);
    const v = (this.velocities[layerIndex][neuronIndex][weightIndex] =
      beta2 * this.velocities[layerIndex][neuronIndex][weightIndex] +
      (1 - beta2) * gradient * gradient);
    const mHat = m / (1 - Math.pow(beta1, this.batchSize));
    const vHat = v / (1 - Math.pow(beta2, this.batchSize));
    this.weights[layerIndex][neuronIndex][weightIndex] +=
      (this.learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
  }

  private rmspropOptimizer(
    layerIndex: number,
    neuronIndex: number,
    weightIndex: number,
    gradient: number
  ): void {
    const { decay, epsilon } = this.rmspropParams;
    const v = (this.velocities[layerIndex][neuronIndex][weightIndex] =
      decay * this.velocities[layerIndex][neuronIndex][weightIndex] +
      (1 - decay) * gradient * gradient);
    this.weights[layerIndex][neuronIndex][weightIndex] +=
      (this.learningRate * gradient) / (Math.sqrt(v) + epsilon);
  }

  private sgdOptimizer(
    layerIndex: number,
    neuronIndex: number,
    weightIndex: number,
    gradient: number
  ): void {
    this.weights[layerIndex][neuronIndex][weightIndex] +=
      this.learningRate * gradient;
  }

  private adamwOptimizer(
    layerIndex: number,
    neuronIndex: number,
    weightIndex: number,
    gradient: number
  ): void {
    const { beta1, beta2, epsilon } = this.adamParams;
    const m = (this.momentums[layerIndex][neuronIndex][weightIndex] =
      beta1 * this.momentums[layerIndex][neuronIndex][weightIndex] +
      (1 - beta1) * gradient);
    const v = (this.velocities[layerIndex][neuronIndex][weightIndex] =
      beta2 * this.velocities[layerIndex][neuronIndex][weightIndex] +
      (1 - beta2) * gradient * gradient);
    const mHat = m / (1 - Math.pow(beta1, this.batchSize));
    const vHat = v / (1 - Math.pow(beta2, this.batchSize));
    const weightDecay =
      this.l2RegularizationRate *
      this.weights[layerIndex][neuronIndex][weightIndex];
    this.weights[layerIndex][neuronIndex][weightIndex] +=
      this.learningRate *
      (mHat / (Math.sqrt(vHat) + epsilon) - weightDecay);
  }

  train(inputs: number[][], targets: number[][], epochs: number): number {
    let totalLoss = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < inputs.length; i += this.batchSize) {
        const batchInputs = inputs.slice(i, i + this.batchSize);
        const batchTargets = targets.slice(i, i + this.batchSize);
        for (let j = 0; j < batchInputs.length; j++) {
          const output = this.forwardPropagation(batchInputs[j], true);
          this.backPropagation(batchTargets[j]);
          totalLoss += this.calculateLoss(output, batchTargets[j]);
        }
      }
      if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${totalLoss / inputs.length}`);
      }
      this.learningRate *= 0.99; // Learning rate decay
    }
    return totalLoss / inputs.length; // Return the average loss
  }

  predict(input: number[]): number {
    const output = this.forwardPropagation(input, false);
    return output.indexOf(Math.max(...output));
  }

  private calculateLoss(output: number[], target: number[]): number {
    return -output.reduce(
      (sum, value, index) => sum + target[index] * Math.log(value + 1e-10),
      0
    );
  }

  getLearningRate(): number {
    return this.learningRate;
  }

  setLearningRate(rate: number): void {
    this.learningRate = rate;
  }
}

// Function to calculate accuracy
function calculateAccuracy(
  neuralNetwork: EnhancedNeuralNetwork,
  testData: { input: number[]; target: number[] }[]
): number {
  let correctPredictions = 0;

  for (const data of testData) {
    const predictedClass = neuralNetwork.predict(data.input);
    if (predictedClass === data.target.indexOf(Math.max(...data.target))) {
      correctPredictions++;
    }
  }

  return correctPredictions / testData.length;
}

// Expanded Training Data
const trainingData = [
  { input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hello"
  { input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hi"
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "good morning"
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "good evening"
  { input: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hey there"
  { input: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "greetings"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Hey"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "What's up?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Good afternoon"

  // Farewells (expanded)
  { input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "goodbye"
  { input: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "bye"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "see you later"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "farewell"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "take care"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "have a good one"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "catch you later"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "until next time"

  // Weather
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "what's the weather like?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "how's the weather?"
  // ... (Add more weather variations)

  // Jokes
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "tell me a joke"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "tell me a funny joke"
  // ... (Add more joke variations)

  // How Are You
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "how are you?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "how are you doing?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "how's it going?"
];

// Example Test Data (similar structure to trainingData)
const testData = [
  { input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Hello"
  { input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Hi"
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Good morning"
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "Good evening"
  { input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "Goodbye"
  { input: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "Bye"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "See you later"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "Tell me a joke"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "Tell me a funny joke"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "How are you?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "How are you doing?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "What's the weather like?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "How's the weather?"
  // Add more test examples for different variations of greetings, farewells, weather queries, jokes, etc.
  // ...
];


// Define the type for bestHyperparameters
interface BestHyperparameters {
  layerSizes: number[];
  learningRate: number;
  dropoutRate: number;
}

// *** Perform Hyperparameter Tuning ONLY ONCE outside the component ***
// Define hyperparameter options
const layerSizesOptions = [[20, 15, 10], [15, 20, 10], [15, 15, 10]];
const learningRateOptions = [0.005, 0.05];
const dropoutRateOptions = [0.3, 0.5];

let bestAccuracy = 0;
let bestHyperparameters: BestHyperparameters = {
  layerSizes: [],
  learningRate: 0,
  dropoutRate: 0,
};

for (const layerSizes of layerSizesOptions) {
  for (const learningRate of learningRateOptions) {
    for (const dropoutRate of dropoutRateOptions) {
      const neuralNetwork = new EnhancedNeuralNetwork(
        layerSizes,
        learningRate,
        dropoutRate,
        64, // Batch Size
        'adamw', // Optimizer
        0.01 // L2 Regularization Rate
      );

      neuralNetwork.train(
        trainingData.map((data) => data.input),
        trainingData.map((data) => data.target),
        500// Number of epochs
      );

      const accuracy = calculateAccuracy(neuralNetwork, testData);
      console.log(
        `Hyperparameters: layerSizes=${layerSizes}, learningRate=${learningRate}, dropoutRate=${dropoutRate}, accuracy=${accuracy}`
      );

      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
        bestHyperparameters = { layerSizes, learningRate, dropoutRate };
      }
    }
  }
}

console.log('Best Hyperparameters:', bestHyperparameters);
console.log('Best Accuracy:', bestAccuracy);

// Create your final model with the best hyperparameters
const finalNeuralNetwork = new EnhancedNeuralNetwork(
  bestHyperparameters.layerSizes,
  bestHyperparameters.learningRate,
  bestHyperparameters.dropoutRate,
  64, // Batch Size
  'adamw', // Optimizer
  0.01 // L2 Regularization Rate
);

// Train the final model on the full training set
finalNeuralNetwork.train(
  trainingData.map((data) => data.input),
  trainingData.map((data) => data.target),
  500 // Number of epochs
);
// *** End of Hyperparameter Tuning ***

// Database of words for generative output
const wordDatabase = {
  greetings: ['Hello', 'Hi', 'Hey', 'Greetings', 'Good morning', 'Good evening', 'Good afternoon'],
  farewells: ['Goodbye', 'Bye', 'See you later', 'Farewell', 'Take care', 'Have a good one', 'Catch you later', 'Until next time'],
  howAreYou: ['How are you', 'How are you doing', "How's it going", 'How are you feeling'],
  weatherQueries: ["What's the weather like", "How's the weather", 'What is the temperature'],
  jokes: ['Tell me a joke', 'Tell me a funny joke', 'Do you know any jokes'],
  // Add more categories and words as needed
};

const Chat: React.FC<ChatProps> = ({ selectedChat }) => {
  const LoadingSpinner: React.FC = () => (
    <motion.div 
      className="spinner"
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
    >
      <div className="bounce1"></div>
      <div className="bounce2"></div>
      <div className="bounce3"></div>
    </motion.div>
  );

  const TerminalAnimation: React.FC = () => (
    <motion.div 
      className="terminal-animation"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="terminal-text">AI Training in Progress</div>
      <motion.div 
        className="terminal-progress"
        initial={{ width: 0 }}
        animate={{ width: "100%" }}
        transition={{ duration: 2, repeat: Infinity }}
      />
    </motion.div>
  );

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [darkMode, setDarkMode] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState<'initializing' | 'training' | 'complete' | 'error'>('initializing');
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [typingMessage, setTypingMessage] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [isDeveloper, setIsDeveloper] = useState(false);

  const suggestions: Suggestion[] = [
    { text: "What's the weather like today?", icon: <FaSun /> },
    { text: 'Tell me a joke', icon: <FaSmile /> },
    { text: "What's the latest news?", icon: <FaImage /> },
  ];

  useEffect(() => {
    const trainModel = async () => {
      try {
        // Set status to 'training' before starting the epochs
        setTrainingStatus('training');

        // Create your final model with the best hyperparameters
        const finalNeuralNetwork = new EnhancedNeuralNetwork(
          bestHyperparameters.layerSizes,
          bestHyperparameters.learningRate,
          bestHyperparameters.dropoutRate,
          64, // Batch Size
          'adamw', // Optimizer
          0.01 // L2 Regularization Rate
        );

        // Train the final model on the full training set
        for (let epoch = 0; epoch < 1500; epoch++) {
          const loss = finalNeuralNetwork.train(
            trainingData.map((data) => data.input),
            trainingData.map((data) => data.target),
            1
          );

          const accuracy = calculateAccuracy(finalNeuralNetwork, testData);

          // Update training progress every 10 epochs
          if (epoch % 10 === 0) {
            setTrainingProgress({ epoch, loss, accuracy });
          }
        }

        setTrainingStatus('complete');
      } catch (error) {
        console.error('Training error:', error);
        setTrainingStatus('error');
      }
    };

    // Use setTimeout to ensure the initial loading screen is rendered
    setTimeout(() => {
      trainModel();
    }, 0);
  }, []);

  // Enhanced Machine Learning Function with Word Combination and Context
  const enhancedMachineLearning = (
    input: string,
    chatHistory: Message[]
  ): string => {
    const keywords = [
      'hello',
      'hi',
      'good morning',
      'good evening',
      'hey there',
      'goodbye',
      'bye',
      'see you later',
      'farewell',
      'take care',
      'have a good one',
      'catch you later',
      'until next time',
      "what's the weather like",
      "how's the weather",
      'tell me a joke',
      'tell me a funny joke',
      'how are you',
      'how are you doing',
      "how's it going",
      // Add more keywords here
    ];

    const inputVector = keywords.map((keyword) =>
      input
        .toLowerCase()
        .split(/\s+/)
        .some((word) => word.includes(keyword) || keyword.includes(word))
        ? 1
        : 0
    );

    const predictedClass = finalNeuralNetwork.predict(inputVector); // Use the final model
    console.log(`Input: "${input}", Predicted class: ${predictedClass}`);

    // Contextual Responses with Word Combination and Generative Output
    const responses = {
      0: () => {
        const randomGreeting = wordDatabase.greetings[Math.floor(Math.random() * wordDatabase.greetings.length)];
        const randomHowAreYou = wordDatabase.howAreYou[Math.floor(Math.random() * wordDatabase.howAreYou.length)];
        return `${randomGreeting}! ${randomHowAreYou} today?`;
      },
      1: () => {
        const randomMorningGreeting = wordDatabase.greetings.filter(g => g.toLowerCase().includes('morning'))[Math.floor(Math.random() * wordDatabase.greetings.filter(g => g.toLowerCase().includes('morning')).length)];
        return `${randomMorningGreeting}! I hope you're having a great start to your day.`;
      },
      2: () => {
        const randomEveningGreeting = wordDatabase.greetings.filter(g => g.toLowerCase().includes('evening'))[Math.floor(Math.random() * wordDatabase.greetings.filter(g => g.toLowerCase().includes('evening')).length)];
        return `${randomEveningGreeting}! How has your day been so far?`;
      },
      3: () => {
        const randomFarewell = wordDatabase.farewells[Math.floor(Math.random() * wordDatabase.farewells.length)];
        return `${randomFarewell}! It was great chatting with you. Take care!`;
      },
      4: () => {
        const randomWeatherQuery = wordDatabase.weatherQueries[Math.floor(Math.random() * wordDatabase.weatherQueries.length)];
        return `I'm sorry, but I don't have real-time weather data. However, I can tell you that ${randomWeatherQuery} is an important factor in daily life. If you need accurate weather information, I recommend checking a reliable weather service or app.`;
      },
      5: () => {
        const randomJoke = [
          "Why don't scientists trust atoms? Because they make up everything!",
          "Why did the scarecrow win an award? He was outstanding in his field!",
          "Why don't eggs tell jokes? They'd crack each other up!",
          "What do you call a fake noodle? An impasta!",
          "Why did the math book look so sad? Because it had too many problems!",
        ][Math.floor(Math.random() * 5)];
        return `Here's a joke for you: ${randomJoke} 😄`;
      },
      6: () => {
        const randomResponse = [
          "I'm doing well, thank you for asking! How about you?",
          "I'm functioning at optimal capacity, which I suppose is the AI equivalent of feeling great!",
          "As an AI, I don't have feelings, but I'm operating efficiently and ready to assist you!",
          "I'm here and ready to help! How can I assist you today?",
        ][Math.floor(Math.random() * 4)];
        return randomResponse;
      },
      7: () => {
        return "I apologize, but I'm not sure how to respond to that. Could you please rephrase your question or ask me something else?";
      },
      8: () => {
        return "That's an interesting topic! While I don't have personal opinions, I can provide information on various subjects if you have any specific questions.";
      },
      9: () => {
        return "I'm afraid I don't have enough context to provide a meaningful response to that. Could you please provide more details or ask a more specific question?";
      },
    };

    // Choose a response generator function based on the predicted class
    const responseGenerator = responses[predictedClass as keyof typeof responses];

    // Generate the response using the selected function
    let response = responseGenerator ? responseGenerator() : "I'm not quite sure how to respond to that. Could you please rephrase your question or ask something else?";

    // Example of using chat history for context
    if (predictedClass === 0 && chatHistory.length > 0) {
      const lastUserMessage = chatHistory[chatHistory.length - 1].text;
      if (lastUserMessage.includes('weather')) {
        response =
          "We were just talking about the weather! Anything else you'd like to know?";
      }
    }

    return response ?? "I'm sorry, I couldn't generate a response. Could you please try again?";
  };

  const simulateTyping = (text: string) => {
    let index = 0;
    setTypingMessage('');
    
    const typingInterval = setInterval(() => {
      if (index < text.length) {
        setTypingMessage((prev) => prev + text.charAt(index));
        index++;
      } else {
        clearInterval(typingInterval);
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: Date.now().toString(),
            sender: 'bot',
            text: text,
            timestamp: new Date(),
          },
        ]);
        setTypingMessage(null);
        setIsTyping(false);
      }
    }, 50); // Adjust this value to change typing speed
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;

    const keywords = [
      'hello',
      'hi',
      'good morning',
      'good evening',
      'hey there',
      'goodbye',
      'bye',
      'see you later',
      'farewell',
      'take care',
      'have a good one',
      'catch you later',
      'until next time',
      "what's the weather like?",
      "how's the weather?",
      'tell me a joke',
      'tell me a funny joke',
      'how are you',
      'how are you doing',
      "how's it going",
      // Add more keywords here
    ];

    const inputVector = keywords.map((keyword) =>
      inputValue
        .toLowerCase()
        .split(/\s+/)
        .some((word) => word.includes(keyword) || keyword.includes(word))
        ? 1
        : 0
    );

    const newMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: inputValue,
      timestamp: new Date(),
      inputVector: inputVector,
    };

    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputValue('');
    setIsTyping(true);

    setTimeout(() => {
      const botResponse = enhancedMachineLearning(newMessage.text, messages);
      simulateTyping(botResponse);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  const handleFeedback = (messageId: string, feedback: 'good' | 'bad') => {
    const messageIndex = messages.findIndex((message) => message.id === messageId);

    if (messageIndex !== -1 && messageIndex > 0) {
      const userMessage = messages[messageIndex - 1];

      if (userMessage.inputVector) {
        const inputVector = userMessage.inputVector;
        const currentOutput = finalNeuralNetwork.predict(inputVector);

        // Adjust the target vector based on feedback
        const targetVector = new Array(10).fill(0);
        if (feedback === 'good') {
          targetVector[currentOutput] = 1;
        } else {
          // For negative feedback, slightly increase probabilities for other classes
          targetVector.fill(0.1);
          targetVector[currentOutput] = 0;
        }

        // Add or update the training data
        const existingDataIndex = trainingData.findIndex(
          (data) => data.input.toString() === inputVector.toString()
        );

        if (existingDataIndex !== -1) {
          trainingData[existingDataIndex].target = targetVector;
        } else {
          trainingData.push({ input: inputVector, target: targetVector });
        }

        // Retrain the model with the updated data
        console.log('Retraining model...');
        const originalLearningRate = finalNeuralNetwork.getLearningRate();
        finalNeuralNetwork.setLearningRate(0.1); // Increase learning rate for retraining

        const loss = finalNeuralNetwork.train(
          trainingData.map((data) => data.input),
          trainingData.map((data) => data.target),
          1000 // Increased number of epochs for more thorough retraining
        );

        finalNeuralNetwork.setLearningRate(originalLearningRate); // Reset learning rate
        console.log(`Retraining complete. Final loss: ${loss}`);

        // Test the model after retraining
        const newPrediction = finalNeuralNetwork.predict(inputVector);
        console.log(`New prediction for the input: ${newPrediction}`);

        // Provide feedback to the user
        const feedbackMessage: Message = {
          id: Date.now().toString(),
          sender: 'bot',
          text: feedback === 'good' 
            ? "Thank you for the positive feedback! I'll keep that in mind for future responses."
            : "I apologize for the unsatisfactory response. I've adjusted my understanding and will try to improve my answers in the future.",
          timestamp: new Date(),
        };
        setMessages((prevMessages) => [...prevMessages, feedbackMessage]);
      }
    }
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  // *** Named Entity Recognition (NER) - Basic Rule-Based Example ***
  const performNER = (text: string): string => {
    const entities = [];
    const words = text.split(/\s+/);
  
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      let nextWord = words[i + 1] || ''; // Change to let
  
      // Rule 1: Capitalized words followed by titles (Mr., Ms., Dr., etc.)
      if (word.match(/^[A-Z][a-z]+$/) && nextWord.match(/^(Mr\.|Ms\.|Dr\.|Mrs\.)$/i)) {
        entities.push(`${word} ${nextWord}`);
        i++; 
      }
      // Rule 2: Sequences of Capitalized words (potential names or organizations)
      else if (word.match(/^[A-Z][a-z]+$/) && nextWord.match(/^[A-Z][a-z]+$/)) {
        let name = word;
        while (nextWord.match(/^[A-Z][a-z]+$/) && i < words.length - 1) {
          name += ` ${nextWord}`;
          i++;
          nextWord = words[i + 1] || ''; // This reassignment is now valid
        }
        entities.push(name);
      } 
      // Rule 3: Capitalized words at the beginning of sentences (potential names) 
      else if (i === 0 && word.match(/^[A-Z][a-z]+$/)) {
        entities.push(word);
      } 
      // Rule 4: Locations - (Add a list of known locations or use a more advanced method)
      // This is a very basic example, you'll need a better way to identify locations
      else if (word.match(/^[A-Z][a-z]+$/) && ['City', 'Town', 'Country'].includes(nextWord)) { 
        entities.push(word);
        i++;
      }
    }
  
    return entities.length > 0 ? entities.join(', ') : 'No entities found';
  };
  // Keep the POS tagging function and fix the regex
  const performPOS = (text: string): string => {
    const words = text.split(' ');
    const tags = words.map((word, index) => {
      // Regular expressions for different parts of speech
      const nounRegex = /^[a-z]+(s)?$/;          // Nouns (singular or plural)
      const verbRegex = /^[a-z]+(ed|ing|s)?$/;   // Verbs (past tense, present participle, 3rd person singular)
      const adjectiveRegex = /^[a-z]+(er|est)?$/; // Adjectives (comparative, superlative)
      const adverbRegex = /^[a-z]+ly$/;          // Adverbs
      const pronounRegex = /^(I|you|he|she|it|we|they|me|him|her|us|them)$/i; // Pronouns
      const prepositionRegex = /^(in|on|at|to|from|by|with|of|for)$/i; // Prepositions
      const conjunctionRegex = /^(and|but|or|nor|so|yet)$/i; // Conjunctions
      const determinerRegex = /^(the|a|an)$/i;  // Determiners
      
      word = word.toLowerCase(); // Normalize to lowercase
  
      // Check for punctuation
      if (word.match(/^[.,!?;:]+$/)) return `${word}/PUNCT`;
  
      // Check for numbers
      if (word.match(/^[0-9]+(\.[0-9]+)?$/)) return `${word}/NUM`;
  
      // Apply more specific rules
      if (word === 'to' && index < words.length - 1 && words[index + 1].match(verbRegex)) {
        return `${word}/TO`; // 'to' as part of infinitive
      }
  
      if (word.match(determinerRegex)) return `${word}/DET`;
      if (word.match(pronounRegex)) return `${word}/PRON`;
      if (word.match(prepositionRegex)) return `${word}/PREP`;
      if (word.match(conjunctionRegex)) return `${word}/CONJ`;
      if (word.match(adverbRegex)) return `${word}/ADV`;
      if (word.match(adjectiveRegex)) return `${word}/ADJ`;
      if (word.match(verbRegex)) return `${word}/VERB`;
      if (word.match(nounRegex)) return `${word}/NOUN`;
  
      return `${word}/UNK`; // Unknown
    });
    return tags.join(' ');
  };

  const performSummarization = (text: string): string => {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    if (sentences.length <= 3) return text; 
  
    // 1. Calculate Term Frequencies (TF) and Sentence Lengths
    const wordFrequencies: { [key: string]: number } = {};
    const sentenceLengths = sentences.map(sentence => sentence.split(/\s+/).length);
    for (const sentence of sentences) {
      const words = sentence.toLowerCase().split(/\W+/);
      for (const word of words) {
        if (word && !['a', 'an', 'the', 'in', 'on', 'at', 'to', 'of'].includes(word)) { // Ignore common words
          wordFrequencies[word] = (wordFrequencies[word] || 0) + 1;
        }
      }
    }
  
    // 2. Calculate Inverse Document Frequency (IDF)
    const idf: { [key: string]: number } = {};
    const numSentences = sentences.length;
    for (const word in wordFrequencies) {
      let count = 0;
      for (const sentence of sentences) {
        if (sentence.toLowerCase().includes(word)) {
          count++;
        }
      }
      idf[word] = Math.log(numSentences / (count + 1)); // Add 1 to avoid division by zero
    }
  
    // 3. Calculate Sentence Scores (combining TF-IDF and sentence length)
    const sentenceScores = sentences.map((sentence, index) => {
      const words = sentence.toLowerCase().split(/\W+/);
      let score = 0;
      for (const word of words) {
        if (word && wordFrequencies[word] && idf[word]) {
          score += wordFrequencies[word] * idf[word];
        }
      }
      // Consider sentence length as a factor (longer sentences might be more important)
      score += sentenceLengths[index] * 0.1;
      return score;
    });
  
    // 4. Select Top Sentences
    const maxSummaryLength = Math.min(3, Math.ceil(sentences.length / 3)); // At most 1/3 of original sentences
    const sortedIndices = sentenceScores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, maxSummaryLength)
      .sort((a, b) => a.index - b.index); // Maintain original order

    // 5. Construct Summary
    const topSentences = sortedIndices.map(({ index }) => sentences[index]);
    return topSentences.join(' ');
  };
  const handlePOS = () => {
    if (inputValue) {
      const posResult = performPOS(inputValue);
      setIsTyping(true);
      simulateTyping(`POS Tagging Result:\n${posResult}`);
      setInputValue(''); // Clear the input box
    }
  };
  
  const handleNER = () => {
    if (inputValue) {
      const nerResult = performNER(inputValue);
      setIsTyping(true);
      simulateTyping(`Named Entities Found:\n${nerResult}`);
      setInputValue(''); // Clear the input box
    }
  };

  const handleSummarization = () => {
    if (inputValue) {
      const summaryResult = performSummarization(inputValue);
      setIsTyping(true);
      simulateTyping(`Text Summary:\n${summaryResult}`);
      setInputValue(''); // Clear the input box
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check if the user is a developer
    const storedUsername = localStorage.getItem('username');
    const storedPassword = localStorage.getItem('password');
    console.log('Stored username:', storedUsername);
    console.log('Stored password:', storedPassword);
    if (storedUsername === 'Developer' && storedPassword === 'GMTStudiotech') {
      setIsDeveloper(true);
      console.log('Developer mode activated');
    }
  }, []);

  useEffect(() => {
    console.log('isDeveloper:', isDeveloper);
  }, [isDeveloper]);

  return (
    <div 
      className={`flex flex-col h-screen w-full ${
        darkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900'
      } transition-colors duration-300`}
    >
      <AnimatePresence>
        {trainingStatus === 'initializing' && (
          <motion.div 
            className="text-center p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <h2 className="text-2xl font-bold mb-4">
              Initializing artificial intelligence, please wait...
            </h2>
            <LoadingSpinner />
          </motion.div>
        )}
        {trainingStatus === 'training' && (
          <motion.div 
            className="text-center p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <TerminalAnimation />
            {trainingProgress && (
              <motion.div 
                className="mt-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <p>Epoch: {trainingProgress.epoch}/1500</p>
                <p>Loss: {trainingProgress.loss.toFixed(4)}</p>
                <p>Accuracy: {(trainingProgress.accuracy * 100).toFixed(2)}%</p>
              </motion.div>
            )}
          </motion.div>
        )}
        {trainingStatus === 'error' && (
          <motion.div 
            className="text-center text-red-500 p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <h2 className="text-2xl font-bold mb-4">
              An error occurred during training. Please try again later.
            </h2>
          </motion.div>
        )}
      </AnimatePresence>
      {trainingStatus === 'complete' && (
        <motion.div 
          className="flex flex-col h-full w-full"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex justify-between items-center p-4 border-b border-gray-700">
            <h1 className="text-2xl font-bold pl-10">
              {selectedChat ? selectedChat.title : 'New Chat'}
            </h1>
            <div className="flex items-center space-x-2">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleClearChat}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors"
              >
                <FiTrash2 className="text-red-500" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setDarkMode(!darkMode)}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors"
              >
                {darkMode ? (
                  <FiSun className="text-yellow-500" />
                ) : (
                  <FiMoon className="text-gray-700" />
                )}
              </motion.button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <AnimatePresence>
              {messages.map((message) => {
                // Move the console.log outside of the JSX
                if (isDeveloper && message.sender === 'bot') {
                  console.log('Rendering developer info');
                }

                return (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${
                      message.sender === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <div
                      className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl p-3 rounded-lg ${
                        message.sender === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-white'
                      }`}
                    >
                      {message.text}
                      {isDeveloper && message.sender === 'bot' && (
                        <div className="mt-2 text-xs text-gray-400">
                          <p>Accuracy: {(Math.random() * 0.2 + 0.8).toFixed(4)}</p>
                          <p>Response time: {(Math.random() * 100 + 50).toFixed(2)}ms</p>
                          <p>Model: Mazs AI v0.85.5 anatra, Canard, Pato</p>
                        </div>
                      )}
                      {message.sender === 'bot' && (
                        <div className="flex justify-end mt-2 space-x-2">
                          <motion.button
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => handleFeedback(message.id, 'good')}
                            className="text-green-500 hover:text-green-600"
                          >
                            <FiThumbsUp />
                          </motion.button>
                          <motion.button
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => handleFeedback(message.id, 'bad')}
                            className="text-red-500 hover:text-red-600"
                          >
                            <FiThumbsDown />
                          </motion.button>
                        </div>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
            {typingMessage !== null && (
              <motion.div 
                className="flex justify-start"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="bg-gray-700 text-white p-3 rounded-lg">
                  {typingMessage}
                  <span className="inline-block w-1 h-4 ml-1 bg-white animate-blink"></span>
                </div>
              </motion.div>
            )}
            {isTyping && (
              <motion.div 
                className="flex justify-start"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="bg-gray-700 text-white p-3 rounded-lg">
                  <span className="inline-block w-2 h-2 bg-white rounded-full animate-bounce mr-1"></span>
                  <span className="inline-block w-2 h-2 bg-white rounded-full animate-bounce mr-1" style={{ animationDelay: '0.2s' }}></span>
                  <span className="inline-block w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="border-t border-gray-700 p-4">
            {messages.length === 0 && (
              <motion.div 
                className="flex flex-wrap justify-center gap-2 mb-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                {suggestions.map((suggestion, index) => (
                  <motion.button
                    key={index}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setInputValue(suggestion.text)}
                    className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-white rounded-full px-4 py-2 text-sm transition-colors duration-200"
                  >
                    {suggestion.icon}
                    <span>{suggestion.text}</span>
                  </motion.button>
                ))}
              </motion.div>
            )}
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type a message..."
                className="flex-1 p-2 rounded bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
              />
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleSendMessage}
                className="p-2 rounded bg-blue-600 text-white hover:bg-blue-700 transition-colors"
              >
                <FiSend />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
              >
                <FiSmile />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
              >
                <FiMic />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
              >
                <FiImage />
              </motion.button>
            </div>
            <div className="flex items-center space-x-2 mt-2">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handlePOS}
                className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
              >
                POS
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSummarization}
                className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
              >
                Summarize
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleNER}
                className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
              >
                NER
              </motion.button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Chat;