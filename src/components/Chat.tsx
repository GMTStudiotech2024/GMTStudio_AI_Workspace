import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FiSend,
  FiMic,
  FiImage,
  FiSun,
  FiMoon,
  FiThumbsUp,
  FiThumbsDown,
  FiTrash2,
  FiCheck,
  FiX,
  FiSettings,
  FiPause
} from 'react-icons/fi';
import { Message as ImportedMessage } from '../types'; // Make sure you have the correct import for your Message type

interface ChatProps {
  selectedChat: { title: string } | null;
}

interface Suggestion {
  text: string;
  icon: React.ReactNode;
}

// Simplified Speech Recognition types
interface SpeechRecognitionResult {
  transcript: string;
}

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResult[][];
}

interface SpeechRecognitionInstance {
  start: () => void;
  onresult: (event: SpeechRecognitionEvent) => void;
  onend: () => void;
  onstart: () => void;
}

// Extend the Window interface
declare global {
  interface Window {
    webkitSpeechRecognition: {
      new (): SpeechRecognitionInstance;
    };
  }
}

// Define a local interface with a different name
interface ChatMessage extends ImportedMessage {
  image?: string;
  inputVector?: number[];
  confirmationType?: 'math' | 'summary';
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
  private attentionWeights: number[][] = []; // Initialize with an empty array
  private useAttention: boolean;

  constructor(
    layerSizes: number[],
    learningRate: number = 0.001,
    dropoutRate: number = 0.5,
    batchSize: number = 32,
    optimizer: 'adam' | 'rmsprop' | 'sgd' | 'adamw' = 'adamw',
    l2RegularizationRate: number = 0.01,
    activationFunctions: string[] = [],
    useAttention: boolean = false
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
    this.useAttention = useAttention;
    if (this.useAttention) {
      this.attentionWeights = Array.from({ length: layerSizes[layerSizes.length - 2] }, () =>
        Array(layerSizes[layerSizes.length - 1]).fill(0).map(() => Math.random())
      );
    }
  }

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

  private attentionMechanism(input: number[]): number[] {
    if (!this.useAttention) return input;

    const attentionScores = this.attentionWeights.map(weights =>
      weights.reduce((sum, weight, i) => sum + weight * input[i], 0)
    );
    const softmaxScores = this.softmax(attentionScores);
    return input.map((value, i) => value * softmaxScores[i]);
  }

  private forwardPropagation(input: number[], isTraining: boolean = true): number[] {
    this.layers[0] = input;
    for (let i = 1; i < this.layers.length; i++) {
      let layerInput = this.layers[i - 1];
      if (i === this.layers.length - 1 && this.useAttention) {
        layerInput = this.attentionMechanism(layerInput);
      }
      for (let j = 0; j < this.layers[i].length; j++) {
        let sum = this.biases[i - 1][j];
        for (let k = 0; k < layerInput.length; k++) {
          sum += layerInput[k] * this.weights[i - 1][j][k];
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
      let layerInput = this.layers[i - 1];
      if (i === this.layers.length - 1 && this.useAttention) {
        layerInput = this.attentionMechanism(layerInput);
      }
      for (let j = 0; j < this.layers[i].length; j++) {
        for (let k = 0; k < layerInput.length; k++) {
          const gradient = deltas[i][j] * layerInput[k];
          this.updateWeight(i - 1, j, k, gradient);
        }
        this.biases[i - 1][j] += this.learningRate * deltas[i][j];
      }
    }

    if (this.useAttention) {
      this.updateAttentionWeights(deltas[this.layers.length - 1]);
    }
  }

  private updateAttentionWeights(outputDeltas: number[]): void {
    const attentionDeltas = this.attentionWeights.map((weights, i) =>
      weights.map((weight, j) => weight * outputDeltas[j] * this.layers[this.layers.length - 2][i])
    );

    for (let i = 0; i < this.attentionWeights.length; i++) {
      for (let j = 0; j < this.attentionWeights[i].length; j++) {
        this.attentionWeights[i][j] += this.learningRate * attentionDeltas[i][j];
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
      (mHat / (Math.sqrt(vHat) + epsilon) - weightDecay)  }

  train(inputs: number[][], targets: number[][], epochs: number): number {
    let totalLoss = 0;
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      for (let i = 0; i < inputs.length; i += this.batchSize) {
        const batchInputs = inputs.slice(i, i + this.batchSize);
        const batchTargets = targets.slice(i, i + this.batchSize);
        let batchLoss = 0;
        for (let j = 0; j < batchInputs.length; j++) {
          const output = this.forwardPropagation(batchInputs[j], true);
          this.backPropagation(batchTargets[j]);
          batchLoss += this.calculateLoss(output, batchTargets[j]);
        }
        epochLoss += batchLoss / batchInputs.length;
      }
      totalLoss = epochLoss / (inputs.length / this.batchSize);
      if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${totalLoss}`);
      }
      this.learningRate *= 0.99; // Learning rate decay
    }
    return totalLoss;
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
  // Greetings (Class 0)
  { input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hello"
  { input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hi"
  { input: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hey there"
  { input: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "greetings"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Hey"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "What's up?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Howdy"
  { input: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Hello there"
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Nice to meet you"

  // Good morning (Class 1)
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "good morning"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0,0,0,0,0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Good afternoon"
  { input: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Hello, good morning"
  { input: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Hi, good morning"
  { input: [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "Good morning, how are you?"

  // Good evening (Class 2)
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "good evening"
  { input: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "Hello, good evening"
  { input: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "Hi, good evening"
  { input: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "Good evening, how are you?"
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "Good evening, what's up?"

  // Farewells (Class 3)
  { input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "goodbye"
  { input: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "bye"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "see you later"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "farewell"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "take care"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "have a good one"
  { input: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "goodbye, bye"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "catch you later"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0,0,0,0,0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "until next time"

  // Weather (Class 4)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "what's the weather like?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "how's the weather?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "Is it going to rain today?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "What's the temperature outside?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "Do I need an umbrella today?"

  // How are you (Class 5)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "how are you?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "how are you doing?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "how's it going?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "How have you been?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "How's your day going?"

  // Jokes (Class 6)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "tell me a joke"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "tell me a funny joke"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "Do you know any jokes?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "Make me laugh"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "I need a good laugh"

  // Time (Class 7)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, // "what time is it?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, // "do you have the time?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, // "What's the current time?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, // "Can you tell me the time?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, // "What hour is it?"

  // Help (Class 8)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, // "can you help me?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, // "I need assistance"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, // "Could you give me a hand?"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, // "I'm having trouble with something"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, // "How do I use this feature?"

  // Thank you (Class 9)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, // "thank you"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, // "thanks a lot"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, // "I appreciate your help"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, // "That's very kind of you"
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, // "Much appreciated"
];

// Define the type for bestHyperparameters
interface BestHyperparameters {
  layerSizes: number[];
  learningRate: number;
  dropoutRate: number;
}
// *** Perform Hyperparameter Tuning ONLY ONCE outside the component ***
// Define hyperparameter options
const layerSizesOptions = [[25, 25, 10], [25, 20, 10], [25, 15, 10]];
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
        50 // Number of epochs
      );

      // Use trainingData for accuracy calculation since testData is not defined
      const accuracy = calculateAccuracy(neuralNetwork, trainingData);
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
  250 // Number of epochs
);
// *** End of Hyperparameter Tuning ***

// Enhanced Database of words for generative output
const wordDatabase = {
  greetings: [
    'Hello',
    'Hi',
    'Hey',
    'Greetings',
    'Good morning',
    'Good afternoon',
    'Good evening',
  ],
  farewells: [
    'Goodbye',
    'Bye',
    'See you later',
    'Farewell',
    'Take care',
    'Have a good one',
    'Catch you later',
    'Until next time',
  ],
  howAreYou: [
    'How are you',
    'How are you doing',
    "How's it going",
    'How are you feeling',
  ],
  weatherQueries: [
    "What's the weather like",
    "How's the weather",
    'What is the temperature',
  ],
  jokes: [
    'Tell me a joke',
    'Tell me a funny joke',
    'Do you know any jokes',
  ],
  positiveAdjectives: [
    'great',
    'wonderful',
    'fantastic',
    'amazing',
    'excellent',
    'superb',
  ],
  timeOfDay: ['morning', 'afternoon', 'evening', 'day'],
  topics: [
    'weather',
    'news',
    'sports',
    'technology',
    'movies',
    'music',
    'books',
  ],
};

// ... (rest of the code)

// Enhanced generative functions
const generateGreeting = (timeOfDay: string): string => {
  const greetings = wordDatabase.greetings.filter((g) =>
    g.toLowerCase().includes(timeOfDay)
  );
  const greeting =
    greetings[Math.floor(Math.random() * greetings.length)] ||
    wordDatabase.greetings[
      Math.floor(Math.random() * wordDatabase.greetings.length)
    ];
  const howAreYou =
    wordDatabase.howAreYou[
      Math.floor(Math.random() * wordDatabase.howAreYou.length)
    ];
  const topic =
    wordDatabase.topics[Math.floor(Math.random() * wordDatabase.topics.length)];
  return `${greeting}! ${howAreYou} this ${timeOfDay}? I hope you're having a great day so far. Is there anything specific you'd like to chat about, perhaps ${topic}?`;
};

const generateFarewell = (): string => {
  const farewell =
    wordDatabase.farewells[
      Math.floor(Math.random() * wordDatabase.farewells.length)
    ];
  const positiveAdjective =
    wordDatabase.positiveAdjectives[
      Math.floor(Math.random() * wordDatabase.positiveAdjectives.length)
    ];
  const topic =
    wordDatabase.topics[Math.floor(Math.random() * wordDatabase.topics.length)];
  return `${farewell}! It was ${positiveAdjective} chatting with you about ${topic}. I hope you have a wonderful rest of your day. Feel free to come back anytime if you want to talk more!`;
};

const generateWeatherResponse = (): string => {
  const weatherQuery =
    wordDatabase.weatherQueries[
      Math.floor(Math.random() * wordDatabase.weatherQueries.length)
    ];
  const topic =
    wordDatabase.topics[Math.floor(Math.random() * wordDatabase.topics.length)];
  return `I'm sorry, but I don't have real-time weather data. However, I can tell you that ${weatherQuery} is an important factor in daily life. If you need accurate weather information, I recommend checking a reliable weather service or app. In the meantime, would you like to chat about how ${topic} might be affected by different weather conditions?`;
};

const generateJoke = (): string => {
  const jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    'Why did the scarecrow win an award? He was outstanding in his field!',
    "Why don't eggs tell jokes? They'd crack each other up!",
    'What do you call a fake noodle? An impasta!',
    'Why did the math book look so sad? Because it had too many problems!',
    'What do you call a bear with no teeth? A gummy bear!',
    'Why did the cookie go to the doctor? Because it was feeling crumbly!',
    'What do you call a sleeping bull? A bulldozer!',
  ];
  const joke = jokes[Math.floor(Math.random() * jokes.length)];
  return `Here's a joke for you: ${joke} 😄 I hope that brought a smile to your face! Would you like to hear another one or perhaps chat about something else?`;
};

const generateHowAreYouResponse = (): string => {
  const responses = [
    "I'm functioning at optimal capacity, which I suppose is the AI equivalent of feeling great! How about you? Is there anything exciting happening in your day?",
    "As an AI, I don't have feelings, but I'm operating efficiently and ready to assist you! What's on your mind today?",
    "I'm here and ready to help! How can I assist you today? Is there a particular topic you'd like to discuss or explore?",
    "I'm doing well, thank you for asking! How about you? Is there anything specific you'd like to chat about or any questions you have?",
    "I'm always excited to learn new things from our conversations. What's been the most interesting part of your day so far?",
  ];
  return responses[Math.floor(Math.random() * responses.length)];
};

const generateTimeResponse = (): string => {
  const currentTime = new Date().toLocaleTimeString();
  return `The current time is ${currentTime}. Time is such a fascinating concept, isn't it? Is there anything time-related you'd like to discuss, like time management or the philosophy of time?`;
};

const generateHelpResponse = (): string => {
  const topics = wordDatabase.topics.slice(0, 3);
  return `I'd be happy to help! I can assist with a variety of topics. For example, we could discuss ${topics.join(', ')}, or any other subject you're interested in. What would you like help with?`;
};

const generateThankYouResponse = (): string => {
  const responses = [
    "You're welcome! I'm glad I could help. Is there anything else you'd like to know or discuss?",
    "It's my pleasure! I enjoy our conversations. Do you have any other questions or topics you'd like to explore?",
    "I'm happy I could assist you. Remember, I'm always here if you need more information or just want to chat!",
    "No problem at all! I'm here to help and learn. Is there a particular subject you're curious about that we could delve into?",
    "I'm glad I could be of assistance. Your questions help me learn and improve. What else would you like to talk about?",
  ];
  return responses[Math.floor(Math.random() * responses.length)];
};

// Update the enhancedMachineLearning function
const enhancedMachineLearning = (input: string): string => {
  // Create inputVector based on the input
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
    "what is the time",
    "what is the date",
    "help me with the math",
    "help me find my airpods",
    "thanks for the help",
    
  ];

  const inputVector = keywords.map((keyword) =>
    input.toLowerCase().includes(keyword) ? 1 : 0
  );

  const predictedClass = finalNeuralNetwork.predict(inputVector);
  console.log(`Input: "${input}", Predicted class: ${predictedClass}`);

  // Get the current time of day
  const hour = new Date().getHours();
  const timeOfDay =
    hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening';

  // Contextual Responses with Word Combination and Generative Output
  const responses = {
    0: () => generateGreeting(timeOfDay),
    1: () => generateGreeting('morning'),
    2: () => generateGreeting('evening'),
    3: () => generateFarewell(),
    4: () => generateWeatherResponse(),
    5: () => generateJoke(),
    6: () => generateHowAreYouResponse(),
    7: () => generateTimeResponse(),
    8: () => generateHelpResponse(),
    9: () => generateThankYouResponse(),
  };

  // Return the appropriate response based on the predicted class
  // If the predicted class is not recognized, use the 9th response
  return (
    responses[predictedClass as keyof typeof responses]?.() || responses[9]()
  );
};

const Chat: React.FC<ChatProps> = ({ selectedChat }) => {
  const LoadingSpinner: React.FC = () => (
    <motion.div
      className="loading-dots"
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
    >
      <span>.</span>
      <span>.</span>
      <span>.</span>
      <span>.</span>
      <span>.</span>
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
        animate={{ width: '100%' }}
        transition={{ duration: 2, repeat: Infinity }}
      />
    </motion.div>
  );

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [pendingConfirmationId, setPendingConfirmationId] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState<
    'initializing' | 'training' | 'complete' | 'error'
  >('initializing');
  const [trainingProgress, setTrainingProgress] = useState<
    TrainingProgress | null
  >(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [typingMessage, setTypingMessage] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [isDeveloper, setIsDeveloper] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(150);
  const [selectedModel, setSelectedModel] = useState('Mazs AI v0.90.1 anatra');
  const [isListening, setIsListening] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [detectedMathExpression, setDetectedMathExpression] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [stopGenerating, setStopGenerating] = useState<(() => void) | null>(null);
  const [partialMessage, setPartialMessage] = useState<string | null>(null);

  const suggestions: Suggestion[] = [
    { text: "What's the weather like today?", icon: <FiSun /> },
    { text: 'Tell me a joke', icon: <FiImage /> },
    { text: "What's the latest news?", icon: <FiImage /> },
  ];

  // Function to get typing speed based on selected model
  const getTypingSpeed = (model: string): number => {
    switch (model) {
      case 'Mazs AI v0.90.1 anatra':
        return 40;
      case 'Mazs AI v0.90.1 canard':
        return 30;
      case 'Mazs AI v0.90.1 pato':
        return 20;
      default:
        return 40;
    }
  };

  const [typingSpeed, setTypingSpeed] = useState(getTypingSpeed(selectedModel));

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
          0.01, // L2 Regularization Rate
          ['relu', 'relu', 'softmax'], // Activation functions
          true // Use attention mechanism
        );

        // Train the final model on the full training set
        for (let epoch = 0; epoch < 1500; epoch++) {
          const loss = finalNeuralNetwork.train(
            trainingData.map((data) => data.input),
            trainingData.map((data) => data.target),
            1
          );

          // Calculate accuracy using the training data instead of testData
          const accuracy = calculateAccuracy(finalNeuralNetwork, trainingData);

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

  useEffect(() => {
    // Check if the user is a developer
    const storedUsername = localStorage.getItem('username');
    const storedPassword = localStorage.getItem('password');
    if (storedUsername === 'Developer' && storedPassword === 'GMTStudiotech') {
      setIsDeveloper(true);
    }
  }, []);

  useEffect(() => {
    setTypingSpeed(getTypingSpeed(selectedModel));
  }, [selectedModel]);

  const simulateTyping = (text: string) => {
    let index = 0;
    let isCancelled = false;
    setTypingMessage('');
    setIsGenerating(true);
    setPartialMessage(null);

    const typingInterval = setInterval(() => {
      if (isCancelled) {
        clearInterval(typingInterval);
        return;
      }

      if (index < text.length) {
        setTypingMessage((prev) => {
          const newMessage = prev + text.charAt(index);
          setPartialMessage(newMessage);
          return newMessage;
        });
        index++;
      } else {
        clearInterval(typingInterval);
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: Date.now().toString(),
            sender: 'bot',
            text: `${selectedModel}: ${text}`,
            timestamp: new Date(),
          },
        ]);
        setTypingMessage(null);
        setPartialMessage(null);
        setIsTyping(false);
        setIsBotResponding(false);
        setIsGenerating(false);
        setStopGenerating(null);
      }
    }, typingSpeed);

    const stopTyping = () => {
      isCancelled = true;
      clearInterval(typingInterval);
      setIsGenerating(false);
      setIsTyping(false);
      setIsBotResponding(false);
      setStopGenerating(null);
      if (partialMessage) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            id: Date.now().toString(),
            sender: 'bot',
            text: `${selectedModel}: ${partialMessage} [Response interrupted]`,
            timestamp: new Date(),
          },
        ]);
        setTypingMessage(null);
        setPartialMessage(null);
      }
    };

    setStopGenerating(() => stopTyping);

    return stopTyping;
  };

  // New function to detect math expressions
  const detectMathExpression = (text: string): string | null => {
    const mathRegex = /(\d+(\s*[+\-*/]\s*\d+)+)/;
    const match = text.match(mathRegex);
    return match ? match[0] : null;
  };

  // New function to calculate math expressions
  const calculateMathExpression = (expression: string): number => {
    // eslint-disable-next-line no-new-func
    return Function(`'use strict'; return (${expression})`)();
  };

  const [isBotResponding, setIsBotResponding] = useState(false);

  // Modified handleSendMessage function
  const handleSendMessage = async () => {
    if (inputValue.trim() === '' && !isGenerating) return;

    if (isGenerating && stopGenerating) {
      // Stop the bot's response generation
      stopGenerating();
      return;
    }

    if (isGenerating) {
      return; // Don't allow sending a new message while generating
    }

    const mathExpression = detectMathExpression(inputValue);
    if (mathExpression) {
      setDetectedMathExpression(mathExpression);
      addConfirmationMessage('math');
      return;
    }

    if (inputValue.split(/\s+/).length > 50) {
      addConfirmationMessage('summary');
      return;
    }

    // Proceed with the original message handling
    sendMessage(inputValue);
  };

  // Add this new function
  const addConfirmationMessage = (type: 'math' | 'summary') => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      sender: 'bot',
      text: type === 'math' ? 'You want to calculate the answer?' : 'You want to summarize this text?',
      timestamp: new Date(),
      confirmationType: type,
    };
    setMessages(prevMessages => [...prevMessages, newMessage]);
    setPendingConfirmationId(newMessage.id);
  };

  // New function to handle math calculation
  const handleMathCalculation = () => {
    setMessages(prevMessages => prevMessages.filter(msg => msg.id !== pendingConfirmationId));
    setPendingConfirmationId(null);
    const result = calculateMathExpression(detectedMathExpression);
    const botResponse = `The result of ${detectedMathExpression} is ${result}.`;
    simulateTyping(botResponse);
    setInputValue('');
  };

  // New function to handle summary
  const handleSummary = () => {
    setMessages(prevMessages => prevMessages.filter(msg => msg.id !== pendingConfirmationId));
    setPendingConfirmationId(null);
    // Implement a basic summarization function
    const summarize = (text: string): string => {
      const sentences = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
      const summary = sentences.slice(0, 3).join('. ') + (sentences.length > 3 ? '...' : '');
      return summary;
    };
    const summary = summarize(inputValue);
    simulateTyping(`Here's a summary of your input:\n${summary}`);
    setInputValue('');
  };

  // New function to proceed with normal message processing
  const proceedWithNormalMessage = () => {
    setMessages(prevMessages => prevMessages.filter(msg => msg.id !== pendingConfirmationId));
    setPendingConfirmationId(null);
    sendMessage(inputValue);
  };

  // Helper function to send a message
  const sendMessage = (text: string) => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: text,
      timestamp: new Date(),
    };

    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputValue('');
    setIsTyping(true);
    setIsBotResponding(true);

    setTimeout(() => {
      const botResponse = enhancedMachineLearning(newMessage.text);
      simulateTyping(botResponse);
    }, 1000);
  };


  const handleFeedback = (messageId: string, feedback: 'good' | 'bad') => {
    const messageIndex = messages.findIndex((message) => message.id === messageId);

    if (messageIndex !== -1 && messageIndex > 0) {
      const userMessage = messages[messageIndex - 1];

      // Create a simple input vector based on the user's message
      const inputVector = createInputVector(userMessage.text);

      // Adjust the target vector based on feedback
      const targetVector = new Array(10).fill(0);
      const predictedClass = finalNeuralNetwork.predict(inputVector);
      
      if (feedback === 'good') {
        targetVector[predictedClass] = 1;
      } else {
        // For negative feedback, slightly increase probabilities for other classes
        targetVector.fill(0.1);
        targetVector[predictedClass] = 0;
      }

      // Add the new training data
      trainingData.push({ input: inputVector, target: targetVector });

      // Retrain the model with the updated data
      console.log('Retraining model...');
      const originalLearningRate = finalNeuralNetwork.getLearningRate();
      finalNeuralNetwork.setLearningRate(0.1); // Increase learning rate for retraining

      const loss = finalNeuralNetwork.train(
        [inputVector],
        [targetVector],
       1000// Reduced number of epochs for faster retraining
      );

      finalNeuralNetwork.setLearningRate(originalLearningRate); // Reset learning rate
      console.log(`Retraining complete. Final loss: ${loss}`);

      // Provide feedback to the user
      const feedbackMessage: ChatMessage = {
        id: Date.now().toString(),
        sender: 'bot',
        text: feedback === 'good'
          ? 'Thank you for the positive feedback! I\'ll keep that in mind for future responses.'
          : "I apologize for the unsatisfactory response. I've adjusted my understanding and will try to improve my answers in the future.",
        timestamp: new Date(),
      };
      setMessages((prevMessages) => [...prevMessages, feedbackMessage]);
    }
  };

  // Helper function to create a simple input vector
  const createInputVector = (text: string): number[] => {
    const keywords = [
      'hello', 'hi', 'good morning', 'good evening', 'hey there',
      'goodbye', 'bye', 'see you later', 'farewell', 'take care',
      'have a good one', 'catch you later', 'until next time',
      "what's the weather like?", "how's the weather?",
      'tell me a joke', 'tell me a funny joke',
      'how are you', 'how are you doing', "how's it going",
    ];

    return keywords.map((keyword) => text.toLowerCase().includes(keyword) ? 1 : 0);
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
      if (
        word.match(/^[A-Z][a-z]+$/) &&
        nextWord.match(/^(Mr\.|Ms\.|Dr\.|Mrs\.)$/i)
      ) {
        entities.push(`${word} ${nextWord}`);
        i++;
      }
      // Rule 2: Sequences of Capitalized words (potential names or organizations)
      else if (
        word.match(/^[A-Z][a-z]+$/) &&
        nextWord.match(/^[A-Z][a-z]+$/)
      ) {
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
      else if (
        word.match(/^[A-Z][a-z]+$/) &&
        ['City', 'Town', 'Country'].includes(nextWord)
      ) {
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
      const nounRegex = /^[a-z]+(s)?$/; // Nouns (singular or plural)
      const verbRegex = /^[a-z]+(ed|ing|s)?$/; // Verbs (past tense, present participle, 3rd person singular)
      const adjectiveRegex = /^[a-z]+(er|est)?$/; // Adjectives (comparative, superlative)
      const adverbRegex = /^[a-z]+ly$/; // Adverbs
      const pronounRegex = /^(I|you|he|she|it|we|they|me|him|her|us|them)$/i; // Pronouns
      const prepositionRegex = /^(in|on|at|to|from|by|with|of|for)$/i; // Prepositions
      const conjunctionRegex = /^(and|but|or|nor|so|yet)$/i; // Conjunctions
      const determinerRegex = /^(the|a|an)$/i; // Determiners

      word = word.toLowerCase(); // Normalize to lowercase

      // Check for punctuation
      if (word.match(/^[.,!?;:]+$/)) return `${word}/PUNCT`;

      // Check for numbers
      if (word.match(/^[0-9]+(\.[0-9]+)?$/)) return `${word}/NUM`;

      // Apply more specific rules
      if (
        word === 'to' &&
        index < words.length - 1 &&
        words[index + 1].match(verbRegex)
      ) {
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

  const performSummarization = async (text: string): Promise<string> => {
    // This is a basic example of summarization
    // In a real-world scenario, you might want to use a more sophisticated algorithm
    // or an external API for better results
    const sentences = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
    const summary = sentences.slice(0, 3).join('. ') + (sentences.length > 3 ? '...' : '');
    return summary;
  };

  const handleSummarization = () => {
    if (inputValue) {
      setIsTyping(true);
      performSummarization(inputValue)
        .then((summaryResult: string) => {
          simulateTyping(`Text Summary:\n${summaryResult}`);
          setInputValue(''); // Clear the input box
        })
        .catch((error: Error) => {
          console.error('Error in summarization:', error);
          simulateTyping('An error occurred during summarization.');
        })
        .finally(() => {
          setIsTyping(false);
        });
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleVoiceInput = () => {
    if ('webkitSpeechRecognition' in window) {
      const recognition = new window.webkitSpeechRecognition();

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[0][0].transcript;
        setInputValue((prevValue) => prevValue + transcript);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.start();
    } else {
      alert('Speech recognition is not supported in your browser.');
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.width = 100; // Thumbnail width
          canvas.height = 100; // Thumbnail height
          ctx?.drawImage(img, 0, 0, 100, 100);
          const thumbnailDataUrl = canvas.toDataURL('image/jpeg');

          // Add image message
          const newMessage: ChatMessage = {
            id: Date.now().toString(),
            sender: 'user',
            text: 'Uploaded image:',
            timestamp: new Date(),
            image: thumbnailDataUrl,
          };
          setMessages((prevMessages) => [...prevMessages, newMessage]);
        };
        img.src = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div
      className={`flex flex-col h-screen w-full ${
        darkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-white'
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
                <p>
                  Accuracy: {(trainingProgress.accuracy * 100).toFixed(2)}%
                </p>
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
              {selectedChat ? selectedChat.title : 'New Chat'} - {selectedModel}
            </h1>
            <div className="flex items-center space-x-2">
              {isDeveloper && (
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setShowSettings(true)}
                  className="p-2 rounded-full hover:bg-gray-700 transition-colors"
                >
                  <FiSettings className="text-gray-400" />
                </motion.button>
              )}
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
              {messages.map((message) => (
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
                    {message.image && (
                      <img
                        src={message.image}
                        alt="Uploaded"
                        className="mt-2 rounded"
                      />
                    )}
                    {message.confirmationType && message.id === pendingConfirmationId && (
                      <div className="flex space-x-2 mt-2">
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={() => message.confirmationType === 'math' ? handleMathCalculation() : handleSummary()}
                          className="p-2 rounded bg-green-600 text-white hover:bg-green-700 transition-colors"
                        >
                          <FiCheck />
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={proceedWithNormalMessage}
                          className="p-2 rounded bg-red-600 text-white hover:bg-red-700 transition-colors"
                        >
                          <FiX />
                        </motion.button>
                      </div>
                    )}
                    {message.sender === 'bot' && (
                      <div className="mt-2 text-xs text-gray-400">
                        {isDeveloper ? (
                          <>
                            <p>
                              Accuracy:{' '}
                              {(Math.random() * 0.2 + 0.8).toFixed(4)}
                            </p>
                            <p>
                              Response time:{' '}
                              {(Math.random() * 100 + 50).toFixed(2)}ms
                            </p>
                            <p>
                              Model: Mazs AI v0.90.1 
                            </p>
                          </>
                        ) : (
                          <p>Model: Mazs AI v0.90.1 anatra</p>
                        )}
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
              ))}
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
                  <span
                    className="inline-block w-2 h-2 bg-white rounded-full animate-bounce mr-1"
                    style={{ animationDelay: '0.2s' }}
                  ></span>
                  <span
                    className="inline-block w-2 h-2 bg-white rounded-full animate-bounce"
                    style={{ animationDelay: '0.4s' }}
                  ></span>
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
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !isBotResponding) {
                    handleSendMessage();
                  }
                }}
                placeholder={isBotResponding ? "Please wait for the bot's response..." : "Type a message..."}
                className={`flex-1 p-2 rounded bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${
                  isBotResponding && !isGenerating ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={isBotResponding && !isGenerating}
              />
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleSendMessage}
                className={`p-2 rounded ${
                  isGenerating ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
                } text-white transition-colors`}
              >
                {isGenerating ? <FiPause /> : <FiSend />}
              </motion.button>
              {isDeveloper && (
                <>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={handleVoiceInput}
                    className={`p-2 rounded ${
                      isListening ? 'bg-red-600' : 'bg-gray-800'
                    } text-white hover:bg-gray-700 transition-colors`}
                  >
                    <FiMic />
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => fileInputRef.current?.click()}
                    className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors"
                  >
                    <FiImage />
                  </motion.button>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleImageUpload}
                    accept="image/*"
                    style={{ display: 'none' }}
                  />
                </>
              )}
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

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-gray-800 p-6 rounded-lg w-96"
          >
            <h2 className="text-xl font-bold mb-4">Model Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Temperature</label>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                />
                <span>{temperature} - {temperature <= 0.3 ? 'Normal' : temperature >= 0.8 ? 'Nonsensical' : 'Creative'}</span>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Max Tokens</label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full bg-gray-700 p-2 rounded"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-gray-700 p-2 rounded"
                >
                  <option value="Mazs AI v0.90.1 anatra">Mazs AI v0.90.1 anatra (40ms)</option>
                  <option value="Mazs AI v0.90.1 canard">Mazs AI v0.90.1 canard (30ms)</option>
                  <option value="Mazs AI v0.90.1 pato">Mazs AI v0.90.1 pato (20ms)</option>
                </select>
              </div>
            </div>
            <div className="mt-6 flex justify-end">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowSettings(false)}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
              >
                Close
              </motion.button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default Chat;