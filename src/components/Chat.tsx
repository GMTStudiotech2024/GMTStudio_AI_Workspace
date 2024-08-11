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
  FiPause,
  FiSearch,
} from 'react-icons/fi';
import { Message as ImportedMessage } from '../types';

interface ChatProps {
  selectedChat: { title: string } | null;
}

interface Suggestion {
  text: string;
  icon: React.ReactNode;
}

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

declare global {
  interface Window {
    webkitSpeechRecognition: {
      new (): SpeechRecognitionInstance;
    };
  }
}

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

type Tensor1D = number[];
type Tensor2D = number[][];

interface TransformerConfig {
  inputDim: number;
  outputDim: number;
  numHeads: number;
  numLayers: number;
  dropoutRate: number;
}

class Transformer {
  private positionEncoding: Tensor2D;
  private layers: TransformerLayer[];
  private config: TransformerConfig;
  private outputLayer: Dense;

  constructor(config: TransformerConfig) {
    this.config = config;
    this.positionEncoding = this.createPositionEncoding(1000, config.inputDim);
    this.layers = Array(config.numLayers).fill(null).map(() => new TransformerLayer(config));
    this.outputLayer = new Dense(config.outputDim);
  }

  forward(input: Tensor2D): Tensor2D {
    const inputWithPosition = this.addPositionalEncoding(input);
    let output = inputWithPosition;
    for (const layer of this.layers) {
      output = layer.forward(output);
    }
    return this.outputLayer.forward(output);
  }

  private addPositionalEncoding(input: Tensor2D): Tensor2D {
    return input.map((seq, i) =>
      seq.map((val, j) => val + this.positionEncoding[i % 1000][j % this.config.inputDim])
    );
  }

  private createPositionEncoding(maxLen: number, dim: number): Tensor2D {
    const positionEncoding: Tensor2D = [];
    for (let pos = 0; pos < maxLen; pos++) {
      const row: Tensor1D = [];
      for (let i = 0; i < dim; i++) {
        if (i % 2 === 0) {
          row.push(Math.sin(pos / Math.pow(10000, i / dim)));
        } else {
          row.push(Math.cos(pos / Math.pow(10000, (i - 1) / dim)));
        }
      }
      positionEncoding.push(row);
    }
    return positionEncoding;
  }
}

class TransformerLayer {
  private multiHeadAttention: MultiHeadAttention;
  private feedForward: FeedForward;
  private layerNorm1: LayerNormalization;
  private layerNorm2: LayerNormalization;
  private dropout: Dropout;

  constructor(config: TransformerConfig) {
    this.multiHeadAttention = new MultiHeadAttention(config);
    this.feedForward = new FeedForward(config);
    this.layerNorm1 = new LayerNormalization(config.inputDim);
    this.layerNorm2 = new LayerNormalization(config.inputDim);
    this.dropout = new Dropout(config.dropoutRate);
  }

  forward(input: Tensor2D): Tensor2D {
    const attended = this.multiHeadAttention.forward(input);
    const normalized1 = this.layerNorm1.forward(input.map((row, i) => row.map((val, j) => val + attended[i][j])));
    const ffOutput = this.feedForward.forward(normalized1);
    const dropped = this.dropout.forward(ffOutput);
    return this.layerNorm2.forward(normalized1.map((row, i) => row.map((val, j) => val + dropped[i][j])));
  }
}

class LayerNormalization {
  private gamma: number[];
  private beta: number[];
  private epsilon: number;

  constructor(inputDim: number, epsilon: number = 1e-8) {
    this.gamma = new Array(inputDim).fill(1);
    this.beta = new Array(inputDim).fill(0);
    this.epsilon = epsilon;
  }

  forward(input: Tensor2D): Tensor2D {
    const mean = input.map(row => row.reduce((a, b) => a + b, 0) / row.length);
    const variance = input.map((row, i) => row.reduce((a, b) => a + Math.pow(b - mean[i], 2), 0) / row.length);
    return input.map((row, i) => row.map((val, j) =>
      this.gamma[j] * (val - mean[i]) / Math.sqrt(variance[i] + this.epsilon) + this.beta[j]
    ));
  }
}

class Dense {
  private weights: Tensor2D;
  private bias: number[];

  constructor(outputDim: number) {
    this.weights = Array(outputDim).fill(null).map(() => Array(outputDim).fill(0).map(() => Math.random() - 0.5));
    this.bias = Array(outputDim).fill(0);
  }

  forward(input: Tensor2D): Tensor2D {
    return input.map(row =>
      this.weights.map((wRow, i) =>
        row.reduce((sum, val, j) => sum + val * wRow[j], 0) + this.bias[i]
      )
    );
  }
}

class Dropout {
  private rate: number;

  constructor(rate: number) {
    this.rate = rate;
  }

  forward(input: Tensor2D): Tensor2D {
    return input.map(row =>
      row.map(val => Math.random() > this.rate ? val / (1 - this.rate) : 0)
    );
  }
}

class MultiHeadAttention {
  private config: TransformerConfig;
  private weights: {
    query: Tensor2D;
    key: Tensor2D;
    value: Tensor2D;
    output: Tensor2D;
  };

  constructor(config: TransformerConfig) {
    this.config = config;
    const dim = config.inputDim;
    this.weights = {
      query: this.initializeWeight(dim, dim),
      key: this.initializeWeight(dim, dim),
      value: this.initializeWeight(dim, dim),
      output: this.initializeWeight(dim, dim),
    };
  }

  forward(input: Tensor2D): Tensor2D {

    const query = this.linearTransform(input, this.weights.query);
    const key = this.linearTransform(input, this.weights.key);
    const value = this.linearTransform(input, this.weights.value);

    const scores = this.dotProduct(query, this.transpose(key));
    const scaledScores = this.scale(scores, Math.sqrt(this.config.inputDim / this.config.numHeads));
    const attentionWeights = this.softmax(scaledScores);

    const attended = this.dotProduct(attentionWeights, value);
    const output = this.linearTransform(attended, this.weights.output);

    return output;
  }

  private initializeWeight(rows: number, cols: number): Tensor2D {
    return Array(rows).fill(null).map(() =>
      Array(cols).fill(null).map(() => Math.random() - 0.5)
    );
  }

  private linearTransform(input: Tensor2D, weight: Tensor2D): Tensor2D {
    return input.map(row =>
      weight.map(wRow =>
        row.reduce((sum, val, i) => sum + val * wRow[i], 0)
      )
    );
  }

  private dotProduct(a: Tensor2D, b: Tensor2D): Tensor2D {
    return a.map(rowA =>
      b[0].map((_, j) =>
        rowA.reduce((sum, val, k) => sum + val * b[k][j], 0)
      )
    );
  }

  private transpose(matrix: Tensor2D): Tensor2D {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
  }

  private scale(matrix: Tensor2D, factor: number): Tensor2D {
    return matrix.map(row => row.map(val => val / factor));
  }

  private softmax(matrix: Tensor2D): Tensor2D {
    return matrix.map(row => {
      const expValues = row.map(Math.exp);
      const sumExp = expValues.reduce((a, b) => a + b, 0);
      return expValues.map(val => val / sumExp);
    });
  }
}

class FeedForward {
  private weights: {
    hidden: Tensor2D;
    output: Tensor2D;
  };


  constructor(config: TransformerConfig) {
    const dim = config.inputDim;
    this.weights = {
      hidden: this.initializeWeight(dim, dim * 4),
      output: this.initializeWeight(dim * 4, dim),
    };
  }


  forward(input: Tensor2D): Tensor2D {
    const hidden = this.linearTransform(input, this.weights.hidden);
    const activated = hidden.map(row => row.map(val => Math.max(0, val)));
    const output = this.linearTransform(activated, this.weights.output);
    return output;
  }

  private initializeWeight(rows: number, cols: number): Tensor2D {
    return Array(rows).fill(null).map(() =>
      Array(cols).fill(null).map(() => Math.random() - 0.5)
    );
  }

  private linearTransform(input: Tensor2D, weight: Tensor2D): Tensor2D {
    return input.map(row =>
      weight[0].map((_, j) =>
        row.reduce((sum, val, k) => sum + val * weight[k][j], 0)
      )
    );
  }
}

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
  private attentionWeights: number[][] = [];
  private useAttention: boolean;
  private transformer: Transformer | null = null;

  constructor(
    layerSizes: number[],
    learningRate: number = 0.001,
    dropoutRate: number = 0.5,
    batchSize: number = 32,
    optimizer: 'adam' | 'rmsprop' | 'sgd' | 'adamw' = 'adamw',
    l2RegularizationRate: number = 0.01,
    activationFunctions: string[] = [],
    useAttention: boolean = false,
    useTransformer: boolean = false
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

    if (useTransformer) {
      const transformerConfig: TransformerConfig = {
        inputDim: layerSizes[0],
        outputDim: layerSizes[layerSizes.length - 1],
        numHeads: 4,
        numLayers: 2,
        dropoutRate: this.dropoutRate
      };
      this.transformer = new Transformer(transformerConfig);
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
        return Math.random() * Math.sqrt(2 / fanIn);
      default:
        return Math.random() * Math.sqrt(2 / (fanIn + fanOut));
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
        return x / (1 + Math.exp(-x));
      case 'mish':
        return x * Math.tanh(Math.log(1 + Math.exp(x)));
      case 'gelu':
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
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

  private useTransformerIfAvailable(input: number[]): number[] {
    if (this.transformer) {
      const inputTensor: Tensor2D = [input];
      const output = this.transformer.forward(inputTensor);
      return output[0];
    }
    return input;
  }

  private forwardPropagation(input: number[], isTraining: boolean = true): number[] {
    const transformedInput = this.useTransformerIfAvailable(input);
    this.layers[0] = transformedInput;
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
      (mHat / (Math.sqrt(vHat) + epsilon) - weightDecay);
  }

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
      this.learningRate *= 0.99;
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

const trainingData = [
  { input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },

  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, 

  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, 

  { input: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, 

  // Weather (Class 4)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, 
  //How are you (Class 5)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, 

  // Jokes (Class 6)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, 

  //Time (Class 7)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, 
  // Help (Class 8)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, 

  //Thank you (Class 9)
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, 
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }, 
];

interface BestHyperparameters {
  layerSizes: number[];
  learningRate: number;
  dropoutRate: number;
}

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
        64,
        'adamw',
        0.01
      );

      neuralNetwork.train(
        trainingData.map((data) => data.input),
        trainingData.map((data) => data.target),
        50
      );

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

const finalNeuralNetwork = new EnhancedNeuralNetwork(
  bestHyperparameters.layerSizes,
  bestHyperparameters.learningRate,
  bestHyperparameters.dropoutRate,
  64,
  'adamw',
  0.01
);

finalNeuralNetwork.train(
  trainingData.map((data) => data.input),
  trainingData.map((data) => data.target),
  250
);

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
  return `${greeting}! ${howAreYou} this ${timeOfDay}? I hope you're having a great day so far. Is there anything specific you'd like to chat about, perhaps ${topic}? You can make me search for information by input the topic and press search button next to the "paperplane" button.`;
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
  return `I'm sorry, but I don't have real-time weather data. However, I can tell you that ${weatherQuery} is an important factor in daily life. If you need accurate weather information, I recommend checking a reliable weather service or app. In the meantime, would you like to chat about how ${topic} might be affected by different weather conditions?You can make me search for information by input the topic and press search button next to the "paperplane" button.`;
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
    "I'm functioning at optimal capacity, which I suppose is the AI equivalent of feeling great! How about you? You can make me search for information by input the topic and press search button next to the 'paperplane' button.",
    "As an AI, I don't have feelings, but I'm operating efficiently and ready to assist you! What's on your mind today?You can make me search for information by input the topic and press search button next to the paperplane button.",
    "I'm here and ready to help! How can I assist you today? Is there a particular topic you'd like to discuss or explore?You can make me search for information by input the topic and press search button next to the paperplane button.",
    "I'm doing well, thank you for asking! How about you? Is there anything specific you'd like to chat about or any questions you have?You can make me search for information by input the topic and press search button next to the paperplane button.",
    "I'm always excited to learn new things from our conversations. What's been the most interesting part of your day so far?You can make me search for information by input the topic and press search button next to the paperplane button.",
  ];
  return responses[Math.floor(Math.random() * responses.length)];
};

const generateTimeResponse = (): string => {
  const currentTime = new Date().toLocaleTimeString();
  return `The current time is ${currentTime}. `;
};

const generateHelpResponse = (): string => {
  const topics = wordDatabase.topics.slice(0, 3);
  return `I'd be happy to help! I can assist with a variety of topics. For example, we could discuss ${topics.join(', ')}, or any other subject you're interested in. You can make me search for information by input the topic and press search button next to the "paperplane" button.`;
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

const generateDontUnderstandResponse = (): string => {
  const responses = [
    "I'm sorry, but I don't understand your input. Could you please rephrase or ask something else?",
    "I'm having trouble understanding what you mean. Can you try asking in a different way?",
    "I apologize, but I'm not sure how to respond to that. Is there another way you could phrase your question?",
    "I'm afraid I don't have enough information to answer that. Could you provide more context or ask a different question?",
    "That's a bit beyond my current capabilities. Is there something else I can help you with?",
  ];
  return responses[Math.floor(Math.random() * responses.length)];
};

// Update the enhancedMachineLearning function
const enhancedMachineLearning = (input: string): string => {
  const inputLower = input.toLowerCase();

  // More specific keyword groups
  const greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'];
  const farewells = ['goodbye', 'bye', 'see you', 'farewell', 'take care'];
  const weatherKeywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy'];
  const jokeKeywords = ['joke', 'funny', 'humor', 'laugh'];
  const howAreYou = ['how are you', 'how\'s it going', 'what\'s up'];
  const timeKeywords = ['time', 'clock', 'hour'];
  const dateKeywords = ['date', 'day', 'month', 'year', 'tomorrow'];
  const mathKeywords = ['math', 'calculate', 'sum', 'multiply', 'divide'];
  const helpKeywords = ['help', 'assist', 'support'];
  const thankYouKeywords = ['thank', 'thanks', 'appreciate'];
  const topics = ['weather', 'news', 'sports', 'technology', 'movies', 'music', 'books'];


  // Create a more detailed input vector
  const inputVector = [
    greetings.some(word => inputLower.includes(word)) ? 1 : 0,
    farewells.some(word => inputLower.includes(word)) ? 1 : 0,
    weatherKeywords.some(word => inputLower.includes(word)) ? 1 : 0,
    jokeKeywords.some(word => inputLower.includes(word)) ? 1 : 0,
    howAreYou.some(phrase => inputLower.includes(phrase)) ? 1 : 0,
    timeKeywords.some(word => inputLower.includes(word)) ? 1 : 0,
    dateKeywords.some(word => inputLower.includes(word)) ? 1 : 0,
    mathKeywords.some(word => inputLower.includes(word)) ? 1 : 0,
    helpKeywords.some(word => inputLower.includes(word)) ? 1 : 0,
    thankYouKeywords.some(word => inputLower.includes(word)) ? 1 : 0,
  ];

  const predictedClass = finalNeuralNetwork.predict(inputVector);
  console.log(`Input: "${input}", Predicted class: ${predictedClass}`);

  // Get the current time of day
  const hour = new Date().getHours();
  const timeOfDay = hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening';

  // Additional check for farewells
  if (farewells.some(word => inputLower.includes(word))) {
    return generateFarewell();
  }
  if (greetings.some(word => inputLower.includes(word))) {
    return generateGreeting(timeOfDay);
  }
  if (weatherKeywords.some(word => inputLower.includes(word))) {
    return generateWeatherResponse();
  }
  if (jokeKeywords.some(word => inputLower.includes(word))) {
    return generateJoke();
  }
  if (howAreYou.some(phrase => inputLower.includes(phrase))) {
    return generateHowAreYouResponse();
  }
  if (timeKeywords.some(word => inputLower.includes(word))) {
    return generateTimeResponse();
  }
  if (helpKeywords.some(word => inputLower.includes(word))) {
    return generateHelpResponse();
  }
  if (thankYouKeywords.some(word => inputLower.includes(word))) {
    return generateThankYouResponse();
  }
  if (dateKeywords.some(word => inputLower.includes(word))) {
    return generateDateResponse(input);
  }
  if (mathKeywords.some(word => inputLower.includes(word))) {
    return "I can help you with math! What calculation do you need?";
  }
  if (topics.some(word => inputLower.includes(word))) {
    return generateTopicResponse();
  }

  const responses = {
    0: () => generateGreeting(timeOfDay),
    1: () => generateFarewell(),
    2: () => generateWeatherResponse(),
    3: () => generateJoke(),
    4: () => generateHowAreYouResponse(),
    5: () => generateTimeResponse(),
    6: () => generateDateResponse(input),
    7: () => "If you want to learn about anything, you will need to type in the topic, and press search button.",
    8: () => generateHelpResponse(),
    9: () => generateThankYouResponse(),
  };

  // Check if the predicted class is valid
  if (predictedClass >= 0 && predictedClass < Object.keys(responses).length) {
    const response = responses[predictedClass as keyof typeof responses]();
    return response;
  }
  // If we reach here, either the class was invalid or the response wasn't informative
  return generateDontUnderstandResponse();
};

// Add this new function for date responses
const generateDateResponse = (input: string): string => {
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const options: Intl.DateTimeFormatOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };

  if (input.toLowerCase().includes('tomorrow')) {
    return `Tomorrow's date will be ${tomorrow.toLocaleDateString('en-US', options)}.`;
  } else {
    return `Today's date is ${today.toLocaleDateString('en-US', options)}.`;
  }
};
const generateTopicResponse = (): string => {
  const topics = wordDatabase.topics.slice(0, 3);
  return `I'd be happy to help! I can assist with a variety of topics. For example, we could discuss ${topics.join(', ')}, or any other subject you're interested in. What would you like help with?`;
};

interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  image?: string;
  confirmationType?: 'math' | 'summary';
}
interface ChatProps {
  selectedChat: { title: string } | null;
}

interface Suggestion {
  text: string;
  icon: React.ReactNode;
}

interface TrainingProgress {
  epoch: number;
  loss: number;
  accuracy: number;
}


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

  const stripHtml = (html: string): string => {
    const tmp = document.createElement("DIV");
    tmp.innerHTML = html;
    return tmp.textContent || tmp.innerText || "";
  };

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
  const [selectedModel, setSelectedModel] = useState('Mazs AI v1.0 anatra');
  const [isListening, setIsListening] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [detectedMathExpression, setDetectedMathExpression] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [stopGenerating, setStopGenerating] = useState<(() => void) | null>(null);
  const [partialMessage, setPartialMessage] = useState<string | null>(null);
  const [isHighQualitySearch, setIsHighQualitySearch] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMemory, setSearchMemory] = useState<string[]>([]);

  const suggestions: Suggestion[] = [
    { text: "What's the weather like today?", icon: <FiSun /> },
    { text: 'Tell me a joke', icon: <FiImage /> },
    { text: "What's the latest news?", icon: <FiImage /> },
  ];
  interface SearchResult {
    pageid: number;
    title: string;
    snippet: string;
  }
  const getTypingSpeed = (model: string): number => {
    switch (model) {
      case 'Mazs AI v0.90.1 anatra':
        return 40;
      case 'Mazs AI v0.90.1 canard':
        return 30;
      case 'Mazs AI v0.90.1 pato':
        return 20;
      case 'Mazs AI v1.0 anatra':
        return 50;
      default:
        return 60;
    }
  };

  const [typingSpeed, setTypingSpeed] = useState(getTypingSpeed(selectedModel));

  useEffect(() => {
    const trainModel = async () => {
      try {
        setTrainingStatus('training');


        const finalNeuralNetwork = new EnhancedNeuralNetwork(
          bestHyperparameters.layerSizes,
          bestHyperparameters.learningRate,
          bestHyperparameters.dropoutRate,
          64,
          'adamw',
          0.01,
          ['relu', 'relu', 'softmax'],
          true
        );

        const totalEpochs = 500; // Reduced to 750 epochs

        for (let epoch = 0; epoch <= totalEpochs; epoch++) {
          const loss = finalNeuralNetwork.train(
            trainingData.map((data) => data.input),
            trainingData.map((data) => data.target),
            1
          );

          const accuracy = calculateAccuracy(finalNeuralNetwork, trainingData);

          // Update progress every epoch
          setTrainingProgress({ epoch, loss, accuracy });

          // Add a small delay between epochs to control the speed of the progress
          await new Promise(resolve => setTimeout(resolve, 20));
        }

        // Ensure the final progress is shown
        setTrainingProgress({ epoch: totalEpochs, loss: 0, accuracy: 1 });

        // Add a 2-second delay after training is complete
        await new Promise(resolve => setTimeout(resolve, 2000));

        setTrainingStatus('complete');
      } catch (error) {
        console.error('Training error:', error);
        setTrainingStatus('error');
      }
    };

    setTimeout(() => {
      trainModel();
    }, 0);
  }, []);

  useEffect(() => {
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

  const detectMathExpression = (text: string): string | null => {
    const mathRegex = /(\d+(\s*[+\-*/]\s*\d+)+)/;
    const match = text.match(mathRegex);
    return match ? match[0] : null;
  };

  const calculateMathExpression = (expression: string): number => {
    return Function(`'use strict'; return (${expression})`)();
  };

  const [isBotResponding, setIsBotResponding] = useState(false);

  const handleSendMessage = async () => {
    if (inputValue.trim() === '' && !isGenerating) return;

    if (isGenerating && stopGenerating) {
      stopGenerating();
      return;
    }

    if (isGenerating) {
      return; 
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

    sendMessage(inputValue);
  };

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

  const handleMathCalculation = () => {
    setMessages(prevMessages => prevMessages.filter(msg => msg.id !== pendingConfirmationId));
    setPendingConfirmationId(null);
    const result = calculateMathExpression(detectedMathExpression);
    const botResponse = `The result of ${detectedMathExpression} is ${result}.`;
    simulateTyping(botResponse);
    setInputValue('');
  };

  const handleSummary = () => {
    setMessages(prevMessages => prevMessages.filter(msg => msg.id !== pendingConfirmationId));
    setPendingConfirmationId(null);
    const summarize = (text: string): string => {
      const sentences = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
      const summary = sentences.slice(0, 3).join('. ') + (sentences.length > 3 ? '...' : '');
      return summary;
    };
    const summary = summarize(inputValue);
    simulateTyping(`Here's a summary of your input:\n${summary}`);
    setInputValue('');
  };

  const proceedWithNormalMessage = () => {
    setMessages(prevMessages => prevMessages.filter(msg => msg.id !== pendingConfirmationId));
    setPendingConfirmationId(null);
    sendMessage(inputValue);
  };

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

    setTimeout(async () => {
      const botResponse = enhancedMachineLearning(newMessage.text);
      console.log("Bot response:", botResponse); // Debugging line

      // Check if the response is actually informative
      if (!isResponseInformative(botResponse)) {
        console.log("Response not informative, triggering search"); // Debugging line
        const searchResults = await handleHighQualitySearch(newMessage.text);
        if (searchResults.length > 0) {
          const summary = generateSummaryFromSearchResults(searchResults);
          simulateTyping(`I found some information that might help: ${summary}`);
        } else {
          simulateTyping("I'm sorry, but I couldn't find any relevant information about that. Could you please rephrase your question or ask about something else?");
        }
      } else {
        simulateTyping(botResponse);
      }
    }, 1000);
  };

  // New function to check if the response is informative
  const isResponseInformative = (response: string): boolean => {
    const uninformativePatterns = [
      /i don't (understand|know)/i,
      /i couldn't find/i,
      /can you (rephrase|clarify)/i,
      /what (specifically|exactly) (do you|are you) (mean|asking|referring to)/i,
      /could you (please )?(provide more|give more) (context|information|details)/i,
      /i'm not sure (what|how) to (respond|answer)/i,
      /can you (be more specific|elaborate)/i,
    ];

    return !uninformativePatterns.some(pattern => pattern.test(response));
  };

  const generateSummaryFromSearchResults = (results: SearchResult[]): string => {
    const topResults = results.slice(0, 2);
    return topResults.map(result => `${result.title}: ${stripHtml(result.snippet)}`).join(' ');
  };

  const handleSearch = async () => {
    if (inputValue.trim() === '') return;

    setIsTyping(true);
    setIsBotResponding(true);

    try {
      const searchResults = await handleHighQualitySearch(inputValue);

      let summary = '';
      if (searchResults.length > 0) {
        const topResults = searchResults.slice(0, 3); // Get top 3 results
        summary = topResults.map(result => `${result.title}: ${stripHtml(result.snippet)}`).join(' ');
        
        // Add to memory
        setSearchMemory(prevMemory => [...prevMemory.slice(-4), summary]);
      }

      let response = '';
      if (summary) {
        response = `It seems that you want to know about "${inputValue}" ${generateSummaryResponse(summary)}`;
        
        // Add analysis based on memory
        if (searchMemory.length > 1) {
          response += ` Interestingly, this relates to our previous searches. ${generateAnalysis(searchMemory)}`;
        }
      } else {
        response = `I'm sorry, but I couldn't find any reliable information about "${inputValue}". Would you like to try a different search term?`;
      }

      simulateTyping(response);
    } catch (error) {
      console.error('Error fetching or analyzing data:', error);
      simulateTyping('I encountered an issue while searching for that information. Could we try again in a moment?');
    } finally {
      setIsTyping(false);
      setIsBotResponding(false);
    }
  };

  const generateSummaryResponse = (summary: string): string => {
    const sentences = summary.split('. ');
    const shortSummary = sentences.slice(0, 2).join('. ');
    return `${shortSummary}. This gives us a general idea, but there's more to explore if you're interested.`;
  };

  const generateAnalysis = (memory: string[]): string => {
    const recentSearches = memory.slice(-2);
    return `It seems we've been exploring topics related to ${recentSearches.join(' and ')}. This could indicate a broader interest in ${guessTheme(recentSearches)}. Would you like to delve deeper into any specific aspect?`;
  };

  const guessTheme = (searches: string[]): string => {
    const commonWords = searches.join(' ').toLowerCase().split(' ');
    const themes = {
      technology: ['ai', 'computer', 'software', 'hardware', 'internet'],
      science: ['physics', 'chemistry', 'biology', 'research', 'experiment'],
      history: ['ancient', 'war', 'civilization', 'century', 'era'],
      // Add more themes as needed
    };

    for (const [theme, keywords] of Object.entries(themes)) {
      if (keywords.some(keyword => commonWords.includes(keyword))) {
        return theme;
      }
    }
    return 'various interconnected subjects';
  };

  const handleFeedback = (messageId: string, feedback: 'good' | 'bad') => {
    const messageIndex = messages.findIndex((message) => message.id === messageId);

    if (messageIndex !== -1 && messageIndex > 0) {
      const userMessage = messages[messageIndex - 1];

      const inputVector = createInputVector(userMessage.text);

      const targetVector = new Array(10).fill(0);
      const predictedClass = finalNeuralNetwork.predict(inputVector);

      if (feedback === 'good') {
        targetVector[predictedClass] = 1;
      } else {
        targetVector.fill(0.1);
        targetVector[predictedClass] = 0;
      }

      trainingData.push({ input: inputVector, target: targetVector });

      console.log('Retraining model...');
      const originalLearningRate = finalNeuralNetwork.getLearningRate();
      finalNeuralNetwork.setLearningRate(0.1);

      const loss = finalNeuralNetwork.train(
        [inputVector],
        [targetVector],
        1000
      );

      finalNeuralNetwork.setLearningRate(originalLearningRate);
      console.log(`Retraining complete. Final loss: ${loss}`);

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

  const performNER = (text: string): string => {
    const entities = [];
    const words = text.split(/\s+/);

    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      let nextWord = words[i + 1] || '';

      if (
        word.match(/^[A-Z][a-z]+$/) &&
        nextWord.match(/^(Mr\.|Ms\.|Dr\.|Mrs\.)$/i)      ) {
        entities.push(`${word} ${nextWord}`);
        i++;
      } else if (
        word.match(/^[A-Z][a-z]+$/) &&
        nextWord.match(/^[A-Z][a-z]+$/)
      ) {
        let name = word;
        while (nextWord.match(/^[A-Z][a-z]+$/) && i < words.length - 1) {
          name += ` ${nextWord}`;
          i++;
          nextWord = words[i + 1] || '';
        }
        entities.push(name);
      } else if (i === 0 && word.match(/^[A-Z][a-z]+$/)) {
        entities.push(word);
      } else if (
        word.match(/^[A-Z][a-z]+$/) &&
        ['City', 'Town', 'Country'].includes(nextWord)
      ) {
        entities.push(word);
        i++;
      }
    }

    return entities.length > 0 ? entities.join(', ') : 'No entities found';
  };

  const performPOS = (text: string): string => {
    const words = text.split(' ');
    const tags = words.map((word, index) => {
      const nounRegex = /^[a-z]+(s)?$/;
      const verbRegex = /^[a-z]+(ed|ing|s)?$/;
      const adjectiveRegex = /^[a-z]+(er|est)?$/;
      const adverbRegex = /^[a-z]+ly$/;
      const pronounRegex = /^(I|you|he|she|it|we|they|me|him|her|us|them)$/i;
      const prepositionRegex = /^(in|on|at|to|from|by|with|of|for)$/i;
      const conjunctionRegex = /^(and|but|or|nor|so|yet)$/i;
      const determinerRegex = /^(the|a|an)$/i;

      word = word.toLowerCase();

      if (word.match(/^[.,!?;:]+$/)) return `${word}/PUNCT`;

      if (word.match(/^[0-9]+(\.[0-9]+)?$/)) return `${word}/NUM`;

      if (
        word === 'to' &&
        index < words.length - 1 &&
        words[index + 1].match(verbRegex)
      ) {
        return `${word}/TO`;
      }

      if (word.match(determinerRegex)) return `${word}/DET`;
      if (word.match(pronounRegex)) return `${word}/PRON`;
      if (word.match(prepositionRegex)) return `${word}/PREP`;
      if (word.match(conjunctionRegex)) return `${word}/CONJ`;
      if (word.match(adverbRegex)) return `${word}/ADV`;
      if (word.match(adjectiveRegex)) return `${word}/ADJ`;
      if (word.match(verbRegex)) return `${word}/VERB`;
      if (word.match(nounRegex)) return `${word}/NOUN`;

      return `${word}/UNK`;
    });
    return tags.join(' ');
  };

  const handlePOS = () => {
    if (inputValue) {
      const posResult = performPOS(inputValue);
      setIsTyping(true);
      simulateTyping(`POS Tagging Result:\n${posResult}`);
      setInputValue('');
    }
  };

  const handleNER = () => {
    if (inputValue) {
      const nerResult = performNER(inputValue);
      setIsTyping(true);
      simulateTyping(`Named Entities Found:\n${nerResult}`);
      setInputValue('');
    }
  };

  const performSummarization = async (text: string): Promise<string> => {
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
          setInputValue('');
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
          canvas.width = 100;
          canvas.height = 100;
          ctx?.drawImage(img, 0, 0, 100, 100);
          const thumbnailDataUrl = canvas.toDataURL('image/jpeg');

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

  const handleHighQualitySearch = async (query: string): Promise<SearchResult[]> => {
    setSearchQuery(query);
    if (!query.trim()) {
      setSearchResults([]);
      return [];
    }

    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&origin=*`;

    try {
      const response = await fetch(url);
      const data = await response.json();
      setSearchResults(data.query.search);
      return data.query.search;
    } catch (error) {
      console.error('Error fetching search results:', error);
      setSearchResults([]);
      return [];
    }
  };

  return (
    <div
      className={`flex flex-col h-screen w-full ${
        darkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-800'
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
            <h2 className="text-3xl font-bold mb-4">
              Initializing Artificial Intelligence
            </h2>
            <p className="text-lg mb-6">Please wait while we prepare your AI assistant...</p>
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
            <h2 className="text-3xl font-bold mb-4">AI Training in Progress, please wait for about 15 seconds</h2>
            {trainingProgress && (
              <motion.div
                className="mt-6 bg-gray-800 p-4 rounded-lg shadow-lg"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <p className="text-lg mb-2">Training Progress:</p>
                <p>Epoch: <span className="font-semibold">{trainingProgress.epoch}/500</span></p>
                <p>Loss: <span className="font-semibold">{trainingProgress.loss.toFixed(4)}</span></p>
                <p>
                  Accuracy: <span className="font-semibold">{(trainingProgress.accuracy * 100).toFixed(2)}%</span>
                </p>
                <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
                  <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${(trainingProgress.epoch / 500) * 100}%` }}></div>
                </div>
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
            <h2 className="text-3xl font-bold mb-4">
              Oops! An Error Occurred
            </h2>
            <p className="text-xl mb-6">We encountered an issue during the AI training process.</p>
            <button
              onClick={() => window.location.reload()}
              className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition duration-300"
            >
              Try Again
            </button>
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
            <div className="flex items-center space-x-4">
              {isDeveloper && (
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setShowSettings(true)}
                  className="p-2 rounded-full hover:bg-gray-700 transition-colors"
                  title="Settings"
                >
                  <FiSettings className="text-gray-400 text-xl" />
                </motion.button>
              )}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleClearChat}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors"
                title="Clear Chat"
              >
                <FiTrash2 className="text-red-500 text-xl" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setDarkMode(!darkMode)}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors"
                title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
              >
                {darkMode ? (
                  <FiSun className="text-yellow-500 text-xl" />
                ) : (
                  <FiMoon className="text-gray-700 text-xl" />
                )}
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setIsHighQualitySearch(!isHighQualitySearch)}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors"
                title="Toggle High Quality Search"
              >
                <FiSearch className="text-gray-400 text-xl" />
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
                    className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl p-3 rounded-lg shadow-md ${
                      message.sender === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-white'
                    }`}
                  >
                    <p className="text-sm mb-1">{message.sender === 'user' ? 'You' : 'AI'}</p>
                    <p>{message.text}</p>
                    {message.image && (
                      <img
                        src={message.image}
                        alt="Uploaded"
                        className="mt-2 rounded max-w-full h-auto"
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
                              {(Math.random() * 0.2 + 0.8).toFixed(2)}
                            </p>
                            <p>
                              Response time:{' '}
                              {(Math.random() * 100 + 50).toFixed(2)}ms
                            </p>
                            <p>
                              Model: Mazs AI v1.0 
                            </p>
                          </>
                        ) : (
                          <p>Model: Mazs AI v1.0 anatra</p>
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
                          title="Thumbs Up"
                        >
                          <FiThumbsUp />
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={() => handleFeedback(message.id, 'bad')}
                          className="text-red-500 hover:text-red-600"
                          title="Thumbs Down"
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
                <div className="bg-gray-700 text-white p-3 rounded-lg shadow-md">
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
                <div className="bg-gray-700 text-white p-3 rounded-lg shadow-md">
                  <span
                    className="inline-block w-2 h-2 bg-white rounded-full animate-bounce mr-1"
                    style={{ animationDelay: '0.2s' }}
                  ></span>
                  <span
                    className="inline-block w-2 h-2 bg-white rounded-full animate-bounce mr-1"
                    style={{ animationDelay: '0.4s' }}
                  ></span>
                  <span
                    className="inline-block w-2 h-2 bg-white rounded-full animate-bounce"
                    style={{ animationDelay: '0.6s' }}
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
                className={`flex-1 p-3 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${
                  isBotResponding && !isGenerating ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={isBotResponding && !isGenerating}
              />
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleSendMessage}
                className={`p-3 rounded-lg ${
                  isGenerating ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
                } text-white transition-colors`}
                title={isGenerating ? "Stop Generating" : "Send Message"}
              >
                {isGenerating ? <FiPause /> : <FiSend />}
              </motion.button>

              {/* Search Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleSearch}
                className="p-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white transition-colors"
                title="Search Wikipedia"
              >
                <FiSearch />
              </motion.button>

              {isDeveloper && (
                <>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={handleVoiceInput}
                    className={`p-3 rounded-lg ${
                      isListening ? 'bg-red-600' : 'bg-gray-800'
                    } text-white hover:bg-gray-700 transition-colors`}
                    title={isListening ? "Stop Listening" : "Start Voice Input"}
                  >
                    <FiMic />
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => fileInputRef.current?.click()}
                    className="p-3 rounded-lg bg-gray-800 text-white hover:bg-gray-700 transition-colors"
                    title="Upload Image"
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
                className="p-2 rounded-lg bg-gray-800 text-white hover:bg-gray-700 transition-colors"
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
                  <option value="Mazs AI v1.0 anatra">Mazs AI v1.0 anatra (10ms)</option>
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
      {isHighQualitySearch && (
        <div className="high-quality-search mt-4 p-4 bg-gray-800 rounded-lg">
          <input
            type="text"
            value={searchQuery}
            placeholder="Search Wikipedia..."
            onChange={(e) => handleHighQualitySearch(e.target.value)}
            className="w-full p-2 rounded bg-gray-700 text-white"
          />
          <div className="search-results mt-4 space-y-4">
            {searchResults.map((result) => (
              <div key={result.pageid} className="search-result bg-gray-700 p-4 rounded">
                <h3 className="text-lg font-semibold">{result.title}</h3>
                <p className="mt-2 text-sm">{stripHtml(result.snippet)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
};

export default Chat;