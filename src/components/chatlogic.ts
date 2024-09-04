
type Tensor1D = number[];
type Tensor2D = number[][];

// Transformer Config Interface
interface TransformerConfig {
  inputDim: number;
  outputDim: number;
  numHeads: number;
  numLayers: number;
  dropoutRate: number;
}
interface SearchResult {
    title: string;
    snippet: string;
  }
// Transformer Class
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

// Transformer Layer Class
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

// Layer Normalization Class
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

// Dense Layer Class
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

// Dropout Class
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

// Multi-Head Attention Class
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

// Feed Forward Class
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

// Calculate Accuracy Function
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

// Training Data
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
  { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  // New training data
  { input: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] },
  { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] },
  { input: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0] },
  { input: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0] },
  { input: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 1, 0, 0, 0, 0] },
  { input: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0] },
  { input: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0] },
];

// Best Hyperparameters
interface BestHyperparameters {
  layerSizes: number[];
  learningRate: number;
  dropoutRate: number;
}

const layerSizesOptions = [[25,45,10],[25, 25, 10], [25, 20, 10], [25, 15, 10]];
const learningRateOptions = [0.005, 0.05];
const dropoutRateOptions = [0.3, 0.5];

let bestAccuracy = 0;
let bestHyperparameters: BestHyperparameters = {
  layerSizes: [],
  learningRate: 0,
  dropoutRate: 0,
};

// Hyperparameter Tuning (You might want to comment this out after finding the best parameters)
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
        10
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

// Final Neural Network with Best Hyperparameters
const finalNeuralNetwork = new EnhancedNeuralNetwork(
  bestHyperparameters.layerSizes,
  bestHyperparameters.learningRate,
  bestHyperparameters.dropoutRate,
  64,
  'adamw',
  0.01,
  ['relu', 'relu', 'softmax'], // Example activation functions
  true // useAttention
);

// Train Final Neural Network
finalNeuralNetwork.train(
  trainingData.map((data) => data.input),
  trainingData.map((data) => data.target),
  250 
);

// Word Database
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

// Response Generation Functions
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
    "Why are you being lazy ? lazy will make your math get C in the test and F in the English class",
    "I afraid that I can't help you with that, you know, i don't have large data set to learn from, but you have, stop making you grade looks like C inside the trash can",
    "I apologize, but I'm not sure how to respond to that. Is there another way you could phrase your question?",
    "I'm afraid I don't have enough information to answer that. Could you provide more context or ask a different question?",
    "That's a bit beyond my current capabilities. Is there something else I can help with?",
  ];
  return responses[Math.floor(Math.random() * responses.length)];
};


// Enhanced Machine Learning Function
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
    ...topics.map(topic => inputLower.includes(topic) ? 1 : 0)
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

// Generate Date Response Function
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

// Generate Topic Response Function
const generateTopicResponse = (): string => {
  const topics = wordDatabase.topics.slice(0, 3);
  return `I'd be happy to help! I can assist with a variety of topics. For example, we could discuss ${topics.join(', ')}, or any other subject you're interested in. What would you like help with?`;
};

// Generate Summary from Search Results Function
const generateSummaryFromSearchResults = (results: SearchResult[]): string => {
  const topResults = results.slice(0, 2);
  return topResults.map(result => `${result.title}: ${stripHtml(result.snippet)}`).join(' ');
};

// Strip HTML Function
const stripHtml = (html: string): string => {
  const tmp = document.createElement("DIV");
  tmp.innerHTML = html;
  return tmp.textContent || tmp.innerText || "";
};

// Handle High Quality Search Function (This might need to be adjusted based on your API)
const handleHighQualitySearch = async (query: string): Promise<SearchResult[]> => {
  // ... your search logic here
  // Example using Wikipedia API:
  if (!query.trim()) {
    return [];
  }

  const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&origin=*`;

  try {
    const response = await fetch(url);
    const data = await response.json();
    return data.query.search;
  } catch (error) {
    console.error('Error fetching search results:', error);
    return [];
  }
};

// Generate Summary Response
const generateSummaryResponse = (summary: string): string => {
  const sentences = summary.split('. ');
  const shortSummary = sentences.slice(0, 2).join('. ');
  return `${shortSummary}. This gives us a general idea, but there's more to explore if you're interested.`;
};

// Generate Analysis
const generateAnalysis = (memory: string[]): string => {
  const recentSearches = memory.slice(-2);
  return `It seems we've been exploring topics related to ${recentSearches.join(' and ')}. This could indicate a broader interest in ${guessTheme(recentSearches)}. Would you like to delve deeper into any specific aspect?`;
};

// Guess Theme
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


// Perform Named Entity Recognition (NER)
const performNER = (text: string): string => {
  // Your NER logic here
  // Example:
  const entities = [];
  const words = text.split(/\s+/);

  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    let nextWord = words[i + 1] || '';

    if (
      word.match(/^[A-Z][a-z]+$/) &&
      nextWord.match(/^(Mr\.|Ms\.|Dr\.|Mrs\.)$/i)
    ) {
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

// Perform Part-of-Speech (POS) Tagging
const performPOS = (text: string): string => {
  // Your POS Tagging logic here
  // Example using simple regex:
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

// Perform Text Summarization
const performSummarization = (text: string): string => {
  const sentences = getSentences(text);
  const wordFrequencies = getWordCounts(text);
  const filteredFrequencies = filterStopWords(wordFrequencies);
  const sortedWords = sortByFreqThenDropFreq(filteredFrequencies);

  const maxSummarySize = 3;
  const summarySentences = new Set<string>();

  // Add the first sentence to the summary
  if (sentences.length > 0) {
    summarySentences.add(formatFirstSentence(sentences[0], sentences));
  }

  // Select sentences based on word frequency
  for (const word of sortedWords) {
    const matchingSentence = search(sentences, word);
    if (matchingSentence && !summarySentences.has(matchingSentence)) {
      summarySentences.add(matchingSentence);
      if (summarySentences.size >= maxSummarySize) {
        break;
      }
    }
  }

  return Array.from(summarySentences).join(' ');
};

// Get Sentences
const getSentences = (text: string): string[] => {
  const cleanedText = text.replace(/Mr\.|Ms\.|Dr\.|Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sep\.|Sept\.|Oct\.|Nov\.|Dec\.|St\.|Prof\.|Mrs\.|Gen\.|Corp\.|Sr\.|Jr\.|cm\.|Ltd\.|Col\.|vs\.|Capt\.|Univ\.|Sgt\.|ft\.|in\.|Ave\.|Lt\.|etc\.|mm\./g, match => match.replace('.', ''))
    .replace(/([A-Z])\./g, '$1')
    .replace(/\n/g, ' ');

  return cleanedText.match(/[^.!?]+[.!?]+/g) || [];
};

// Get Word Counts
const getWordCounts = (text: string): Map<string, number> => {
  const words = text.toLowerCase().match(/\b\w+\b/g) || [];
  const wordCounts = new Map<string, number>();
  for (const word of words) {
    wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
  }
  return wordCounts;
};

// Filter Stop Words
const filterStopWords = (wordCounts: Map<string, number>): Map<string, number> => {
  const stopWords = new Set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
    'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']);
  
  for (const word of stopWords) {
    wordCounts.delete(word);
  }
  return wordCounts;
};

// Sort By Frequency Then Drop Frequency
const sortByFreqThenDropFreq = (wordFrequencies: Map<string, number>): string[] => {
  return Array.from(wordFrequencies.entries())
    .sort((a, b) => b[1] - a[1])
    .map(entry => entry[0]);
};

// Search For Sentence Containing Word
const search = (sentences: string[], word: string): string | undefined => {
  return sentences.find(sentence => sentence.toLowerCase().includes(word.toLowerCase()));
};


// Format First Sentence
const formatFirstSentence = (firstSentence: string, sentences: string[]): string => {
  const datePattern = /(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)\s\d{1,2}\s(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\s\d{1,2}\.\d{2}(\sEST|\sPST)/;
  
  firstSentence = firstSentence.replace('Last modified on', '').replace(datePattern, '');

  const bbcArticles = 'Share this with Email Facebook Messenger Messenger Twitter Pinterest WhatsApp LinkedIn Copy this link These are external links and will open in a new window';
  const guardianArticles = 'We use cookies to improve your experience on our site and to show you personalised advertising';

  if (firstSentence.includes(bbcArticles)) {
    firstSentence = firstSentence.replace(bbcArticles, '');
  }

  if (firstSentence.includes('First published on') || firstSentence.includes(guardianArticles)) {
    firstSentence = firstSentence.replace('First published on', '');
    firstSentence = sentences[2] || firstSentence;
  }

  if (firstSentence.includes('MailOnline')) {
    const words = firstSentence.split(' ');
    const commentsIndex = words.indexOf('comments');
    if (commentsIndex !== -1) {
      firstSentence = words.slice(commentsIndex + 1).join(' ').trim();
    }
  }

  return firstSentence.trim();
};

// Export all functions and variables
export { 
  enhancedMachineLearning, 
  calculateAccuracy, 
  trainingData, 
  bestHyperparameters, 
  EnhancedNeuralNetwork,
  generateDateResponse,
  generateSummaryFromSearchResults, 
  stripHtml, 
  handleHighQualitySearch,
  generateSummaryResponse,
  generateAnalysis,
  guessTheme,
  performNER, 
  performPOS, 
  performSummarization,
  formatFirstSentence,
  search,
  sortByFreqThenDropFreq,
  filterStopWords,
  getWordCounts,
  getSentences,
  finalNeuralNetwork
};