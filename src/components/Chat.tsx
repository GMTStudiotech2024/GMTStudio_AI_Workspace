import React, { useState, useRef, useEffect } from 'react';
import {
  FaPaperPlane,
  FaSmile,
  FaMicrophone,
  FaImage,
  FaSun,
  FaMoon,
  FaThumbsUp,
  FaThumbsDown,
  FaTrashAlt,
} from 'react-icons/fa';
import { Message, Suggestion } from '../types';
import { motion } from 'framer-motion';

interface ChatProps {
  selectedChat: { title: string } | null;
}

const Chat: React.FC<ChatProps> = ({ selectedChat }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestions: Suggestion[] = [
    { text: "What's the weather like today?", icon: <FaSun /> },
    { text: 'Tell me a joke', icon: <FaSmile /> },
    { text: "What's the latest news?", icon: <FaImage /> },
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

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
              .map(() => this.initializeWeight(layerSizes[i - 1], layerSizes[i]))
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

    private initializeWeight(fanIn: number, fanOut: number): number {
      return Math.random() * Math.sqrt(2 / (fanIn + fanOut));
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

    private forwardPropagation(input: number[], isTraining: boolean = true): number[] {
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

    train(inputs: number[][], targets: number[][], epochs: number): void {
      for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;
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
  }

  // Initialize the neural network
  const neuralNetwork = new EnhancedNeuralNetwork(
    [15, 50,50, 10], // Example network architecture (adjust if needed)
    0.001, // Example learning rate
    0.3, // Example dropout rate
    64, // Example batch size
    'adamw', // Example optimizer
    0.01 // Example L2 regularization rate
  );

  // Expanded Training Data (More Examples and Intents)
  const trainingData = [
    // Greetings
    { input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hello"
    { input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hi"
    { input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }, // "good morning"
    { input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] }, // "good evening"
    { input: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }, // "hey there"

    // Farewells
    { input: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "goodbye"
    { input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "bye"
    { input: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }, // "see you later"

    // Weather
    { input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "what's the weather like?"
    { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], target: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }, // "how's the weather?"

    // Jokes
    { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "tell me a joke"
    { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], target: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }, // "tell me a funny joke"

    // How are you?
    { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], target: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }, // "how are you?"
    { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], target: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }, // "how are you doing?"
    { input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], target: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }, // "how's it going?"
  ];

  // Train the neural network
  const epochs = 50;
  neuralNetwork.train(
    trainingData.map((data) => data.input),
    trainingData.map((data) => data.target),
    epochs
  );

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
      "what's the weather like?",
      "how's the weather?",
      'tell me a joke',
      'tell me a funny joke',
      'How are you?',
      'How are you doing?',
      'Hows it going?',
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

    const predictedClass = neuralNetwork.predict(inputVector);

    // Contextual Responses with Word Combination
    const responses = {
      0: [
        'Hello! How are you doing today?',
        'Hi there! Do you need any help?',
        'Hey there! How can I assist you?',
      ],
      1: [
        'Good morning! How can I assist you today?',
        'Good morning to you! What are you up to?',
      ],
      2: [
        'Good evening! How was your day?',
        'Good evening to you! What can I do for you today?',
      ],
      3: [
        'Goodbye! Have a great day!',
        'See you later! Talk to you soon!',
      ],
      4: [
        "I'm afraid I don't have real-time weather data.",
        "I'm not able to provide weather information.",
      ],
      5: [
        'Why did the scarecrow win an award? Because he was outstanding in his field!',
        'What do you call a lazy kangaroo? Pouch potato!',
      ],
      6: [ // Responses for "How are you?"
        "I'm doing well, thank you for asking!",
        "I'm a chatbot, so I don't have feelings, but I'm functioning as expected!",
      ],
      7: [ // Responses for "How are you doing?"
        "I'm doing well, thank you for asking!",
        "I'm a chatbot, so I don't have feelings, but I'm functioning as expected!",
      ],
      8: [ // Responses for "How's it going?"
        "I'm doing well, thank you for asking!",
        "I'm a chatbot, so I don't have feelings, but I'm functioning as expected!",
      ],
      9:[ // Responses for not knowing the answer
        "I'm sorry, I don't know the answer to that question.",
        "I'm not sure how to respond to that. Can you please rephrase your question?",
      ]
      // Add more responses based on predicted classes
    };

    // Choose a random response from the array for the predicted class
    const randomResponseIndex = Math.floor(
      Math.random() * responses[predictedClass as keyof typeof responses].length
    );
    let response =
      responses[predictedClass as keyof typeof responses][randomResponseIndex];

    // Example of using chat history for context
    if (predictedClass === 0 && chatHistory.length > 0) {
      const lastUserMessage = chatHistory[chatHistory.length - 1].text;
      if (lastUserMessage.includes('weather')) {
        response =
          "We were just talking about the weather! Anything else you'd like to know?";
      }
    }

    return (
      response ||
      "I'm not quite sure how to respond to that. Could you please rephrase your question or ask something else?"
    );
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
      "what's the weather like?",
      "how's the weather?",
      'tell me a joke',
      'tell me a funny joke',
      'How are you?',
      'How are you doing?',
      'Hows it going?',
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
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: enhancedMachineLearning(newMessage.text, messages),
        timestamp: new Date(),
      };
      setMessages((prevMessages) => [...prevMessages, botResponse]);
      setIsTyping(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  const handleFeedback = (messageId: string, feedback: 'good' | 'bad') => {
    const messageIndex = messages.findIndex(
      (message) => message.id === messageId
    );

    if (messageIndex !== -1) {
      const inputVector = messages[messageIndex - 1].inputVector;

      if (inputVector) {
        const targetVector = trainingData.find(
          (data) => data.input.toString() === inputVector.toString()
        )?.target;

        if (targetVector) {
          const predictedClass = neuralNetwork.predict(inputVector);
          if (feedback === 'good') {
            targetVector[predictedClass] += 0.1;
          } else {
            targetVector[predictedClass] -= 0.1;
          }

          neuralNetwork.train(
            trainingData.map((data) => data.input),
            trainingData.map((data) => data.target),
            10
          );
        }
      }
    }
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  return (
<div
      className={`flex flex-col h-screen ${
        darkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900'
      }`}
    >
      <div className="flex justify-between items-center p-4 border-b border-gray-700">
        <h1 className="text-2xl font-bold pl-10">
          {selectedChat ? selectedChat.title : 'New Chat'}
        </h1>
        <div className="flex items-center">
          {/* Clear Chat Button */}
          <button
            onClick={handleClearChat}
            className="p-2 rounded-full hover:bg-gray-700 transition-colors mr-2"
          >
            <FaTrashAlt className="text-red-500" />
          </button>
          {/* Dark Mode Button */}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-full hover:bg-gray-700 transition-colors"
          >
            {darkMode ? (
              <FaSun className="text-yellow-500" />
            ) : (
              <FaMoon className="text-gray-700" />
            )}
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
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
              {/* Add feedback buttons for bot messages */}
              {message.sender === 'bot' && (
                <div className="flex justify-end mt-2 space-x-2">
                  <button
                    onClick={() => handleFeedback(message.id, 'good')}
                    className="text-green-500 hover:text-green-600"
                  >
                    <FaThumbsUp />
                  </button>
                  <button
                    onClick={() => handleFeedback(message.id, 'bad')}
                    className="text-red-500 hover:text-red-600"
                  >
                    <FaThumbsDown />
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-gray-700 text-white p-3 rounded-lg">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="border-t border-gray-700 p-4">
        {messages.length === 0 && (
          <div className="flex flex-wrap justify-center gap-2 mb-4">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => setInputValue(suggestion.text)}
                className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-white rounded-full px-4 py-2 text-sm transition-colors duration-200"
              >
                {suggestion.icon}
                <span>{suggestion.text}</span>
              </button>
            ))}
          </div>
        )}
        <div className="flex items-center space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            className="flex-1 p-2 rounded bg-gray-800 text-white border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSendMessage}
            className="p-2 rounded bg-blue-600 text-white hover:bg-blue-700 transition-colors"
          >
            <FaPaperPlane />
          </button>
          <button className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors">
            <FaSmile />
          </button>
          <button className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors">
            <FaMicrophone />
          </button>
          <button className="p-2 rounded bg-gray-800 text-white hover:bg-gray-700 transition-colors">
            <FaImage />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chat;