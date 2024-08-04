import React, { useState, useRef, useEffect } from 'react';
import { FaPaperPlane, FaSmile, FaMicrophone, FaImage, FaSun, FaMoon } from 'react-icons/fa';
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
    { text: "Tell me a joke", icon: <FaSmile /> },
    { text: "What's the latest news?", icon: <FaImage /> },
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Enhanced Neural Network implementation with advanced techniques
  class EnhancedNeuralNetwork {
    private layers: number[][];
    private weights: number[][][];
    private biases: number[][];
    private learningRate: number;
    private velocities: number[][][];
    private dropoutRate: number;
    private batchSize: number;
    private optimizer: 'adam' | 'rmsprop' | 'sgd' = 'adam';
    private adamParams: { beta1: number; beta2: number; epsilon: number };
    private rmspropParams: { decay: number; epsilon: number };
    private l2RegularizationRate: number;

    constructor(
      layerSizes: number[],
      learningRate: number = 0.001,
      dropoutRate: number = 0.5,
      batchSize: number = 32,
      optimizer: 'adam' | 'rmsprop' | 'sgd' = 'adam',
      l2RegularizationRate: number = 0.01
    ) {
      this.layers = layerSizes.map(size => new Array(size).fill(0));
      this.weights = [];
      this.biases = [];
      this.velocities = [];
      this.learningRate = learningRate;
      this.dropoutRate = dropoutRate;
      this.batchSize = batchSize;
      this.optimizer = optimizer;
      this.adamParams = { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
      this.rmspropParams = { decay: 0.9, epsilon: 1e-8 };
      this.l2RegularizationRate = l2RegularizationRate;

      for (let i = 1; i < layerSizes.length; i++) {
        this.weights.push(Array.from({ length: layerSizes[i] }, () => 
          Array(layerSizes[i-1]).fill(0).map(() => this.initializeWeight(layerSizes[i-1]))
        ));
        this.biases.push(Array(layerSizes[i]).fill(0));
        this.velocities.push(Array.from({ length: layerSizes[i] }, () => 
          Array(layerSizes[i-1]).fill(0)
        ));
      }
    }

    private initializeWeight(fanIn: number): number {
      return Math.random() * Math.sqrt(2 / fanIn); // He initialization
    }

    private activation(x: number, type: string): number {
      switch (type) {
        case 'sigmoid': return 1 / (1 + Math.exp(-x));
        case 'tanh': return Math.tanh(x);
        case 'relu': return Math.max(0, x);
        case 'leakyRelu': return x > 0 ? x : 0.01 * x;
        case 'elu': return x >= 0 ? x : (Math.exp(x) - 1);
        case 'swish': return x / (1 + Math.exp(-x));
        default: return x;
      }
    }

    private activationDerivative(x: number, type: string): number {
      switch (type) {
        case 'sigmoid': return x * (1 - x);
        case 'tanh': return 1 - x * x;
        case 'relu': return x > 0 ? 1 : 0;
        case 'leakyRelu': return x > 0 ? 1 : 0.01;
        case 'elu': return x >= 0 ? 1 : x + 1;
        case 'swish': return x * (1 / (1 + Math.exp(-x))) + (1 / (1 + Math.exp(-x))) * (1 - x * (1 / (1 + Math.exp(-x))));
        default: return 1;
      }
    }

    private softmax(arr: number[]): number[] {
      const expValues = arr.map(val => Math.exp(val - Math.max(...arr)));
      const sumExpValues = expValues.reduce((a, b) => a + b, 0);
      return expValues.map(val => val / sumExpValues);
    }

    private dropout(layer: number[]): number[] {
      return layer.map(neuron => Math.random() > this.dropoutRate ? neuron / (1 - this.dropoutRate) : 0);
    }

    private forwardPropagation(input: number[], isTraining: boolean = true): number[] {
      this.layers[0] = input;
      for (let i = 1; i < this.layers.length; i++) {
        for (let j = 0; j < this.layers[i].length; j++) {
          let sum = this.biases[i-1][j];
          for (let k = 0; k < this.layers[i-1].length; k++) {
            sum += this.layers[i-1][k] * this.weights[i-1][j][k];
          }
          this.layers[i][j] = i === this.layers.length - 1 ? sum : this.activation(sum, ['tanh', 'relu', 'elu', 'leakyRelu', 'swish'][i % 5]);
        }
        if (isTraining && i < this.layers.length - 1) {
          this.layers[i] = this.dropout(this.layers[i]);
        }
      }
      this.layers[this.layers.length - 1] = this.softmax(this.layers[this.layers.length - 1]);
      return this.layers[this.layers.length - 1];
    }

    private backPropagation(target: number[]): void {
      const deltas: number[][] = new Array(this.layers.length).fill(0).map(() => []);
      
      for (let i = 0; i < this.layers[this.layers.length - 1].length; i++) {
        deltas[this.layers.length - 1][i] = target[i] - this.layers[this.layers.length - 1][i];
      }
      
      for (let i = this.layers.length - 2; i > 0; i--) {
        for (let j = 0; j < this.layers[i].length; j++) {
          let error = 0;
          for (let k = 0; k < this.layers[i + 1].length; k++) {
            error += deltas[i + 1][k] * this.weights[i][k][j];
          }
          deltas[i][j] = error * this.activationDerivative(this.layers[i][j], ['tanh', 'relu', 'elu', 'leakyRelu', 'swish'][i % 5]);
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

    private updateWeight(layerIndex: number, neuronIndex: number, weightIndex: number, gradient: number): void {
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
      }
      this.weights[layerIndex][neuronIndex][weightIndex] -= this.l2RegularizationRate * this.weights[layerIndex][neuronIndex][weightIndex];
    }

    private adamOptimizer(layerIndex: number, neuronIndex: number, weightIndex: number, gradient: number): void {
      const { beta1, beta2, epsilon } = this.adamParams;
      const m = this.velocities[layerIndex][neuronIndex][weightIndex] = beta1 * this.velocities[layerIndex][neuronIndex][weightIndex] + (1 - beta1) * gradient;
      const v = this.velocities[layerIndex][neuronIndex][weightIndex] = beta2 * this.velocities[layerIndex][neuronIndex][weightIndex] + (1 - beta2) * gradient * gradient;
      const mHat = m / (1 - Math.pow(beta1, this.batchSize));
      const vHat = v / (1 - Math.pow(beta2, this.batchSize));
      this.weights[layerIndex][neuronIndex][weightIndex] += this.learningRate * mHat / (Math.sqrt(vHat) + epsilon);
    }

    private rmspropOptimizer(layerIndex: number, neuronIndex: number, weightIndex: number, gradient: number): void {
      const { decay, epsilon } = this.rmspropParams;
      const v = this.velocities[layerIndex][neuronIndex][weightIndex] = decay * this.velocities[layerIndex][neuronIndex][weightIndex] + (1 - decay) * gradient * gradient;
      this.weights[layerIndex][neuronIndex][weightIndex] += this.learningRate * gradient / (Math.sqrt(v) + epsilon);
    }

    private sgdOptimizer(layerIndex: number, neuronIndex: number, weightIndex: number, gradient: number): void {
      this.weights[layerIndex][neuronIndex][weightIndex] += this.learningRate * gradient;
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
      return output.reduce((sum, value, index) => sum - target[index] * Math.log(value), 0);
    }
  }

  const neuralNetwork = new EnhancedNeuralNetwork([1250, 1600, 3200, 1600, 800, 400, 125], 0.0025, 0.5, 80, 'adam', 0.01);

  const enhancedMachineLearning = (input: string): string => {
    const keywords = [
      'hello', 'hi', 'hey', 'greetings',
      'how are you', 'how\'s it going', 'what\'s up',
      'weather', 'temperature', 'forecast', 'sunny', 'rainy', 'cloudy',
      'joke', 'funny', 'humor', 'laugh',
      'news', 'current events', 'headlines', 'latest',
      'bye', 'goodbye', 'see you', 'farewell',
      'thanks', 'thank you', 'appreciate', 'grateful',
      'help', 'assist', 'support', 'guidance',
      'time', 'date', 'day', 'hour', 'minute',
      'name', 'who are you', 'what are you', 'identity',
      'music', 'song', 'artist', 'genre',
      'movie', 'film', 'actor', 'director',
      'book', 'author', 'read', 'literature',
      'food', 'recipe', 'cuisine', 'restaurant',
      'travel', 'vacation', 'destination', 'trip',
      'sports', 'team', 'player', 'game',
      'technology', 'gadget', 'innovation', 'app'
    ];

    const responses = [
      'Hello! How can I assist you today?',
      "I'm doing well, thank you for asking! How about you?",
      "I'm afraid I don't have real-time weather data. You might want to check a reliable weather service for the most up-to-date information.",
      "Here's a joke for you: Why don't scientists trust atoms? Because they make up everything!",
      "I apologize, but I don't have access to current news. For the latest updates, please check a reputable news website.",
      'Goodbye! It was a pleasure chatting with you. Have a great day!',
      "You're welcome! I'm always here to help.",
      "I'd be happy to help. What do you need assistance with?",
      "I'm an AI assistant, so I don't track time, but you can check your device for the current time and date.",
      "I'm an AI language model created by GMTStudio. It's nice to meet you!",
      "Music is a universal language! What kind of music do you enjoy?",
      "Movies are a great form of entertainment. Do you have a favorite film or genre?",
      "Reading is a wonderful hobby. Is there a particular book or author you'd recommend?",
      "Food is such a diverse topic! Do you have a favorite cuisine or dish?",
      "Traveling can be very exciting! Do you have any dream destinations?",
      "Sports can be thrilling to watch and play. Do you follow any particular teams or athletes?",
      "Technology is advancing rapidly. Is there a recent innovation that has caught your attention?"
    ];

    const words = input.toLowerCase().split(/\s+/);
    const inputVector = keywords.map(keyword => 
      words.some(word => word.includes(keyword) || keyword.includes(word)) ? 1 : 0
    );
    const prediction = neuralNetwork.predict(inputVector);

    let response = responses[prediction];
    if (words.some(word => ['weather', 'temperature', 'forecast'].includes(word))) {
      const timeWords = words.filter(word => ['today', 'tomorrow', 'week'].includes(word));
      const conditionWords = words.filter(word => ['sunny', 'rainy', 'cloudy'].includes(word));
      if (timeWords.length > 0 || conditionWords.length > 0) {
        response += ` It seems you're asking about ${timeWords.join(' and ')} ${conditionWords.join(' and ')} weather. For the most accurate information, I recommend checking your local weather app.`;
      }
    }
    if (words.includes('joke') && words.includes('another')) {
      response += " Here's another one: Why did the scarecrow win an award? Because he was outstanding in his field!";
    }
    if (words.some(word => ['news', 'headlines', 'current'].includes(word)) && words.some(word => ['latest', 'recent', 'today'].includes(word))) {
      response += " For the most recent news, I suggest visiting a trusted news website or using a news aggregator app that provides real-time updates.";
    }

    return response || "I'm not quite sure how to respond to that. Could you please rephrase your question or ask something else?";
  };
  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;

    const newMessage: Message = {
      id: Date.now().toString(),
      sender: 'user',
      text: inputValue,
      timestamp: new Date(),
    };

    setMessages(prevMessages => [...prevMessages, newMessage]);
    setInputValue('');
    setIsTyping(true);

    // Use the enhanced neural network to generate a response
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: enhancedMachineLearning(newMessage.text),
        timestamp: new Date(),
      };
      setMessages(prevMessages => [...prevMessages, botResponse]);
      setIsTyping(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className={`flex flex-col h-screen ${darkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900'}`}>
      <div className="flex justify-between items-center p-4 border-b border-gray-700">
        <h1 className="text-2xl font-bold pl-10">{selectedChat ? selectedChat.title : 'New Chat'}</h1>
        <button onClick={() => setDarkMode(!darkMode)} className="p-2 rounded-full hover:bg-gray-700 transition-colors">
          {darkMode ? <FaSun className="text-yellow-500" /> : <FaMoon className="text-gray-700" />}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl p-3 rounded-lg ${
                message.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-white'
              }`}
            >
              {message.text}
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
          <button onClick={handleSendMessage} className="p-2 rounded bg-blue-600 text-white hover:bg-blue-700 transition-colors">
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