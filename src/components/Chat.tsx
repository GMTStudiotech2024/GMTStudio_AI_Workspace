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
  FiCheck,
  FiX,
  FiSettings,
  FiPause,
  FiSearch,
  FiSmile,
  FiGlobe,
  FiCpu,
  FiEdit,
} from 'react-icons/fi';
import { enhancedMachineLearning, calculateAccuracy, trainingData, generateSummaryFromSearchResults, stripHtml, handleHighQualitySearch, generateSummaryResponse,
  generateAnalysis, performNER, performPOS, performSummarization,  } from './chatlogic';
import logo from '../assets/GMTStudio_.png';
import { finalNeuralNetwork } from './chatlogic';

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

interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  image?: string;
  confirmationType?: 'math' | 'summary';
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognition extends EventTarget {
  start(): void;
  stop(): void;
  onstart: (event: Event) => void;
  onresult: (event: SpeechRecognitionEvent) => void;
  onend: (event: Event) => void;
}

declare global {
  interface Window {
    webkitSpeechRecognition: new () => SpeechRecognition;
    SpeechRecognition: new () => SpeechRecognition;
  }
}

const Chat: React.FC<ChatProps> = ({ selectedChat }) => {
  interface LoadingSpinnerProps {
    size?: number;
    color?: string;
  }

  const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ size = 60, color = "#ffffff" }) => (
    <motion.div
      className="loading-dots"
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
      style={{ fontSize: `${size / 10}px`, color: color }}
    >
      <span>.</span>
      <span>.</span>
      <span>.</span>
      <span>.</span>
      <span>.</span>
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
  const [selectedModel, setSelectedModel] = useState('Mazs AI v1.0 anatra');
  const [isListening, setIsListening] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [detectedMathExpression, setDetectedMathExpression] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [stopGenerating, setStopGenerating] = useState<(() => void) | null>(null);
  const [partialMessage, setPartialMessage] = useState<string | null>(null);
  const [isHighQualitySearch] = useState(false);
  const [searchMemory, setSearchMemory] = useState<string[]>([]);

  const [showInitialView, setShowInitialView] = useState(true);
  const [showCustomInput, setShowCustomInput] = useState(false);

  const suggestions: Suggestion[] = [
    { text: "What's the weather like today?", icon: <FiSun /> },
    { text: 'Tell me a joke', icon: <FiSmile /> },
    { text: "What's the latest news?", icon: <FiGlobe /> },
    { text: "What is AI", icon: <FiCpu /> },
  ];

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

        const totalEpochs = 250; // Reduced to 250 epochs

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
        await new Promise(resolve => setTimeout(resolve, 1000));

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

    setShowInitialView(false);

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

  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Array<{ title: string; snippet: string }>>([]);

  const handleSearch = async () => {
    if (inputValue.trim() === '') return;

    setShowInitialView(false);
    setIsTyping(true);
    setIsBotResponding(true);

    try {
      const results = await handleHighQualitySearch(inputValue);
      setSearchQuery(inputValue);
      setSearchResults(results);

      let summary = '';
      if (results.length > 0) {
        const topResults = results.slice(0, 3);
        summary = topResults.map(result => `${result.title}: ${stripHtml(result.snippet)}`).join(' ');

        setSearchMemory(prevMemory => [...prevMemory.slice(-4), summary]);
      }

      let response = '';
      if (summary) {
        response = ` ${generateSummaryResponse(summary)}`;

        if (searchMemory.length > 1) {
          response += ` Interestingly, this relates to our previous searches. ${generateAnalysis(searchMemory)}`;
        }
      } else {
        response = `I'm sorry, but I couldn't find any reliable information about "${inputValue}". Would you like to try a different search term?`;
      }

      const newMessage: ChatMessage = {
        id: Date.now().toString(),
        sender: 'user',
        text: `Search: ${inputValue}`,
        timestamp: new Date(),
      };
      setMessages(prevMessages => [...prevMessages, newMessage]);

      simulateTyping(response);
    } catch (error) {
      console.error('Error fetching or analyzing data:', error);
      simulateTyping('I encountered an issue while searching for that information. Could we try again in a moment?');
    } finally {
      setIsTyping(false);
      setIsBotResponding(false);
      setInputValue('');
    }
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
    setShowInitialView(true);
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

  

  const handleSummarization = () => {
    if (inputValue) {
      setIsTyping(true);
      const summaryResult = performSummarization(inputValue);
      simulateTyping(`Text Summary:\n${summaryResult}`);
      setInputValue('');
      setIsTyping(false);
    }
  };

  const handleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();

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

  

  return (
    <div
      className={`flex flex-col h-screen w-full ${
        darkMode ? 'bg-gray-900 text-gray-100' : 'bg-gray-100 text-gray-800'
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
            <h2 className="text-4xl font-bold mb-4">
              Initializing Artificial Intelligence
            </h2>
            <p className="text-xl mb-6">Please wait while we prepare your AI assistant...</p>
            <LoadingSpinner size={60} color={darkMode ? "#ffffff" : "#000000"} />
          </motion.div>
        )}
        {trainingStatus === 'training' && (
          <motion.div
            className="text-center p-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <h2 className="text-4xl font-bold mb-4">AI Training in Progress</h2>
            <p className="text-xl mb-6">This process will take approximately 15 seconds.</p>
            {trainingProgress && (
              <motion.div
                className="mt-6 bg-gray-800 p-6 rounded-lg shadow-lg max-w-md mx-auto"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <p className="text-2xl mb-4">Training Progress:</p>
                <div className="flex justify-between items-center mb-2">
                  <span>Epoch:</span>
                  <span className="font-semibold">{trainingProgress.epoch}/250</span>
                </div>
                <div className="flex justify-between items-center mb-2">
                  <span>Loss:</span>
                  <span className="font-semibold">{trainingProgress.loss.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center mb-4">
                  <span>Accuracy:</span>
                  <span className="font-semibold">{(trainingProgress.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-4">
                  <motion.div
                    className="bg-blue-600 h-4 rounded-full"
                    style={{ width: `${(trainingProgress.epoch / 250) * 100}%` }}
                    initial={{ width: 0 }}
                    animate={{ width: `${(trainingProgress.epoch / 250) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
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
            <h2 className="text-4xl font-bold mb-4">
              Oops! An Error Occurred
            </h2>
            <p className="text-xl mb-6">We encountered an issue during the AI training process.</p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => window.location.reload()}
              className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-full transition duration-300 text-lg"
            >
              Try Again
            </motion.button>
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
            <h1 className="text-3xl font-bold pl-10">
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
                  <FiSettings className="text-gray-400 text-2xl" />
                </motion.button>
              )}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleClearChat}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors"
                title="Clear Chat"
              >
                <FiEdit className="text-gray-400 text-2xl" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setDarkMode(!darkMode)}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors"
                title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
              >
                {darkMode ? (
                  <FiSun className="text-yellow-500 text-2xl" />
                ) : (
                  <FiMoon className="text-gray-700 text-2xl" />
                )}
              </motion.button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            <div className="flex flex-col p-4 space-y-4">
              {showInitialView ? (
                <div className="h-full flex flex-col items-center justify-center">
                  <motion.img
                    src={logo}
                    alt="GMTStudio AI Studio Logo"
                    className="w-40 h-40 mb-8"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 260, damping: 20 }}
                  />
                  <motion.h2
                    className="text-4xl font-bold mb-6 text-center"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    Welcome to GMTStudio AI Studio
                  </motion.h2>
                  <motion.p
                    className="text-xl mb-8 text-center text-gray-300 max-w-lg"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                  >
                    Explore the power of AI with our advanced chat interface. What would you like to discuss today?
                  </motion.p>
                  <div className="w-full max-w-4xl">
                    <motion.div
                      className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.6 }}
                    >
                      {suggestions.map((suggestion, index) => (
                        <motion.button
                          key={index}
                          whileHover={{ scale: 1.05, backgroundColor: "#4A5568" }}
                          whileTap={{ scale: 0.95 }}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.1 * index }}
                          onClick={() => {
                            setInputValue(suggestion.text);
                            handleSendMessage();
                          }}
                          className="flex items-center justify-start space-x-3 bg-gray-800 text-white rounded-lg px-6 py-4 text-sm transition-all duration-200 shadow-lg hover:shadow-xl border border-gray-700"
                        >
                          <div className="text-3xl">{suggestion.icon}</div>
                          <span className="flex-1 text-left font-medium">{suggestion.text}</span>
                        </motion.button>
                      ))}
                    </motion.div>
                  </div>
                  <motion.div
                    className="mt-12 text-center"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                  >
                    <p className="text-gray-400 mb-2">You can use advanced Search by typing in the topic you want to know about and clicking the search button next to the send button.</p>
                    <button
                      onClick={() => setShowCustomInput(true)}
                      className="text-blue-400 hover:text-blue-300 transition-colors duration-200 font-semibold text-lg"
                    >
                      Ask a custom question
                    </button>
                  </motion.div>
                </div>
              ) : messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center">
                  <p className="text-2xl mb-6">Your chat is empty. Start a conversation!</p>
                  <div className="w-full max-w-3xl">
                    <div className="grid grid-cols-2 gap-4">
                      {suggestions.map((suggestion, index) => (
                        <motion.button
                          key={index}
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => {
                            setInputValue(suggestion.text);
                            handleSendMessage();
                          }}
                          className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg px-4 py-3 text-lg transition-colors duration-200"
                        >
                          {suggestion.icon}
                          <span>{suggestion.text}</span>
                        </motion.button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
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
                        className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl p-4 rounded-lg shadow-md ${
                          message.sender === 'user'
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-700 text-white'
                        }`}
                      >
                        <p className="text-sm mb-2 font-semibold">{message.sender === 'user' ? 'You' : 'AI'}</p>
                        <p className="text-lg">{message.text}</p>
                        {message.image && (
                          <img
                            src={message.image}
                            alt="Uploaded"
                            className="mt-3 rounded-lg max-w-full h-auto"
                          />
                        )}
                        {message.confirmationType && message.id === pendingConfirmationId && (
                          <div className="flex space-x-3 mt-3">
                            <motion.button
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.9 }}
                              onClick={() => message.confirmationType === 'math' ? handleMathCalculation() : handleSummary()}
                              className="p-2 rounded-full bg-green-600 text-white hover:bg-green-700 transition-colors"
                            >
                              <FiCheck className="text-xl" />
                            </motion.button>
                            <motion.button
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.9 }}
                              onClick={proceedWithNormalMessage}
                              className="p-2 rounded-full bg-red-600 text-white hover:bg-red-700 transition-colors"
                            >
                              <FiX className="text-xl" />
                            </motion.button>
                          </div>
                        )}
                        {message.sender === 'bot' && (
                          <div className="mt-3 text-xs text-gray-400">
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
                          <div className="flex justify-end mt-3 space-x-3">
                            <motion.button
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.9 }}
                              onClick={() => handleFeedback(message.id, 'good')}
                              className="text-green-500 hover:text-green-600"
                              title="Thumbs Up"
                            >
                              <FiThumbsUp className="text-xl" />
                            </motion.button>
                            <motion.button
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.9 }}
                              onClick={() => handleFeedback(message.id, 'bad')}
                              className="text-red-500 hover:text-red-600"
                              title="Thumbs Down"
                            >
                              <FiThumbsDown className="text-xl" />
                            </motion.button>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              )}
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
          </div>
          <div className="border-t border-gray-700 p-4 pt-1">
            <div className="flex items-center space-x-2 pb-3">
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
                  <option value="Mazs AI v1.0 anatra">Mazs AI v1.0 anatra (50ms)</option>
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
            {searchResults.map((result: { title: string; snippet: string }) => (
              <div key={result.title} className="search-result bg-gray-700 p-4 rounded">
                <h3 className="text-lg font-semibold">{result.title}</h3>
                <p className="mt-2 text-sm">{stripHtml(result.snippet)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Custom Input Modal */}
      {showCustomInput && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="bg-gray-800 p-6 rounded-lg w-96"
          >
            <h2 className="text-xl font-bold mb-4">Ask a Custom Question</h2>
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your question here..."
              className="w-full h-32 p-2 bg-gray-700 text-white rounded mb-4"
            />
            <div className="flex justify-end space-x-2">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowCustomInput(false)}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
              >
                Cancel
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => {
                  handleSendMessage();
                  setShowCustomInput(false);
                  setShowInitialView(false);
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              >
                Send
              </motion.button>
                    </div>
          </motion.div>
        </div>
                    )}

                  </div>
  );
};

export default Chat;