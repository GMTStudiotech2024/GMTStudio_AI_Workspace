import React, { useState, useRef, useEffect } from 'react';
import { FaPaperPlane, FaRobot, FaUser, FaMicrophone, FaImage, FaSmile, FaMoon, FaSun } from 'react-icons/fa';
import '../components/animations.css';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';

interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

interface Suggestion {
  text: string;
  icon: React.ReactNode;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);

  const suggestions: Suggestion[] = [
    { text: "Who are you", icon: <span>🤔</span> },
    { text: "Python script for daily email reports", icon: <span>📊</span> },
    { text: "Message to comfort a friend", icon: <span>💌</span> },
    { text: "Plan a relaxing day", icon: <span>🏖️</span> }
  ];

  const handleSendMessage = async () => {
    if (inputValue.trim() !== '') {
      const newMessage: Message = {
        id: Date.now().toString(),
        sender: 'user',
        text: inputValue,
        timestamp: new Date()
      };
      setMessages([...messages, newMessage]);
      setInputValue('');
      await generateBotResponse(newMessage.text);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const generateBotResponse = async (userMessage: string) => {
    setIsTyping(true);

    const apiKey = process.env.REACT_APP_GEMINI_API_KEY;
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: "gemini-1.0-pro",
    });

    const generationConfig = {
      temperature: 0.9,
      topP: 1,
      topK: 0,
      maxOutputTokens: 2048,
      responseMimeType: "text/plain",
    };

    const chatSession = model.startChat({
      generationConfig,
      history: messages.map(msg => ({ sender: msg.sender, text: msg.text })),
    });

    const result = await chatSession.sendMessage(userMessage);
    const botResponseText = result.response.text();

    const botResponse: Message = {
      id: Date.now().toString(),
      sender: 'bot',
      text: botResponseText,
      timestamp: new Date()
    };
    setMessages(prevMessages => [...prevMessages, botResponse]);
    setIsTyping(false);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const formatTimestamp = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleEmojiClick = (emoji: string) => {
    setInputValue(prevValue => prevValue + emoji);
    setShowEmojiPicker(false);
  };

  const renderEmojiPicker = () => (
    <div className="absolute bottom-16 right-0 bg-gray-800 rounded-lg p-2 shadow-lg">
      {['😊', '😂', '🤔', '👍', '❤️', '🎉'].map(emoji => (
        <button
          key={emoji}
          onClick={() => handleEmojiClick(emoji)}
          className="p-1 hover:bg-gray-700 rounded"
        >
          {emoji}
        </button>
      ))}
    </div>
  );

  return (
    <div className={`flex flex-col h-screen ${darkMode ? 'bg-black text-gray-100' : 'bg-white text-gray-900'}`}>
      <div className="flex justify-between p-4">
        <h1 className="text-2xl font-bold">Mazs AI</h1>
        <button onClick={() => setDarkMode(!darkMode)} className="p-2">
          {darkMode ? <FaSun className="text-yellow-500" /> : <FaMoon className="text-gray-700" />}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} message`}
          >
            <div
              className={`flex flex-col max-w-xs ${
                message.sender === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              <div
                className={`flex items-start space-x-2 rounded-lg p-3 ${
                  message.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-white'
                }`}
              >
                {message.sender === 'user' ? <FaUser className="mt-1" /> : <FaRobot className="mt-1" />}
                <p className="break-words">{message.text}</p>
              </div>
              <span className="text-xs text-gray-500 mt-1">
                {formatTimestamp(message.timestamp)}
              </span>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start message">
            <div className="flex flex-col max-w-xs items-start">
              <div className="flex items-center space-x-2 rounded-lg p-3 bg-gray-800 text-white">
                <FaRobot className="mt-1" />
                <span>...</span>
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.7s' }}></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="border-t border-gray-800 p-4">
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
        <div className="p-4 flex items-center border-t border-gray-700 relative">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            className="flex-1 p-2 rounded bg-gray-800 text-white border border-gray-700 focus:outline-none"
          />
          <button onClick={handleSendMessage} className="ml-2 p-2 rounded bg-blue-600 text-white">
            <FaPaperPlane />
          </button>
          <button onClick={() => setShowEmojiPicker(!showEmojiPicker)} className="ml-2 p-2 rounded bg-gray-800 text-white">
            <FaSmile />
          </button>
          {showEmojiPicker && renderEmojiPicker()}
          <button className="ml-2 p-2 rounded bg-gray-800 text-white">
            <FaMicrophone />
          </button>
          <button className="ml-2 p-2 rounded bg-gray-800 text-white">
            <FaImage />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chat;
