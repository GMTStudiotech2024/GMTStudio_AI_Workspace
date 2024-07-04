import React, { useState, useRef, useEffect } from 'react';
import { FaPaperPlane, FaRobot, FaUser, FaMicrophone, FaImage, FaSmile, FaMoon, FaSun } from 'react-icons/fa';
import '../components/animations.css';

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

const initialContext = {
  topic: '',
  quizState: { active: false, question: '', answer: '' },
  userName: ''
};

const responses: { [key: string]: string } = {
  greeting: "Hello! How can I assist you today?",
  help: "Sure, I'm here to help! What do you need assistance with?",
  who_are_you: "I'm an AI assistant. I'm here to help you with your questions and tasks.",
  name: "I'm Mazs AI v0.1.5, your virtual assistant.",
  quiz_capitals: "Sure! Let's start a quiz. What is the capital of France?",
  python_script: "I can help you with Python scripts. For example, here's a script for sending daily email reports.",
  comfort_friend: "Here's a message to comfort a friend: 'I'm here for you, always.'",
  plan_relaxing_day: "To plan a relaxing day, start with a good breakfast, a walk in nature, and some meditation.",
  weather_today: "I'm not connected to the internet, but you can check your local weather forecast online.",
  tell_joke: "Why don't scientists trust atoms? Because they make up everything!"
};

const patterns: { [key: string]: RegExp } = {
  greeting: /\b(hi|hello|hey)\b/i,
  help: /\b(help|assist)\b/i,
  who_are_you: /\b(who are you|what are you)\b/i,
  name: /\b(your name|who are you)\b/i,
  quiz_capitals: /\b(quiz me on world capitals|capitals quiz)\b/i,
  python_script: /\b(python script for daily email reports)\b/i,
  comfort_friend: /\b(message to comfort a friend)\b/i,
  plan_relaxing_day: /\b(plan a relaxing day)\b/i,
  my_name_is: /\b(my name is (\w+))\b/i,
  weather_today: /\b(weather today|current weather)\b/i,
  tell_joke: /\b(tell me a joke|make me laugh)\b/i
};

const synonyms: { [key: string]: string[] } = {
  hello: ['hi', 'hey', 'greetings'],
  help: ['assist', 'aid', 'support'],
  name: ['who are you', 'what are you'],
  quiz: ['quiz me on world capitals', 'capitals quiz'],
  python: ['python script for daily email reports'],
  comfort: ['message to comfort a friend'],
  relax: ['plan a relaxing day'],
  weather: ['weather today', 'current weather'],
  joke: ['tell me a joke', 'make me laugh']
};

const negatePattern = /\b(no|not|don't|never|none)\b/i;

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [context, setContext] = useState(initialContext);

  const suggestions: Suggestion[] = [
    { text: "Quiz me on world capitals", icon: <span>🌎</span> },
    { text: "Python script for daily email reports", icon: <span>📊</span> },
    { text: "Message to comfort a friend", icon: <span>💌</span> },
    { text: "Plan a relaxing day", icon: <span>🏖️</span> }
  ];

  const handleSendMessage = () => {
    if (inputValue.trim() !== '') {
      const newMessage: Message = {
        id: Date.now().toString(),
        sender: 'user',
        text: inputValue,
        timestamp: new Date()
      };
      setMessages([...messages, newMessage]);
      setInputValue('');
      generateBotResponse(newMessage.text);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const preprocessMessage = (message: string) => {
    return message.toLowerCase().replace(/[^\w\s]/gi, '');
  };

  const stemWord = (word: string) => {
    return word.replace(/(ing|ed|s)$/, '');
  };

  const lemmatizeWord = (word: string) => {
    const lemmas: { [key: string]: string } = {
      'am': 'be',
      'are': 'be',
      'is': 'be',
      'was': 'be',
      'were': 'be',
      'being': 'be',
      'has': 'have',
      'have': 'have',
      'had': 'have',
      'having': 'have'
    };
    return lemmas[word] || word;
  };

  const handleNegation = (message: string) => {
    if (negatePattern.test(message)) {
      return message.split(' ').map(word => (negatePattern.test(word) ? 'not' : word)).join(' ');
    }
    return message;
  };

  const handleSynonyms = (message: string) => {
    const words = message.split(' ');
    return words.map(word => {
      const synonymKey = Object.keys(synonyms).find(key =>
        synonyms[key].includes(word)
      );
      return synonymKey || word;
    }).join(' ');
  };

  const tokenizeAndProcess = (message: string) => {
    const tokens = message.split(' ');
    return tokens.map(token => lemmatizeWord(stemWord(token)));
  };

  const generateBotResponse = (userMessage: string) => {
    setIsTyping(true);

    const getBotResponse = (message: string) => {
      const preprocessedMessage = preprocessMessage(message);
      const withSynonyms = handleSynonyms(preprocessedMessage);
      const withNegation = handleNegation(withSynonyms);
      const tokens = tokenizeAndProcess(withNegation);
      const lowerCaseMessage = tokens.join(' ');

      let response = `I'm an AI assistant. You said: "${message}". How can I help you further?`;

      if (context.quizState.active) {
        if (lowerCaseMessage.includes(context.quizState.answer.toLowerCase())) {
          response = "Correct! Do you want another question?";
          setContext({ ...context, quizState: { active: false, question: '', answer: '' } });
        } else {
          response = `Incorrect. The correct answer is ${context.quizState.answer}. Do you want another question?`;
          setContext({ ...context, quizState: { active: false, question: '', answer: '' } });
        }
      } else {
        const foundPattern = Object.keys(patterns).find(key =>
          patterns[key].test(lowerCaseMessage)
        );

        if (foundPattern) {
          response = responses[foundPattern];
          if (foundPattern === "quiz_capitals") {
            setContext({
              ...context,
              quizState: { active: true, question: "What is the capital of France?", answer: "Paris" }
            });
          }
          if (foundPattern === "my_name_is") {
            const nameMatch = message.match(patterns.my_name_is);
            const userName = nameMatch ? nameMatch[2] : '';
            setContext({ ...context, userName });
            response = `Nice to meet you, ${userName}! How can I assist you today?`;
          }
        } else {
          response = `I'm not sure how to respond to that. Can you please clarify or ask something else?`;
        }
      }

      return response;
    };

    setTimeout(() => {
      const botResponse: Message = {
        id: Date.now().toString(),
        sender: 'bot',
        text: getBotResponse(userMessage),
        timestamp: new Date()
      };
      setMessages(prevMessages => [...prevMessages, botResponse]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1500);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const formatTimestamp = (date: Date) => {
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
