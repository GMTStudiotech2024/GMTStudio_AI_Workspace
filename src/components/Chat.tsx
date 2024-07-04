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
  who_are_you: "I'm Mazs AI v0.1.5, your virtual assistant. How can I help you?",
  name: "I'm Mazs AI v0.1.5, your virtual assistant.",
  quiz_capitals: "Sure! Let's start a quiz. What is the capital of France?",
  python_script: "I can help you with Python scripts. For example, here's a script for sending daily email reports.",
  comfort_friend: "Here's a message to comfort a friend: 'I'm here for you, always.'",
  plan_relaxing_day: "To plan a relaxing day, start with a good breakfast, a walk in nature, and some meditation.",
  weather_today: "I'm not connected to the internet, but you can check your local weather forecast online.",
  tell_joke: "Why don't scientists trust atoms? Because they make up everything!",
  book_recommendation: "I recommend 'To Kill a Mockingbird' by Harper Lee. It's a classic!",
  favorite_movie: "One of my favorite movies is 'Inception' directed by Christopher Nolan.",
  news_update: "I'm not connected to the internet, but you can check the latest news on your preferred news website.",
  health_tip: "Remember to stay hydrated, exercise regularly, and get enough sleep.",
  motivational_quote: "Here's a motivational quote: 'The only way to do great work is to love what you do.' - Steve Jobs",
  programming_help: "I can assist with programming questions. What do you need help with?",
  math_problem: "Sure, I can help with math problems. What do you need assistance with?",
  translate: "I can help translate text. What do you need translated and into which language?",
  favorite_book: "One of my favorite books is '1984' by George Orwell.",
  time_management: "To manage your time effectively, prioritize tasks, set clear goals, and take breaks.",
  random_fact: "Did you know? Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.",
  technology_trend: "A current technology trend is the rise of artificial intelligence and machine learning in various industries.",
  history_question: "Sure, I can help with history questions. What do you need to know?",
  cooking_recipe: "I can help with recipes. What dish are you interested in cooking?",
  travel_recommendation: "I recommend visiting Kyoto, Japan. It's known for its beautiful temples, gardens, and traditional tea houses.",
  workout_routine: "For a balanced workout routine, include cardio, strength training, and flexibility exercises.",
  mental_health_tip: "Remember to take breaks, practice mindfulness, and seek support when needed.",
  productivity_tip: "To boost productivity, break tasks into smaller steps and take regular breaks.",
  fun_fact: "Fun fact: A group of flamingos is called a 'flamboyance'.",
  science_fact: "Science fact: The speed of light is approximately 299,792 kilometers per second.",
  space_fact: "Space fact: Jupiter is the largest planet in our solar system.",
  animal_fact: "Animal fact: An octopus has three hearts and blue blood.",
  geography_fact: "Geography fact: Australia is both a country and a continent.",
  music_recommendation: "I recommend listening to 'Bohemian Rhapsody' by Queen. It's a timeless classic.",
  art_recommendation: "I recommend looking into the works of Vincent van Gogh. 'Starry Night' is particularly famous.",
  movie_recommendation: "I recommend watching 'The Shawshank Redemption'. It's a great movie."
};

const patterns: { [key: string]: RegExp } = {
  greeting: /\b(hi|hello|hey|hola)\b/i,
  help: /\b(help|assist)\b/i,
  who_are_you: /\b(who are you|what are you|who is this)\b/i,
  name: /\b(your name|who are you|what are you called)\b/i,
  quiz_capitals: /\b(quiz me on world capitals|capitals quiz)\b/i,
  python_script: /\b(python script for daily email reports|help with python)\b/i,
  comfort_friend: /\b(message to comfort a friend|comforting message)\b/i,
  plan_relaxing_day: /\b(plan a relaxing day|relaxing day plan)\b/i,
  my_name_is: /\b(my name is (\w+))\b/i,
  weather_today: /\b(weather today|current weather)\b/i,
  tell_joke: /\b(tell me a joke|make me laugh)\b/i,
  book_recommendation: /\b(recommend me a book|book recommendation)\b/i,
  favorite_movie: /\b(favorite movie|movie you like)\b/i,
  news_update: /\b(latest news|news update)\b/i,
  health_tip: /\b(health tip|health advice)\b/i,
  motivational_quote: /\b(motivational quote|inspire me)\b/i,
  programming_help: /\b(programming help|code help|programming question)\b/i,
  math_problem: /\b(math problem|help with math|solve this math)\b/i,
  translate: /\b(translate|translation|translate this)\b/i,
  favorite_book: /\b(favorite book|book you like)\b/i,
  time_management: /\b(time management|manage time|time tips)\b/i,
  random_fact: /\b(random fact|tell me a fact|interesting fact)\b/i,
  technology_trend: /\b(technology trend|latest tech|tech update)\b/i,
  history_question: /\b(history question|ask about history|history fact)\b/i,
  cooking_recipe: /\b(cooking recipe|recipe for|how to cook)\b/i,
  travel_recommendation: /\b(travel recommendation|place to visit|travel tip)\b/i,
  workout_routine: /\b(workout routine|exercise plan|workout tips)\b/i,
  mental_health_tip: /\b(mental health tip|mental well-being|mental health advice)\b/i,
  productivity_tip: /\b(productivity tip|boost productivity|productivity advice)\b/i,
  fun_fact: /\b(fun fact|interesting fact|did you know)\b/i,
  science_fact: /\b(science fact|science information|science trivia)\b/i,
  space_fact: /\b(space fact|space information|space trivia)\b/i,
  animal_fact: /\b(animal fact|animal information|animal trivia)\b/i,
  geography_fact: /\b(geography fact|geography information|geography trivia)\b/i,
  music_recommendation: /\b(recommend me a song|music recommendation|song you like)\b/i,
  art_recommendation: /\b(recommend me art|art recommendation|art you like)\b/i,
  movie_recommendation: /\b(recommend me a movie|movie recommendation|movie you like)\b/i
};


const synonyms: { [key: string]: string[] } = {
  hello: ['hi', 'hey', 'greetings','hola'],
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
    { text: "who are you", icon: <span>🤔</span> },
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
      setIsTyping(false); // Stop typing animation after response
    }, 1000 + Math.random() * 1500); // Adjust timing as needed
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
