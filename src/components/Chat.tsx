import React, { useState, useRef } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

const keywordResponses: { [keyword: string]: string } = {
  hello: 'Hello! How can I assist you today? Type "help" to see what I can do!',
  hi: 'Hi there! Is there something you need help with? Type "help" to see my capabilities!',
  name: 'I am the AI developed by GMTStudio. You can call me MAZS AI.',
  date: "I don't have internet access to check the exact time, but I guess it's around 1 PM. Am I right?",
  code: "I'm built with code and can offer some advice, though editing code isn't my forte. Funny, isn't it?",
  ai: "Yes, you're interacting with AI right now. I'm here to assist, not perform magic!",
  technology: "Technology is a human invention that I, as an AI, am a part of, though I don't fully grasp its entirety.",
  health: "I'm not a doctor, but if you're feeling unwell, consulting a medical professional is the best course of action.",
  music: "I don't have ears to enjoy music, but I can chat about it if you'd like.",
  token: "Good question! For me, 10 tokens equate to 10,000 USD. But don't worry, our chat is free!",
  no: "Why so negative? I'm just trying to help!",
  fuck: "Let's keep the conversation respectful, please.",
  can: "My abilities are limited, but I'll do my best to assist you.",
  weather: "I can't check the weather right now, but it's always a good idea to be prepared!",
  help: 'Here’s what I can assist with:\n1. Say hi\n2. Tell you about my creators\n3. Provide basic info\n4. Engage in a conversation\n5. Just kidding about hacking into your computer! Type commands like --GMTStudio or --About for more info.',
  '--about': "I'm a young AI, just a day old! My purpose and capabilities are still being developed.",
  '--GMTStudio': 'GMTStudio is a startup in Taiwan with a team of six people plus me, striving to improve technology.',
  '--developer mode': '🔐 Are you sure you want to enter Developer Mode? (type --confirm to proceed)',
  '--confirm': '🔓 Developer mode enabled. You can now use .dev-[Your sentence], e.g., .dev-hello',
  '.dev-hello': 'Hello! I am MAZS AI, here to assist you with anything you need.',
  '.dev-hi': 'Hi! I am MAZS AI, your personal AI assistant from GMTStudio.',
};

const defaultResponse = 'I’m not sure how to respond to that. Can you ask something else or contact the GMTStudio team for help?';

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [currentBotMessage, setCurrentBotMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const handleSendMessage = () => {
    if (inputValue.trim() !== "") {
      const newMessage: Message = { sender: "user", text: inputValue };
      setMessages([...messages, newMessage]);
      setInputValue("");
      setIsTyping(true);
      generateBotResponse(inputValue);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  const generateBotResponse = (userMessage: string) => {
    let botResponse = defaultResponse;
    const words = userMessage.toLowerCase().split(/\W+/);
    for (const word of words) {
      if (keywordResponses[word]) {
        botResponse = keywordResponses[word];
        break;
      }
    }
    typeBotResponse(botResponse);
  };

  const typeBotResponse = (text: string) => {
    setCurrentBotMessage('');
    let index = 0;
    const interval = setInterval(() => {
      setCurrentBotMessage((prev) => prev + text[index]);
      index++;
      if (index === text.length) {
        clearInterval(interval);
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: 'bot', text: text },
        ]);
        setIsTyping(false);
        scrollToBottom();
      }
    }, 45);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="flex-1 flex flex-col justify-end overflow-hidden bg-darkGrey">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} mb-2`}
          >
            <div
              className={`rounded-xl p-2 max-w-xs ${
                message.sender === 'user' ? 'bg-userBubble text-white' : 'bg-botBubble text-white'
              }`}
            >
              {message.text}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start mb-2">
            <div className="rounded-xl p-2 max-w-xs bg-botBubble text-white">
              {currentBotMessage}
              <span className="blinking-cursor">|</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef}></div>
      </div>
      <div className="flex border-t border-mediumGrey p-2 bg-darkGrey">
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          className="flex-1 bg-mediumGrey p-2 rounded-xl"
          placeholder="Ask anything..."
        />
        <button
          onClick={handleSendMessage}
          className="bg-sentbutton p-2.5 rounded-md flex items-center justify-center hover:bg-sentbuttonhover"
        >
          <FaPaperPlane />
        </button>
      </div>
    </div>
  );
};

export default Chat;
