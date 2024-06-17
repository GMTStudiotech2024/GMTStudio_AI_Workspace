import React, { useState, useRef } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

const keywordResponses: { [keyword: string]: string } = {
  // ... (your keywordResponses here)
};

const defaultResponse = 'Looks like ';

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
      }
    }, 50);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  return (
    <div className="flex-1 flex flex-col justify-end overflow-hidden bg-darkGrey">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.sender === 'user' ? 'justify-end' : 'justify-start'
            } mb-2`}
          >
            <div
              className={`rounded-xl p-2 max-w-xs ${
                message.sender === 'user'
                  ? 'bg-userBubble text-white'
                  : 'bg-botBubble text-white'
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
