import React, { useState, useRef } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

const keywordResponses: { [keyword: string]: string } = {
 hello:'Hello ! Is there  anything I can help with ? You can type help to see what I can do !',
 hi:'Hi ? Anything wrong ? You can type --help see what I can do !',
name:'I am the AI who works in GMTStudio, They called me MAZS AI',
date:'Since I cannot access to the internet, I dont know the time, so I am guessing, maybe 13 o clock, right ?',
code:'well I am built with code, so I dont know how i can help you with and help you edit you code, this is irony isnt it? ',
Ai :"AI ? that is the things you are using right now, basically I am a computer, I dont know what you want me to do, I am not a witch or harry potter!",
technology:"this is a words created by humans , which i don't actually know about it ",
Health:"I am not a doctor, but i can give you some adivse :) for example, if you have headach, you have cancer, your nose are stuck, you have cancer! your hands made a loud noise, not cancer but you have broken your bones ",
music:"since I don't have ears, i don't know music",


  "help": 'I can help you with the following things : \n\n 1.  Say hi to you \n 2.  tell you about the team or studio or company which created me  \n 3.  tell you basic information \n 4.  err.... dont know but maybe talk to you \n 5. I can hack into your computer and use your file as my data ! just kidding I cant, but I will do it if our developer are smart enough. \n 6. You can type some command like --GMTStudio and --About to see more ! ',
  "--about":'About me ? Hmmm that is indead a great mystery, First, I am just one day old how dare you ask me that question ? and Yes I admit that I can answer that answer but My boss says no.',
  "--GMTStudio":'GMTStudio is a startup company based in the Taiwan. they are a group of 6 people plus one AI which is me, who are trying to make the technology better. ',
  "--developer mode":'🔐 Are you sure you want to open Developer Mode ? (type --confirm  to Open) ',
  "--confirm":" 🔓 Developer mode enable, for now on please type (.dev-[Your sentence]), for example : .dev-hello",
  ".dev-hello":" Hello, I am Mazs AI, I am here to help you with your daily life ! I can guide you to an advanture of GMTStudio",
  ".dev-hi":" Hi, I am Mazs AI, Your personal AI helper, I can guide you to an advanture of GMTStudio ! ",



};

const defaultResponse = 'Looks like I do not have that in database ! would you contact the Team to fix it ?  ';

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
