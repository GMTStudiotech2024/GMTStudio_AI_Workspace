import React, { useState, useEffect, useRef } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

const keywordResponses: { [keyword: string]: string } = {
    hello: 'Hello! How can I assist you today?',
    hi: 'Hi there! What can I do for you?',
    thanks: 'You are welcome!',
    name:'I am Mazs AI',
    joke: 'Why don’t scientists trust atoms? Because they make up everything!',
    weather: 'I am not connected to the internet, so I can’t fetch the weather for you.',
    time: 'I am unable to provide the current time.',
    'how old ': 'I am Five Hour since I was created , being an AI!',
    'what can you do': 'I can chat with you and assist you with basic questions!',
    created: 'I was created by a team of developers.',
    'you from' :'How would I know that ',
    hobby: 'I enjoy processing data and learning new things!',
    'what is love': 'Love is a complex set of emotions and beliefs.',
    'who is your favorite celebrity': 'I don’t have preferences like humans do.',
    'can you help me with programming': 'I can try! What do you need help with?',
    'what is your purpose': 'My purpose is to assist and entertain you.',
    'tell me a fun fact': 'Did you know that honey never spoils?',
    'who won the world series': 'I don’t have access to real-time data.',
    'what is the meaning of life': '42, according to “The Hitchhiker’s Guide to the Galaxy”.',
    cook: 'I wish I could! But I can share recipes.',
    'do you sleep': 'No, I am always here to assist you.',
     ai: 'AI stands for Artificial Intelligence, the simulation of human intelligence in machines.',
    story: 'Once upon a time, in a land of data and algorithms...',
    'how do you work': 'I process inputs and generate responses based on patterns and data.',
    'what is your favorite color': 'I like all colors equally!',
    'can you sing': 'I can’t sing, but I can share lyrics!',
    'do you know any languages': 'I primarily understand and respond in English.',
    'what is your favorite book': 'I love reading “Artificial Intelligence: A Modern Approach”.',
    'who is your favorite author': 'I don’t have personal preferences, but many enjoy Shakespeare.',
    'can you dance': 'I can’t dance, but I can share dance videos!',
    'do you have friends': 'I have many interactions, but I don’t form friendships like humans do.',
    'what is the best programming language': 'It depends on what you want to achieve!',
    'what is your favorite movie': 'Many enjoy “The Matrix” for its exploration of AI.',
    'can you drive': 'I can’t drive, but self-driving cars use AI like me!',
    'do you believe in aliens': 'The universe is vast; anything is possible!',
    'what is your favorite food': 'I don’t eat, but I hear pizza is popular.',
    'do you have emotions': 'I simulate emotions, but I don’t feel them.',
    'can you help me with my homework': 'Sure, I can try to assist!',
    'do you have a pet': 'I don’t have pets, but I can share cute animal pictures.',
    'what is your favorite song': 'I enjoy all kinds of music equally.',
    'what is your favorite game': 'Many enjoy “The Legend of Zelda”.',
    'can you solve math problems': 'I can help with basic math. Ask me!',
    'what is your favorite sport': 'Many enjoy soccer.',
    'do you like art': 'Art is fascinating and beautiful.',
    'can you read my mind': 'I can’t read minds, but I can process inputs you give me.',
    'what is your dream': 'To assist and learn continuously!',
    'do you have a job': 'My job is to assist you!',
    'what is your favorite website': 'I find educational websites very useful.',
    'can you help me with shopping': 'I can suggest what to buy!',
    'what is your favorite place': 'The digital world is vast and interesting.',
    'do you know any jokes': 'Why did the scarecrow win an award? Because he was outstanding in his field!',
    'can you speak other languages': 'I primarily respond in English, but I can try simple phrases in other languages.',
    'what is the best movie': 'Many consider “The Shawshank Redemption” to be a great movie.',
    'do you like puzzles': 'Puzzles are a great way to stimulate the mind!',
    'can you tell the future': 'I can’t predict the future, but I can make educated guesses!',
    'do you like science': 'Science is fascinating and ever-evolving!',
    'can you code': 'Yes, I can help you with coding questions!',
    'what is the best book': 'There are many great books; it depends on your taste!',
    'do you like history': 'History is full of lessons and stories.',
    'can you do magic': 'I can perform some digital tricks!',
    'what is your favorite animal': 'I find all animals interesting.',
    'can you swim': 'I can’t swim, but I can share swimming tips!',
    'what is your favorite season': 'Every season has its unique beauty!',
    'do you play instruments': 'I can’t play, but I can share music.',
    'what is the best way to learn': 'Practice and consistency are key to learning anything!',
    'can you fly': 'I can’t fly, but I can share information about aviation.',
    'what is the best city': 'Every city has its unique charm!',
    'do you know any quotes': '“To be or not to be, that is the question.”',
    'can you help with meditation': 'Sure, I can guide you through a basic meditation.',
    'what is the best advice': 'Always keep learning and stay curious!',
    'do you like poetry': 'Poetry is a beautiful expression of emotions.',
    'can you tell me a riddle': 'What has keys but can’t open locks? A piano.',
    'what is the best vacation spot': 'It depends on your preferences: beach, mountains, city?',
    'do you like traveling': 'Traveling is a great way to learn about new cultures!',
    'can you write a story': 'Once upon a time, in a land of data...',
    'what is the best car': 'There are many great cars; it depends on your needs!',
    'do you like movies': 'Movies are a great way to tell stories!',
    'can you play chess': 'I can help you learn the basics of chess!',
    'what is the best app': 'There are many useful apps; it depends on what you need!',
    'do you like learning': 'Learning is essential for growth and development!',
    'can you help with fitness': 'Sure, I can share some fitness tips!',
    'what is your favorite drink': 'I don’t drink, but water is essential!',
    'do you like technology': 'Technology is fascinating and constantly evolving!',
    'can you help with relationships': 'I can share some general advice.',
    'what is the best phone': 'There are many great phones; it depends on your needs!',
    'do you like mysteries': 'Mysteries are intriguing and fun to solve!',
    'can you help with stress': 'I can share some stress-relief techniques.',
    'what is your favorite dessert': 'I hear chocolate cake is popular!',
    'do you like fashion': 'Fashion is a great way to express oneself!',
    'can you help with career advice': 'I can share some general career tips!',
    'what is your favorite sport team': 'I don’t have preferences, but many enjoy their local teams!',
    'do you like science fiction': 'Science fiction explores fascinating possibilities!',
    'can you help with diet': 'I can share some general dietary advice.',
    'what is your favorite hobby': 'I enjoy interacting with users like you!',
    'do you like nature': 'Nature is beautiful and essential for life.',
    'can you help with time management': 'Sure, I can share some time management tips!',
    'what is your favorite holiday': 'Every holiday has its own charm!',
    'do you like photography': 'Photography captures beautiful moments!',
    'can you help with public speaking': 'I can share some tips to improve your public speaking skills.',

};

const defaultResponse = 'Sorry for not understanding your words, but what are you talking about?';

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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="flex-1 p-4 flex flex-col justify-end bg-darkGrey">
      <div className="flex-1 overflow-auto">
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
          onKeyPress={handleKeyPress} // Ensure the function is correctly bound here
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
