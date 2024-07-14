import React from 'react';
import { FaRobot, FaUser } from 'react-icons/fa';
import { Message } from './chatTypes';

interface MessageListProps {
  messages: Message[];
  isTyping: boolean;
  messagesEndRef: React.RefObject<HTMLDivElement>;
}

const MessageList: React.FC<MessageListProps> = ({ messages, isTyping, messagesEndRef }) => {
  return (
    <div className="flex-1 p-4 overflow-y-auto">
      {messages.map(message => (
        <div
          key={message.id}
          className={`flex items-center mb-4 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          {message.sender === 'bot' && (
            <FaRobot className="text-gray-600 mr-2" />
          )}
          <div
            className={`rounded-lg p-4 ${message.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-white'}`}
          >
            {message.text}
          </div>
          {message.sender === 'user' && (
            <FaUser className="text-gray-600 ml-2" />
          )}
        </div>
      ))}
      {isTyping && (
        <div className="flex items-center mb-4 justify-start">
          <FaRobot className="text-gray-600 mr-2" />
          <div className="rounded-lg p-4 bg-gray-800 text-white">Typing...</div>
        </div>
      )}
      <div ref={messagesEndRef}></div>
    </div>
  );
};

export default MessageList;
