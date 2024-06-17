import React, { useState, useEffect } from 'react';
import { FaBars, FaTrash } from 'react-icons/fa';
import logo from '../assets/GMTStudio-AI_studio.png';

interface Chat {
  id: number;
  title: string;
  lastMessage: string;
}

const Sidebar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState<Chat[]>([]);

  // Load chat history from localStorage
  useEffect(() => {
    const storedChats = localStorage.getItem('chatHistory');
    if (storedChats) {
      setChatHistory(JSON.parse(storedChats));
    }
  }, []);

  // Save chat history to localStorage
  const saveChatHistory = (chats: Chat[]) => {
    localStorage.setItem('chatHistory', JSON.stringify(chats));
  };

  const addNewChat = () => {
    const newChatTitle = prompt('Enter the chat title:');
    if (newChatTitle) {
      const newChat: Chat = {
        id: chatHistory.length > 0 ? chatHistory[chatHistory.length - 1].id + 1 : 1,
        title: newChatTitle,
        lastMessage: '',
      };
      const updatedChatHistory = [...chatHistory, newChat];
      setChatHistory(updatedChatHistory);
      saveChatHistory(updatedChatHistory);
    }
  };

  const deleteChat = (id: number) => {
    const updatedChatHistory = chatHistory.filter(chat => chat.id !== id);
    setChatHistory(updatedChatHistory);
    saveChatHistory(updatedChatHistory);
  };

  return (
    <>
      <div className="w-64 bg-background p-4 flex flex-col lg:flex border-b border-mediumGrey">
        <div className="flex items-center space-x-2 mb-10 border-b-mediumGrey pb-4">
          <img src={logo} alt="GMT Studio AI Dev" className="max-w-12" />
          <div className="text-2xl font-bold">GMTStudio AI Workspace</div>
        </div>

        <button onClick={addNewChat} className="bg-white p-2 mb-5 rounded-xl text-black">+ New Chat</button>

        <div className="w-45 mb-4">
          <div className="relative w-full min-w-[200px] h-10">
            <input
              className="peer w-full h-full bg-transparent text-white font-sans font-normal outline outline-0 focus:outline-0 disabled:bg-white disabled:border-0 transition-all placeholder-shown:border placeholder-shown:border-white placeholder-shown:border-t-white border focus:border-2 border-t-transparent focus:border-t-transparent text-sm px-3 py-2.5 rounded-[7px] border-white focus:border-white"
              placeholder=" "
            />
            <label className="flex w-full h-full select-none pointer-events-none absolute left-0 font-normal !overflow-visible truncate peer-placeholder-shown:text-white leading-tight peer-focus:leading-tight peer-disabled:text-transparent peer-disabled:peer-placeholder-shown:text-white transition-all -top-1.5 peer-placeholder-shown:text-sm text-[11px] peer-focus:text-[11px] before:content[' '] before:block before:box-border before:w-2.5 before:h-1.5 before:mt-[6.5px] before:mr-1 peer-placeholder-shown:before:border-transparent before:rounded-tl-md before:border-t peer-focus:before:border-t-2 before:border-l peer-focus:before:border-l-2 before:pointer-events-none before:transition-all peer-disabled:before:border-transparent after:content[' '] after:block after:flex-grow after:box-border after:w-2.5 after:h-1.5 after:mt-[6.5px] after:ml-1 peer-placeholder-shown:after:border-transparent after:rounded-tr-md after:border-t peer-focus:after:border-t-2 after:border-r peer-focus:after:border-r-2 after:pointer-events-none after:transition-all peer-disabled:after:border-transparent peer-placeholder-shown:leading-[3.75] text-white peer-focus:text-white before:border-white peer-focus:before:!border-white after:border-white peer-focus:after:!border-white">
              Search Chat History
            </label>
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          {chatHistory.length > 0 ? (
            <ul>
              {chatHistory.map((chat) => (
                <li key={chat.id} className="mb-4 p-2 border-b border-lightGrey cursor-pointer hover:bg-hoverGrey rounded-md flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="bg-primary w-10 h-10 flex items-center justify-center rounded-full text-white font-bold mr-3">
                      {chat.title.charAt(0)}
                    </div>
                    <div>
                      <div className="font-semibold text-white">{chat.title}</div>
                      <div className="text-sm text-lightGrey">{chat.lastMessage}</div>
                    </div>
                  </div>
                  <button onClick={() => deleteChat(chat.id)} className="text-red-500 hover:text-red-700">
                    <FaTrash />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-lightGrey">No chats.</div>
          )}
        </div>
      </div>

      <button className="lg:hidden p-4" onClick={() => setIsOpen(!isOpen)}>
        <FaBars />
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex z-50">
          <div className="w-64 bg-darkGrey p-4 flex flex-col">
            <div className="flex items-center space-x-2 mb-4">
              <img src={logo} alt="GMT Studio AI Dev" className="w-10 h-10" />
              <span className="text-2xl font-bold">GMT Studio AI Dev</span>
            </div>
            <button onClick={addNewChat} className="bg-primary p-2 mb-4 rounded-xl">+ New Chat</button>
            <div className="flex-1 overflow-auto">
              {chatHistory.length > 0 ? (
                <ul>
                  {chatHistory.map((chat) => (
                    <li key={chat.id} className="mb-4 p-2 border-b border-lightGrey cursor-pointer hover:bg-hoverGrey rounded-md flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="bg-primary w-10 h-10 flex items-center justify-center rounded-full text-white font-bold mr-3">
                          {chat.title.charAt(0)}
                        </div>
                        <div>
                          <div className="font-semibold text-white">{chat.title}</div>
                          <div className="text-sm text-lightGrey">{chat.lastMessage}</div>
                        </div>
                      </div>
                      <button onClick={() => deleteChat(chat.id)} className="text-red-500 hover:text-red-700">
                        <FaTrash />
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-lightGrey">No chats.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Sidebar;
