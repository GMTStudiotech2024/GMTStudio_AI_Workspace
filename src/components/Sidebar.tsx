import React, { useState } from 'react';
import { FaPlus, FaChevronDown, FaSearch, FaTimes } from 'react-icons/fa';
import './animations.css';

interface ChatItem {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: string;  // Add timestamp to the ChatItem interface
}

interface SidebarProps {
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isSidebarOpen, toggleSidebar }) => {
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  const addNewChat = () => {
    const currentTimestamp = new Date().toLocaleString();  // Get the current timestamp
    const newChat = {
      id: Date.now().toString(),
      title: 'New chat',
      lastMessage: 'No messages yet',
      timestamp: currentTimestamp  // Set the timestamp for the new chat
    };
    setChats([...chats, newChat]);
  };

  const filteredChats = chats.filter(chat => 
    chat.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      <div className={`fixed inset-y-0 left-0 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-200 ease-in-out md:relative md:translate-x-0 bg-gray-900 text-white w-64 h-screen sidebar`}>
        <div className="flex items-center justify-between p-4">
          <h1 className="text-xl font-bold">GMTStudio AI Studio</h1>
          <button onClick={addNewChat} className="p-2">
            <FaPlus />
          </button>
          <button onClick={toggleSidebar} className="p-2 md:hidden">
            <FaTimes />
          </button>
        </div>
        <div className="relative p-4">
          <input
            type="text"
            placeholder="Search chats..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-800 text-white p-2 rounded"
          />
          <FaSearch className="absolute right-6 top-7 text-gray-500" />
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {filteredChats.map(chat => (
            <div key={chat.id} className="flex items-center justify-between p-2 bg-gray-800 rounded-lg">
              <div>
                <h2 className="font-semibold">{chat.title}</h2>
                <p className="text-sm text-gray-400">{chat.lastMessage}</p>
                <p className="text-xs text-gray-500">{chat.timestamp}</p> {/* Display the timestamp */}
              </div>
              <button className="p-2">
                <FaChevronDown />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
