import React, { useState, useEffect } from 'react';
import { FaPlus, FaSearch, FaTimes, FaChevronDown, FaLock, FaTrash } from 'react-icons/fa';
import './animations.css';

interface ChatItem {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: string;
  category: 'Personal' | 'Work';
  messages: string[];
}

interface SidebarProps {
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  onSelectChat: (chat: ChatItem) => void;
}

const MAX_CHATS = 10;

const Sidebar: React.FC<SidebarProps> = ({ isSidebarOpen, toggleSidebar, onSelectChat }) => {
  const [chats, setChats] = useState<ChatItem[]>(() => {
    const savedChats = localStorage.getItem('chats');
    return savedChats ? JSON.parse(savedChats) : [];
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [isPersonalOpen, setIsPersonalOpen] = useState(true);
  const [isWorkOpen, setIsWorkOpen] = useState(true);
  const [isAddingChat, setIsAddingChat] = useState(false);
  const [newChatTitle, setNewChatTitle] = useState('');
  const [newChatCategory, setNewChatCategory] = useState<'Personal' | 'Work'>('Personal');

  useEffect(() => {
    localStorage.setItem('chats', JSON.stringify(chats.slice(0, MAX_CHATS)));
  }, [chats]);

  const addNewChat = () => {
    if (newChatTitle.trim() === '') {
      alert('Please enter a chat title');
      return;
    }

    if (chats.length >= MAX_CHATS) {
      alert("You've reached the maximum number of chats. Upgrade to Pro for more space.");
      return;
    }

    const currentTimestamp = new Date().toLocaleString();
    const newChat: ChatItem = {
      id: Date.now().toString(),
      title: newChatTitle,
      lastMessage: 'No messages yet',
      timestamp: currentTimestamp,
      category: newChatCategory,
      messages: []
    };
    setChats(prevChats => [newChat, ...prevChats]);
    setIsAddingChat(false);
    setNewChatTitle('');
  };

  const handleAddNewChatClick = (category: 'Personal' | 'Work') => {
    setNewChatCategory(category);
    setIsAddingChat(true);
  };

  const handleDeleteChat = (chatId: string) => {
    setChats(prevChats => prevChats.filter(chat => chat.id !== chatId));
  };

  const filteredChats = chats.filter(chat =>
    chat.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const renderChatList = (category: 'Personal' | 'Work') => (
    <div className="space-y-2 mt-2">
      {filteredChats.filter(chat => chat.category === category).map(chat => (
        <div 
          key={chat.id} 
          className="p-2 bg-gray-800 rounded-lg cursor-pointer hover:bg-gray-700 flex justify-between items-center"
        >
          <div onClick={() => onSelectChat(chat)}>
            <h3 className="font-semibold">{chat.title}</h3>
            <p className="text-sm text-gray-400">{chat.lastMessage}</p>
            <p className="text-xs text-gray-500">{chat.timestamp}</p>
          </div>
          <button onClick={() => handleDeleteChat(chat.id)} className="p-1 text-red-500 hover:text-red-700">
            <FaTrash />
          </button>
        </div>
      ))}
      <button 
        onClick={() => handleAddNewChatClick(category)} 
        className="p-2 w-full text-left bg-gray-700 rounded-lg flex items-center space-x-2 hover:bg-gray-600"
      >
        <FaPlus className="text-sm" />
        <span className="text-sm">New Chat</span>
      </button>
    </div>
  );

  return (
    <div className={`fixed inset-y-0 left-0 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-200 ease-in-out md:relative md:translate-x-0 bg-gray-900 text-white w-64 h-screen sidebar`}>
      <div className="flex items-center justify-between p-4 bg-gray-800">
        <div className="flex items-center space-x-2">
          <img
            src="https://via.placeholder.com/40"
            alt="Profile"
            className="w-10 h-10 rounded-full"
          />
          <div>
            <p className="text-sm font-semibold">GMTStudio Test Account</p>
            <p className="text-xs text-gray-400">account@GMTStudio.com</p>
          </div>
        </div>
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
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <section>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Personal</h2>
            <button onClick={() => setIsPersonalOpen(!isPersonalOpen)} className="p-1">
              <FaChevronDown className={`transition-transform ${isPersonalOpen ? 'rotate-180' : 'rotate-0'}`} />
            </button>
          </div>
          {isPersonalOpen && renderChatList('Personal')}
        </section>
        
        <section>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Work</h2>
            <button onClick={() => setIsWorkOpen(!isWorkOpen)} className="p-1">
              <FaChevronDown className={`transition-transform ${isWorkOpen ? 'rotate-180' : 'rotate-0'}`} />
            </button>
          </div>
          {isWorkOpen && renderChatList('Work')}
        </section>
      </div>

      {chats.length >= MAX_CHATS && (
        <div className="p-4 bg-yellow-500 text-black text-sm flex items-center justify-center">
          <FaLock className="mr-2" />
          Upgrade to Pro for more space
        </div>
      )}

      {/* New Chat Modal */}
      {isAddingChat && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
          <div className="bg-gray-800 p-6 rounded-lg w-64 space-y-4">
            <h2 className="text-lg font-semibold">Create New {newChatCategory} Chat</h2>
            <input
              type="text"
              placeholder="Enter chat name"
              value={newChatTitle}
              onChange={(e) => setNewChatTitle(e.target.value)}
              className="w-full bg-gray-700 text-white p-2 rounded"
            />
            <div className="flex justify-end space-x-2">
              <button onClick={() => setIsAddingChat(false)} className="bg-gray-600 px-3 py-1 rounded">Cancel</button>
              <button onClick={addNewChat} className="bg-blue-500 px-3 py-1 rounded">Add</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
