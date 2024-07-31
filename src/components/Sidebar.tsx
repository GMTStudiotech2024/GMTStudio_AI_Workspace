import React, { useState, useEffect, useRef } from 'react';
import { FaPlus, FaSearch, FaTimes, FaTrash, FaUser, FaPencilAlt, FaCog } from 'react-icons/fa';
import SettingModal from './SettingsModal'; 

export interface ChatItem {
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

const Sidebar: React.FC<SidebarProps> = ({ isSidebarOpen, toggleSidebar, onSelectChat }) => {
  const [chats, setChats] = useState<ChatItem[]>(() => {
    const savedChats = localStorage.getItem('chats');
    return savedChats ? JSON.parse(savedChats) : [];
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [isAddingChat, setIsAddingChat] = useState(false);
  const [newChatTitle, setNewChatTitle] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<'Personal' | 'Work'>('Personal');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    localStorage.setItem('chats', JSON.stringify(chats));
  }, [chats]);

  const addNewChat = () => {
    if (newChatTitle.trim() === '') {
      alert('Please enter a chat title');
      return;
    }

    const currentTimestamp = new Date().toLocaleString();
    const newChat: ChatItem = {
      id: Date.now().toString(),
      title: newChatTitle,
      lastMessage: 'New conversation started',
      timestamp: currentTimestamp,
      category: selectedCategory,
      messages: []
    };
    setChats(prevChats => [newChat, ...prevChats]);
    setIsAddingChat(false);
    setNewChatTitle('');
  };

  const handleDeleteChat = (chatId: string) => {
    setChats(prevChats => prevChats.filter(chat => chat.id !== chatId));
  };

  const handleEditChat = (chatId: string) => {
    setEditingChatId(chatId);
    const chatToEdit = chats.find(chat => chat.id === chatId);
    if (chatToEdit) {
      setNewChatTitle(chatToEdit.title);
      setSelectedCategory(chatToEdit.category);
    }
  };

  const saveEditedChat = () => {
    if (editingChatId) {
      setChats(prevChats => prevChats.map(chat =>
        chat.id === editingChatId
          ? { ...chat, title: newChatTitle, category: selectedCategory }
          : chat
      ));
      setEditingChatId(null);
      setNewChatTitle('');
    }
  };

  const filteredChats = chats.filter(chat =>
    chat.title.toLowerCase().includes(searchTerm.toLowerCase()) &&
    chat.category === selectedCategory
  );

  const toggleSettingsModal = () => {
    setIsSettingsOpen(!isSettingsOpen);
  };

  return (
    <div className={`fixed inset-y-0 left-0 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out md:relative md:translate-x-0 bg-gray-900 text-white w-64 h-screen flex flex-col shadow-lg`}>
      <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
        <h1 className="text-xl font-bold">Mazs AI v0.61.2</h1>
        <button onClick={toggleSidebar} className="p-2 md:hidden focus:outline-none">
          <FaTimes />
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-4">
          <button
            onClick={() => setIsAddingChat(true)}
            className="w-full p-2 bg-blue-500 text-white rounded-lg flex items-center justify-center space-x-2 hover:bg-blue-600 focus:outline-none focus:ring focus:ring-blue-300"
          >
            <FaPlus />
            <span>New Chat</span>
          </button>

          <div className="relative">
            <input
              type="text"
              placeholder="Search chats..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-gray-800 text-white p-2 pl-10 rounded focus:outline-none focus:ring focus:ring-blue-300"
              ref={searchInputRef}
              aria-label="Search chats"
            />
            <FaSearch className="absolute left-2 top-3 text-gray-500" />
          </div>

          <div className="flex space-x-2">
            <button
              onClick={() => setSelectedCategory('Personal')}
              className={`flex-1 p-2 rounded ${selectedCategory === 'Personal' ? 'bg-blue-500' : 'bg-gray-800'} hover:bg-blue-600 focus:outline-none focus:ring focus:ring-blue-300`}
              aria-pressed={selectedCategory === 'Personal'}
            >
              Personal
            </button>
            <button
              onClick={() => setSelectedCategory('Work')}
              className={`flex-1 p-2 rounded ${selectedCategory === 'Work' ? 'bg-blue-500' : 'bg-gray-800'} hover:bg-blue-600 focus:outline-none focus:ring focus:ring-blue-300`}
              aria-pressed={selectedCategory === 'Work'}
            >
              Work
            </button>
          </div>

          <div className="space-y-2">
            {filteredChats.map(chat => (
              <div
                key={chat.id}
                className="p-2 bg-gray-800 rounded-lg cursor-pointer hover:bg-gray-700 flex justify-between items-center transition-colors duration-200 ease-in-out"
                onClick={() => onSelectChat(chat)}
              >
                <div>
                  <h3 className="font-semibold">{chat.title}</h3>
                  <p className="text-xs text-gray-400">{chat.lastMessage}</p>
                </div>
                <div className="flex space-x-2">
                  <button onClick={(e) => { e.stopPropagation(); handleEditChat(chat.id); }} className="text-gray-400 hover:text-white focus:outline-none">
                    <FaPencilAlt size={14} />
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat.id); }} className="text-gray-400 hover:text-red-500 focus:outline-none">
                    <FaTrash size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="p-4 bg-gray-800 border-t border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FaUser className="w-8 h-8 rounded-full bg-gray-600 p-1" />
            <div>
              <p className="text-sm font-semibold">User</p>
              <p className="text-xs text-gray-400">Free Plan</p>
            </div>
          </div>
          <button
            onClick={toggleSettingsModal}
            className="text-gray-400 hover:text-white focus:outline-none"
            aria-label="Open settings"
          >
            <FaCog />
          </button>
        </div>
      </div>

      {isAddingChat && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg w-64 space-y-4 shadow-lg">
            <h2 className="text-lg font-semibold">{editingChatId ? 'Edit Chat' : 'Create New Chat'}</h2>
            <input
              type="text"
              placeholder="Enter chat name"
              value={newChatTitle}
              onChange={(e) => setNewChatTitle(e.target.value)}
              className="w-full bg-gray-700 text-white p-2 rounded focus:outline-none focus:ring focus:ring-blue-300"
              aria-label="Chat name"
            />
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedCategory('Personal')}
                className={`flex-1 p-2 rounded ${selectedCategory === 'Personal' ? 'bg-blue-500' : 'bg-gray-700'} focus:outline-none focus:ring focus:ring-blue-300`}
                aria-pressed={selectedCategory === 'Personal'}
              >
                Personal
              </button>
              <button
                onClick={() => setSelectedCategory('Work')}
                className={`flex-1 p-2 rounded ${selectedCategory === 'Work' ? 'bg-blue-500' : 'bg-gray-700'} focus:outline-none focus:ring focus:ring-blue-300`}
                aria-pressed={selectedCategory === 'Work'}
              >
                Work
              </button>
            </div>
            <div className="flex justify-end space-x-2">
              <button onClick={() => { setIsAddingChat(false); setEditingChatId(null); }} className="bg-gray-600 px-3 py-1 rounded focus:outline-none focus:ring focus:ring-blue-300">Cancel</button>
              <button onClick={editingChatId ? saveEditedChat : addNewChat} className="bg-blue-500 px-3 py-1 rounded focus:outline-none focus:ring focus:ring-blue-300">
                {editingChatId ? 'Save' : 'Add'}
              </button>
            </div>
          </div>
        </div>
      )}

      {isSettingsOpen && (
        <SettingModal
          onClose={toggleSettingsModal}
          title="Latest Updates"
          version="v0.62.0"
          description="Update The Icon"
        />
      )}
    </div>
  );
};

export default Sidebar;