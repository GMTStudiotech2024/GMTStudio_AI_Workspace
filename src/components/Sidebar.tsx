import React, { useState, useEffect } from 'react';
import { FaPlus, FaSearch, FaTrash, FaUser, FaPencilAlt, FaCog, FaChevronLeft, FaChevronRight, FaHistory } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatItem } from '../types';
import { useNavigate } from 'react-router-dom';

interface SidebarProps {
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  onSelectChat: (chat: ChatItem) => void;
  onNewChat: (chat: ChatItem) => void;
  isDeveloper: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ isSidebarOpen, toggleSidebar, onSelectChat, onNewChat, isDeveloper }) => {
  const [chats, setChats] = useState<ChatItem[]>(() => {
    const savedChats = localStorage.getItem('chats');
    return savedChats ? JSON.parse(savedChats) : [];
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<'All' | 'Personal' | 'Work'>('All');
  const [newChatTitle, setNewChatTitle] = useState('');
  const [newChatCategory, setNewChatCategory] = useState<'Personal' | 'Work'>('Personal');
  const [isAddingNewChat, setIsAddingNewChat] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    localStorage.setItem('chats', JSON.stringify(chats));
  }, [chats]);

  const handleDeleteChat = (chatId: string) => {
    setChats(prevChats => prevChats.filter(chat => chat.id !== chatId));
  };

  const handleAddNewChat = () => {
    setIsAddingNewChat(true);
  };

  const handleCreateNewChat = () => {
    if (newChatTitle.trim()) {
      const newChat: ChatItem = {
        id: Date.now().toString(),
        title: newChatTitle.trim(),
        lastMessage: '',
        category: newChatCategory,
        timestamp: new Date(),
      };
      setChats(prevChats => [...prevChats, newChat]);
      onNewChat(newChat);
      onSelectChat(newChat);
      setNewChatTitle('');
      setNewChatCategory('Personal');
      setIsAddingNewChat(false);
    }
  };

  const handleLogout = () => {
    // Clear any user-related data from localStorage
    localStorage.removeItem('username');
    localStorage.removeItem('password');
    // Navigate to the login page
    navigate('/LandingPage');
  };

  const filteredChats = chats.filter(chat =>
    chat.title.toLowerCase().includes(searchTerm.toLowerCase()) &&
    (selectedCategory === 'All' || chat.category === selectedCategory)
  );

  return (
    <AnimatePresence>
      {isSidebarOpen && (
        <motion.div
          initial={{ x: '-100%' }}
          animate={{ x: 0 }}
          exit={{ x: '-100%' }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="fixed inset-y-0 left-0 z-50 w-72 bg-gray-900 text-white shadow-lg overflow-hidden flex flex-col"
        >
          <div className="flex items-center justify-between p-4 bg-gray-800">
            <h1 className="text-xl font-bold">Mazs AI</h1>
            <button onClick={toggleSidebar} className="p-2 rounded-full hover:bg-gray-700 transition-colors">
              <FaChevronLeft />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            <div className="p-4 space-y-4">
              {isAddingNewChat ? (
                <div className="space-y-2">
                  <input
                    type="text"
                    value={newChatTitle}
                    onChange={(e) => setNewChatTitle(e.target.value)}
                    placeholder="Enter chat name..."
                    className="w-full p-2 bg-gray-800 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setNewChatCategory('Personal')}
                      className={`flex-1 p-2 rounded-lg ${newChatCategory === 'Personal' ? 'bg-blue-600' : 'bg-gray-800'} hover:bg-blue-700 transition-colors`}
                    >
                      Personal
                    </button>
                    <button
                      onClick={() => setNewChatCategory('Work')}
                      className={`flex-1 p-2 rounded-lg ${newChatCategory === 'Work' ? 'bg-blue-600' : 'bg-gray-800'} hover:bg-blue-700 transition-colors`}
                    >
                      Work
                    </button>
                  </div>
                  <button
                    onClick={handleCreateNewChat}
                    className="w-full p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Create
                  </button>
                </div>
              ) : (
                <button
                  onClick={handleAddNewChat}
                  className="w-full flex items-center justify-center space-x-2 bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <FaPlus />
                  <span>New Chat</span>
                </button>
              )}

              <div className="relative">
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full bg-gray-800 text-white p-2 pl-10 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <FaSearch className="absolute left-3 top-3 text-gray-400" />
              </div>

              <div className="flex space-x-2">
                {['All', 'Personal', 'Work'].map((category) => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category as 'All' | 'Personal' | 'Work')}
                    className={`flex-1 p-2 rounded-lg ${selectedCategory === category ? 'bg-blue-600' : 'bg-gray-800'} hover:bg-blue-700 transition-colors`}
                  >
                    {category}
                  </button>
                ))}
              </div>

              <div className="space-y-2">
                {filteredChats.map(chat => (
                  <motion.div
                    key={chat.id}
                    whileHover={{ scale: 1.02 }}
                    className="p-3 bg-gray-800 rounded-lg cursor-pointer hover:bg-gray-700 transition-colors"
                    onClick={() => onSelectChat(chat)}
                  >
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-3">
                        <FaHistory className="text-gray-400" />
                        <div>
                          <h3 className="font-semibold truncate">{chat.title}</h3>
                          <p className="text-xs text-gray-400 truncate">{chat.lastMessage}</p>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        <button onClick={(e) => { e.stopPropagation(); /* handleEditChat(chat.id) */ }} className="text-gray-400 hover:text-white">
                          <FaPencilAlt size={14} />
                        </button>
                        <button onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat.id); }} className="text-gray-400 hover:text-red-500">
                          <FaTrash size={14} />
                        </button>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>

          <div className="p-4 bg-gray-800 border-t border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center">
                  <FaUser className="text-white" />
                </div>
                <div>
                  <p className="text-sm font-semibold">User</p>
                  <p className="text-xs text-gray-400">{isDeveloper ? 'Developer' : 'Free Plan'}</p>
                </div>
              </div>
              <button 
                className="text-gray-400 hover:text-white p-2 rounded-full hover:bg-gray-700 transition-colors"
                onClick={handleLogout}
              >
                <FaCog />
              </button>
            </div>
          </div>
        </motion.div>
      )}
      {!isSidebarOpen && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed top-4 left-4 z-50 p-2 bg-gray-800 text-white rounded-full shadow-lg hover:bg-gray-700 transition-colors"
          onClick={toggleSidebar}
        >
          <FaChevronRight />
        </motion.button>
      )}
    </AnimatePresence>
  );
};

export default Sidebar;