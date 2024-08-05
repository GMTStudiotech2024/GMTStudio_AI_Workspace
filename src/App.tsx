import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import UpdateInfoModal from './components/SettingsModal';
import { SpeedInsights } from "@vercel/speed-insights/react";
import { FaBars } from 'react-icons/fa';
import Login from './components/Login';
import LandingPage from './components/LandingPage';
import { ChatItem } from './types';

const App: React.FC = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showLandingPage, setShowLandingPage] = useState(true);
  const [selectedChat, setSelectedChat] = useState<ChatItem | null>(null);

  const toggleSettingsModal = () => {
    setIsSettingsOpen(!isSettingsOpen);
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const handleLogin = (username: string, password: string) => {
    if (username && password) {
      setIsLoggedIn(true);
    } else {
      alert('Invalid credentials');
    }
  };

  const handleGetStarted = () => {
    setShowLandingPage(false);
  };

  const handleSelectChat = (chat: ChatItem) => {
    setSelectedChat(chat);
  };

  const handleNewChat = () => {
    setSelectedChat(null);
  };

  if (showLandingPage) {
    return <LandingPage onGetStarted={handleGetStarted} />;
  }

  if (!isLoggedIn) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="flex h-screen bg-background text-white overflow-hidden">
      <Sidebar 
        isSidebarOpen={isSidebarOpen}
        toggleSidebar={toggleSidebar}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
      />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="flex justify-between items-center p-4 border-b border-mediumGrey bg-darkGrey">
          <div className="text-2xl font-bold flex items-center space-x-2">
            <button onClick={toggleSidebar} className="md:hidden p-2">
              <FaBars />
            </button>
            <span className="pl-10">Mazs AI v0.70.1</span>
          </div>
          <button
            onClick={toggleSettingsModal}
            className="bg-primary p-2 rounded"
          >
            Settings
          </button>
        </header>
        <Chat selectedChat={selectedChat} />
        {isSettingsOpen && (
          <UpdateInfoModal
            onClose={toggleSettingsModal}
            title="Latest Updates"
            version="v0.70.1"
            description="AI improvements"
          />
        )}
      </main>
      <SpeedInsights />
    </div>
  );
};

export default App;