import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import UpdateInfoModal from './components/SettingsModal';
import { SpeedInsights } from "@vercel/speed-insights/react";
import Login from './components/Login';
import LandingPage from './components/LandingPage';
import { ChatItem } from './types';
import { BrowserRouter as Router } from 'react-router-dom';

const App: React.FC = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showLandingPage, setShowLandingPage] = useState(true);
  const [selectedChat, setSelectedChat] = useState<ChatItem | null>(null);
  const [isDeveloper] = useState(false);

  const toggleSettingsModal = () => {
    setIsSettingsOpen(!isSettingsOpen);
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const handleLogin = (username: string, password: string) => {
    if (username && password) {
      localStorage.setItem('username', username);
      localStorage.setItem('password', password);
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
    <Router> {/* Wrap the entire app with Router */}
      <div className="flex h-screen bg-background text-white overflow-hidden">
        <Sidebar 
          isSidebarOpen={isSidebarOpen}
          toggleSidebar={toggleSidebar}
          onSelectChat={handleSelectChat}
          onNewChat={handleNewChat}
          isDeveloper={isDeveloper} // Add this line
        />
        <main className="flex-1 flex flex-col overflow-hidden">
          <header className="hidden md:flex justify-between items-center p-4 border-b border-mediumGrey bg-darkGrey">
            <div className="text-2xl font-bold flex items-center space-x-2">
              <span className="pl-10">Mazs AI v0.90.1</span>
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
              version="v0.90.1"
              description="AI improvements"
            />
          )}
        </main>
        <SpeedInsights />
      </div>
    </Router>
  );
};

export default App;