// App.tsx
import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import UpdateInfoModal from './components/SettingsModal';  // Adjust import to match the correct path
import { SpeedInsights } from "@vercel/speed-insights/react";
import { FaBars } from 'react-icons/fa';

const App: React.FC = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);


  const toggleSettingsModal = () => {
    setIsSettingsOpen(!isSettingsOpen);
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };






  return (
    <div className="flex h-screen bg-background text-white overflow-hidden">
      <Sidebar 
        isSidebarOpen={isSidebarOpen}
  toggleSidebar={toggleSidebar} onSelectChat={() => {}}      />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="flex justify-between items-center p-4 border-b border-mediumGrey bg-darkGrey">
          <div className="text-2xl font-bold flex items-center space-x-2">
            <button onClick={toggleSidebar} className="md:hidden p-2">
              <FaBars />
            </button>
            <span>Mazs AI v0.61.1</span>
          </div>
          <button
            onClick={toggleSettingsModal}
            className="bg-primary p-2 rounded"
          >
            Settings
          </button>
        </header>
        <Chat />
        {isSettingsOpen && (
          <UpdateInfoModal
            onClose={toggleSettingsModal}
            title="Latest Updates"
            version="v0.1.5"
            description="We expanded the training data from 250 to 500 words, which make him a bit smarter"
          />
        )}
      </main>
      <SpeedInsights />
    </div>
  );
};

export default App;