import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import SettingsModal from './components/SettingsModal';
import { SpeedInsights } from "@vercel/speed-insights/react";
import {FaRegFileCode} from 'react-icons/fa';
const App: React.FC = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const toggleSettingsModal = () => {
    setIsSettingsOpen(!isSettingsOpen);
  };

  return (
    <div className="flex h-screen bg-background text-white overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="flex justify-between items-center p-4 border-b border-mediumGrey bg-darkGrey">
          <div className="text-2xl font-bold flex items-center space-x-2">
            <span className="">Mazs AI v0.1.a </span>
          </div>
          <button
            onClick={toggleSettingsModal}
            className="bg-primary p-2 rounded"
          >
            Settings
          </button>
        </header>
        <Chat />
        {isSettingsOpen && <SettingsModal onClose={toggleSettingsModal} />}
      </main>
      <SpeedInsights/>
    </div>
  );
};

export default App;
