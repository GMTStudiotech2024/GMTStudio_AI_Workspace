import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import UpdateInfoModal from './components/SettingsModal';  // Adjust import to match the correct path
import { SpeedInsights } from "@vercel/speed-insights/react";

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
            <span>Mazs AI v0.1.c</span>
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
            version="v0.1.c"
            description="We expanded the training data from 250 to 500 words, which make him a bit smarter"
          />
        )}
      </main>
      <SpeedInsights />
    </div>
  );
};

export default App;
