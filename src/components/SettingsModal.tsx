import React, { useState } from 'react';

interface SettingsModalProps {
  onClose: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ onClose }) => {
  const [themeColor, setThemeColor] = useState('dark');
  const [model, setModel] = useState('default');

  const handleSave = () => {
    // Save settings logic here
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-darkGrey p-6 rounded shadow-lg w-96">
        <h2 className="text-xl mb-4">Settings</h2>
        <div className="mb-4">
          <label className="block mb-2">Theme Color</label>
          <select value={themeColor} onChange={(e) => setThemeColor(e.target.value)} className="w-full bg-mediumGrey p-2 rounded">
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </select>
        </div>
        <div className="mb-4">
          <label className="block mb-2">Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)} className="w-full bg-mediumGrey p-2 rounded">
            <option value="default">Default</option>
            <option value="advanced">Advanced🔐</option>
          </select>
        </div>
        <div className="flex justify-end">
          <button onClick={onClose} className="bg-mediumGrey p-2 rounded mr-2">Cancel</button>
          <button onClick={handleSave} className="bg-accent p-2 rounded">Save</button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
