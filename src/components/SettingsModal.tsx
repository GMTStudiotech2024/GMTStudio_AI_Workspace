import React from 'react';
import logo from '../assets/GMTStudio-AI_studio.png';

interface UpdateInfoModalProps {
  onClose: () => void;
  title: string;
  version: string;
  description: string;
}

const UpdateInfoModal: React.FC<UpdateInfoModalProps> = ({ onClose, title, version, description }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-darkGrey p-6 rounded shadow-lg w-96">
        <div className="flex justify-center mb-4">
          <img src={logo} alt="GMT Studio" className="h-12" />
        </div>
        <h2 className="text-xl mb-2 text-center">{title}</h2>
        <h3 className="text-md mb-4 text-center bg-gradient-to-r from-amber-200 to-yellow-500 bg-clip-text text-transparent">Version {version}</h3>
        <p className="mb-4">{description}</p>
        <div className="flex justify-end">
          <button onClick={onClose} className="bg-gradient-to-r from-cyan-500 to-blue-500 p-2 rounded">Close</button>
        </div>
      </div>
    </div>
  );
};

export default UpdateInfoModal;
