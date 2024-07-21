import React from 'react';
import { FaTimes } from 'react-icons/fa';

interface UpdateInfoModalProps {
  onClose: () => void;
  title: string;
  version: string;
  description: string;
}

const UpdateInfoModal: React.FC<UpdateInfoModalProps> = ({ onClose, title, version, description }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-gray-800 p-6 rounded-lg w-96 max-w-full mx-4 space-y-4 text-white">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">{title}</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <FaTimes />
          </button>
        </div>
        <div className="space-y-2">
          <p className="font-semibold">Current Version: {version}</p>
          <p>{description}</p>
        </div>
        <div className="space-y-2">
          <h3 className="font-semibold">What's New:</h3>
          <ul className="list-disc list-inside space-y-1">
            <li>Updated user interface for better usability</li>
            <li>Improved performance for faster response times</li>
            <li>Added new features to enhance productivity</li>
            <li>Fixed various bugs and issues</li>
          </ul>
        </div>
        <div className="pt-4">
          <button
            onClick={onClose}
            className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default UpdateInfoModal;