import React from 'react';

interface EmojiPickerProps {
  onEmojiClick: (emoji: string) => void;
}

const emojis = ['😀', '😂', '🥲', '😊', '😍', '😘', '😜', '🤔', '🤩', '😎'];

const EmojiPicker: React.FC<EmojiPickerProps> = ({ onEmojiClick }) => {
  return (
    <div className="absolute bottom-16 right-4 bg-white rounded shadow-lg p-2">
      <div className="flex flex-wrap">
        {emojis.map((emoji, index) => (
          <button
            key={index}
            onClick={() => onEmojiClick(emoji)}
            className="p-2 hover:bg-gray-200 rounded"
          >
            {emoji}
          </button>
        ))}
      </div>
    </div>
  );
};

export default EmojiPicker;
