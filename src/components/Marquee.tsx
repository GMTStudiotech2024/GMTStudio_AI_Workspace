// Marquee.tsx
import React from 'react';

const Marquee: React.FC<{ text: string }> = ({ text }) => {
  return (
    <div className="marquee-container">
      <div className="marquee">
        <span>{text}</span>
      </div>
    </div>
  );
};

export default Marquee;
