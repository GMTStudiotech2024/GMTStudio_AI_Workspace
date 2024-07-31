export interface Message {
    id: string;
    sender: 'user' | 'bot';
    text: string;
    timestamp: Date;
  }
  
  export interface Suggestion {
    text: string;
    icon: React.ReactNode;
  }