export interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
  inputVector?: number[]; // Add this line
}

export interface Suggestion {
  text: string;
  icon: React.ReactNode;
}
  export interface ChatItem {
    id: string;
    title: string;
    lastMessage: string;
    category: 'Personal' | 'Work';
    timestamp: Date;
  }
  export type Tensor1D = number[];
export type Tensor2D = number[][];