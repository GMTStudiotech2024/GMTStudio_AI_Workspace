export interface Message {
    id: string;
    sender: 'user' | 'bot';
    text: string;
    timestamp: Date;
  }
  
  export interface Context {
    quizState: {
      active: boolean;
      question: string;
      answer: string;
    };
    userName: string;
  }
  
  export interface Suggestion {
    text: string;
    icon: JSX.Element;
  }
  