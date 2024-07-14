import { Suggestion, Context } from './chatTypes';
import {
  FaRobot, FaSmile,  FaSun, FaQuestionCircle, FaLaptopCode, FaCloud
} from 'react-icons/fa';

export const initialContext: Context = {
  quizState: {
    active: false,
    question: '',
    answer: ''
  },
  userName: ''
};

export const suggestions: Suggestion[] = [
  { text: 'Hello', icon: <FaSmile /> },
  { text: 'Tell me a joke', icon: <FaQuestionCircle /> },
  { text: 'Quiz me on world capitals', icon: <FaQuestionCircle /> },
  { text: 'What is your name?', icon: <FaRobot /> },
  { text: 'Write a Python script for daily email reports', icon: <FaLaptopCode /> },
  { text: 'Send a message to comfort a friend', icon: <FaSmile /> },
  { text: 'Plan a relaxing day', icon: <FaSun /> },
  { text: 'What is the weather today?', icon: <FaCloud /> },
];

export const patterns: { [key: string]: RegExp } = {
  greeting: /\b(hello|hi|hey|hola)\b/i,
  tell_joke: /\b(joke|funny)\b/i,
  quiz_capitals: /\b(quiz|capitals)\b/i,
  my_name_is: /\b(my name is|i am|i'm)\s(\w+)\b/i,
  python_script: /\b(python script)\b/i,
  comfort_friend: /\b(comfort|friend)\b/i,
  relaxing_day: /\b(relax|day)\b/i,
  weather: /\b(weather|today's weather)\b/i,
};

export const responses: { [key: string]: string } = {
  greeting: 'Hello! How can I assist you today?',
  tell_joke: 'Why don’t scientists trust atoms? Because they make up everything!',
  quiz_capitals: 'Sure! Let\'s start. What is the capital of France?',
  my_name_is: 'Nice to meet you!',
  python_script: 'You can write a Python script for daily email reports using libraries like smtplib and email. Do you need an example?',
  comfort_friend: 'I\'m sorry to hear that. Here is a message to comfort your friend: "I\'m here for you, and I care about you. Please don\'t hesitate to reach out if you need anything."',
  relaxing_day: 'How about starting with a nice walk in the park, followed by a spa session and ending the day with a good book or movie?',
  weather: 'Let me check the weather for you.',
};
