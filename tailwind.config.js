/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkGrey: '#1f1f1f',
        mediumGrey: '#2a2a2a',
        lightGrey: '#3a3a3a',
        primary: '#4a4a4a',
        accent: '#5a5a5a',
        userBubble: '#4A90E2',
        botBubble: '#333333',
        background: '#121212',
        sentbutton:'#1e88e5',
        sentbuttonhover:'#1565c0',
      },
    },
  },
  plugins: [],
}
