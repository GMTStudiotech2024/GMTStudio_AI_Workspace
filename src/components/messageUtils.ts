export const preprocessMessage = (message: string): string => {
    return message.toLowerCase().replace(/[^\w\s]/gi, '');
  };
  
  export const stemWord = (word: string): string => {
    return word.replace(/(ing|ed|s)$/, '');
  };
  
  export const lemmatizeWord = (word: string): string => {
    const lemmas: { [key: string]: string } = {
      'am': 'be',
      'are': 'be',
      'is': 'be',
      'was': 'be',
      'were': 'be',
      'being': 'be',
      'has': 'have',
      'have': 'have',
      'had': 'have',
      'having': 'have'
    };
    return lemmas[word] || word;
  };
  
  export const handleNegation = (message: string): string => {
    const negatePattern = /\b(no|not|don't|never|none)\b/i;
    if (negatePattern.test(message)) {
      return message.split(' ').map(word => (negatePattern.test(word) ? 'not' : word)).join(' ');
    }
    return message;
  };
  
  export const handleSynonyms = (message: string): string => {
    const synonyms: { [key: string]: string[] } = {
      hello: ['hi', 'hey', 'greetings', 'hola'],
      help: ['assist', 'aid', 'support'],
      name: ['who are you', 'what are you'],
      quiz: ['quiz me on world capitals', 'capitals quiz'],
      python: ['python script for daily email reports'],
      comfort: ['message to comfort a friend'],
      relax: ['plan a relaxing day'],
      weather: ['weather today', 'today’s weather']
    };
  
    for (const [key, values] of Object.entries(synonyms)) {
      const regex = new RegExp(`\\b(${values.join('|')})\\b`, 'gi');
      message = message.replace(regex, key);
    }
    return message;
  };
  
  export const tokenizeAndProcess = (message: string): string[] => {
    const tokens = message.split(' ').map(word => lemmatizeWord(stemWord(word)));
    return tokens;
  };
  