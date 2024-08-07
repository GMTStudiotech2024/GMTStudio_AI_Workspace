import React, { useState, useEffect } from 'react';
import { FaBrain, FaLock, FaChartLine, FaArrowRight, FaGithub, FaTwitter, FaLinkedin, FaPlay, FaRobot, FaCode, FaDatabase, FaServer, FaUserCog, FaComments, FaBars } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';

interface LandingPageProps {
  onGetStarted: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <div className="bg-gradient-to-b from-gray-100 to-white text-gray-900 min-h-screen">
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${isScrolled ? 'bg-white shadow-md' : 'bg-transparent'}`}>
        <div className="container mx-auto px-4 py-4">
          <nav className="flex justify-between items-center">
            <motion.h1 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-3xl font-bold text-blue-600"
            >
              Mazs AI
            </motion.h1>
            <div className="hidden md:flex items-center space-x-6">
              <a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Features</a>
              <a href="#architecture" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Architecture</a>
              <a href="#future" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Future</a>
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onGetStarted} 
                className="bg-blue-600 text-white px-4 py-2 rounded-md transition-colors duration-200 flex items-center"
              >
                Get Started <FaArrowRight className="ml-2" />
              </motion.button>
            </div>
            <button onClick={toggleMobileMenu} className="md:hidden text-gray-600">
              <FaBars size={24} />
            </button>
          </nav>
        </div>
      </header>

      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed inset-0 bg-white z-40 md:hidden"
          >
            <div className="flex flex-col items-center justify-center h-full space-y-6">
              <a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors duration-200" onClick={toggleMobileMenu}>Features</a>
              <a href="#architecture" className="text-gray-600 hover:text-blue-600 transition-colors duration-200" onClick={toggleMobileMenu}>Architecture</a>
              <a href="#future" className="text-gray-600 hover:text-blue-600 transition-colors duration-200" onClick={toggleMobileMenu}>Future</a>
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => { onGetStarted(); toggleMobileMenu(); }}
                className="bg-blue-600 text-white px-4 py-2 rounded-md transition-colors duration-200 flex items-center"
              >
                Get Started <FaArrowRight className="ml-2" />
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="container mx-auto px-4 pt-32 pb-16">
        <section className="text-center mb-24">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 text-gray-900"
          >
            Mazs AI: Neural Network-Powered Chatbot
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="text-lg md:text-xl lg:text-2xl mb-10 max-w-3xl mx-auto text-gray-600"
          >
            A technical deep dive into our advanced conversational AI system.
          </motion.p>
          <div className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onGetStarted} 
              className="bg-blue-600 text-white px-8 py-4 rounded-md text-lg font-semibold shadow-lg transition-all duration-200 flex items-center justify-center"
            >
              Start Exploring <FaArrowRight className="ml-2" />
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gray-200 text-gray-800 px-8 py-4 rounded-md text-lg font-semibold shadow-lg transition-all duration-200 flex items-center justify-center"
            >
              Watch Demo <FaPlay className="ml-2" />
            </motion.button>
          </div>
        </section>

        <section id="features" className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8 mb-24">
          <FeatureCard
            icon={<FaRobot className="text-5xl mb-6 text-blue-600" />}
            title="Intent Classification"
            description="Utilizes a three-layer neural network for accurate intent classification"
          />
          <FeatureCard
            icon={<FaBrain className="text-5xl mb-6 text-blue-600" />}
            title="ReLU Activation"
            description="Employs ReLU activation in the hidden layer for improved learning"
          />
          <FeatureCard
            icon={<FaLock className="text-5xl mb-6 text-blue-600" />}
            title="Dropout Regularization"
            description="Implements dropout for better generalization and reduced overfitting"
          />
          <FeatureCard
            icon={<FaChartLine className="text-5xl mb-6 text-blue-600" />}
            title="AdamW Optimizer"
            description="Uses AdamW optimizer for efficient weight updates and regularization"
          />
          <FeatureCard
            icon={<FaCode className="text-5xl mb-6 text-blue-600" />}
            title="Softmax Output"
            description="Softmax activation in the output layer for multi-class classification"
          />
          <FeatureCard
            icon={<FaDatabase className="text-5xl mb-6 text-blue-600" />}
            title="Expandable Dataset"
            description="Designed to accommodate growing training data for improved performance"
          />
        </section>

        <section id="architecture" className="text-center mb-24">
          <h3 className="text-3xl md:text-4xl font-bold mb-6">Neural Network Architecture</h3>
          <p className="text-lg md:text-xl mb-8">Mazs AI employs a sophisticated three-layer feedforward neural network:</p>
          <div className="grid md:grid-cols-3 gap-8">
            <ArchitectureCard title="Input Layer" description="Variable size, encoding user input as binary vectors" />
            <ArchitectureCard title="Hidden Layer" description="Variable size with ReLU activation for complex pattern learning" />
            <ArchitectureCard title="Output Layer" description="10 neurons with Softmax activation for intent classification" />
          </div>
        </section>

        <section id="future" className="text-center mb-24">
          <h3 className="text-3xl md:text-4xl font-bold mb-6">Future Enhancements</h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-8">
            <EnhancementCard
              icon={<FaComments className="text-5xl mb-6 text-blue-600" />}
              title="Advanced NLP Integration"
              description="Implementing sophisticated NLP techniques for improved language understanding and context awareness"
            />
            <EnhancementCard
              icon={<FaUserCog className="text-5xl mb-6 text-blue-600" />}
              title="Personalized User Experiences"
              description="Developing adaptive response systems based on individual user preferences and interaction history"
            />
            <EnhancementCard
              icon={<FaServer className="text-5xl mb-6 text-blue-600" />}
              title="External API Integration"
              description="Connecting to various external services for real-time information retrieval and task execution capabilities"
            />
          </div>
        </section>

        <section className="text-center mb-24">
          <h3 className="text-3xl md:text-4xl font-bold mb-6">Ready to explore the future of AI chatbots?</h3>
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onGetStarted} 
            className="bg-blue-600 text-white px-8 py-4 rounded-md text-lg font-semibold shadow-lg transition-all duration-200"
          >
            Get Started with Mazs AI
          </motion.button>
        </section>
      </main>

      <footer className="bg-gray-100 py-12">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div>
              <h4 className="text-2xl font-bold mb-4 text-blue-600">Mazs AI</h4>
              <p className="text-gray-600 mb-4">Advancing conversational AI through neural network innovation.</p>
              <div className="flex space-x-4">
                <a href="#" className="text-gray-400 hover:text-blue-600 transition-colors duration-200"><FaTwitter size={24} /></a>
                <a href="#" className="text-gray-400 hover:text-blue-600 transition-colors duration-200"><FaLinkedin size={24} /></a>
                <a href="#" className="text-gray-400 hover:text-blue-600 transition-colors duration-200"><FaGithub size={24} /></a>
              </div>
            </div>
            <div>
              <h5 className="text-lg font-semibold mb-4">Quick Links</h5>
              <ul className="space-y-2">
                <li><a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Features</a></li>
                <li><a href="#architecture" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Architecture</a></li>
                <li><a href="#future" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Future Enhancements</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Contact</a></li>
              </ul>
            </div>
            <div>
              <h5 className="text-lg font-semibold mb-4">Resources</h5>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Documentation</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">API Reference</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Research Papers</a></li>
              </ul>
            </div>
            <div>
              <h5 className="text-lg font-semibold mb-4">Legal</h5>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Privacy Policy</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Terms of Service</a></li>
              </ul>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-gray-300 text-center">
            <p className="text-gray-600">&copy; 2023 Mazs AI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

const FeatureCard: React.FC<{ icon: React.ReactNode; title: string; description: string }> = ({ icon, title, description }) => {
  return (
    <motion.div 
      whileHover={{ scale: 1.05 }}
      className="bg-white p-6 md:p-8 rounded-lg text-center shadow-lg transition-all duration-200"
    >
      {icon}
      <h3 className="text-xl md:text-2xl font-semibold mb-4 text-gray-900">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </motion.div>
  );
};

const EnhancementCard: React.FC<{ icon: React.ReactNode; title: string; description: string }> = ({ icon, title, description }) => {
  return (
    <motion.div 
      whileHover={{ scale: 1.05 }}
      className="bg-gray-100 p-6 md:p-8 rounded-lg text-center shadow-md transition-all duration-200"
    >
      {icon}
      <h4 className="text-xl md:text-2xl font-semibold mb-4 text-gray-900">{title}</h4>
      <p className="text-gray-600">{description}</p>
    </motion.div>
  );
};

const ArchitectureCard: React.FC<{ title: string; description: string }> = ({ title, description }) => {
  return (
    <motion.div 
      whileHover={{ scale: 1.05 }}
      className="bg-white p-6 md:p-8 rounded-lg text-center shadow-md transition-all duration-200"
    >
      <h4 className="text-xl md:text-2xl font-semibold mb-4 text-gray-900">{title}</h4>
      <p className="text-gray-600">{description}</p>
    </motion.div>
  );
};

export default LandingPage;