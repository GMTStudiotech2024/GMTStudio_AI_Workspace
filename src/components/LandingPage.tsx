import React, { useState, useEffect } from 'react';
import { FaRocket, FaBrain, FaLock, FaChartLine, FaArrowRight, FaGithub, FaTwitter, FaLinkedin, FaCheckCircle, FaPlay } from 'react-icons/fa';
import { motion } from 'framer-motion';

interface LandingPageProps {
  onGetStarted: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="bg-white text-gray-900 min-h-screen">
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
            <div className="flex items-center space-x-6">
              <a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Features</a>
              <a href="#pricing" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Pricing</a>
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onGetStarted} 
                className="bg-blue-600 text-white px-4 py-2 rounded-md transition-colors duration-200 flex items-center"
              >
                Get Started <FaArrowRight className="ml-2" />
              </motion.button>
            </div>
          </nav>
        </div>
      </header>

      <main className="container mx-auto px-4 pt-32 pb-16">
        <section className="text-center mb-24">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            className="text-6xl font-bold mb-6 text-gray-900"
          >
            AI-Powered Solutions for the Modern World
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="text-2xl mb-10 max-w-3xl mx-auto text-gray-600"
          >
            Mazs AI brings cutting-edge artificial intelligence to your fingertips, revolutionizing the way you work and create.
          </motion.p>
          <div className="flex justify-center space-x-4">
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onGetStarted} 
              className="bg-blue-600 text-white px-8 py-4 rounded-md text-lg font-semibold shadow-lg transition-all duration-200 flex items-center"
            >
              Start Free Trial <FaArrowRight className="ml-2" />
            </motion.button>
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gray-200 text-gray-800 px-8 py-4 rounded-md text-lg font-semibold shadow-lg transition-all duration-200 flex items-center"
            >
              Watch Demo <FaPlay className="ml-2" />
            </motion.button>
          </div>
        </section>

        <section id="features" className="grid md:grid-cols-2 lg:grid-cols-3 gap-12 mb-24">
          <FeatureCard
            icon={<FaRocket className="text-5xl mb-6 text-blue-600" />}
            title="Advanced AI"
            description="State-of-the-art language models for human-like interactions that adapt to your needs"
          />
          <FeatureCard
            icon={<FaBrain className="text-5xl mb-6 text-blue-600" />}
            title="Continuous Learning"
            description="Our AI evolves and improves with every interaction, providing increasingly valuable insights"
          />
          <FeatureCard
            icon={<FaLock className="text-5xl mb-6 text-blue-600" />}
            title="Secure & Private"
            description="Your data is protected with enterprise-grade security, ensuring confidentiality and trust"
          />
          <FeatureCard
            icon={<FaChartLine className="text-5xl mb-6 text-blue-600" />}
            title="Boost Productivity"
            description="Streamline your workflow with AI-powered assistance, saving time and enhancing output quality"
          />
          <FeatureCard
            icon={<FaRocket className="text-5xl mb-6 text-blue-600" />}
            title="Scalable Solutions"
            description="From startups to enterprises, our AI adapts to your business needs and grows with you"
          />
          <FeatureCard
            icon={<FaBrain className="text-5xl mb-6 text-blue-600" />}
            title="Intuitive Interface"
            description="User-friendly design ensures smooth interaction with our advanced AI capabilities"
          />
        </section>

        <section className="text-center mb-24">
          <h3 className="text-4xl font-bold mb-6">Ready to transform your work with AI?</h3>
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onGetStarted} 
            className="bg-blue-600 text-white px-8 py-4 rounded-md text-lg font-semibold shadow-lg transition-all duration-200"
          >
            Start Free Trial
          </motion.button>
        </section>

        <section id="pricing" className="mb-24">
          <h3 className="text-4xl font-bold text-center mb-12">Flexible Pricing for Every Need</h3>
          <div className="grid md:grid-cols-3 gap-8">
            <PricingCard
              title="Basic"
              price="$0.01"
              features={["1,000 AI queries/month", "Basic support", "1 user"]}
            />
            <PricingCard
              title="Pro"
              price="$0.5"
              features={["10,000 AI queries/month", "Priority support", "5 users", "Advanced analytics"]}
              highlighted={true}
            />
            <PricingCard
              title="Enterprise"
              price="Custom"
              features={["Unlimited AI queries", "24/7 dedicated support", "Unlimited users", "Custom integrations"]}
            />
          </div>
        </section>
      </main>

      <footer className="bg-gray-100 py-12">
        <div className="container mx-auto px-4">
          <div className="flex flex-wrap justify-between items-start">
            <div className="w-full md:w-1/4 mb-6 md:mb-0">
              <h4 className="text-2xl font-bold mb-4 text-blue-600">Mazs AI</h4>
              <p className="text-gray-600 mb-4">Empowering the future with intelligent solutions.</p>
              <div className="flex space-x-4">
                <a href="#" className="text-gray-400 hover:text-blue-600 transition-colors duration-200"><FaTwitter size={24} /></a>
                <a href="#" className="text-gray-400 hover:text-blue-600 transition-colors duration-200"><FaLinkedin size={24} /></a>
                <a href="#" className="text-gray-400 hover:text-blue-600 transition-colors duration-200"><FaGithub size={24} /></a>
              </div>
            </div>
            <div className="w-full md:w-1/4 mb-6 md:mb-0">
              <h5 className="text-lg font-semibold mb-4">Quick Links</h5>
              <ul className="space-y-2">
                <li><a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Features</a></li>
                <li><a href="#pricing" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Pricing</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">About Us</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Contact</a></li>
              </ul>
            </div>
            <div className="w-full md:w-1/4 mb-6 md:mb-0">
              <h5 className="text-lg font-semibold mb-4">Resources</h5>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Documentation</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">API Reference</a></li>
                <li><a href="#" className="text-gray-600 hover:text-blue-600 transition-colors duration-200">Blog</a></li>
              </ul>
            </div>
            <div className="w-full md:w-1/4">
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
      className="bg-white p-8 rounded-lg text-center shadow-lg transition-all duration-200"
    >
      {icon}
      <h3 className="text-2xl font-semibold mb-4 text-gray-900">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </motion.div>
  );
};

const PricingCard: React.FC<{ title: string; price: string; features: string[]; highlighted?: boolean }> = ({ title, price, features, highlighted }) => {
  return (
    <motion.div 
      whileHover={{ scale: 1.05 }}
      className={`p-8 rounded-lg text-center shadow-lg transition-all duration-200 ${
        highlighted ? 'bg-blue-600 text-white' : 'bg-white text-gray-900'
      }`}
    >
      <h4 className="text-2xl font-bold mb-4">{title}</h4>
      <p className="text-4xl font-bold mb-6">{price}<span className="text-lg">/month</span></p>
      <ul className="space-y-2 mb-8">
        {features.map((feature, index) => (
          <li key={index} className="flex items-center justify-center">
            <FaCheckCircle className={`mr-2 ${highlighted ? 'text-white' : 'text-green-500'}`} /> {feature}
          </li>
        ))}
      </ul>
      <button className={`px-6 py-2 rounded-md font-semibold transition-colors duration-200 ${
        highlighted ? 'bg-white text-blue-600 hover:bg-gray-100' : 'bg-blue-600 text-white hover:bg-blue-700'
      }`}>
        Choose Plan
      </button>
    </motion.div>
  );
};

export default LandingPage;