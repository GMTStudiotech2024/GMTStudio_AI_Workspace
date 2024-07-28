import React from 'react';
import { FaRocket, FaBrain, FaLock, FaChartLine } from 'react-icons/fa';

interface LandingPageProps {
  onGetStarted: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  return (
    <div className="bg-gray-900 text-white min-h-screen">
      <header className="container mx-auto px-4 py-8">
        <nav className="flex justify-between items-center">
          <h1 className="text-2xl font-bold">Mazs AI</h1>
          <div>
            <a href="#features" className="mr-4 hover:text-blue-400">Features</a>
            <a href="#" className="mr-4 hover:text-blue-400">Pricing</a>
            <button onClick={onGetStarted} className="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded">Get Started</button>
          </div>
        </nav>
      </header>

      <main className="container mx-auto px-4 py-16">
        <section className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-4">Experience the Future of AI</h2>
          <p className="text-xl mb-8">Mazs AI is a cutting-edge platform that brings the power of artificial intelligence to your fingertips.</p>
          <button onClick={onGetStarted} className="bg-blue-500 hover:bg-blue-600 px-6 py-3 rounded-lg text-lg font-semibold">Try Mazs AI Now</button>
        </section>

        <section id="features" className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          <FeatureCard
            icon={<FaRocket className="text-4xl mb-4" />}
            title="Advanced AI"
            description="State-of-the-art language models for human-like interactions"
          />
          <FeatureCard
            icon={<FaBrain className="text-4xl mb-4" />}
            title="Continuous Learning"
            description="Our AI evolves and improves with every interaction"
          />
          <FeatureCard
            icon={<FaLock className="text-4xl mb-4" />}
            title="Secure & Private"
            description="Your data is protected with enterprise-grade security"
          />
          <FeatureCard
            icon={<FaChartLine className="text-4xl mb-4" />}
            title="Boost Productivity"
            description="Streamline your workflow with AI-powered assistance"
          />
        </section>

        <section className="text-center mb-16">
          <h3 className="text-3xl font-bold mb-4">Ready to transform your work with AI?</h3>
          <button onClick={onGetStarted} className="bg-green-500 hover:bg-green-600 px-6 py-3 rounded-lg text-lg font-semibold">Get Started for Free</button>
        </section>
      </main>

      <footer className="bg-gray-800 py-8">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; 2023 Mazs AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

const FeatureCard: React.FC<{ icon: React.ReactNode; title: string; description: string }> = ({ icon, title, description }) => {
  return (
    <div className="bg-gray-800 p-6 rounded-lg text-center">
      {icon}
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p>{description}</p>
    </div>
  );
};

export default LandingPage;