import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import {
  CodeBracketIcon,
  DocumentTextIcon,
  ChartBarIcon,
  HomeIcon,
  DocumentDuplicateIcon,
} from '@heroicons/react/24/outline';

import HomePage from './components/HomePage';
import ComparePage from './components/ComparePage';
import DocsPage from './components/DocsPage';

const App: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      setIsDarkMode(savedTheme === 'dark');
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = !isDarkMode ? 'dark' : 'light';
    setIsDarkMode(newTheme);
    localStorage.setItem('theme', newTheme);
  };

  return (
    <div className={`min-h-screen ${isDarkMode ? 'dark' : 'light'}`}>
      {/* Theme Toggle */}
      <div className="fixed top-4 right-4 z-50">
        <button
          onClick={toggleTheme}
          className={`
            p-2 rounded-lg shadow-sm
            ${
              isDarkMode
                ? 'bg-gray-800 text-yellow-400 hover:bg-gray-700'
                : 'bg-white text-gray-800 hover:bg-gray-100'
            }
            transition-colors duration-200
          `}
        >
          {isDarkMode ? '☀️' : '🌙'}
        </button>
      </div>

      {/* Main App */}
      <Router>
        <div className="min-h-screen">
          {/* Header */}
          <header className="bg-white dark:bg-gray-900 shadow-sm border-b border-gray-200 dark:border-gray-700">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-16">
                <div className="flex items-center">
                  <CodeBracketIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
                  <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                    AI Code Review Platform
                  </h1>
                  <span className="ml-3 text-sm text-gray-500 dark:text-gray-400">
                    v1_stable • v2_experimental • v3_deprecated
                  </span>
                </div>
              </div>

              {/* Navigation */}
              <nav className="hidden md:flex space-x-8">
                <a
                  href="/"
                  className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                >
                  <HomeIcon className="h-5 w-5 mr-2" />
                  Home
                </a>
                <a
                  href="/compare"
                  className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                >
                  <ChartBarIcon className="h-5 w-5 mr-2" />
                  Compare
                </a>
                <a
                  href="/docs"
                  className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
                >
                  <DocumentTextIcon className="h-5 w-5 mr-2" />
                  Documentation
                </a>
              </nav>
            </div>
          </header>

          {/* Page Content */}
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/docs" element={<DocsPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>

          {/* Footer */}
          <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
              <div className="md:flex md:items-center md:justify-between">
                <div className="text-center md:text-left">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    © 2024 AI Code Review Platform. Built with React, TypeScript, and Tailwind CSS.
                  </p>
                </div>
                <div className="flex space-x-6 md:mt-0">
                  <a
                    href="#"
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    Privacy
                  </a>
                  <span className="text-gray-300 dark:text-gray-600">•</span>
                  <a
                    href="#"
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    Terms
                  </a>
                  <span className="text-gray-300 dark:text-gray-600">•</span>
                  <a
                    href="#"
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    Support
                  </a>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </Router>
    </div>
  );
};

export default App;
