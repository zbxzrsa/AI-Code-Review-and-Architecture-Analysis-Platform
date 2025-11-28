import React, { useState, useRef } from 'react';
import { CloudArrowUpIcon, DocumentTextIcon, CodeBracketIcon } from '@heroicons/react/24/outline';
import MonacoEditor from './MonacoEditor';
import VersionSelector from './VersionSelector';
import ReviewResults from './ReviewResults';

interface HomePageProps {
  onVersionChange?: (version: string) => void;
}

const HomePage: React.FC<HomePageProps> = ({ onVersionChange }) => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('python');
  const [fileName, setFileName] = useState('example.py');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [reviewResults, setReviewResults] = useState<any>(null);
  const [selectedVersion, setSelectedVersion] = useState('v1');
  const [focusAreas, setFocusAreas] = useState<string[]>(['security', 'quality']);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleVersionChange = (version: string) => {
    setSelectedVersion(version);
    if (onVersionChange) {
      onVersionChange(version);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = e => {
      const content = e.target?.result as string;
      if (content) {
        setCode(content);
        setFileName(file.name);

        // Auto-detect language from file extension
        const extension = file.name.split('.').pop()?.toLowerCase();
        const languageMap: Record<string, string> = {
          py: 'python',
          js: 'javascript',
          jsx: 'javascript',
          ts: 'typescript',
          tsx: 'typescript',
          java: 'java',
          cpp: 'cpp',
          c: 'c',
          go: 'go',
          rs: 'rust',
        };

        setLanguage(languageMap[extension] || 'python');
      }
    };
    reader.readAsText(file);
  };

  const handleAnalyze = async () => {
    if (!code.trim()) {
      alert('Please enter some code to analyze');
      return;
    }

    setIsAnalyzing(true);
    setReviewResults(null);

    try {
      const response = await fetch('/api/review', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code,
          language,
          focus_areas: focusAreas,
          version: selectedVersion,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setReviewResults(result);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFocusAreaToggle = (area: string) => {
    setFocusAreas(prev => (prev.includes(area) ? prev.filter(a => a !== area) : [...prev, area]));
  };

  const sampleCode = {
    python: `def calculate_factorial(n):
    """Calculate factorial of a number"""
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result`,
    javascript: `function calculateTotal(items) {
    let total = 0;
    for(let i = 0; i < items.length; i++) {
        total += items[i].price;
    }
    return total;
}

function getUserData(userId) {
    // Simulating API call
    var users = {
        "123": {name: "John", email: "john@example.com", password: "secret123"},
        "456": {name: "Jane", email: "jane@example.com", password: "password456"}
    };
    
    return users[userId];
}`,
    typescript: `interface User {
    id: string;
    name: string;
    email: string;
}

class UserService {
    private users: Map<string, User> = new Map();

    constructor() {
        this.users.set("1", { id: "1", name: "John", email: "john@example.com" });
        this.users.set("2", { id: "2", name: "Jane", email: "jane@example.com" });
    }

    getUser(id: string): User | undefined {
        return this.users.get(id);
    }
}`,
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">AI Code Review Platform</h1>
              <p className="ml-4 text-gray-500">Analyze your code with AI-powered insights</p>
            </div>
            <VersionSelector
              selectedVersion={selectedVersion}
              onVersionChange={handleVersionChange}
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Code Input Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <CodeBracketIcon className="h-6 w-6 mr-2" />
                  Code Input
                </h2>

                {/* File Upload */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Upload File
                  </label>
                  <div className="flex items-center justify-center w-full">
                    <label className="flex flex-col items-center justify-center w-full h-32 px-4 py-6 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-gray-400 bg-gray-50 hover:bg-gray-100 transition-colors">
                      <CloudArrowUpIcon className="h-8 w-8 text-gray-400 mb-2" />
                      <span className="text-sm text-gray-600">Choose a file or drag and drop</span>
                      <span className="text-xs text-gray-500">
                        .py, .js, .ts, .java, .cpp, .go, .rs
                      </span>
                    </label>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".py,.js,.jsx,.ts,.tsx,.java,.cpp,.c,.go,.rs"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                  {fileName && (
                    <div className="mt-2 text-sm text-gray-600">
                      Selected: <span className="font-medium">{fileName}</span>
                    </div>
                  )}
                </div>

                {/* Language Selection */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Programming Language
                  </label>
                  <select
                    value={language}
                    onChange={e => setLanguage(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="python">Python</option>
                    <option value="javascript">JavaScript</option>
                    <option value="typescript">TypeScript</option>
                    <option value="java">Java</option>
                    <option value="cpp">C++</option>
                    <option value="c">C</option>
                    <option value="go">Go</option>
                    <option value="rust">Rust</option>
                  </select>
                </div>

                {/* Sample Code Buttons */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Or try sample code:
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => {
                        setCode(sampleCode.python);
                        setLanguage('python');
                        setFileName('sample.py');
                      }}
                      className="px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      Python Example
                    </button>
                    <button
                      onClick={() => {
                        setCode(sampleCode.javascript);
                        setLanguage('javascript');
                        setFileName('sample.js');
                      }}
                      className="px-3 py-2 text-sm bg-green-100 text-green-700 rounded-md hover:bg-green-200 focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      JavaScript Example
                    </button>
                  </div>
                </div>
              </div>

              {/* Monaco Editor */}
              <div className="mb-4">
                <MonacoEditor
                  value={code}
                  onChange={setCode}
                  language={language}
                  height="400px"
                  placeholder="Enter your code here or upload a file..."
                />
              </div>

              {/* Focus Areas */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-3">Focus Areas</label>
                <div className="space-y-2">
                  {['security', 'quality', 'performance', 'style', 'architecture'].map(area => (
                    <label key={area} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={focusAreas.includes(area)}
                        onChange={() => handleFocusAreaToggle(area)}
                        className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700 capitalize">{area}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || !code.trim()}
                className="w-full flex justify-center items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018 8a8 8 0 018-8 0 4-4-4-4-4-4-4-4-4zm-2 0a6 6 0 016 6 6 0 016-6 4-4-4-4-4-4-4-4-4zm4 0a6 6 0 016 6 6 0 016-6 4-4-4-4-4-4-4-4-4z"
                      />
                    </svg>
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <DocumentTextIcon className="h-5 w-5 mr-2" />
                    Analyze Code
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <DocumentTextIcon className="h-6 w-6 mr-2" />
                Analysis Results
              </h2>

              {reviewResults ? (
                <ReviewResults
                  issues={reviewResults.issues || []}
                  score={reviewResults.score || 0}
                  processingTime={reviewResults.processing_time || 0}
                  version={selectedVersion}
                  modelUsed={reviewResults.model_used || 'unknown'}
                  language={language}
                  focusAreas={focusAreas}
                  cached={reviewResults.cached || false}
                />
              ) : (
                <div className="text-center py-12">
                  <div className="text-gray-400 mb-4">
                    <svg
                      className="mx-auto h-12 w-12 text-gray-300"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M9 12h6m-6 4h6m2 5H8m-6 4h6m2 5H8"
                      />
                    </svg>
                  </div>
                  <p className="text-gray-500">No analysis results yet.</p>
                  <p className="text-sm text-gray-400">
                    Enter code above and click "Analyze Code" to get started.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
