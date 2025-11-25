import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';

import './App.css';

function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <Router>
        <div className="App">
          <header className="App-header">
            <h1>AI Code Review Platform</h1>
            <p>Enhanced with Ant Design 5, React Router 6, and advanced features</p>
          </header>
          <main>
            <Routes>
              <Route
                path="/"
                element={
                  <div style={{ padding: '20px' }}>
                    <h2>🚀 AI Code Review & Architecture Analysis Platform</h2>
                    <p>Welcome to the next-generation code analysis platform!</p>

                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                        gap: '20px',
                        marginTop: '30px',
                      }}
                    >
                      <div
                        style={{
                          padding: '20px',
                          border: '1px solid #d9d9d9',
                          borderRadius: '8px',
                        }}
                      >
                        <h3>🔍 Code Analysis</h3>
                        <p>Advanced static and dynamic analysis with AI-powered insights</p>
                      </div>
                      <div
                        style={{
                          padding: '20px',
                          border: '1px solid #d9d9d9',
                          borderRadius: '8px',
                        }}
                      >
                        <h3>🤖 AI Integration</h3>
                        <p>Multiple AI providers with intelligent suggestion generation</p>
                      </div>
                      <div
                        style={{
                          padding: '20px',
                          border: '1px solid #d9d9d9',
                          borderRadius: '8px',
                        }}
                      >
                        <h3>📊 Quality Metrics</h3>
                        <p>Comprehensive code quality tracking and reporting</p>
                      </div>
                      <div
                        style={{
                          padding: '20px',
                          border: '1px solid #d9d9d9',
                          borderRadius: '8px',
                        }}
                      >
                        <h3>🔒 Security Scanning</h3>
                        <p>Vulnerability detection and security best practices</p>
                      </div>
                      <div
                        style={{
                          padding: '20px',
                          border: '1px solid #d9d9d9',
                          borderRadius: '8px',
                        }}
                      >
                        <h3>📈 Performance Monitoring</h3>
                        <p>Real-time performance metrics and optimization suggestions</p>
                      </div>
                    </div>

                    <div
                      style={{
                        marginTop: '30px',
                        padding: '20px',
                        backgroundColor: '#f6f8ff',
                        borderRadius: '8px',
                      }}
                    >
                      <h3>✨ Platform Status</h3>
                      <p>
                        <strong>Backend:</strong> Enhanced with FastAPI, SQLAlchemy, and
                        comprehensive APIs
                      </p>
                      <p>
                        <strong>Frontend:</strong> Rebuilt with React 18, Ant Design 5, and modern
                        tooling
                      </p>
                      <p>
                        <strong>Database:</strong> PostgreSQL with Redis caching and Neo4j graph
                        storage
                      </p>
                      <p>
                        <strong>CI/CD:</strong> GitHub Actions with comprehensive testing and
                        deployment
                      </p>
                    </div>
                  </div>
                }
              />
              <Route
                path="/dashboard"
                element={
                  <div style={{ padding: '20px' }}>
                    <h2>📊 Dashboard</h2>
                    <p>Project overview and analytics dashboard</p>
                    <p>Features: KPI metrics, trend analysis, team performance</p>
                  </div>
                }
              />
              <Route
                path="/projects"
                element={
                  <div style={{ padding: '20px' }}>
                    <h2>📁 Projects</h2>
                    <p>Project management with repository integration</p>
                    <p>Features: Multi-tenant support, GitHub integration, team collaboration</p>
                  </div>
                }
              />
              <Route
                path="/analysis"
                element={
                  <div style={{ padding: '20px' }}>
                    <h2>🔍 Code Analysis</h2>
                    <p>Advanced code analysis with AI-powered insights</p>
                    <p>
                      Features: Static analysis, AI suggestions, architecture graphs, security
                      scanning
                    </p>
                  </div>
                }
              />
              <Route
                path="/settings"
                element={
                  <div style={{ padding: '20px' }}>
                    <h2>⚙️ Settings</h2>
                    <p>Platform configuration and user preferences</p>
                    <p>Features: Provider management, policy configuration, team settings</p>
                  </div>
                }
              />
            </Routes>
          </main>
        </div>
      </Router>
    </ConfigProvider>
  );
}

export default App;
