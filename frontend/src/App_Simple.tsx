import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Code Review Platform</h1>
        <p>Platform is running successfully!</p>
        <div style={{ marginTop: '20px' }}>
          <a href="/api/status" target="_blank" rel="noopener noreferrer">
            Check API Status
          </a>
        </div>
      </header>
    </div>
  );
}

export default App;
