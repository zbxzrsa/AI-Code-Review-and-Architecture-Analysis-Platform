import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AuthProvider } from './contexts/AuthContext';
import { SimpleThemeProvider } from './app/providers/SimpleThemeProvider';
import './App.css';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);

root.render(
  <React.StrictMode>
    <SimpleThemeProvider>
      <AuthProvider>
        <App />
      </AuthProvider>
    </SimpleThemeProvider>
  </React.StrictMode>
);
