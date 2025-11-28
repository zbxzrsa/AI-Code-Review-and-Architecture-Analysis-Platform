import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App_Complete';
import { AuthProvider } from './contexts/AuthContext';
import { SimpleThemeProvider } from './app/providers/SimpleThemeProvider';
import './index.css';

const root = createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <SimpleThemeProvider>
      <AuthProvider>
        <App />
      </AuthProvider>
    </SimpleThemeProvider>
  </React.StrictMode>
);
