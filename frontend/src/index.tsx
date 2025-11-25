import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App_Simple';
import './App_Simple.css';

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
