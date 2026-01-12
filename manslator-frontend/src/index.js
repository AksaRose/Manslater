import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Wait for fonts to load before showing text to prevent flash
if ('fonts' in document) {
  // Wait for all fonts to be ready
  document.fonts.ready.then(() => {
    // Check if Shopie font is loaded
    if (document.fonts.check('1em Shopie')) {
      document.body.classList.add('fonts-loaded');
    } else {
      // Font not loaded yet, wait a bit more
      setTimeout(() => {
        document.body.classList.add('fonts-loaded');
      }, 300);
    }
  });
  
  // Fallback: show text after max 1.5 seconds even if font hasn't loaded
  setTimeout(() => {
    document.body.classList.add('fonts-loaded');
  }, 1500);
} else {
  // Fallback for browsers without Font Loading API - show after short delay
  setTimeout(() => {
    document.body.classList.add('fonts-loaded');
  }, 200);
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
