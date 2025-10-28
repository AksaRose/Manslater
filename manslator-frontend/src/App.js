import React, { useState } from 'react';
import './App.css';
import { Analytics } from "@vercel/analytics/react"

function App() {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleTranslate = async () => {
    setIsLoading(true);
    setTranslatedText(''); // Clear previous translation
    try {
      const response = await fetch('https://manslater.onrender.com/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      const data = await response.json();
      if (response.ok) {
        setTranslatedText(data.translatedText);
      } else {
        setTranslatedText(`Error: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error translating text:', error);
      setTranslatedText('Error: Could not reach the translation server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <div className="App">
        <header className="App-header">
          <h1>Manslater</h1>
          <div className="translator-container">
            <textarea
              className="input-text"
              placeholder="ex: I'm fine"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={isLoading}
            ></textarea>
            <button className="translate-button" onClick={handleTranslate} disabled={isLoading}>
              {isLoading ? 'Translating Fun...' : 'Translate'}
            </button>
            {isLoading && <p className="loading-text">Wait, lemme fix it...</p>}
            {!isLoading && translatedText && (
              <div className="output-text">
                <p>{translatedText}</p>
              </div>
            )}
          </div>
          <div className="buy-me-a-coffee">
            <a href="https://buymeacoffee.com/aksarose" target="_blank" rel="noopener noreferrer">
              <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cpath fill='%235D4037' d='M83.4 20.8h-11.7v-1.7c0-2.3-1.9-4.2-4.2-4.2H32.5c-2.3 0-4.2 1.9-4.2 4.2v1.7H16.6c-2.3 0-4.2 1.9-4.2 4.2v25c0 2.3 1.9 4.2 4.2 4.2h12.5v16.7c0 2.3 1.9 4.2 4.2 4.2h33.3c2.3 0 4.2-1.9 4.2-4.2V54.2h12.5c2.3 0 4.2-1.9 4.2-4.2V25c0-2.3-1.9-4.2-4.2-4.2zM36.7 19.1h26.7v1.7H36.7v-1.7zM79.2 45.8h-8.3V25h8.3v20.8z'/%3E%3Cpath fill='%23FFDD00' d='M70.8 50c0 1.9-1.5 3.3-3.3 3.3H32.5c-1.9 0-3.3-1.5-3.3-3.3V25c0-1.9 1.5-3.3 3.3-3.3h35c1.9 0 3.3 1.5 3.3 3.3v25z'/%3E%3C/svg%3E" alt="Buy Me a Coffee logo" class="bmc-logo" /> Buy Me a Coffee
            </a>
          </div>
        </header>
      </div>
      <Analytics />
    </>
  );
}

export default App;
