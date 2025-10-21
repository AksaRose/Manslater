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
        </header>
      </div>
      <Analytics />
    </>
  );
}

export default App;
