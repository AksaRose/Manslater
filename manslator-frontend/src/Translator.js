import React, { useState } from "react";
import "./App.css";

function Translator() {
  const [inputText, setInputText] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      return;
    }

    const newUserMessage = { text: inputText, sender: "user" };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setInputText("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5001/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      });
      const data = await response.json();
      if (response.ok) {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: data.translatedText, sender: "manslater" },
        ]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: `Error: ${data.error || "Unknown error"}`, sender: "manslater" },
        ]);
      }
    } catch (error) {
      console.error("Error translating text:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Error: Could not reach the translation server.", sender: "manslater" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Manslater</h1>
        <div className="chat-container">
          <div className="messages-display">
            {messages.map((msg, index) => (
              <div key={index} className={`message-bubble ${msg.sender}`}>
                <p>{msg.text}</p>
              </div>
            ))}
            {isLoading && (
              <div className="message-bubble manslater loading">
                <p>Wait, lemme fix it...</p>
              </div>
            )}
          </div>
          <div className="input-area">
            <textarea
              className="input-text"
              placeholder="ex: I'm fine"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={isLoading}
              onKeyPress={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleTranslate();
                }
              }}
            ></textarea>
            <button
              className="translate-button"
              onClick={handleTranslate}
              disabled={isLoading}
            >
              {isLoading ? "Translating Fun..." : "Translate"}
            </button>
          </div>
        </div>
      </header>
    </div>
  );
}

export default Translator;
