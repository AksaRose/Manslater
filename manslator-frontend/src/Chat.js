import React, { useState, useRef, useEffect } from "react";
import "./Convo.css";

const Chat = () => {
  const [messages, setMessages] = useState([
    {
      role: "ai",
      content: "Alright genius, what did she say that's got you panicking?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);
  const API_URL = "http://localhost:8000";

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput("");

    // Add user message to chat
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();

      // Store session ID for future requests
      if (!sessionId) {
        setSessionId(data.session_id);
      }

      // Add AI response to chat
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: data.response,
        },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: "Oops, something went wrong on my end. Try again?",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = async () => {
    if (sessionId) {
      try {
        await fetch(`${API_URL}/session/${sessionId}`, {
          method: "DELETE",
        });
      } catch (error) {
        console.error("Error clearing session:", error);
      }
    }

    setMessages([
      {
        role: "ai",
        content: "Alright genius, what did she say that's got you panicking?",
      },
    ]);
    setSessionId(null);
  };

  return (
    <>
      <div className="convo-container">
        <div className="convo-header">
          <button className="clear-btn" onClick={clearChat}>
            Clear Chat
          </button>
        </div>

        <div className="messages-container">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message ${
                msg.role === "user" ? "user-message" : "ai-message"
              }`}
            >
              <div className="message-content">{msg.content}</div>
            </div>
          ))}

          {isLoading && (
            <div className="message ai-message">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="input-container-fixed">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={isLoading}
          rows="1"
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || !input.trim()}
          className="send-arrow-btn"
          aria-label="Send message"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </>
  );
};

export default Chat;
