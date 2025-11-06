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
  const API_URL = "https://manslater.onrender.com";
  // Typing indicator phrases and animation state
  const typingPhrases = [
    "Wait, genius — I'm thinking 🤔",
    "It's women, you know it's hard",
    "Almost there",
  ];
  const [typingIndex, setTypingIndex] = useState(0);
  const [dotCount, setDotCount] = useState(0);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Manage typing indicator animation while loading
  useEffect(() => {
    if (!isLoading) {
      // reset when not loading
      setTypingIndex(0);
      setDotCount(0);
      return;
    }

    // interval for cycling dots
    const dotsInterval = setInterval(() => {
      setDotCount((c) => (c + 1) % 4); // 0..3
    }, 500);

    // interval for cycling phrases
    const phraseInterval = setInterval(() => {
      setTypingIndex((i) => (i + 1) % typingPhrases.length);
    }, 2000);

    return () => {
      clearInterval(dotsInterval);
      clearInterval(phraseInterval);
    };
  }, [isLoading]);

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

      const data = await response.json();

      // Handle rate limit (429) or other errors
      if (!response.ok) {
        // Extract error message from response
        const errorMessage =
          data.detail?.message || data.detail || "Failed to get response";

        // Split error message by newline if it contains multiple parts
        const errorParts = errorMessage
          .split("\n")
          .filter((part) => part.trim());

        // Add each error part as a separate AI message
        setMessages((prev) => [
          ...prev,
          ...errorParts.map((part) => ({
            role: "ai",
            content: part,
          })),
        ]);
        return;
      }

      // Store session ID for future requests
      if (!sessionId) {
        setSessionId(data.session_id);
      }

      // Split the response by newline to separate roast and advice
      const responseParts = data.response
        .split("\n")
        .filter((part) => part.trim());

      // Add each part as a separate AI message bubble
      setMessages((prev) => [
        ...prev,
        ...responseParts.map((part) => ({
          role: "ai",
          content: part,
        })),
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
          <button
            className="clear-btn"
            onClick={clearChat}
            aria-label="Clear chat"
          >
            {/* Trash / bin icon */}
            <svg
              viewBox="0 0 24 24"
              width="18"
              height="18"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
              <path d="M10 11v6" />
              <path d="M14 11v6" />
              <path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2" />
            </svg>
          </button>
        </div>

        <div className="messages-container">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message-wrapper ${
                msg.role === "user" ? "user" : "manslater"
              }`}
            >
              {msg.role === "ai" && (
                <img
                  src={process.env.PUBLIC_URL + "/images/dfds.png"}
                  className="avatar"
                  alt="AI Assistant Profile"
                />
              )}
              <div
                className={`message-bubble ${
                  msg.role === "user" ? "user" : "manslater"
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message-wrapper manslater">
              <img
                src={process.env.PUBLIC_URL + "/images/dfds.png"}
                className="avatar"
                alt="AI Assistant Profile"
              />
              <div
                className="message-bubble manslater loading"
                aria-live="polite"
              >
                <div className="typing-row">
                  <div className="typing-text">
                    {typingPhrases[typingIndex]}
                    {".".repeat(dotCount)}
                  </div>
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
