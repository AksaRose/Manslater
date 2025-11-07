import React, { useState, useRef, useEffect, useCallback } from "react";
import html2canvas from "html2canvas";
import "./Convo.css";
import ShareButton from "./ShareButton";
import "./ShareButton.css";

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
  const containerRef = useRef(null);
  const textareaRef = useRef(null);
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

  const captureScreen = useCallback(async () => {
    if (!containerRef.current) return;

    try {
      // Reset any transformations before capture
      containerRef.current.style.transform = "none";

      // Capture the conversation
      const conversationCanvas = await html2canvas(containerRef.current, {
        backgroundColor: null,
        scale: 2,
      });

      // Create a new canvas for the final composition
      const finalCanvas = document.createElement("canvas");
      const ctx = finalCanvas.getContext("2d");

      // Load the template image
      const templateImage = new Image();
      templateImage.crossOrigin = "anonymous";
      templateImage.src = `${process.env.PUBLIC_URL}/images/template.jpg`;

      return new Promise((resolve, reject) => {
        templateImage.onload = () => {
          finalCanvas.width = templateImage.width;
          finalCanvas.height = templateImage.height;

          // Draw template first
          ctx.drawImage(templateImage, 0, 0);

          // Calculate optimal positioning for conversation
          const isMobile = window.innerWidth <= 768;
          const verticalPadding =
            templateImage.height * (isMobile ? 0.15 : 0.25);
          const horizontalPadding =
            templateImage.width * (isMobile ? 0.05 : 0.1);

          const availableWidth = templateImage.width - horizontalPadding * 2;
          const availableHeight = templateImage.height - verticalPadding * 2;

          const scale =
            Math.min(
              availableWidth / conversationCanvas.width,
              availableHeight / conversationCanvas.height
            ) * (isMobile ? 1.1 : 0.95);

          const scaledWidth = conversationCanvas.width * scale;
          const scaledHeight = conversationCanvas.height * scale;

          const x = (templateImage.width - scaledWidth) / 2;
          const y = templateImage.height * 0.45 - scaledHeight / 2;

          // Draw conversation on top
          ctx.drawImage(conversationCanvas, x, y, scaledWidth, scaledHeight);

          const image = finalCanvas.toDataURL("image/png");
          resolve(image);
        };
        templateImage.onerror = reject;
      });
    } catch (error) {
      console.error("Error capturing conversation:", error);
      throw error;
    }
  }, []);

  const handleShare = useCallback(async () => {
    try {
      const image = await captureScreen();
      const response = await fetch(image);
      const blob = await response.blob();

      const file = new File([blob], "story.png", { type: "image/png" });

      // 1) Prefer Web Share API with files (best experience on Android/Chrome)
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        try {
          await navigator.share({ files: [file], title: "Share" });
          return;
        } catch (err) {
          console.warn("navigator.share failed:", err);
        }
      }

      // 2) Upload image to backend so we have a public HTTPS URL Instagram can access
      try {
        const form = new FormData();
        form.append("file", file);

        const uploadRes = await fetch(`${API_URL}/upload`, {
          method: "POST",
          body: form,
        });
        if (!uploadRes.ok) throw new Error("Upload failed");
        const uploadData = await uploadRes.json();
        const publicUrl = uploadData.url;

        // Try to copy the public URL to clipboard so user can paste if needed
        if (navigator.clipboard && navigator.clipboard.writeText) {
          try {
            await navigator.clipboard.writeText(publicUrl);
          } catch (e) {
            // ignore clipboard errors
          }
        }

        // Open Instagram app - user may need to add the uploaded image manually from the URL or camera roll
        // There's no reliable cross-platform deep link that programmatically attaches a web-hosted image from a URL.
        // We'll open Instagram and provide the public URL in the clipboard to make attaching easier.
        window.location.href = "instagram://story-camera";
        return;
      } catch (uploadErr) {
        console.warn("Upload fallback failed:", uploadErr);
      }

      // 3) Final fallback: try to write to clipboard as an image (modern Chromium) then open IG
      if (navigator.clipboard && window.ClipboardItem) {
        try {
          await navigator.clipboard.write([
            new ClipboardItem({ [blob.type]: blob }),
          ]);
          window.location.href = "instagram://story-camera";
          return;
        } catch (err) {
          console.warn("clipboard image write failed:", err);
        }
      }

      // 4) If all else fails, trigger a download so the user can attach the image manually
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = "manslater-story.png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(downloadUrl), 3000);
    } catch (error) {
      console.error("Error sharing:", error);
    }
  }, [captureScreen]);

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
  }, [isLoading, typingPhrases.length]); // Added typingPhrases.length as dependency

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput("");
    // ensure body class removed when message is sent programmatically
    if (document && document.body)
      document.body.classList.remove("keyboard-open");

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

  // Adjust textarea height to fit content
  const adjustTextareaHeight = useCallback(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    // reset height to allow shrinking
    ta.style.height = "auto";
    // set to scrollHeight (limit controlled via CSS max-height)
    ta.style.height = ta.scrollHeight + "px";
  }, []);

  // Keep textarea height in sync with value
  useEffect(() => {
    adjustTextareaHeight();
  }, [input, adjustTextareaHeight]);

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
      <div className="convo-container" ref={containerRef}>
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
              <div className="message-content-wrapper">
                <div
                  className={`message-bubble ${
                    msg.role === "user" ? "user" : "manslater"
                  }`}
                >
                  {msg.content}
                </div>
                {msg.role === "ai" &&
                  index === messages.length - 1 &&
                  !isLoading && (
                    <ShareButton onCapture={handleShare} visible={true} />
                  )}
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
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onFocus={() => document.body.classList.add("keyboard-open")}
          onBlur={() => document.body.classList.remove("keyboard-open")}
          onInput={adjustTextareaHeight}
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
