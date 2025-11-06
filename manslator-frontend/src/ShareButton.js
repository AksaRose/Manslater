import React, { useState, useCallback } from "react";
import "./ShareButton.css";

const ShareButton = ({ onCapture, visible }) => {
  const [isGenerating, setIsGenerating] = useState(false);

  const handleClick = useCallback(async () => {
    if (isGenerating) return;
    setIsGenerating(true);

    try {
      await onCapture();
    } catch (error) {
      console.error("Error capturing:", error);
    } finally {
      setIsGenerating(false);
    }
  }, [onCapture, isGenerating]);

  return (
    <div className={`share-wrapper ${visible ? "visible" : ""}`}>
      <button
        className={`share-button ${visible ? "visible" : ""} ${
          isGenerating ? "loading" : ""
        }`}
        onClick={handleClick}
        aria-label="Share"
        disabled={isGenerating}
      >
        {isGenerating ? (
          <svg
            className="loading-spinner"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <circle
              className="spinner-path"
              cx="12"
              cy="12"
              r="10"
              fill="none"
              strokeWidth="3"
            />
          </svg>
        ) : (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
            <polyline points="16 6 12 2 8 6" />
            <line x1="12" y1="2" x2="12" y2="15" />
          </svg>
        )}
      </button>
    </div>
  );
};

export default ShareButton;
