import React, { useState, useCallback } from "react";
import "./ShareButton.css";

const ShareButton = ({ onCapture, visible, imageUrl }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [showOptions, setShowOptions] = useState(false);

  const handleShare = useCallback(
    async (platform) => {
      if (isGenerating) return;
      setIsGenerating(true);

      try {
        const image = await onCapture();

        if (platform === "instagram") {
          // Instagram Story sharing
          const dataUrl = image.replace(
            /^data:image\/(png|jpg|jpeg);base64,/,
            ""
          );
          const blob = await fetch(`data:image/png;base64,${dataUrl}`).then(
            (res) => res.blob()
          );
          const filesArray = [
            new File([blob], "manslater-story.png", { type: "image/png" }),
          ];

          if (navigator.share && navigator.canShare({ files: filesArray })) {
            await navigator.share({
              files: filesArray,
              title: "Share to Instagram Story",
              text: "Check out this conversation on Manslater!",
            });
          } else {
            // Fallback for browsers that don't support Web Share API
            const link = document.createElement("a");
            link.href = `instagram://story-camera?media=${encodeURIComponent(
              image
            )}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
          }
        } else {
          // Regular download
          const link = document.createElement("a");
          link.href = image;
          link.download = "manslater-conversation.png";
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      } catch (error) {
        console.error("Error sharing:", error);
      } finally {
        setIsGenerating(false);
        setShowOptions(false);
      }
    },
    [onCapture, isGenerating]
  );

  const toggleOptions = useCallback(() => {
    setShowOptions((prev) => !prev);
  }, []);

  return (
    <div className={`share-wrapper ${visible ? "visible" : ""}`}>
      <button
        className="share-button"
        onClick={toggleOptions}
        aria-label="Share options"
        disabled={isGenerating}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
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
      </button>

      {showOptions && (
        <div className="share-options">
          <button
            className="share-option instagram"
            onClick={() => handleShare("instagram")}
            aria-label="Share to Instagram Story"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <rect x="2" y="2" width="20" height="20" rx="5" />
              <circle cx="12" cy="12" r="4" />
              <circle
                cx="18"
                cy="6"
                r="1.5"
                fill="currentColor"
                stroke="none"
              />
            </svg>
            <span>Instagram Story</span>
          </button>
          <button
            className="share-option download"
            onClick={() => handleShare("download")}
            aria-label="Download image"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            <span>Download</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default ShareButton;
