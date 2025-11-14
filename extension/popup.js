const checkbox = document.getElementById("extToggle");
const stateText = document.getElementById("stateText");

async function loadState() {
  const { enabled = false } = await chrome.storage.local.get("enabled");
  checkbox.checked = enabled;
  stateText.textContent = enabled ? "Enabled" : "Disabled";
}

checkbox.addEventListener("change", async () => {
  const enabled = checkbox.checked;
  await chrome.storage.local.set({ enabled });
  stateText.textContent = enabled ? "Enabled" : "Disabled";

  // Debug log to verify state toggle
  console.log(`Extension state toggled: ${enabled ? "Enabled" : "Disabled"}`);

  chrome.runtime.sendMessage({ type: "EXTENSION_STATE_CHANGED", enabled });
});

loadState();

// filepath: [background.js](http://_vscodecontentref_/0)
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get("enabled", (data) => {
    if (typeof data.enabled === "undefined") {
      chrome.storage.local.set({ enabled: false });
    }
  });
});

chrome.runtime.onMessage.addListener((msg, _sender, _sendResponse) => {
  if (msg.type === "EXTENSION_STATE_CHANGED") {
    // Optional: do background tasks when toggled.
    // Example: console.log('Extension enabled?', msg.enabled);
  }
});
