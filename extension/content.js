// filepath: c:\Users\habee\OneDrive\Desktop\Manslater\extension\content.js
(async function () {
  const { enabled = false } = await chrome.storage.local.get("enabled");
  if (!enabled) return;

  // Example active behavior: add a subtle data attribute to document
  document.documentElement.setAttribute("data-manslater-active", "true");
})();
