// filepath: c:\Users\habee\OneDrive\Desktop\Manslater\extension\readme.md
# Manslater Extension (Basic Toggle)

## Features
- Popup with enable/disable switch.
- State stored in `chrome.storage.local`.
- Content script runs only when enabled (checks stored flag).
- Background listens for state changes.

## Install (Chrome)
1. Open `chrome://extensions`.
2. Enable Developer Mode.
3. Click “Load unpacked”.
4. Select the `extension` folder.

## Files
- `manifest.json` – Extension definition.
- `popup.html/.css/.js` – UI for toggle.
- `background.js` – Initialization & message listener.
- `content.js` – Example page behavior when enabled.

## Customization
Replace the logic inside `content.js` with actual functionality. Use `chrome.storage.local.get('enabled')` everywhere you need to respect the toggle.

## Uninstall
Remove from `chrome://extensions`.
