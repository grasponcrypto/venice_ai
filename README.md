# Venice AI
Home Assistant Venice AI Conversation Integration

## Overview
The Venice AI integration allows you to enhance your Home Assistant setup with advanced conversational capabilities powered by Venice AI. This integration enables seamless interaction with your smart home devices through natural language processing.

## Features
- Natural language understanding for smart home commands
- Customizable responses based on user preferences
- Integration with various Home Assistant components
- Dynamic model selection from available Venice AI models

## Installation

### Option 1: HACS (Recommended)
1. Make sure you have [HACS](https://hacs.xyz/) installed in your Home Assistant instance.
2. Click on HACS in the sidebar.
3. Go to "Integrations".
4. Click the three dots in the top right corner and select "Custom repositories".
5. Add `https://github.com/grasponcrypto/venice_ai` as a repository with category "Integration".
6. Click "Add".
7. Search for "Venice AI" in the integrations tab.
8. Click "Download" and follow the installation instructions.
9. Restart Home Assistant.

### Option 2: Manual Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/grasponcrypto/venice_ai.git
   ```

2. **Copy the integration files:**
   Place the `venice_ai` folder in your Home Assistant `custom_components` directory.

3. **Restart Home Assistant:**
   After copying the files, restart your Home Assistant instance to load the new integration.

## Configuration
To configure the Venice AI integration:

1. Go to Settings → Devices & Services
2. Click "Add Integration" and search for "Venice AI"
3. Enter your Venice AI API key
4. Configure additional options as needed:
   - Select your preferred model
   - Adjust temperature, max tokens, and other parameters
   - Customize the system prompt if desired

## Support
If you encounter any issues or have feature requests, please open an issue on our [GitHub Issues page](https://github.com/grasponcrypto/venice_ai/issues).

## License
This project is licensed under the MIT License - see the LICENSE file for details.
