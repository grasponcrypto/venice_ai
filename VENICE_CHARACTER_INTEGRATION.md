# Venice.ai Native Character Integration

## Overview

This document describes the Venice.ai native character integration feature that allows users to select and use Venice.ai's built-in characters within the Home Assistant Venice AI integration.

## Features

### ✅ **Native Venice.ai Character Support**
- Connects directly to Venice.ai's character system
- Supports all available Venice.ai characters
- Preserves 100% of original functionality including tool calls and Home Assistant control

### ✅ **Dual Character Selection Methods**
- **Dropdown Selection**: Choose from dynamically fetched character list
- **Manual ID Input**: Enter specific character IDs (e.g., `character-chat/1DdptTVK`)

### ✅ **Seamless Integration**
- Character selection appears in Home Assistant integration options
- No disruption to existing configurations
- Maintains backward compatibility

## User Interface

### Configuration Options

When configuring the Venice AI integration in Home Assistant, users will see:

1. **Character (Optional)**: Dropdown selector with available characters
   - Fetches list dynamically from Venice.ai API
   - Shows character names for easy selection
   - Includes "No Character" option for default behavior

2. **Character ID**: Manual text input field
   - For entering specific character IDs directly
   - Supports Venice.ai's character ID format
   - Useful for characters not in the dropdown or for testing

### Priority System

- **Character ID field** takes precedence if populated
- **Character dropdown** is used if ID field is empty
- **No character selected** uses default system prompt

## Technical Implementation

### API Integration

The integration includes a new `Characters` API client class:

```python
class Characters:
    """Characters API for Venice AI."""

    async def list(self) -> list[dict]:
        """List available characters."""

    async def get(self, character_id: str) -> dict:
        """Get character details by ID."""
```

### Character Parameter Usage

Characters are integrated into the Venice API call via the `venice_parameters`:

```python
api_request_payload = {
    "model": model_name,
    "messages": messages,
    "venice_parameters": {
        "include_venice_system_prompt": False,
        "character": character_id  # Venice.ai character parameter
    },
    # ... other parameters
}
```

### Configuration Flow

The options flow dynamically fetches characters:

```python
# Fetch characters for dropdown
characters_response = await client.characters.list()
for character in characters_response:
    character_id = character.get("id")
    character_name = character.get("name", character_id)
    # Add to dropdown options
```

## Usage Examples

### Example 1: Using Your Specific Character

For your character `character-chat/1DdptTVK`:

1. Go to **Settings → Devices & Services → Venice AI → Configure**
2. Leave **Character** dropdown as "No Character"
3. Enter `character-chat/1DdptTVK` in **Character ID** field
4. Save configuration
5. All conversations will now use your character

### Example 2: Selecting from Dropdown

1. Go to **Settings → Devices & Services → Venice AI → Configure**
2. Choose a character from the **Character** dropdown
3. Leave **Character ID** field empty
4. Save configuration

### Example 3: Combining with Custom Prompt

You can use characters alongside custom system prompts:

1. Select a character (via dropdown or ID)
2. Add your custom prompt in the **System Prompt** field
3. The character's personality will guide responses while your prompt provides context

## Compatibility

### ✅ **Maintained Features**
- All existing tool calling functionality
- Home Assistant device control
- Conversation history
- TTS integration
- Model selection and parameters
- Custom system prompts

### ✅ **Backward Compatibility**
- Existing configurations continue working unchanged
- No character selected = original behavior
- All settings remain optional

## Configuration Details

### New Constants

```python
# Character configuration
CONF_CHARACTER = "character"
CONF_CHARACTER_ID = "character_id"
RECOMMENDED_CHARACTER = ""  # No character by default
```

### New UI Labels

```json
"character": "Character (Optional)",
"character_id": "Character ID"
```

## Troubleshooting

### Character Not Appearing in Dropdown

**Cause**: Character may not be in the public list

**Solution**: Use the **Character ID** field to enter it manually

### Character ID Not Working

**Check**: Verify the character ID format is correct
- Should match Venice.ai's character ID format
- Example: `character-chat/1DdptTVK`

**Debug**: Check Home Assistant logs for character API errors

### API Errors

**Check**: Network connectivity and API key validity
- Ensure Venice AI API key is valid
- Check network connectivity to api.venice.ai

## API Endpoints Used

### Characters API
- `GET /characters` - List available characters
- `GET /characters/{id}` - Get specific character details

Both endpoints use standard Venice.ai authentication with the user's API key.

## Implementation Notes

1. **Character Priority**: Manual Character ID overrides dropdown selection
2. **API Integration**: Uses Venice.ai's native character parameter
3. **Error Handling**: Graceful fallback if character fetching fails
4. **Logging**: Comprehensive logging for debugging character usage
5. **Performance**: Characters fetched once per configuration session

## Testing

### Testing Your Character

1. Configure your character ID: `character-chat/1DdptTVK`
2. Start a conversation with the Venice AI assistant
3. Verify the character's personality comes through
4. Test that tool calls still work (ask to control a device)

### Testing Dropdown Selection

1. Select any character from the dropdown
2. Observe character-specific responses
3. Verify Home Assistant functionality remains intact

## Future Enhancements

### Potential Improvements
- Character search functionality
- Character preview/descriptions in UI
- Character-specific parameter customization
- Character import/export functionality

### API Considerations
- Cache character list for performance
- Handle character API rate limits
- Add character subscription status checking

---

**This integration provides true Venice.ai character support while maintaining all existing Home Assistant functionality.**
