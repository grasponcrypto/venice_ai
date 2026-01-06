"""Test script for character validation."""

import asyncio
import sys
import os

# Add the custom_components path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "custom_components"))

from venice_ai.client import AsyncVeniceAIClient


async def test_character_validation():
    """Test character validation with your character ID."""
    
    # You'll need to set your API key here
    api_key = "your-api-key-here"  # Replace with your actual API key
    
    if api_key == "your-api-key-here":
        print("❌ Please set your actual API key in the test script")
        return
    
    client = AsyncVeniceAIClient(api_key=api_key)
    
    try:
        # Test with your character ID
        character_id = "1DdptTVK"  # Your character ID
        
        print(f"🔍 Testing character validation for: {character_id}")
        
        # Call the character validation
        character_data = await client.characters.get(character_id)
        
        if character_data:
            print("✅ Character validation successful!")
            print(f"Character data: {character_data}")
        else:
            print("❌ Character not found or invalid")
            
        # Test with invalid character
        print(f"\n🔍 Testing with invalid character ID: invalid-id")
        invalid_data = await client.characters.get("invalid-id")
        
        if invalid_data is None:
            print("✅ Invalid character properly handled (returned None)")
        else:
            print("❌ Invalid character should have returned None")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_character_validation())
