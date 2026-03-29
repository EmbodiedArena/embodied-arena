#!/usr/bin/env python3
"""
Test script to verify API configuration
"""
import sys
import os
sys.path.append('OST-Bench-main')

from models.utils.openai_api import API_keys, get_client

def test_api_config():
    """Test if API keys are properly configured"""
    print("🔍 Testing API Configuration...")
    print()

    # Test API keys
    print("📋 API Keys Status:")
    for key_type, key_value in API_keys.items():
        status = "✅ Set" if key_value else "❌ Empty"
        print(f"  {key_type}: {status}")
    print()

    # Test client creation
    test_models = [
        ("gpt-4o", "gpt"),
        ("o3", "gpt"),
        ("gemini-2.5-pro", "gemini"),
        ("claude-3-7-sonnet-20250219", "claude"),
        ("qwen-vl-max", "qwen")
    ]

    print("🔧 Testing Client Creation:")
    for model_name, expected_key_type in test_models:
        try:
            client = get_client(model_name)
            actual_key_type = None

            # Determine which key type was used
            if hasattr(client, '_api_key') and client._api_key == API_keys['gpt']:
                actual_key_type = 'gpt'
            elif hasattr(client, 'api_key') and client.api_key == API_keys['claude']:
                actual_key_type = 'claude'
            elif hasattr(client, '_api_key') and client._api_key == API_keys['gemini']:
                actual_key_type = 'gemini'
            elif hasattr(client, '_api_key') and client._api_key == API_keys['qwen']:
                actual_key_type = 'qwen'

            if actual_key_type == expected_key_type:
                print(f"  ✅ {model_name}: Using {actual_key_type} key")
            else:
                print(f"  ⚠️  {model_name}: Expected {expected_key_type}, got {actual_key_type}")

        except Exception as e:
            print(f"  ❌ {model_name}: Failed to create client - {str(e)}")

    print()
    print("🎯 Configuration complete! You can now run the OST-Bench API scripts.")

if __name__ == "__main__":
    test_api_config()
