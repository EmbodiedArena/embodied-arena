#!/usr/bin/env python3
"""
Test API connection
"""
import sys
import os
sys.path.insert(0, 'OST-Bench-main')

try:
    from models.utils.openai_api import get_client, API_keys

    print("🔍 Testing API Connection...")
    print()

    # Test different models
    test_models = ['gpt', 'claude', 'gemini', 'qwen']

    for model_type in test_models:
        if API_keys[model_type]:
            print(f"📋 Testing {model_type.upper()} API:")
            try:
                client = get_client(f"{model_type}-test-model")
                print(f"  ✅ Client created successfully for {model_type}")
            except Exception as e:
                print(f"  ❌ Failed to create client for {model_type}: {str(e)}")
        else:
            print(f"📋 {model_type.upper()} API: No key configured")
        print()

    print("🎯 API connection test completed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct environment with required packages installed.")
