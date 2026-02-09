#!/usr/bin/env python3
"""
Test script for the Emotion Detection API
Tests both local and deployed endpoints
"""

import requests
import sys
from pathlib import Path
import json

def test_api(base_url: str):
    """Test the emotion detection API"""
    
    print(f"\n{'='*60}")
    print(f"Testing Emotion Detection API")
    print(f"Base URL: {base_url}")
    print(f"{'='*60}\n")
    
    # Test 1: Root endpoint
    print("Test 1: Root Endpoint")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Root endpoint working\n")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}\n")
        return False
    
    # Test 2: Health check
    print("Test 2: Health Check")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Health check working\n")
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
        return False
    
    # Test 3: Get emotions list
    print("Test 3: Get Emotions List")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/emotions")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Emotions endpoint working\n")
    except Exception as e:
        print(f"❌ Emotions endpoint failed: {e}\n")
        return False
    
    # Test 4: Model info
    print("Test 4: Model Information")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/model/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Model info endpoint working\n")
    except Exception as e:
        print(f"❌ Model info endpoint failed: {e}\n")
        return False
    
    # Test 5: Image prediction (if test image exists)
    print("Test 5: Image Prediction")
    print("-" * 40)
    
    # Create a simple test image if none exists
    test_image_path = Path("test_face.jpg")
    
    if test_image_path.exists():
        try:
            with open(test_image_path, 'rb') as f:
                files = {'file': ('test_face.jpg', f, 'image/jpeg')}
                response = requests.post(f"{base_url}/predict", files=files)
                print(f"Status: {response.status_code}")
                print(f"Response: {json.dumps(response.json(), indent=2)}")
                print("✅ Prediction endpoint working\n")
        except Exception as e:
            print(f"❌ Prediction endpoint failed: {e}\n")
            return False
    else:
        print("⚠️ No test image found (test_face.jpg)")
        print("   Skipping prediction test\n")
    
    print(f"{'='*60}")
    print("✅ All tests completed successfully!")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        # Default to local
        base_url = "http://localhost:8000"
    
    # Remove trailing slash
    base_url = base_url.rstrip('/')
    
    success = test_api(base_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
