"""
Example: Test the API with a sample image
"""

import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"  # Change to your Azure URL
# API_URL = "https://your-app.azurewebsites.net"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_single_image(image_path: str):
    """Test single image prediction"""
    print(f"Testing prediction with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    if result['success']:
        print(f"\n✅ Detected {result['faces_detected']} face(s)")
        for face in result['results']:
            print(f"   Face {face['face_id']}: {face['emotion']} ({face['confidence']:.2%})")
    print()

def test_batch_images(image_paths: list):
    """Test batch prediction"""
    print(f"Testing batch prediction with {len(image_paths)} images...")
    
    files = []
    for path in image_paths:
        files.append(('files', (Path(path).name, open(path, 'rb'), 'image/jpeg')))
    
    response = requests.post(f"{API_URL}/predict/batch", files=files)
    
    # Close file handles
    for _, (_, file_obj, _) in files:
        file_obj.close()
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    print()

def main():
    """Run example tests"""
    print("="*60)
    print("Emotion Detection API - Example Usage")
    print("="*60)
    print()
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Single image (you'll need to provide your own image)
    # Uncomment and modify the path:
    # test_single_image("path/to/your/test_image.jpg")
    
    # Test 3: Batch images
    # Uncomment and modify the paths:
    # test_batch_images([
    #     "path/to/image1.jpg",
    #     "path/to/image2.jpg",
    #     "path/to/image3.jpg"
    # ])
    
    print("="*60)
    print("Example complete!")
    print("="*60)

if __name__ == "__main__":
    main()
