import requests
import argparse
import json
import os

def test_extraction(image_path: str):
    url = "http://localhost:8000/api/v1/extract"
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    print(f"Testing API with image: {image_path}")
    print(f"Sending POST request to {url}...\n")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            print("✅ Success! Extracted Data:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Is it running on http://localhost:8000?")
    except Exception as e:
        print(f"❌ Error during request: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Invoice Extraction API")
    parser.add_argument("--image", type=str, required=True, help="Path to the receipt image")
    args = parser.parse_args()
    
    test_extraction(args.image)
