import requests
import json

def test_post_endpoint():
    """Test the POST endpoint that was failing"""
    try:
        # Test POST request to /detect-anomalies
        url = 'http://localhost:5000/detect-anomalies'
        
        # Sample data that might be sent by Java application
        test_data = {
            "image_path": "test_thermal_image.jpg",
            "analysis_type": "anomaly_detection"
        }
        
        print("Testing POST request to /detect-anomalies...")
        response = requests.post(url, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("\n✅ SUCCESS: POST request works correctly!")
        else:
            print(f"\n❌ FAILED: Expected 200, got {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_get_endpoint():
    """Test the GET endpoint for comparison"""
    try:
        print("\n" + "="*50)
        print("Testing GET request to /detect-anomalies...")
        
        response = requests.get('http://localhost:5000/detect-anomalies')
        print(f"Status Code: {response.status_code}")
        print("Response preview (first 200 chars):")
        response_text = response.text
        print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
        
        if response.status_code == 200:
            print("✅ SUCCESS: GET request works correctly!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_post_endpoint()
    test_get_endpoint()