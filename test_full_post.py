import requests
import os
import json

def test_post_request():
    """Test the POST request with actual image files"""
    
    # Check if we have test images
    baseline_path = "test_baseline.jpg"  # You'll need actual image files
    candidate_path = "test_candidate.jpg"
    
    url = "http://localhost:5000/detect-anomalies"
    
    try:
        # Create dummy image files for testing if they don't exist
        if not os.path.exists(baseline_path):
            print("⚠️  Creating dummy baseline image file for testing...")
            with open(baseline_path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x02}\x01\x17\x00\x00\x00\x00IEND\xaeB`\x82')
        
        if not os.path.exists(candidate_path):
            print("⚠️  Creating dummy candidate image file for testing...")
            with open(candidate_path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x02}\x01\x17\x00\x00\x00\x00IEND\xaeB`\x82')
        
        print("🧪 Testing POST request to /detect-anomalies...")
        
        # Prepare files for upload
        with open(baseline_path, 'rb') as baseline_file, open(candidate_path, 'rb') as candidate_file:
            files = {
                'baseline': ('baseline.jpg', baseline_file, 'image/jpeg'),
                'candidate': ('candidate.jpg', candidate_file, 'image/jpeg')
            }
            
            response = requests.post(url, files=files)
            
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Content-Type: {response.headers.get('content-type')}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: POST request works!")
            print("📄 Response preview:")
            try:
                json_response = response.json()
                print(json.dumps(json_response, indent=2)[:500] + "...")
            except:
                print(response.text[:500] + "...")
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server.")
        print("Make sure the Flask server is running on http://localhost:5000")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        # Cleanup test files
        for f in [baseline_path, candidate_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"🧹 Cleaned up {f}")
                except:
                    pass

def test_missing_files():
    """Test error handling when files are missing"""
    print("\n" + "="*50)
    print("🧪 Testing error handling (missing files)...")
    
    try:
        response = requests.post("http://localhost:5000/detect-anomalies")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 400:
            print("✅ SUCCESS: Proper error handling for missing files!")
        else:
            print("⚠️  Unexpected status code for missing files")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_post_request()
    test_missing_files()