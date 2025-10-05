import requests
import json

def test_server():
    try:
        # Test home endpoint
        response = requests.get('http://localhost:5000/')
        print("Home endpoint status:", response.status_code)
        print("Home response:", json.dumps(response.json(), indent=2))
        print("\n" + "="*50 + "\n")
        
        # Test fault-regions endpoint
        response = requests.get('http://localhost:5000/fault-regions')
        print("Fault-regions endpoint status:", response.status_code)
        print("Fault-regions response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_server()