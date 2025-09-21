import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing OMR Grader API")
    print("=" * 50)
    
    # Test 1: Health Check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health Check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health Check failed: {e}")
    
    # Test 2: List Submissions
    try:
        response = requests.get(f"{base_url}/submissions")
        print(f"✅ List Submissions: {response.status_code}")
        submissions = response.json()
        print(f"   Found {len(submissions)} submissions")
    except Exception as e:
        print(f"❌ List Submissions failed: {e}")
    
    # Test 3: API Documentation
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"✅ API Docs: {response.status_code}")
        print(f"   Swagger UI is available at {base_url}/docs")
    except Exception as e:
        print(f"❌ API Docs failed: {e}")

if __name__ == "__main__":
    test_api()