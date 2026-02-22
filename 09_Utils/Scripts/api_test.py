"""
Quick test for Xeno-Canto API v3
Save as: test_api_v3.py
"""

import requests
import time

print("=" * 60)
print("TESTING XENO-CANTO API v3")
print("=" * 60)

test_id = "475302"

# Test different URL formats
urls_to_test = [
    f"https://xeno-canto.org/api/3/recordings/{test_id}",
    f"https://xeno-canto.org/api/3/recordings?query=nr:{test_id}",
    f"https://www.xeno-canto.org/api/3/recordings/{test_id}",
    f"https://xeno-canto.org/{test_id}",  # Direct page (for scraping)
]

for url in urls_to_test:
    print(f"\n🔗 Testing: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        
        if response.status_code == 200:
            if 'json' in response.headers.get('Content-Type', ''):
                data = response.json()
                print(f"   ✅ JSON Response!")
                print(f"   Keys: {list(data.keys())[:5]}")
            elif 'html' in response.headers.get('Content-Type', ''):
                print(f"   📄 HTML Response (can webscrape)")
                if 'Blue-winged Warbler' in response.text:
                    print(f"   ✅ Found species name in HTML!")
        else:
            print(f"   Response: {response.text[:150]}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    time.sleep(1)

print("\n" + "=" * 60)
print("Test complete!")