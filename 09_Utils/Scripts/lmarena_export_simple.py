"""
LM Arena Data Export Tool
Extracts chat/arena data from arena.ai

The "Auth session missing" error means arena.ai API requires authentication.
Solution: Get your API key from arena.ai and use it directly.
"""

import os
import json
from pathlib import Path

OUTPUT_FILE = "lmarena_chat.txt"

print("="*70)
print("🦁 LM ARENA DATA EXPORT TOOL")
print("="*70)
print()

# Ask user for their setup preference
print("How would you like to export data?\n")
print("Option 1: Using Playwright Browser (interactive)")
print("         - Opens browser, you stay logged in")
print("         - Paste your API key when prompted")
print("         - Script extracts data\n")

print("Option 2: Using API Key Directly (faster)")
print("         - Just provide your API key")
print("         - Script handles authentication")
print("         - No browser window needed\n")

choice = input("Choose 1 or 2, or 'Q' to quit: ").strip().upper()

if choice == 'Q':
    print("Exiting...")
    exit(0)

if choice == '2':
    # Direct API approach
    print("\n" + "="*70)
    print("API MODE: Direct Data Extraction")
    print("="*70)
    
    api_key = input("\n🔑 Paste your arena.ai API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        exit(1)
    
    import requests
    
    print("📡 Connecting to arena.ai API...")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try different API endpoints
        endpoints = [
            "https://api.arena.ai/v1/chat/history",
            "https://api.arena.ai/v1/conversations",
            "https://api.lmarena.ai/chat/history",
        ]
        
        data = None
        for endpoint in endpoints:
            try:
                print(f"Trying: {endpoint}")
                response = requests.get(endpoint, headers=headers, timeout=10)
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Success!")
                    break
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        if data:
            # Save JSON data
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"\n✅ Exported data to {OUTPUT_FILE}")
        else:
            print("\n❌ Could not connect to API")
            print("\nTroubleshooting:")
            print("  1. Verify your API key is correct")
            print("  2. Check arena.ai docs for current API endpoint")
            print("  3. Ensure your account has API access enabled")
    
    except ImportError:
        print("❌ requests library not installed")
        print("Install with: pip install requests")

else:
    # Browser approach
    print("\n" + "="*70)
    print("BROWSER MODE: Interactive Extraction")
    print("="*70)
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright not installed")
        print("Install with: pip install playwright")
        print("Then run: playwright install")
        exit(1)
    
    import time
    
    CONTEXT_DIR = "browser_context"
    Path(CONTEXT_DIR).mkdir(exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        
        try:
            context = browser.new_context(storage_state=f"{CONTEXT_DIR}/state.json")
            print("\n✅ Using saved session (already logged in)")
        except:
            print("\n📝 Starting fresh (will need login)")
            context = browser.new_context()
        
        page = context.new_page()
        
        print("🌐 Opening arena.ai...")
        try:
            page.goto("https://arena.ai/", timeout=60000, wait_until="networkidle")
        except:
            print("⚠️ Page load timeout (Cloudflare/network delay)")
        
        time.sleep(3)
        
        print("\n" + "="*70)
        print("⚠️  ACTION REQUIRED IN BROWSER WINDOW")
        print("="*70)
        print("""
YOUR TO-DO:
  1. If NOT logged in:
     - Click "Sign In" 
     - Enter your credentials
     - Complete any 2FA if needed
  
  2. Navigate to the page with your chat data
  
  3. Return here and press ENTER

STUCK AT CLOUDFLARE?
  - Wait 10-15 seconds, it auto-checks

NEED API KEY approach?
  - Close this and restart, choose Option 2
""")
        
        input("Press ENTER when ready and page is loaded: ")
        
        print("\n📜 Scrolling to load more data...")
        try:
            for i in range(10):
                page.mouse.wheel(0, -3000)
                time.sleep(0.5)
                print(f"   Scrolled {i+1}/10")
        except:
            print("   (Could not scroll)")
        
        time.sleep(2)
        
        # Get content
        try:
            html = page.content()
            print(f"✅ Got page HTML ({len(html)} bytes)")
            
            # Save for inspection
            with open("arena_debug.html", "w", encoding="utf-8") as f:
                f.write(html)
            
            # Try to extract text
            body_text = page.locator("body").inner_text()
            
            lines = [line.strip() for line in body_text.split("\n") if line.strip()]
            
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            
            print(f"✅ Exported {len(lines)} lines to {OUTPUT_FILE}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"Saved debug HTML to arena_debug.html")
        
        # Save session
        try:
            context.storage_state(path=f"{CONTEXT_DIR}/state.json")
            print("💾 Session saved (no re-login next time)")
        except:
            pass
        
        context.close()
        browser.close()

print("\n" + "="*70)
print("✅ Done!")
print("="*70)
print(f"\n📁 Output files:")
print(f"   - {OUTPUT_FILE}")
if os.path.exists("arena_debug.html"):
    print(f"   - arena_debug.html")
print("\n💡 Next steps:")
print("   - Check the output file")
print("   - If data looks wrong, try the other mode")
print("   - Contact arena.ai support for API issues")
