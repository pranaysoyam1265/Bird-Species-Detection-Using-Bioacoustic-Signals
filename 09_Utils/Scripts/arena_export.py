"""
LM Arena Data Extractor
Extracts visible content from arena.ai using browser automation
No API key needed - just browser login
"""

from playwright.sync_api import sync_playwright
import time
import json
from pathlib import Path

OUTPUT_FILE = "lmarena_chat.txt"
CONTEXT_DIR = "browser_context"
Path(CONTEXT_DIR).mkdir(exist_ok=True)

print("="*70)
print("🦁 LM ARENA DATA EXTRACTOR")
print("="*70)
print("\nThis tool extracts visible chat/results data from arena.ai")
print("by automating a browser. No API key needed!\n")

with sync_playwright() as p:
    # Launch browser
    browser = p.chromium.launch(headless=False)
    
    # Try to load saved session
    try:
        context = browser.new_context(storage_state=f"{CONTEXT_DIR}/state.json")
        print("✅ Using saved login session (you should already be logged in)")
        print("   If not logged in, close all windows and delete 'browser_context' folder\n")
    except FileNotFoundError:
        print("📝 Starting fresh session (you'll need to log in)\n")
        context = browser.new_context()
    
    page = context.new_page()
    
    # Intercept failed API requests to see what's happening
    failed_requests = []
    
    def handle_response(response):
        if response.status >= 400:
            failed_requests.append({
                'url': response.url,
                'status': response.status,
                'method': response.request.method
            })
    
    page.on("response", handle_response)
    
    # Navigate to arena
    print("🌐 Opening arena.ai...")
    try:
        page.goto("https://arena.ai/", timeout=60000)
        print("✅ Page loaded")
    except Exception as e:
        print(f"⚠️  Navigation took a while: {e}")
    
    time.sleep(5)
    
    # Check if logged in
    print("\n🔍 Checking login status...")
    try:
        # Look for login button
        logout_btn = page.locator("text=/logout|sign out|profile/i").first
        is_logged_in = False
        try:
            is_logged_in = logout_btn.is_visible(timeout=2000)
        except:
            pass
        
        if is_logged_in:
            print("✅ You appear to be logged in")
        else:
            print("⚠️  Login status unclear - check browser window")
    except:
        print("⚠️ Could not determine login status")
    
    print("\n" + "="*70)
    print("BROWSER IS OPEN - WHAT TO DO:")
    print("="*70)
    print("""
STEP 1 - LOGIN (if not already logged in):
   - If you see a login screen, enter your credentials
   - Complete any 2FA or verification
   
STEP 2 - NAVIGATE TO DATA:
   - Click on "Results" or "Arena" or similar
   - Wait for your chat/battle results to load
   - You should see a list of matches/conversations
   
STEP 3 - COME BACK:
   - Once data is loaded, return to this terminal
   - Press ENTER below when ready
   - Script will extract everything visible on the page
""")
    
    input("🎯 Press ENTER when your data is loaded and visible: ")
    
    print("\n📡 Analyzing page...")
    time.sleep(2)
    
    # Method 1: Get all text content
    print("✓ Extracting visible text...")
    body_text = ""
    try:
        body_text = page.locator("body").inner_text()
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
    
    # Method 2: Get all card/item elements
    print("✓ Looking for data cards...")
    cards_data = []
    
    selectors_to_try = [
        "div[class*='card']",
        "div[class*='item']",
        "div[class*='battle']",
        "div[class*='arena']",
        "article",
        "[role='article']",
    ]
    
    for selector in selectors_to_try:
        try:
            cards = page.locator(selector).all()
            if cards and len(cards) > 0:
                print(f"  ✅ Found {len(cards)} items with: {selector}")
                for i, card in enumerate(cards[:100], 1):  # Limit to 100
                    try:
                        text = card.inner_text().strip()
                        if text and len(text) > 10:
                            cards_data.append(f"--- ITEM {i} ---\n{text}")
                    except:
                        pass
                break
        except:
            continue
    
    # Method 3: Get JSON/script data
    print("✓ Checking for data in page scripts...")
    script_data = {}
    try:
        scripts = page.locator("script").all()
        for script in scripts:
            try:
                content = script.inner_text()
                if "arena" in content.lower() or "battle" in content.lower():
                    # Try to find JSON
                    if "{" in content and "}" in content:
                        script_data[f"script_{len(script_data)}"] = content[:1000]  # First 1000 chars
            except:
                pass
    except:
        pass
    
    # Method 4: Check for API errors in failed requests
    print("✓ Checking for failed API calls...")
    if failed_requests:
        print(f"  ⚠️ Found {len(failed_requests)} failed requests:")
        for req in failed_requests[:5]:
            print(f"     - {req['status']} {req['method']} {req['url'][:50]}...")
    
    # Compile all data
    all_data = []
    
    if body_text:
        print(f"\n✅ Got page text ({len(body_text)} characters)")
        all_data.append("=== PAGE TEXT CONTENT ===\n")
        all_data.append(body_text[:10000])  # First 10k chars
    
    if cards_data:
        print(f"✅ Got {len(cards_data)} card items")
        all_data.append("\n\n=== EXTRACTED CARDS ===\n")
        all_data.extend(cards_data)
    
    if script_data:
        print(f"✅ Got data from {len(script_data)} scripts")
        all_data.append("\n\n=== SCRIPT DATA ===\n")
        all_data.append(json.dumps(script_data, indent=2))
    
    # Save all extracted data
    if all_data:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(all_data))
        print(f"\n✅ Exported data to: {OUTPUT_FILE}")
        
        # Show preview
        print("\n📋 Preview (first 500 chars):")
        print("-" * 70)
        print("".join(all_data)[:500])
        print("-" * 70)
    else:
        print("\n❌ No data extracted!")
        print("\nPossible reasons:")
        print("  1. Page didn't load properly")
        print("  2. Data is still loading (try clicking around first)")
        print("  3. You need to scroll down to see content")
        print("  4. Page structure is different than expected")
        print("\n💡 Try:")
        print("  - Click some buttons in the browser")
        print("  - Scroll down to load more content")
        print("  - Wait longer for data to load")
        print("  - Run again with more time")
    
    # Save session for next time
    print("\n💾 Saving session (you won't need to re-login next time)...")
    try:
        context.storage_state(path=f"{CONTEXT_DIR}/state.json")
        print("✅ Session saved")
    except Exception as e:
        print(f"⚠️ Could not save session: {e}")
    
    context.close()
    browser.close()

print("\n" + "="*70)
print("✅ EXTRACTION COMPLETE")
print("="*70)
print(f"\n📁 Files created:")
print(f"   📄 {OUTPUT_FILE} - Extracted data")
print(f"   📂 browser_context/ - Saved login session\n")
print("Run again next time - you'll skip the login!")
