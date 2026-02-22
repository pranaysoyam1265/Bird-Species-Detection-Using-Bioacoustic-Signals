from playwright.sync_api import sync_playwright
import time
import os
import json
from pathlib import Path

OUTPUT_FILE = "lmarena_chat.txt"
CONTEXT_DIR = "browser_context"

Path(CONTEXT_DIR).mkdir(exist_ok=True)

def extract_auth_tokens(page):
    """Extract auth tokens from browser storage"""
    try:
        # Get localStorage
        local_storage = page.evaluate("""
            () => {
                let data = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    data[key] = localStorage.getItem(key);
                }
                return data;
            }
        """)
        
        if local_storage:
            print("✅ Found localStorage data")
            for key in local_storage:
                if 'token' in key.lower() or 'auth' in key.lower() or 'session' in key.lower():
                    print(f"   Found: {key}")
        
        # Get cookies
        cookies = page.context.cookies()
        auth_cookies = {}
        for cookie in cookies:
            if 'session' in cookie['name'].lower() or 'auth' in cookie['name'].lower() or 'token' in cookie['name'].lower():
                auth_cookies[cookie['name']] = cookie['value']
                print(f"✅ Found cookie: {cookie['name']}")
        
        return local_storage, auth_cookies
    except Exception as e:
        print(f"⚠️ Could not extract auth: {e}")
        return {}, {}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    
    try:
        context = browser.new_context(storage_state=f"{CONTEXT_DIR}/state.json")
        print("📂 Using saved session...")
    except:
        print("📝 Creating new session...")
        context = browser.new_context()
    
    page = context.new_page()
    
    print("🌐 Opening arena.ai...")
    try:
        page.goto("https://arena.ai/", timeout=60000, wait_until="networkidle")
    except Exception as e:
        print(f"⚠️ Navigation timeout: {e}")
        pass
    
    time.sleep(3)
    
    print("\n" + "="*60)
    print("⚠️  IMPORTANT: You need to be logged in!")
    print("="*60)
    print("\n✋ Please follow these steps in the BROWSER WINDOW:")
    print("  1. Click on your profile/account in the top right")
    print("  2. Go to Settings or Account settings")
    print("  3. Look for 'API Key', 'Auth Token', or 'Session'")
    print("  4. Copy the API key/token value")
    print("\n  OR if there's no obvious way:")
    print("  5. Open browser DevTools (F12)")
    print("  6. Go to Application → LocalStorage → arena.ai")
    print("  7. Look for keys containing 'token', 'auth', or 'session'")
    print("  8. Copy the value\n")
    
    # Give user time to find and copy auth token
    auth_token = None
    print("Paste your API key/auth token below and press ENTER")
    print("(If you can't find it, try logging out and back in):")
    try:
        auth_token = input("API Key/Token: ").strip()
    except:
        pass
    
    if not auth_token:
        print("\n❌ No auth token provided. Trying to extract from browser...")
        time.sleep(5)
        
        print("\n📡 Checking browser storage...")
        local_storage, auth_cookies = extract_auth_tokens(page)
        
        if auth_cookies:
            # Use cookies from browser
            print("\n✅ Found auth in cookies! Using browser session...")
        else:
            print("\n⚠️ No auth found in browser. Script may fail.")
            print("💡 Solution: Get your API key from arena.ai settings")
            auth_token = input("\nTry pasting API key now: ").strip()

    # Scroll up to load older entries
    print("\n📜 Scrolling to load older entries...")
    for i in range(25):
        page.mouse.wheel(0, -4000)
        time.sleep(0.3)
        if (i + 1) % 5 == 0:
            print(f"   Scrolled {i+1}/25...")

    # Try to find data cards
    print("🔍 Searching for data...")
    cards = None
    
    # Try multiple selectors
    selectors = [
        "div:has-text('Model is Working Correctly')",
        "[class*='card']",
        "[class*='message']",
        "[class*='entry']",
        "div[role='article']"
    ]
    
    for selector in selectors:
        try:
            cards = page.locator(selector).all()
            if cards and len(cards) > 0:
                print(f"✅ Found {len(cards)} entries using: {selector}")
                break
        except:
            continue
    
    if not cards or len(cards) == 0:
        print("❌ No data found! Possible issues:")
        print("   - Not fully logged in")
        print("   - Page structure changed")
        print("   - No chat data available")
        context.close()
        browser.close()
        exit(1)
    
    # Extract data
    extracted = []
    for i, card in enumerate(cards, 1):
        try:
            text = card.inner_text().strip()
            if len(text) > 20:  # Lowered threshold
                extracted.append(f"----- ENTRY {i} -----\n{text}")
        except Exception as e:
            print(f"   ⚠️ Could not extract entry {i}: {e}")
            continue

    # Save results
    if extracted:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n\n".join(extracted))
        print(f"\n✅ Exported {len(extracted)} entries to {OUTPUT_FILE}")
        
        # Save login session for next time
        context.storage_state(path=f"{CONTEXT_DIR}/state.json")
        print("💾 Session saved for next time (no re-login needed)")
    else:
        print("❌ No data extracted after processing cards.")
    
    context.close()
    browser.close()
