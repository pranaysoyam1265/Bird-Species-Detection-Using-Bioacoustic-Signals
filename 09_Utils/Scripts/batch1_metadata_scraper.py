"""
Bird Audio Metadata Retrieval Script - Selenium Web Scraping
=============================================================
Phase 0: Xeno-Canto Metadata Construction
Batch 1: First 1000 Recording IDs

Configuration:
- Method: WebDriver Manager + Selenium
- Delay: 2 seconds between requests
- Browser: Visible mode
- Checkpoint: Every 100 recordings
- Resume: Enabled
"""

import time
import os
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
AUDIO_FOLDER = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\Audio Recordings"
OUTPUT_CSV = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\bird_metadata_batch1.csv"
CHECKPOINT_CSV = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\bird_metadata_batch1_checkpoint.csv"
LOG_FILE = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\scraping_log.txt"

# Scraping settings
REQUEST_DELAY = 2  # seconds between requests
CHECKPOINT_EVERY = 100  # save progress every N recordings
PAGE_LOAD_TIMEOUT = 30  # max seconds to wait for page load

# ============================================================
# BATCH 1: FIRST 1000 RECORDING IDs
# ============================================================

RECORDING_IDS = [
    "101288", "101292", "101293", "101580", "101581", "101591", "101593", "101614",
    "102972", "104520", "104521", "109162", "109190", "109191", "109192", "109299",
    "109300", "109301", "109305", "109306", "109314", "109601", "109602", "109603",
    "109606", "109651", "109666", "109668", "109768", "109852", "109920", "109921",
    "109922", "109923", "109946", "109947", "109948", "109949", "109950", "109951",
    "110078", "110079", "110080", "110081", "110813", "111040", "111041", "111042",
    "111043", "111138", "111139", "112499", "112501", "112502", "112579", "112598",
    "112829", "113485", "113486", "113721", "113723", "113724", "113808", "113814",
    "113820", "113821", "113827", "114061", "114073", "114081", "114085", "114086",
    "114087", "114342", "114550", "114551", "114552", "114554", "114555", "114556",
    "114557", "114558", "114559", "114560", "114561", "11476", "114930", "115430",
    "115587", "115594", "115938", "115939", "115940", "116054", "116055", "116056",
    "116372", "116373", "116374", "11722", "11847", "118749", "119593", "120285",
    "120809", "120842", "120843", "120844", "120855", "120868", "120878", "121042",
    "121043", "121044", "121045", "121046", "121068", "121396", "121435", "122259",
    "122442", "122453", "122455", "122640", "122641", "122806", "122807", "122809",
    "122810", "122811", "122812", "122813", "122814", "122815", "122816", "122817",
    "122818", "122819", "122820", "122821", "122822", "122824", "122825", "122826",
    "122892", "122895", "122896", "122898", "122900", "122904", "122908", "122909",
    "122910", "122911", "122912", "122913", "122914", "122915", "122916", "122918",
    "122922", "122924", "123156", "123302", "123306", "123343", "123363", "123364",
    "123370", "123405", "123506", "124067", "124268", "124312", "124431", "124433",
    "124438", "124439", "124441", "124484", "124590", "124591", "124593", "124595",
    "124992", "125090", "125091", "125093", "125094", "125095", "125096", "125168",
    "125169", "125171", "125173", "125176", "125178", "125207", "125512", "125540",
    "125542", "125544", "125546", "125547", "125549", "125550", "125553", "125555",
    "125556", "125557", "125558", "125559", "125560", "125561", "125564", "125567",
    "125569", "125571", "125573", "125576", "125579", "125581", "125727", "125729",
    "125844", "126022", "127371", "127418", "127600", "127999", "128001", "128002",
    "128099", "128249", "128490", "128705", "128942", "129225", "129313", "129386",
    "129387", "129650", "129798", "130058", "130170", "130499", "130500", "130507",
    "130508", "130510", "130511", "130512", "130513", "130525", "131093", "131442",
    "131445", "131447", "131452", "131462", "131549", "131559", "131566", "131567",
    "131637", "131879", "131881", "131882", "131925", "131951", "131952", "131971",
    "132145", "132216", "132217", "132219", "132220", "132223", "132224", "132226",
    "132229", "132236", "132237", "132239", "132241", "132243", "132246", "132248",
    "132250", "132251", "132254", "132255", "132256", "132257", "132258", "132259",
    "132277", "132278", "132280", "132281", "132282", "132283", "132284", "132285",
    "132286", "132287", "132288", "132319", "132320", "132321", "132322", "132323",
    "132324", "132325", "132328", "132331", "132333", "132334", "132335", "132338",
    "132341", "132342", "132344", "132347", "132348", "132350", "132352", "132353",
    "132355", "132358", "132359", "132363", "132365", "132373", "132381", "132382",
    "132383", "132384", "132386", "132692", "132693", "132695", "132819", "132821",
    "132823", "132824", "132826", "132827", "132828", "132830", "132831", "132832",
    "132833", "132835", "132836", "132837", "132838", "132839", "132842", "132844",
    "132845", "132846", "132847", "132848", "132849", "132850", "132851", "132852",
    "132853", "132854", "132855", "132856", "132857", "132858", "132859", "132861",
    "132864", "132870", "132872", "132873", "132886", "132888", "132889", "132890",
    "132891", "132892", "132894", "132896", "132901", "132902", "132903", "132904",
    "132905", "132906", "132908", "132909", "132910", "132911", "132913", "132914",
    "132916", "132917", "132918", "132920", "133039", "133048", "133080", "133167",
    "133392", "133469", "133500", "133564", "133565", "133569", "133590", "133792",
    "133794", "133902", "133923", "133969", "133979", "133993", "133996", "133998",
    "134014", "134017", "134100", "134185", "134349", "134439", "134495", "134496",
    "134499", "134501", "134502", "134504", "134505", "134872", "134874", "134876",
    "134994", "135026", "135052", "135053", "135054", "135055", "135067", "135075",
    "135086", "135087", "135110", "135117", "135120", "135284", "135440", "135451",
    "135454", "135455", "135456", "135457", "135459", "135460", "135462", "135463",
    "135466", "135468", "135476", "135477", "135478", "135492", "135494", "135495",
    "135496", "135497", "135498", "135499", "135501", "135502", "135503", "135680",
    "135688", "135883", "136021", "136035", "136051", "136052", "136053", "136054",
    "136055", "136115", "136140", "136330", "137040", "137041", "137042", "137043",
    "137044", "137045", "137527", "137570", "137604", "137605", "137610", "137618",
    "137637", "137714", "137726", "137741", "137774", "137865", "138054", "138063",
    "138065", "138066", "138068", "138069", "138073", "138084", "138085", "138093",
    "138140", "138154", "138159", "138517", "138610", "138633", "138639", "138706",
    "138718", "138873", "139055", "139086", "139089", "139144", "139171", "139433",
    "139438", "139439", "139446", "139577", "139595", "139598", "139603", "139605",
    "139608", "139610", "139613", "139614", "139619", "139728", "139729", "139739",
    "139740", "139798", "139829", "139831", "139856", "139859", "139880", "139897",
    "139908", "139915", "139916", "139921", "139932", "139937", "139938", "139961",
    "139987", "140280", "140282", "140283", "140286", "140298", "140465", "141035",
    "141213", "141214", "141316", "141320", "141344", "141345", "141412", "141439",
    "141441", "141444", "141445", "141458", "141469", "141677", "141749", "141896",
    "141897", "141898", "142065", "142066", "142067", "142068", "142069", "142186",
    "142218", "142244", "142329", "142466", "142592", "142594", "142632", "142649",
    "142650", "142722", "142731", "142977", "143748", "143934", "143936", "143939",
    "144024", "144069", "144071", "144072", "144080", "144101", "144540", "144655",
    "144658", "144664", "144666", "144672", "144674", "144675", "144676", "144696",
    "144899", "145054", "145055", "146080", "146082", "146083", "146086", "146148",
    "146167", "146297", "146599", "146604", "147059", "147067", "147069", "147070",
    "147071", "147072", "147073", "147272", "147273", "147278", "147286", "147287",
    "147288", "147289", "147290", "147291", "147292", "147879", "147880", "148239",
    "148500", "148502", "148503", "148504", "149087", "149225", "149253", "149269",
    "149270", "149271", "149273", "149275", "149276", "149277", "149283", "149284",
    "149285", "149286", "149289", "149290", "149305", "149337", "149345", "149386",
    "149387", "149388", "149390", "149391", "149392", "149393", "149395", "149397",
    "149398", "149399", "149401", "149402", "149403", "149405", "149406", "149591",
    "149911", "149941", "150015", "150063", "150215", "150216", "150220", "150427",
    "150428", "151103", "151283", "152339", "152396", "152413", "152456", "152476",
    "152477", "152601", "152873", "152886", "153020", "153035", "153179", "153246",
    "153402", "153403", "153409", "153410", "153411", "153412", "153417", "153438",
    "153440", "153442", "153443", "153474", "153475", "153476", "153478", "153480",
    "153483", "153552", "153554", "153556", "153640", "153786", "153860", "153879",
    "154092", "154310", "154449", "154891", "155039", "155357", "155439", "155440",
    "155441", "155548", "156075", "156195", "156257", "156532", "156536", "156643",
    "156746", "156788", "156822", "157452", "157457", "157459", "157462", "157611",
    "157628", "157638", "157666", "157677", "157759", "157776", "157777", "157778",
    "157779", "157888", "157890", "158380", "158384", "158401", "158415", "158517",
    "158534", "158795", "158797", "158801", "160878", "160880", "160904", "160906",
    "160907", "160908", "160910", "160911", "160912", "160913", "160914", "160915",
    "160917", "160978", "160983", "160984", "160985", "160987", "16111", "161152",
    "161897", "162091", "162851", "16305", "163060", "163061", "163062", "163063",
    "163064", "163065", "163066", "163067", "163068", "163209", "163210", "163612",
    "163633", "163684", "163734", "163751", "163942", "164259", "164549", "164729",
    "164791", "165034", "165267", "165272", "165280", "165284", "165285", "165291",
    "165292", "165294", "165295", "165298", "165299", "165300", "165302", "165303",
    "165304", "165314", "165317", "165319", "165320", "165321", "165322", "165323",
    "165324", "165325", "165450", "165566", "165672", "166076", "166083", "166084",
    "166085", "166220", "166398", "166419", "166584", "167153", "167210", "167212",
    "167605", "167606", "167609", "167611", "167789", "167791", "167792", "167793",
    "167817", "167818", "167819", "167820", "167821", "167900", "167901", "167902",
    "167903", "167904", "167952", "167970", "168080", "168081", "168082", "168092",
    "168094", "168397", "168573", "168584", "168609", "168751", "168752", "168753",
    "168795", "168796", "168876", "168882", "16891", "16892", "16893", "168958",
    "16897", "16900", "169069", "16908", "169080", "169081", "169082", "169083",
    "169084", "169085", "169137", "169144", "169145", "169146", "169151", "169152",
    "16917", "16924", "16925", "169404", "169443", "169596", "169597", "16967",
    "16971", "169752", "16991", "16992", "170063", "170238", "170239", "170393",
    "170415", "170783", "170895", "17094", "170982", "170983", "170984", "170985",
    "170988", "17100", "17105", "171092", "171094", "17120"
]

# ============================================================
# BROWSER SETUP
# ============================================================

def create_browser():
    """Create and configure Chrome browser"""
    
    print("🌐 Setting up Chrome browser...")
    
    options = Options()
    
    # Visible mode (not headless)
    # options.add_argument('--headless')  # Uncomment to run hidden
    
    # Performance and stability
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    # Anti-detection
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # User agent
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.7559.110 Safari/537.36')
    
    # Download ChromeDriver
    print("📥 Downloading/verifying ChromeDriver...")
    service = Service(ChromeDriverManager().install())
    
    # Create browser
    driver = webdriver.Chrome(service=service, options=options)
    
    # Additional anti-detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    print("✅ Browser ready!")
    return driver

# ============================================================
# METADATA EXTRACTION
# ============================================================

def safe_find_text(driver, by, value, default=''):
    """Safely find element and return text"""
    try:
        element = driver.find_element(by, value)
        return element.text.strip()
    except:
        return default

def safe_find_attribute(driver, by, value, attribute, default=''):
    """Safely find element and return attribute"""
    try:
        element = driver.find_element(by, value)
        return element.get_attribute(attribute) or default
    except:
        return default

def extract_from_page_text(page_text, label):
    """Extract value following a label from page text"""
    try:
        lines = page_text.split('\n')
        for i, line in enumerate(lines):
            if label.lower() in line.lower():
                # Return the next non-empty line or same line after label
                if ':' in line:
                    return line.split(':', 1)[1].strip()
                elif i + 1 < len(lines):
                    return lines[i + 1].strip()
        return ''
    except:
        return ''

def parse_title_for_species(title):
    """Parse page title to extract species info"""
    # Title format: "XC101288 Blue-winged Teal (Spatula discors) :: xeno-canto"
    try:
        # Remove XC number and xeno-canto suffix
        title = re.sub(r'^XC\d+\s*', '', title)
        title = re.sub(r'\s*::\s*xeno-canto.*$', '', title, flags=re.IGNORECASE)
        
        # Extract scientific name from parentheses
        scientific_match = re.search(r'\(([^)]+)\)', title)
        scientific_name = scientific_match.group(1) if scientific_match else ''
        
        # English name is before parentheses
        english_name = re.sub(r'\s*\([^)]+\)\s*', '', title).strip()
        
        # Split scientific name
        genus = ''
        species = ''
        subspecies = ''
        if scientific_name:
            parts = scientific_name.split()
            if len(parts) >= 1:
                genus = parts[0]
            if len(parts) >= 2:
                species = parts[1]
            if len(parts) >= 3:
                subspecies = ' '.join(parts[2:])
        
        return english_name, scientific_name, genus, species, subspecies
    except:
        return '', '', '', '', ''

def scrape_recording_page(driver, recording_id):
    """Scrape metadata from a single Xeno-Canto recording page"""
    
    url = f"https://xeno-canto.org/{recording_id}"
    
    # Initialize metadata with defaults
    metadata = {
        'recording_id': f"XC{recording_id}",
        'scientific_name': '',
        'genus': '',
        'species': '',
        'subspecies': '',
        'english_name': '',
        'call_type': '',
        'sex': '',
        'life_stage': '',
        'duration_seconds': '',
        'date': '',
        'time': '',
        'country': '',
        'location': '',
        'latitude': '',
        'longitude': '',
        'altitude': '',
        'quality': '',
        'recordist': '',
        'license': '',
        'remarks': '',
        'background_species': '',
        'status': 'success'
    }
    
    try:
        # Navigate to page
        driver.get(url)
        
        # Wait for page to load (check title doesn't contain anubis)
        max_wait = PAGE_LOAD_TIMEOUT
        start = time.time()
        while time.time() - start < max_wait:
            title = driver.title.lower()
            if 'anubis' not in title and 'xeno-canto' in title:
                break
            time.sleep(0.5)
        
        # Additional wait for content
        time.sleep(1)
        
        # Check if page loaded successfully
        page_title = driver.title
        if 'anubis' in page_title.lower() or not page_title:
            metadata['status'] = 'blocked_by_anubis'
            return metadata
        
        if '404' in page_title or 'not found' in page_title.lower():
            metadata['status'] = 'not_found'
            return metadata
        
        # ===== Extract from Title =====
        english_name, scientific_name, genus, species, subspecies = parse_title_for_species(page_title)
        metadata['english_name'] = english_name
        metadata['scientific_name'] = scientific_name
        metadata['genus'] = genus
        metadata['species'] = species
        metadata['subspecies'] = subspecies
        
        # ===== Get full page text =====
        try:
            body = driver.find_element(By.TAG_NAME, "body")
            page_text = body.text
        except:
            page_text = ''
        
        # ===== Extract Call Type =====
        # Usually appears after species name, like "call" or "song"
        call_type_match = re.search(r'·\s*(song|call|alarm|flight|duet|drumming|display|begging|[a-z\s]+call|[a-z\s]+song)', page_text.lower())
        if call_type_match:
            metadata['call_type'] = call_type_match.group(1).strip()
        
        # ===== Extract from "Basic data" section =====
        # Look for common patterns in page text
        
        # Recordist
        recordist_match = re.search(r'Recordist\s*\n\s*([^\n]+)', page_text)
        if recordist_match:
            metadata['recordist'] = recordist_match.group(1).strip()
        
        # Date
        date_match = re.search(r'Date\s*\n?\s*(\d{4}-\d{2}-\d{2})', page_text)
        if date_match:
            metadata['date'] = date_match.group(1)
        
        # Time
        time_match = re.search(r'Time\s*\n?\s*(\d{1,2}:\d{2})', page_text)
        if time_match:
            metadata['time'] = time_match.group(1)
        
        # Latitude
        lat_match = re.search(r'Latitude\s*\n?\s*([-\d.]+)', page_text)
        if lat_match:
            metadata['latitude'] = lat_match.group(1)
        
        # Longitude
        lng_match = re.search(r'Longitude\s*\n?\s*([-\d.]+)', page_text)
        if lng_match:
            metadata['longitude'] = lng_match.group(1)
        
        # Location
        loc_match = re.search(r'Location\s*\n\s*([^\n]+)', page_text)
        if loc_match:
            metadata['location'] = loc_match.group(1).strip()
        
        # Country - usually at end of location or separate
        country_match = re.search(r'Country\s*\n\s*([^\n]+)', page_text)
        if country_match:
            metadata['country'] = country_match.group(1).strip()
        elif metadata['location']:
            # Try to extract from location (last part after comma)
            parts = metadata['location'].split(',')
            if len(parts) > 1:
                metadata['country'] = parts[-1].strip()
        
        # Elevation/Altitude
        alt_match = re.search(r'(?:Elevation|Altitude)\s*\n?\s*(\d+)\s*m', page_text)
        if alt_match:
            metadata['altitude'] = alt_match.group(1)
        
        # ===== Quality Rating =====
        # Look for quality rating pattern (A, B, C, D, E)
        quality_match = re.search(r'Rating[^\n]*\n[^\n]*([ABCDE])\s', page_text)
        if quality_match:
            metadata['quality'] = quality_match.group(1)
        else:
            # Alternative pattern
            quality_match2 = re.search(r'quality[:\s]+([ABCDE])', page_text, re.IGNORECASE)
            if quality_match2:
                metadata['quality'] = quality_match2.group(1).upper()
        
        # ===== License =====
        license_match = re.search(r'(Creative Commons[^\n]+)', page_text)
        if license_match:
            metadata['license'] = license_match.group(1).strip()
        
        # ===== Remarks =====
        remarks_match = re.search(r'Remarks from the Recordist\s*\n\s*([^\n]+(?:\n[^\n]+)*?)(?=\nLocation|\nRating|\nCitation|$)', page_text)
        if remarks_match:
            metadata['remarks'] = remarks_match.group(1).strip()[:500]  # Limit length
        
        # ===== Duration =====
        # Look for time format like "0:11" in the player area
        duration_match = re.search(r'(\d+):(\d{2})\s*$', page_text[:500], re.MULTILINE)
        if duration_match:
            minutes = int(duration_match.group(1))
            seconds = int(duration_match.group(2))
            metadata['duration_seconds'] = str(minutes * 60 + seconds)
        
        # ===== Sex and Life Stage =====
        if 'male' in page_text.lower():
            if 'female' in page_text.lower():
                metadata['sex'] = 'male, female'
            else:
                metadata['sex'] = 'male'
        elif 'female' in page_text.lower():
            metadata['sex'] = 'female'
        
        if 'juvenile' in page_text.lower():
            metadata['life_stage'] = 'juvenile'
        elif 'immature' in page_text.lower():
            metadata['life_stage'] = 'immature'
        elif 'adult' in page_text.lower():
            metadata['life_stage'] = 'adult'
        
        # ===== Background Species =====
        bg_match = re.search(r'(?:Background|Also recorded)[:\s]*([^\n]+)', page_text, re.IGNORECASE)
        if bg_match:
            metadata['background_species'] = bg_match.group(1).strip()[:200]
        
        return metadata
        
    except Exception as e:
        metadata['status'] = f'error: {str(e)[:100]}'
        return metadata

# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def load_checkpoint():
    """Load existing checkpoint if available"""
    if os.path.exists(CHECKPOINT_CSV):
        try:
            df = pd.read_csv(CHECKPOINT_CSV)
            processed_ids = set(df['recording_id'].str.replace('XC', '').tolist())
            print(f"📂 Loaded checkpoint: {len(processed_ids)} recordings already processed")
            return df.to_dict('records'), processed_ids
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
            return [], set()
    return [], set()

def save_checkpoint(metadata_list):
    """Save current progress to checkpoint file"""
    try:
        df = pd.DataFrame(metadata_list)
        df.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"⚠️ Could not save checkpoint: {e}")

def log_message(message):
    """Write message to log file"""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    except:
        pass

# ============================================================
# LOCAL FILE CHECKING
# ============================================================

def add_local_file_info(metadata_list):
    """Add local file path and existence info"""
    for item in metadata_list:
        recording_id = item.get('recording_id', '')
        filename = f"{recording_id}.wav"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        item['local_file_path'] = filepath
        item['file_exists'] = os.path.exists(filepath)
    return metadata_list

# ============================================================
# SUMMARY REPORT
# ============================================================

def generate_summary_report(metadata_df):
    """Generate a summary report of the metadata"""
    
    print("\n" + "="*60)
    print("📊 METADATA SUMMARY REPORT - BATCH 1")
    print("="*60)
    
    total = len(metadata_df)
    successful = len(metadata_df[metadata_df['status'] == 'success'])
    failed = total - successful
    
    print(f"\n📁 Total recordings processed: {total}")
    print(f"✅ Successfully retrieved: {successful} ({100*successful/total:.1f}%)")
    print(f"❌ Failed/Not found: {failed} ({100*failed/total:.1f}%)")
    
    if successful > 0:
        success_df = metadata_df[metadata_df['status'] == 'success']
        
        # Species count
        if 'scientific_name' in success_df.columns:
            species_counts = success_df['scientific_name'].value_counts()
            unique_species = len(species_counts[species_counts.index != ''])
            print(f"\n🐦 Unique species found: {unique_species}")
            
            if unique_species > 0:
                print(f"\n📋 Top 15 species by recording count:")
                print("-"*45)
                for i, (sp, count) in enumerate(species_counts.head(15).items(), 1):
                    if sp:
                        print(f"   {i:2d}. {sp}: {count}")
        
        # Country distribution
        if 'country' in success_df.columns:
            country_counts = success_df['country'].value_counts()
            unique_countries = len(country_counts[country_counts.index != ''])
            print(f"\n🌍 Countries represented: {unique_countries}")
            
            if unique_countries > 0:
                print(f"\n🗺️ Top 10 countries:")
                print("-"*45)
                for country, count in country_counts.head(10).items():
                    if country:
                        print(f"   {country}: {count}")
        
        # Quality distribution
        if 'quality' in success_df.columns:
            quality_counts = success_df['quality'].value_counts()
            print(f"\n⭐ Quality distribution:")
            print("-"*45)
            for quality, count in sorted(quality_counts.items()):
                if quality:
                    print(f"   Quality {quality}: {count}")
        
        # Call type distribution
        if 'call_type' in success_df.columns:
            call_types = success_df['call_type'].value_counts()
            print(f"\n🎵 Call types (top 10):")
            print("-"*45)
            for call_type, count in call_types.head(10).items():
                if call_type:
                    print(f"   {call_type}: {count}")
    
    # Local files
    if 'file_exists' in metadata_df.columns:
        existing = metadata_df['file_exists'].sum()
        print(f"\n💾 Local WAV files found: {existing}/{total} ({100*existing/total:.1f}%)")
    
    # Failed recordings
    if failed > 0:
        print(f"\n⚠️ Failed recordings status breakdown:")
        print("-"*45)
        failed_df = metadata_df[metadata_df['status'] != 'success']
        status_counts = failed_df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
    
    print("\n" + "="*60)

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("🐦 BIRD AUDIO METADATA RETRIEVAL - SELENIUM")
    print("   Phase 0: Xeno-Canto Metadata Construction")
    print("   Batch 1: First 1000 Recording IDs")
    print("="*60)
    print(f"\n📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Audio folder: {AUDIO_FOLDER}")
    print(f"📄 Output CSV: {OUTPUT_CSV}")
    print(f"⏱️ Request delay: {REQUEST_DELAY} seconds")
    print(f"💾 Checkpoint every: {CHECKPOINT_EVERY} recordings")
    
    log_message("="*50)
    log_message("Starting Batch 1 metadata scraping")
    
    # Check if audio folder exists
    if not os.path.exists(AUDIO_FOLDER):
        print(f"\n⚠️ Warning: Audio folder does not exist: {AUDIO_FOLDER}")
        print("   Local file checking will show all files as not found.")
    
    # Load checkpoint if exists
    all_metadata, processed_ids = load_checkpoint()
    
    # Filter out already processed IDs
    ids_to_process = [rid for rid in RECORDING_IDS if rid not in processed_ids]
    
    if len(ids_to_process) == 0:
        print("\n✅ All recordings already processed! Loading from checkpoint...")
        metadata_df = pd.DataFrame(all_metadata)
    else:
        print(f"\n🔄 Recordings to process: {len(ids_to_process)}")
        if len(processed_ids) > 0:
            print(f"   (Resuming from checkpoint, {len(processed_ids)} already done)")
        
        estimated_time = len(ids_to_process) * (REQUEST_DELAY + 1) / 60
        print(f"⏳ Estimated time: ~{estimated_time:.0f} minutes\n")
        
        input("Press ENTER to start scraping (browser will open)...")
        
        # Create browser
        driver = None
        try:
            driver = create_browser()
            
            # Process recordings with progress bar
            for i, rec_id in enumerate(tqdm(ids_to_process, desc="Scraping metadata")):
                try:
                    # Scrape metadata
                    metadata = scrape_recording_page(driver, rec_id)
                    all_metadata.append(metadata)
                    
                    # Log progress
                    status = metadata.get('status', 'unknown')
                    if status != 'success':
                        log_message(f"XC{rec_id}: {status}")
                    
                    # Save checkpoint periodically
                    if len(all_metadata) % CHECKPOINT_EVERY == 0:
                        save_checkpoint(all_metadata)
                        tqdm.write(f"   💾 Checkpoint saved: {len(all_metadata)} recordings")
                        log_message(f"Checkpoint saved: {len(all_metadata)} recordings")
                    
                    # Respect rate limiting
                    time.sleep(REQUEST_DELAY)
                    
                except Exception as e:
                    error_msg = f"Error processing XC{rec_id}: {str(e)[:100]}"
                    tqdm.write(f"   ⚠️ {error_msg}")
                    log_message(error_msg)
                    all_metadata.append({
                        'recording_id': f"XC{rec_id}",
                        'status': f'error: {str(e)[:100]}'
                    })
                    time.sleep(REQUEST_DELAY)
            
            # Final checkpoint save
            save_checkpoint(all_metadata)
            
        except Exception as e:
            print(f"\n❌ Critical error: {e}")
            log_message(f"Critical error: {e}")
            save_checkpoint(all_metadata)
            
        finally:
            # Close browser
            if driver:
                print("\n🔒 Closing browser...")
                try:
                    driver.quit()
                except:
                    pass
    
    # Add local file info
    print("\n🔍 Checking local file existence...")
    all_metadata = add_local_file_info(all_metadata)
    
    # Convert to DataFrame
    metadata_df = pd.DataFrame(all_metadata)
    
    # Ensure all columns are present
    expected_columns = [
        'recording_id', 'scientific_name', 'genus', 'species', 'subspecies',
        'english_name', 'call_type', 'sex', 'life_stage', 'duration_seconds',
        'date', 'time', 'country', 'location', 'latitude', 'longitude',
        'altitude', 'quality', 'recordist', 'license', 'remarks',
        'background_species', 'local_file_path', 'file_exists', 'status'
    ]
    
    for col in expected_columns:
        if col not in metadata_df.columns:
            metadata_df[col] = ''
    
    # Reorder columns
    metadata_df = metadata_df[expected_columns]
    
    # Save final CSV
    metadata_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ Metadata saved to: {OUTPUT_CSV}")
    log_message(f"Metadata saved to: {OUTPUT_CSV}")
    
    # Generate summary report
    generate_summary_report(metadata_df)
    
    # Save summary to text file
    summary_file = OUTPUT_CSV.replace('.csv', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("METADATA SUMMARY REPORT - BATCH 1\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total recordings: {len(metadata_df)}\n")
        f.write(f"Successfully retrieved: {len(metadata_df[metadata_df['status'] == 'success'])}\n")
        if 'scientific_name' in metadata_df.columns:
            unique_species = metadata_df[metadata_df['status'] == 'success']['scientific_name'].nunique()
            f.write(f"Unique species: {unique_species}\n")
        if 'file_exists' in metadata_df.columns:
            f.write(f"Local files found: {metadata_df['file_exists'].sum()}\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*60)
    print("✅ BATCH 1 COMPLETE!")
    print("="*60)
    
    return metadata_df

# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    result_df = main()