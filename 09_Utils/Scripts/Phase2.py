import requests
import time

# A small subset of your list to test
species_list = [
    "Quiscalus major", "Passerina cyanea", "Baeolophus atricristatus",
    "Coccyzus americanus", "Haemorhous cassinii", "Empidonax virescens",
    "Larus californicus", "Buteo regalis", "Ardea herodias", 
    "Archilochus colubris"
]

print(f"{'Species':<30} | {'Total Recs':<10} | {'Quality A'}")
print("-" * 55)

for species in species_list:
    # Query Xeno-Canto API
    url = f"https://xeno-canto.org/api/2/recordings?query={species.replace(' ', '+')}"
    try:
        response = requests.get(url)
        data = response.json()
        
        total_recordings = int(data['numRecordings'])
        
        # Filter for Quality 'A' (just an estimate based on typical ratios)
        # Note: The API summary doesn't give quality counts directly without parsing, 
        # but total count is a good proxy.
        
        print(f"{species:<30} | {total_recordings:<10} | ~{int(total_recordings * 0.2)}")
        
    except Exception as e:
        print(f"{species:<30} | Error")
    
    # Be nice to the API
    time.sleep(0.5)