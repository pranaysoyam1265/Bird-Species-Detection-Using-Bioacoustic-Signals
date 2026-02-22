# Save as: 08_Deployment/pages/4_🐦_Species_Browser.py

"""
🐦 Species Browser Page
=======================
Browse all 87 bird species with information and examples
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import *

# Page config
st.set_page_config(page_title="Species Browser | Bird Detector", page_icon="🐦", layout="wide")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_species_data():
    """Load all species information"""
    
    # Load label mapping
    with open(LABEL_MAPPING_PATH, 'r') as f:
        label_mapping = json.load(f)
    
    # Load test results for accuracy info
    try:
        with open(TEST_RESULTS_PATH, 'r') as f:
            test_results = json.load(f)
        per_class_accuracy = test_results.get('per_class_accuracy', {})
        per_class_samples = test_results.get('per_class_samples', {})
    except:
        per_class_accuracy = {}
        per_class_samples = {}
    
    # Build species list
    species_list = []
    for scientific_name, info in label_mapping.items():
        species_list.append({
            'index': info['index'],
            'scientific_name': scientific_name,
            'english_name': info['english_name'],
            'accuracy': per_class_accuracy.get(scientific_name, 0),
            'samples': per_class_samples.get(scientific_name, 0),
        })
    
    return sorted(species_list, key=lambda x: x['english_name'])

species_data = load_species_data()

# ============================================================
# PAGE HEADER
# ============================================================

st.title("🐦 Species Browser")
st.markdown(f"Explore all **{len(species_data)}** bird species our model can identify.")

# ============================================================
# FILTERS & SEARCH
# ============================================================

st.markdown("### 🔍 Search & Filter")

col1, col2, col3 = st.columns(3)

with col1:
    search_query = st.text_input(
        "Search species",
        placeholder="Type species name...",
        help="Search by common or scientific name"
    )

with col2:
    accuracy_filter = st.select_slider(
        "Minimum accuracy",
        options=[0, 50, 60, 70, 80, 85, 90, 95],
        value=0,
        help="Filter species by model accuracy"
    )

with col3:
    sort_by = st.selectbox(
        "Sort by",
        ["Name (A-Z)", "Name (Z-A)", "Accuracy (High-Low)", "Accuracy (Low-High)", "Samples (High-Low)"]
    )

# Apply filters
filtered_species = species_data.copy()

if search_query:
    search_lower = search_query.lower()
    filtered_species = [
        s for s in filtered_species
        if search_lower in s['english_name'].lower() or search_lower in s['scientific_name'].lower()
    ]

if accuracy_filter > 0:
    filtered_species = [s for s in filtered_species if s['accuracy'] >= accuracy_filter]

# Apply sorting
if sort_by == "Name (A-Z)":
    filtered_species.sort(key=lambda x: x['english_name'])
elif sort_by == "Name (Z-A)":
    filtered_species.sort(key=lambda x: x['english_name'], reverse=True)
elif sort_by == "Accuracy (High-Low)":
    filtered_species.sort(key=lambda x: x['accuracy'], reverse=True)
elif sort_by == "Accuracy (Low-High)":
    filtered_species.sort(key=lambda x: x['accuracy'])
elif sort_by == "Samples (High-Low)":
    filtered_species.sort(key=lambda x: x['samples'], reverse=True)

st.markdown(f"Showing **{len(filtered_species)}** species")

st.markdown("---")

# ============================================================
# SPECIES GRID
# ============================================================

def get_accuracy_badge(accuracy):
    """Get colored badge based on accuracy"""
    if accuracy >= 95:
        return "🟢", "#28a745"
    elif accuracy >= 90:
        return "🟡", "#ffc107"
    elif accuracy >= 85:
        return "🟠", "#fd7e14"
    else:
        return "🔴", "#dc3545"

def get_species_emoji(english_name):
    """Get emoji based on species type"""
    name_lower = english_name.lower()
    
    if 'owl' in name_lower:
        return "🦉"
    elif 'eagle' in name_lower or 'hawk' in name_lower:
        return "🦅"
    elif 'duck' in name_lower or 'teal' in name_lower or 'mallard' in name_lower:
        return "🦆"
    elif 'hummingbird' in name_lower:
        return "🌺"
    elif 'woodpecker' in name_lower or 'flicker' in name_lower:
        return "🪶"
    elif 'crow' in name_lower or 'raven' in name_lower:
        return "⬛"
    elif 'jay' in name_lower or 'magpie' in name_lower:
        return "💙"
    elif 'cardinal' in name_lower:
        return "❤️"
    elif 'sparrow' in name_lower or 'finch' in name_lower:
        return "🐦"
    elif 'warbler' in name_lower:
        return "💛"
    elif 'thrush' in name_lower or 'robin' in name_lower:
        return "🧡"
    else:
        return "🐦"

# Display as grid
cols_per_row = 3
rows = [filtered_species[i:i+cols_per_row] for i in range(0, len(filtered_species), cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    
    for i, species in enumerate(row):
        with cols[i]:
            emoji = get_species_emoji(species['english_name'])
            badge, badge_color = get_accuracy_badge(species['accuracy'])
            
            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: transform 0.2s, box-shadow 0.2s;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 2rem; margin-right: 0.5rem;">{emoji}</span>
                    <div>
                        <h4 style="margin: 0; color: #333;">{species['english_name']}</h4>
                        <p style="margin: 0; color: #666; font-style: italic; font-size: 0.9rem;">{species['scientific_name']}</p>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div>
                        <span style="color: {badge_color}; font-weight: bold;">{badge} {species['accuracy']:.1f}%</span>
                        <span style="color: #888; font-size: 0.8rem;"> accuracy</span>
                    </div>
                    <div style="color: #888; font-size: 0.9rem;">
                        {species['samples']} samples
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Expandable details
            with st.expander("View Details"):
                st.markdown(f"""
                **Common Name:** {species['english_name']}
                
                **Scientific Name:** *{species['scientific_name']}*
                
                **Model Accuracy:** {species['accuracy']:.1f}%
                
                **Training Samples:** {species['samples']}
                
                **Class Index:** {species['index']}
                """)
                
                # Links
                xc_url = f"https://xeno-canto.org/species/{species['scientific_name'].replace(' ', '-')}"
                ebird_url = f"https://ebird.org/species/{species['scientific_name'].replace(' ', '').lower()[:6]}"
                
                st.markdown(f"""
                🔗 **External Links:**
                - [Xeno-Canto]({xc_url}) - Listen to recordings
                - [eBird](https://ebird.org) - Distribution & sightings
                - [Wikipedia](https://en.wikipedia.org/wiki/{species['scientific_name'].replace(' ', '_')}) - More info
                """)

# ============================================================
# STATISTICS SECTION
# ============================================================

st.markdown("---")
st.markdown("## 📊 Species Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Accuracy Distribution")
    
    # Create histogram
    import plotly.express as px
    
    accuracy_values = [s['accuracy'] for s in species_data]
    
    fig = px.histogram(
        x=accuracy_values,
        nbins=20,
        labels={'x': 'Accuracy (%)', 'y': 'Number of Species'},
        title='Model Accuracy Distribution Across Species'
    )
    fig.add_vline(x=85, line_dash="dash", line_color="red", annotation_text="Target (85%)")
    fig.add_vline(x=sum(accuracy_values)/len(accuracy_values), line_dash="dash", 
                  line_color="blue", annotation_text=f"Mean ({sum(accuracy_values)/len(accuracy_values):.1f}%)")
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📈 Top & Bottom Performers")
    
    # Top 5
    st.markdown("**Top 5 Species (Highest Accuracy)**")
    top_5 = sorted(species_data, key=lambda x: x['accuracy'], reverse=True)[:5]
    for s in top_5:
        st.markdown(f"- 🟢 **{s['english_name']}**: {s['accuracy']:.1f}%")
    
    st.markdown("")
    
    # Bottom 5
    st.markdown("**Bottom 5 Species (Needs Improvement)**")
    bottom_5 = sorted(species_data, key=lambda x: x['accuracy'])[:5]
    for s in bottom_5:
        badge, _ = get_accuracy_badge(s['accuracy'])
        st.markdown(f"- {badge} **{s['english_name']}**: {s['accuracy']:.1f}%")

# ============================================================
# DOWNLOAD SPECIES LIST
# ============================================================

st.markdown("---")
st.markdown("### 📥 Download Species List")

# Create DataFrame
species_df = pd.DataFrame(species_data)
species_df = species_df[['index', 'english_name', 'scientific_name', 'accuracy', 'samples']]
species_df.columns = ['Index', 'Common Name', 'Scientific Name', 'Accuracy (%)', 'Training Samples']

csv = species_df.to_csv(index=False)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.download_button(
        "📥 Download CSV",
        csv,
        "bird_species_list.csv",
        "text/csv"
    )

with col2:
    json_data = json.dumps(species_data, indent=2)
    st.download_button(
        "📥 Download JSON",
        json_data,
        "bird_species_list.json",
        "application/json"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🐦 Species data from Xeno-Canto | Model trained on {0} recordings
</div>
""".format(sum(s['samples'] for s in species_data)), unsafe_allow_html=True)