# Save as: 08_Deployment/pages/5_⚙️_Settings.py

"""
⚙️ Settings Page
================
User preferences and application settings
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import *

# Page config
st.set_page_config(page_title="Settings | Bird Detector", page_icon="⚙️", layout="wide")

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

default_settings = {
    'theme': 'Light',
    'default_confidence_threshold': 0.5,
    'default_top_k': 5,
    'auto_noise_reduction': False,
    'show_scientific_names': True,
    'chunk_duration': 5.0,
    'auto_save_history': True,
    'spectrogram_colormap': 'magma',
    'audio_sample_rate': 22050,
}

if 'settings' not in st.session_state:
    st.session_state.settings = default_settings.copy()

# ============================================================
# PAGE HEADER
# ============================================================

st.title("⚙️ Settings")
st.markdown("Customize your Bird Detector experience.")

# ============================================================
# APPEARANCE SETTINGS
# ============================================================

st.markdown("## 🎨 Appearance")

col1, col2 = st.columns(2)

with col1:
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark", "Auto"],
        index=["Light", "Dark", "Auto"].index(st.session_state.settings.get('theme', 'Light')),
        help="Choose your preferred color theme"
    )
    st.session_state.settings['theme'] = theme
    
    if theme == "Dark":
        st.info("🌙 Dark mode enabled. Note: Full dark mode requires Streamlit config changes.")

with col2:
    spectrogram_colormap = st.selectbox(
        "Spectrogram Colormap",
        ["magma", "viridis", "plasma", "inferno", "cividis", "gray"],
        index=["magma", "viridis", "plasma", "inferno", "cividis", "gray"].index(
            st.session_state.settings.get('spectrogram_colormap', 'magma')
        ),
        help="Color scheme for spectrogram visualization"
    )
    st.session_state.settings['spectrogram_colormap'] = spectrogram_colormap

st.markdown("---")

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

st.markdown("## 🔍 Analysis Defaults")

col1, col2 = st.columns(2)

with col1:
    default_confidence = st.slider(
        "Default Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings.get('default_confidence_threshold', 0.5),
        step=0.05,
        help="Default minimum confidence for displaying predictions"
    )
    st.session_state.settings['default_confidence_threshold'] = default_confidence
    
    default_top_k = st.slider(
        "Default Number of Predictions",
        min_value=1,
        max_value=10,
        value=st.session_state.settings.get('default_top_k', 5),
        help="Default number of top predictions to show"
    )
    st.session_state.settings['default_top_k'] = default_top_k

with col2:
    chunk_duration = st.select_slider(
        "Audio Chunk Duration",
        options=[3.0, 4.0, 5.0, 6.0, 7.0],
        value=st.session_state.settings.get('chunk_duration', 5.0),
        help="Duration of audio segments for analysis"
    )
    st.session_state.settings['chunk_duration'] = chunk_duration
    
    sample_rate = st.selectbox(
        "Audio Sample Rate",
        [16000, 22050, 44100],
        index=[16000, 22050, 44100].index(
            st.session_state.settings.get('audio_sample_rate', 22050)
        ),
        help="Sample rate for audio processing"
    )
    st.session_state.settings['audio_sample_rate'] = sample_rate

st.markdown("---")

# ============================================================
# FEATURE TOGGLES
# ============================================================

st.markdown("## 🔧 Features")

col1, col2 = st.columns(2)

with col1:
    auto_noise = st.toggle(
        "Auto Noise Reduction",
        value=st.session_state.settings.get('auto_noise_reduction', False),
        help="Automatically apply noise reduction to uploaded files"
    )
    st.session_state.settings['auto_noise_reduction'] = auto_noise
    
    show_scientific = st.toggle(
        "Show Scientific Names",
        value=st.session_state.settings.get('show_scientific_names', True),
        help="Display scientific names alongside common names"
    )
    st.session_state.settings['show_scientific_names'] = show_scientific

with col2:
    auto_save = st.toggle(
        "Auto-Save History",
        value=st.session_state.settings.get('auto_save_history', True),
        help="Automatically save analysis results to history"
    )
    st.session_state.settings['auto_save_history'] = auto_save
    
    show_uncertainty = st.toggle(
        "Show Uncertainty Warnings",
        value=st.session_state.settings.get('show_uncertainty', True),
        help="Display warnings for low-confidence predictions"
    )
    st.session_state.settings['show_uncertainty'] = show_uncertainty

st.markdown("---")

# ============================================================
# DATA MANAGEMENT
# ============================================================

st.markdown("## 💾 Data Management")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📋 History")
    history_count = len(st.session_state.get('analysis_history', []))
    st.metric("Saved Analyses", history_count)
    
    if st.button("🗑️ Clear History"):
        st.session_state.analysis_history = []
        st.success("History cleared!")
        st.rerun()

with col2:
    st.markdown("### 📤 Export Settings")
    
    settings_json = json.dumps(st.session_state.settings, indent=2)
    st.download_button(
        "📥 Export Settings",
        settings_json,
        "bird_detector_settings.json",
        "application/json"
    )

with col3:
    st.markdown("### 📥 Import Settings")
    
    uploaded_settings = st.file_uploader(
        "Upload settings file",
        type=['json'],
        key="settings_upload"
    )
    
    if uploaded_settings:
        try:
            imported_settings = json.load(uploaded_settings)
            st.session_state.settings.update(imported_settings)
            st.success("Settings imported!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to import settings: {e}")

st.markdown("---")

# ============================================================
# RESET SETTINGS
# ============================================================

st.markdown("## 🔄 Reset")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Reset to Defaults", type="secondary"):
        st.session_state.settings = default_settings.copy()
        st.success("Settings reset to defaults!")
        st.rerun()

with col2:
    if st.button("🧹 Clear All Data", type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            if key != 'settings':
                del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()

st.markdown("---")

# ============================================================
# ABOUT / HELP
# ============================================================

st.markdown("## ℹ️ About")

with st.expander("📖 About Bird Detector"):
    st.markdown("""
    ### 🐦 Bird Species Detection System
    
    A confidence-aware, explainable multi-species bird detection system using bioacoustic signals and deep learning.
    
    **Model:**
    - Architecture: EfficientNet-B2
    - Accuracy: 96.06% (test set)
    - Species: 87 North American birds
    - Training Data: 180,751 spectrograms
    
    **Data Source:**
    - [Xeno-Canto](https://xeno-canto.org) - Bird sound recordings
    
    **Built With:**
    - PyTorch
    - Streamlit
    - librosa
    - timm
    """)

with st.expander("❓ Help & FAQ"):
    st.markdown("""
    ### Frequently Asked Questions
    
    **Q: What audio formats are supported?**
    A: WAV, MP3, FLAC, OGG, and M4A files are supported.
    
    **Q: How long can my recordings be?**
    A: We recommend recordings between 5 seconds and 10 minutes for best results.
    
    **Q: Why is my confidence low?**
    A: Low confidence can occur when:
    - The recording has background noise
    - Multiple species are vocalizing
    - The species isn't in our database
    - The recording quality is poor
    
    **Q: How can I improve accuracy?**
    A: Try these tips:
    - Use high-quality recordings
    - Enable noise reduction
    - Record in quiet environments
    - Focus on one bird at a time
    
    **Q: Is my data private?**
    A: Yes! All processing happens locally in your browser session. We don't store your audio files.
    """)

with st.expander("📧 Contact & Feedback"):
    st.markdown("""
    ### Get in Touch
    
    - **GitHub:** [github.com/yourusername/bird-detector](https://github.com)
    - **Issues:** Report bugs or request features on GitHub
    - **Email:** your.email@example.com
    
    We appreciate your feedback to improve the system!
    """)

# ============================================================
# CURRENT SETTINGS DISPLAY
# ============================================================

with st.expander("🔧 Current Settings (JSON)"):
    st.json(st.session_state.settings)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    ⚙️ Settings are saved in your browser session
</div>
""", unsafe_allow_html=True)