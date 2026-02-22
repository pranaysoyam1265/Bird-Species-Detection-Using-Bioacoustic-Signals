# Save as: 08_Deployment/app.py

"""
🐦 Bird Species Detection System
================================
A confidence-aware, explainable multi-species bird detection system
using bioacoustic signals and deep learning.

Model: EfficientNet-B2 (96.06% accuracy on 87 species)
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="🐦 Bird Species Detector",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/bird-detector',
        'Report a bug': 'https://github.com/yourusername/bird-detector/issues',
        'About': """
        ## 🐦 Bird Species Detection System
        
        A deep learning-powered tool for identifying bird species from audio recordings.
        
        **Model Performance:**
        - 96.06% Test Accuracy
        - 98.74% Top-5 Accuracy
        - 87 North American Species
        
        Built with ❤️ using PyTorch and Streamlit
        """
    }
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    .stTitle {
        color: #1E88E5;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Species cards */
    .species-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .species-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Confidence bars */
    .confidence-bar {
        height: 24px;
        border-radius: 12px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        transition: width 0.5s ease;
    }
    
    .confidence-low {
        background: linear-gradient(90deg, #f44336, #ff5722);
    }
    
    .confidence-medium {
        background: linear-gradient(90deg, #ff9800, #ffc107);
    }
    
    .confidence-high {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success/Error messages */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
    
    /* Audio player */
    audio {
        width: 100%;
        border-radius: 30px;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #f8f9fa;
        padding: 0.5rem;
        text-align: center;
        font-size: 0.8rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================================
# LANDING PAGE
# ============================================================

def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0;">🐦 Bird Species Detector</h1>
            <p style="font-size: 1.2rem; color: #666; margin-top: 0.5rem;">
                AI-Powered Bioacoustic Identification System
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">96.1%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <div class="metric-value">87</div>
            <div class="metric-label">Bird Species</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);">
            <div class="metric-value">98.7%</div>
            <div class="metric-label">Top-5 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%);">
            <div class="metric-value">180K+</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("## 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Upload Audio
        Upload a bird recording in WAV, MP3, or FLAC format. 
        We support recordings up to 10 minutes long.
        """)
        
    with col2:
        st.markdown("""
        ### 2️⃣ AI Analysis
        Our deep learning model analyzes the audio,
        detecting and classifying bird vocalizations.
        """)
        
    with col3:
        st.markdown("""
        ### 3️⃣ Get Results
        View predictions with confidence scores,
        spectrograms, and species information.
        """)
    
    # CTA Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🎵 Start Analyzing →", type="primary", use_container_width=True):
            st.switch_page("pages/1_🎵_Analyze.py")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features section
    st.markdown("---")
    st.markdown("## ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 High Accuracy Detection
        - **96.06%** overall test accuracy
        - **98.74%** top-5 accuracy
        - Real-time processing
        
        #### 📊 Confidence Visualization
        - Per-prediction confidence scores
        - Uncertainty quantification
        - Similar species warnings
        
        #### 🔊 Audio Processing
        - Noise reduction
        - Multiple format support
        - Batch processing (up to 50 files)
        """)
        
    with col2:
        st.markdown("""
        #### 🔬 Explainability
        - Mel-spectrogram visualization
        - Temporal detection timeline
        - Model attention heatmaps
        
        #### 📱 User Experience
        - Mobile-friendly design
        - Dark/Light mode
        - Export to PDF/CSV
        
        #### 🐦 Species Information
        - 87 North American species
        - Photos and descriptions
        - Similar species comparison
        """)
    
    # Sample species
    st.markdown("---")
    st.markdown("## 🐦 Sample Species We Can Identify")
    
    sample_species = [
        ("Northern Cardinal", "100%", "🔴"),
        ("American Robin", "97.2%", "🟠"),
        ("Blue Jay", "98.1%", "🔵"),
        ("Bald Eagle", "100%", "🦅"),
        ("Barred Owl", "93.6%", "🦉"),
        ("Common Yellowthroat", "100%", "💛"),
    ]
    
    cols = st.columns(6)
    for i, (name, acc, emoji) in enumerate(sample_species):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                <div style="font-size: 2rem;">{emoji}</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">{name}</div>
                <div style="color: #28a745; font-weight: bold;">{acc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using PyTorch, Streamlit, and EfficientNet</p>
        <p style="font-size: 0.8rem;">
            Data source: <a href="https://xeno-canto.org" target="_blank">Xeno-Canto</a> | 
            Model: EfficientNet-B2 | 
            Training: 180,751 spectrograms
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()