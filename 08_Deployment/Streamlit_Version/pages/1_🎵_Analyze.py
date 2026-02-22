# Save as: 08_Deployment/pages/1_🎵_Analyze.py

"""
🎵 Analyze Page - Main Detection Interface
==========================================
Unified interface for:
- General bird species detection
- Targeted species search
- Audio analysis and visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import time
import plotly.graph_objects as go

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import *
from utils.model import load_model, load_label_mapping, predict_single, calculate_uncertainty
from utils.audio import (load_uploaded_audio, process_audio_to_spectrograms, 
                          analyze_audio_quality, reduce_noise)
from utils.visualization import (plot_spectrogram_plotly, plot_top_predictions,
                                  plot_confidence_gauge, plot_detection_timeline,
                                  plot_waveform, get_confidence_color)

# Page config
st.set_page_config(page_title="Analyze | Bird Detector", page_icon="🎵", layout="wide")

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def initialize_model():
    """Load model and label mapping"""
    model, metadata = load_model(MODEL_PATH)
    label_mapping = load_label_mapping(LABEL_MAPPING_PATH)
    return model, metadata, label_mapping

with st.spinner("Loading AI model..."):
    model, model_metadata, label_mapping = initialize_model()

# Create species lookup lists
all_species_list = []
for scientific, info in label_mapping['raw'].items():
    all_species_list.append({
        'idx': info['index'],
        'scientific': scientific,
        'english': info['english_name'],
        'display': f"{info['english_name']} ({scientific})"
    })
all_species_list = sorted(all_species_list, key=lambda x: x['english'])
all_species_english = [s['english'] for s in all_species_list]

# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎵 Analyze Bird Audio")

# Model status in a compact format
status_cols = st.columns([2, 1, 1, 1])
with status_cols[0]:
    st.markdown("Upload a recording to identify bird species or search for a specific bird.")
with status_cols[1]:
    st.caption(f"✅ Model loaded")
with status_cols[2]:
    st.caption(f"📊 {model_metadata.get('val_acc', 0):.1f}% accuracy")
with status_cols[3]:
    st.caption(f"🐦 {len(label_mapping['idx_to_species'])} species")

st.markdown("---")

# ============================================================
# SIDEBAR - ALL SETTINGS
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Analysis mode
    st.markdown("### 🎯 Analysis Mode")
    analysis_mode = st.radio(
        "What would you like to do?",
        ["🔍 Detect All Species", "🎯 Search for Specific Bird"],
        help="Choose detection mode"
    )
    
    # Species search (only show when in search mode)
    target_species = None
    if analysis_mode == "🎯 Search for Specific Bird":
        st.markdown("### 🐦 Target Species")
        
        search_query = st.text_input(
            "Search species:",
            placeholder="Type name...",
            help="Start typing to filter"
        )
        
        # Filter species based on search
        if search_query:
            filtered_species = [s for s in all_species_list 
                               if search_query.lower() in s['english'].lower() 
                               or search_query.lower() in s['scientific'].lower()]
        else:
            filtered_species = all_species_list
        
        # Species selector
        selected_display = st.selectbox(
            "Select species:",
            options=[""] + [s['display'] for s in filtered_species],
            format_func=lambda x: x if x else "Choose a species..."
        )
        
        if selected_display:
            target_species = next((s for s in all_species_list if s['display'] == selected_display), None)
            
            if target_species:
                st.success(f"🎯 Searching for: **{target_species['english']}**")
        
        # Search sensitivity
        st.markdown("### 🎚️ Search Sensitivity")
        search_threshold = st.slider(
            "Detection threshold",
            min_value=0.01,
            max_value=0.30,
            value=0.05,
            step=0.01,
            format="%.0f%%",
            help="Lower = more sensitive (may have false positives)"
        )
    
    st.markdown("---")
    
    # Audio settings
    st.markdown("### 🔊 Audio Processing")
    
    apply_noise_reduction = st.checkbox(
        "Noise Reduction",
        value=False,
        help="Apply noise reduction (good for noisy recordings)"
    )
    
    chunk_duration = st.select_slider(
        "Chunk Duration",
        options=[3.0, 4.0, 5.0, 6.0, 7.0],
        value=5.0,
        format_func=lambda x: f"{x}s"
    )
    
    st.markdown("---")
    
    # Display settings
    st.markdown("### 📊 Display Options")
    
    confidence_threshold = st.slider(
        "Min. Confidence",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        format="%.0f%%"
    )
    
    top_k = st.slider(
        "Top Predictions",
        min_value=3,
        max_value=10,
        value=5
    )
    
    show_spectrograms = st.checkbox("Show Spectrograms", value=True)
    show_waveform = st.checkbox("Show Waveform", value=False)
    show_timeline = st.checkbox("Show Timeline", value=True)

# ============================================================
# MAIN CONTENT - FILE UPLOAD
# ============================================================

# Show target species banner if in search mode
if analysis_mode == "🎯 Search for Specific Bird" and target_species:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    ">
        <div>
            <span style="font-size: 1.5rem;">🎯</span>
            <strong style="margin-left: 0.5rem; font-size: 1.2rem;">Searching for: {target_species['english']}</strong>
            <span style="opacity: 0.8; margin-left: 0.5rem; font-style: italic;">({target_species['scientific']})</span>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 0.3rem 0.8rem; border-radius: 15px;">
            Sensitivity: {search_threshold*100:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload audio file",
    type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
    help="Supported: WAV, MP3, FLAC, OGG, M4A (max 10 min)"
)

# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file:
    
    # Load and analyze audio
    with st.spinner("Loading audio..."):
        audio, sr, temp_path = load_uploaded_audio(uploaded_file)
    
    if audio is None:
        st.error("❌ Failed to load audio file. Please try a different file.")
    else:
        # Audio info bar
        quality = analyze_audio_quality(audio, sr)
        duration = quality['duration']
        
        info_cols = st.columns([2, 1, 1, 1, 1])
        
        with info_cols[0]:
            st.audio(uploaded_file)
        with info_cols[1]:
            st.metric("Duration", f"{duration:.1f}s")
        with info_cols[2]:
            st.metric("Sample Rate", f"{sr//1000}kHz")
        with info_cols[3]:
            q_emoji = "🟢" if quality['quality_level'] in ["Excellent", "Good"] else "🟡"
            st.metric("Quality", f"{q_emoji} {quality['quality_level']}")
        with info_cols[4]:
            st.metric("SNR", f"{quality['snr_estimate']:.0f}dB")
        
        # Waveform (optional)
        if show_waveform:
            with st.expander("📈 Waveform", expanded=False):
                waveform_fig = plot_waveform(audio, sr, title="")
                waveform_fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(waveform_fig, use_container_width=True)
        
        # Apply noise reduction
        if apply_noise_reduction:
            with st.spinner("Applying noise reduction..."):
                audio = reduce_noise(audio, sr)
            st.success("✅ Noise reduction applied")
        
        st.markdown("---")
        
        # Process audio
        with st.spinner("Analyzing audio..."):
            # Generate spectrograms
            spectrograms = process_audio_to_spectrograms(
                audio, sr, 
                chunk_duration=chunk_duration,
                overlap=0.5
            )
            
            # Run predictions
            all_predictions = []
            progress = st.progress(0, text="Analyzing segments...")
            
            for i, (spec, start_time, end_time) in enumerate(spectrograms):
                progress.progress((i + 1) / len(spectrograms), 
                                  text=f"Analyzing segment {i+1}/{len(spectrograms)}...")
                
                pred = predict_single(model, spec)
                pred['start_time'] = start_time
                pred['end_time'] = end_time
                pred['spectrogram'] = spec
                pred['uncertainty'] = calculate_uncertainty(pred['probabilities'])
                
                all_predictions.append(pred)
            
            progress.empty()
        
        st.session_state.current_predictions = all_predictions
        
        # ============================================================
        # RESULTS - SPECIES SEARCH MODE
        # ============================================================
        
        if analysis_mode == "🎯 Search for Specific Bird" and target_species:
            
            target_idx = target_species['idx']
            target_name = target_species['english']
            
            # Find all detections of target species
            detections = []
            for pred in all_predictions:
                prob = pred['probabilities'][target_idx]
                if prob >= search_threshold:
                    detections.append({
                        'start': pred['start_time'],
                        'end': pred['end_time'],
                        'confidence': prob,
                        'is_top': pred['predicted_class'] == target_idx,
                        'spectrogram': pred['spectrogram'],
                        'rank': (list(pred['top_indices']).index(target_idx) + 1) 
                                if target_idx in pred['top_indices'][:10] else None
                    })
            
            # Results header
            st.markdown("## 🔍 Search Results")
            
            if detections:
                max_conf = max(d['confidence'] for d in detections)
                num_top_pred = sum(1 for d in detections if d['is_top'])
                
                # Status banner
                if max_conf >= 0.5:
                    st.success(f"""
                    ### ✅ {target_name} DETECTED!
                    Found in **{len(detections)}** segment(s) with up to **{max_conf*100:.1f}%** confidence.
                    """)
                elif max_conf >= 0.2:
                    st.warning(f"""
                    ### ⚠️ {target_name} POSSIBLY PRESENT
                    Found in **{len(detections)}** segment(s) with moderate confidence (max: {max_conf*100:.1f}%).
                    Review the detections below.
                    """)
                else:
                    st.info(f"""
                    ### 🔍 {target_name} - Weak Detection
                    Found in **{len(detections)}** segment(s) with low confidence (max: {max_conf*100:.1f}%).
                    This may be a false positive.
                    """)
                
                # Metrics row
                m_cols = st.columns(4)
                with m_cols[0]:
                    st.metric("Detections", len(detections))
                with m_cols[1]:
                    st.metric("Max Confidence", f"{max_conf*100:.1f}%")
                with m_cols[2]:
                    st.metric("Top-1 Matches", num_top_pred)
                with m_cols[3]:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                
                # Detection timeline
                st.markdown("### ⏱️ Detection Timeline")
                st.caption(f"Segments where {target_name} was detected (colored by confidence)")
                
                fig = go.Figure()
                
                # Background bar (full recording)
                fig.add_trace(go.Bar(
                    x=[duration],
                    y=[""],
                    orientation='h',
                    marker_color='#e9ecef',
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                # Detection bars
                for det in detections:
                    color = get_confidence_color(det['confidence'])
                    fig.add_trace(go.Bar(
                        x=[det['end'] - det['start']],
                        y=[""],
                        base=[det['start']],
                        orientation='h',
                        marker_color=color,
                        marker_line_color='white',
                        marker_line_width=1,
                        hovertemplate=f"<b>{target_name}</b><br>" +
                                      f"Time: {det['start']:.1f}s - {det['end']:.1f}s<br>" +
                                      f"Confidence: {det['confidence']*100:.1f}%<extra></extra>",
                        showlegend=False
                    ))
                
                fig.update_layout(
                    xaxis_title="Time (seconds)",
                    xaxis=dict(range=[0, duration]),
                    barmode='overlay',
                    height=100,
                    margin=dict(l=0, r=0, t=10, b=30),
                    yaxis=dict(showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Top detections detail
                st.markdown("### 🎯 Top Detections")
                
                # Sort by confidence
                sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:3]
                
                det_cols = st.columns(len(sorted_detections))
                
                for i, det in enumerate(sorted_detections):
                    with det_cols[i]:
                        conf_color = get_confidence_color(det['confidence'])
                        rank_text = f"Rank #{det['rank']}" if det['rank'] else "Rank >10"
                        
                        st.markdown(f"""
                        <div style="
                            border: 2px solid {conf_color};
                            border-radius: 10px;
                            padding: 1rem;
                            text-align: center;
                        ">
                            <div style="font-size: 0.9rem; color: #666;">
                                {det['start']:.1f}s - {det['end']:.1f}s
                            </div>
                            <div style="font-size: 2rem; font-weight: bold; color: {conf_color};">
                                {det['confidence']*100:.1f}%
                            </div>
                            <div style="font-size: 0.8rem; color: #888;">
                                {rank_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Play button
                        if st.button(f"▶️ Play", key=f"play_{i}"):
                            start_sample = int(det['start'] * sr)
                            end_sample = int(det['end'] * sr)
                            segment = audio[start_sample:end_sample]
                            
                            import soundfile as sf
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                                sf.write(tmp.name, segment, sr)
                                st.audio(tmp.name)
                        
                        # Spectrogram
                        if show_spectrograms:
                            spec_fig = plot_spectrogram_plotly(det['spectrogram'], sr=sr, title="")
                            spec_fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(spec_fig, use_container_width=True)
                
                # Full detection table
                with st.expander("📋 All Detections"):
                    det_df = pd.DataFrame([
                        {
                            "Time": f"{d['start']:.1f}s - {d['end']:.1f}s",
                            "Confidence": f"{d['confidence']*100:.1f}%",
                            "Rank": d['rank'] if d['rank'] else ">10",
                            "Top Prediction": "✓" if d['is_top'] else "",
                        }
                        for d in sorted(detections, key=lambda x: x['start'])
                    ])
                    st.dataframe(det_df, use_container_width=True, hide_index=True)
            
            else:
                # No detections
                st.error(f"""
                ### ❌ {target_name} NOT DETECTED
                
                The target species was not found above the {search_threshold*100:.0f}% threshold.
                """)
                
                # Show max probability found
                max_prob = max(pred['probabilities'][target_idx] for pred in all_predictions)
                st.info(f"**Maximum probability found:** {max_prob*100:.2f}%")
                
                st.markdown("""
                **Tips to improve detection:**
                - Lower the sensitivity threshold in the sidebar
                - Enable noise reduction
                - Try a longer recording
                - The species may not be present in this recording
                """)
            
            # Also show what else was detected
            st.markdown("---")
            with st.expander("🐦 Other Species Detected", expanded=False):
                # Aggregate predictions
                aggregated_probs = np.mean([p['probabilities'] for p in all_predictions], axis=0)
                top_indices = np.argsort(aggregated_probs)[::-1][:5]
                
                for idx in top_indices:
                    if idx != target_idx:
                        species = label_mapping['idx_to_english'].get(idx, f"Species {idx}")
                        prob = aggregated_probs[idx]
                        color = get_confidence_color(prob)
                        
                        st.markdown(f"""
                        <div style="
                            display: flex;
                            justify-content: space-between;
                            padding: 0.5rem;
                            margin: 0.25rem 0;
                            background: #f8f9fa;
                            border-radius: 5px;
                            border-left: 3px solid {color};
                        ">
                            <span>{species}</span>
                            <strong style="color: {color};">{prob*100:.1f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
        
        # ============================================================
        # RESULTS - GENERAL DETECTION MODE
        # ============================================================
        
        else:
            st.markdown("## 🎯 Detection Results")
            
            # Find best overall prediction
            best_pred = max(all_predictions, key=lambda x: x['confidence'])
            best_idx = best_pred['predicted_class']
            best_species = label_mapping['idx_to_english'].get(best_idx, f"Species {best_idx}")
            best_scientific = label_mapping['idx_to_species'].get(best_idx, "")
            
            # Main result card
            result_cols = st.columns([2, 1])
            
            with result_cols[0]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    color: white;
                ">
                    <div style="font-size: 1rem; opacity: 0.9;">Top Detection</div>
                    <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                        🐦 {best_species}
                    </div>
                    <div style="font-style: italic; opacity: 0.8;">{best_scientific}</div>
                    <div style="font-size: 2.5rem; font-weight: bold; margin-top: 1rem;">
                        {best_pred['confidence']*100:.1f}%
                    </div>
                    <div style="opacity: 0.8;">confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_cols[1]:
                gauge_fig = plot_confidence_gauge(best_pred['confidence'], title="")
                gauge_fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Uncertainty warning
            if not best_pred['uncertainty']['is_confident']:
                st.warning(f"""
                ⚠️ **Low Confidence Warning:** This prediction has {best_pred['uncertainty']['confidence_level'].lower()} confidence.
                Review the alternatives below.
                """)
            
            # Top predictions
            st.markdown("### 📊 Top Predictions")
            
            # Aggregate across all segments
            aggregated_probs = np.mean([p['probabilities'] for p in all_predictions], axis=0)
            agg_prediction = {
                'probabilities': aggregated_probs,
                'top_indices': np.argsort(aggregated_probs)[::-1][:top_k],
                'top_probs': np.sort(aggregated_probs)[::-1][:top_k],
            }
            
            pred_fig = plot_top_predictions(agg_prediction, label_mapping, top_k=top_k)
            pred_fig.update_layout(height=max(200, top_k * 40))
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Timeline
            if show_timeline and len(spectrograms) > 1:
                st.markdown("### ⏱️ Detection Timeline")
                
                detections = [(pred, pred['start_time'], pred['end_time']) 
                              for pred in all_predictions 
                              if pred['confidence'] >= confidence_threshold]
                
                if detections:
                    timeline_fig = plot_detection_timeline(detections, duration, label_mapping)
                    st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Spectrograms
            if show_spectrograms:
                st.markdown("### 🎼 Spectrogram")
                
                # Show best detection spectrogram
                spec_fig = plot_spectrogram_plotly(
                    best_pred['spectrogram'],
                    sr=sr,
                    title=f"Best detection: {best_pred['start_time']:.1f}s - {best_pred['end_time']:.1f}s"
                )
                st.plotly_chart(spec_fig, use_container_width=True)
                
                # Option to see all
                if len(spectrograms) > 1:
                    with st.expander(f"📊 View All {len(spectrograms)} Segments"):
                        seg_cols = st.columns(3)
                        for i, pred in enumerate(all_predictions[:9]):
                            with seg_cols[i % 3]:
                                species = label_mapping['idx_to_english'].get(pred['predicted_class'], "Unknown")
                                st.caption(f"{pred['start_time']:.1f}s - {pred['end_time']:.1f}s")
                                st.caption(f"{species} ({pred['confidence']*100:.0f}%)")
                                
                                mini_fig = plot_spectrogram_plotly(pred['spectrogram'], sr=sr, title="")
                                mini_fig.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0))
                                st.plotly_chart(mini_fig, use_container_width=True)
        
        # ============================================================
        # SAVE TO HISTORY
        # ============================================================
        
        # Determine what to save
        if analysis_mode == "🎯 Search for Specific Bird" and target_species:
            history_entry = {
                'filename': uploaded_file.name,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'mode': 'search',
                'target_species': target_species['english'],
                'found': len(detections) > 0 if 'detections' in dir() else False,
                'max_confidence': max(d['confidence'] for d in detections) if detections else 0,
                'duration': duration,
                'num_segments': len(spectrograms),
            }
        else:
            history_entry = {
                'filename': uploaded_file.name,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'mode': 'detect',
                'best_species': best_species,
                'best_scientific': best_scientific,
                'confidence': best_pred['confidence'],
                'duration': duration,
                'num_segments': len(spectrograms),
            }
        
        st.session_state.analysis_history.append(history_entry)

# ============================================================
# EMPTY STATE
# ============================================================

else:
    # No file uploaded yet
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        border: 2px dashed #dee2e6;
        margin: 2rem 0;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎵</div>
        <h2 style="color: #495057; margin-bottom: 0.5rem;">Upload an Audio File</h2>
        <p style="color: #6c757d; max-width: 400px; margin: 0 auto;">
            Drag and drop a bird recording above, or click to browse.<br>
            <small>Supported: WAV, MP3, FLAC, OGG, M4A</small>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick tips
    st.markdown("### 💡 Tips for Best Results")
    
    tip_cols = st.columns(3)
    
    with tip_cols[0]:
        st.markdown("""
        **🎙️ Recording Quality**
        - Use recordings with minimal background noise
        - Closer recordings work better
        - 5-30 second clips are ideal
        """)
    
    with tip_cols[1]:
        st.markdown("""
        **🔍 Detection Mode**
        - Use "Detect All" to discover what's in a recording
        - Use "Search" to find a specific species
        - Adjust sensitivity for search
        """)
    
    with tip_cols[2]:
        st.markdown("""
        **⚙️ Settings**
        - Enable noise reduction for field recordings
        - Lower confidence threshold to see more results
        - Check the timeline for long recordings
        """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("🐦 Bird Species Detector | 96.06% accuracy | 87 species | Powered by EfficientNet-B2")