# Save as: 08_Deployment/pages/2_📊_Results.py

"""
📊 Results & History Page
=========================
View past analyses, compare results, and export reports
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
import io

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import *
from utils.visualization import (plot_top_predictions, plot_spectrogram_plotly,
                                  plot_species_comparison, get_confidence_color)

# Page config
st.set_page_config(page_title="Results | Bird Detector", page_icon="📊", layout="wide")

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'selected_results' not in st.session_state:
    st.session_state.selected_results = []

# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Results & History")
st.markdown("View your analysis history, compare results, and export reports.")

# ============================================================
# QUICK STATS
# ============================================================

history = st.session_state.analysis_history

if history:
    st.markdown("### 📈 Session Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(history))
    
    with col2:
        unique_species = set(h['best_species'] for h in history)
        st.metric("Unique Species", len(unique_species))
    
    with col3:
        avg_confidence = np.mean([h['confidence'] for h in history])
        st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
    
    with col4:
        total_duration = sum(h['duration'] for h in history)
        st.metric("Total Audio", f"{total_duration:.1f}s")
    
    st.markdown("---")

# ============================================================
# HISTORY TABLE
# ============================================================

st.markdown("## 📋 Analysis History")

if not history:
    st.info("No analyses yet. Go to the **Analyze** page to process some audio files.")
    
    # Demo data option
    if st.button("📊 Load Demo History"):
        # Create demo history
        demo_history = [
            {
                'filename': 'demo_cardinal.wav',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'best_species': 'Northern Cardinal',
                'best_scientific': 'Cardinalis cardinalis',
                'confidence': 0.95,
                'duration': 12.5,
                'num_segments': 3,
                'predictions': None,
            },
            {
                'filename': 'demo_robin.wav',
                'timestamp': (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                'best_species': 'American Robin',
                'best_scientific': 'Turdus migratorius',
                'confidence': 0.88,
                'duration': 8.3,
                'num_segments': 2,
                'predictions': None,
            },
            {
                'filename': 'demo_bluejay.wav',
                'timestamp': (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                'best_species': 'Blue Jay',
                'best_scientific': 'Cyanocitta cristata',
                'confidence': 0.92,
                'duration': 15.7,
                'num_segments': 4,
                'predictions': None,
            },
        ]
        st.session_state.analysis_history = demo_history
        st.rerun()

else:
    # Filters
    st.markdown("### 🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Species filter
        all_species = sorted(set(h['best_species'] for h in history))
        selected_species = st.multiselect(
            "Filter by Species",
            options=all_species,
            default=[],
            help="Select species to filter"
        )
    
    with col2:
        # Confidence filter
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
    
    with col3:
        # Search
        search_query = st.text_input(
            "Search Filename",
            placeholder="Search..."
        )
    
    # Filter history
    filtered_history = history.copy()
    
    if selected_species:
        filtered_history = [h for h in filtered_history if h['best_species'] in selected_species]
    
    if min_confidence > 0:
        filtered_history = [h for h in filtered_history if h['confidence'] >= min_confidence]
    
    if search_query:
        filtered_history = [h for h in filtered_history 
                           if search_query.lower() in h['filename'].lower()]
    
    st.markdown(f"Showing **{len(filtered_history)}** of **{len(history)}** results")
    
    # Display as table
    if filtered_history:
        # Create DataFrame
        df = pd.DataFrame([
            {
                '📁 Filename': h['filename'],
                '🕐 Time': h['timestamp'],
                '🐦 Species': h['best_species'],
                '📊 Confidence': f"{h['confidence']*100:.1f}%",
                '⏱️ Duration': f"{h['duration']:.1f}s",
                '📈 Segments': h['num_segments'],
            }
            for h in filtered_history
        ])
        
        # Add selection column
        df.insert(0, '✅ Select', False)
        
        # Editable dataframe for selection
        edited_df = st.data_editor(
            df,
            column_config={
                "✅ Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select for comparison or export",
                    default=False,
                ),
                "📊 Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            disabled=['📁 Filename', '🕐 Time', '🐦 Species', '📊 Confidence', '⏱️ Duration', '📈 Segments'],
            hide_index=True,
            use_container_width=True
        )
        
        # Get selected indices
        selected_indices = edited_df[edited_df['✅ Select']].index.tolist()
        st.session_state.selected_results = [filtered_history[i] for i in selected_indices]
        
        if selected_indices:
            st.success(f"✅ {len(selected_indices)} result(s) selected")

# ============================================================
# DETAILED VIEW
# ============================================================

if history:
    st.markdown("---")
    st.markdown("## 🔍 Detailed View")
    
    # Select a result to view details
    result_options = {f"{h['filename']} ({h['timestamp']})": i for i, h in enumerate(history)}
    
    selected_result_key = st.selectbox(
        "Select a result to view details:",
        options=list(result_options.keys())
    )
    
    if selected_result_key:
        selected_idx = result_options[selected_result_key]
        result = history[selected_idx]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ### 📄 {result['filename']}
            
            | Property | Value |
            |----------|-------|
            | **Analyzed** | {result['timestamp']} |
            | **Duration** | {result['duration']:.1f} seconds |
            | **Segments** | {result['num_segments']} |
            | **Best Match** | {result['best_species']} |
            | **Scientific Name** | *{result['best_scientific']}* |
            | **Confidence** | {result['confidence']*100:.1f}% |
            """)
        
        with col2:
            # Confidence indicator
            confidence = result['confidence']
            color = get_confidence_color(confidence)
            
            st.markdown(f"""
            <div style="
                background: {color};
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                color: white;
            ">
                <h2 style="margin: 0;">{confidence*100:.1f}%</h2>
                <p style="margin: 0;">Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show predictions if available
        if result.get('predictions'):
            with st.expander("📊 All Segment Predictions"):
                for i, pred in enumerate(result['predictions']):
                    st.markdown(f"**Segment {i+1}** ({pred['start_time']:.1f}s - {pred['end_time']:.1f}s)")
                    st.markdown(f"- Predicted: {pred['predicted_class']} ({pred['confidence']*100:.1f}%)")

# ============================================================
# COMPARISON MODE
# ============================================================

if len(st.session_state.selected_results) >= 2:
    st.markdown("---")
    st.markdown("## 🔄 Compare Selected Results")
    
    selected = st.session_state.selected_results
    
    st.markdown(f"Comparing **{len(selected)}** results:")
    
    # Comparison table
    comparison_data = []
    for h in selected:
        comparison_data.append({
            'Filename': h['filename'],
            'Species': h['best_species'],
            'Confidence': f"{h['confidence']*100:.1f}%",
            'Duration': f"{h['duration']:.1f}s",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Species distribution
    st.markdown("### 🐦 Species Distribution")
    
    species_counts = pd.Series([h['best_species'] for h in selected]).value_counts()
    
    import plotly.express as px
    fig = px.pie(
        values=species_counts.values,
        names=species_counts.index,
        title="Species Distribution in Selected Results"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EXPORT OPTIONS
# ============================================================

if history:
    st.markdown("---")
    st.markdown("## 📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 CSV Export")
        
        # Prepare CSV data
        export_data = []
        for h in history:
            export_data.append({
                'Filename': h['filename'],
                'Timestamp': h['timestamp'],
                'Species': h['best_species'],
                'Scientific_Name': h['best_scientific'],
                'Confidence': h['confidence'],
                'Duration_Seconds': h['duration'],
                'Num_Segments': h['num_segments'],
            })
        
        export_df = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"bird_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("### 📋 JSON Export")
        
        # Prepare JSON (without numpy arrays)
        json_data = []
        for h in history:
            json_entry = {
                'filename': h['filename'],
                'timestamp': h['timestamp'],
                'best_species': h['best_species'],
                'best_scientific': h['best_scientific'],
                'confidence': float(h['confidence']),
                'duration': float(h['duration']),
                'num_segments': h['num_segments'],
            }
            json_data.append(json_entry)
        
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"bird_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        st.markdown("### 📊 Summary Report")
        
        # Generate summary report
        report = f"""
# Bird Detection Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- Total Analyses: {len(history)}
- Unique Species: {len(set(h['best_species'] for h in history))}
- Total Audio Duration: {sum(h['duration'] for h in history):.1f} seconds
- Average Confidence: {np.mean([h['confidence'] for h in history])*100:.1f}%

## Species Detected
{chr(10).join(f"- {species}: {sum(1 for h in history if h['best_species'] == species)} detection(s)" for species in sorted(set(h['best_species'] for h in history)))}

## Detection Details
{chr(10).join(f"- {h['filename']}: {h['best_species']} ({h['confidence']*100:.1f}%)" for h in history)}
"""
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"bird_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# ============================================================
# CLEAR HISTORY
# ============================================================

if history:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col3:
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.analysis_history = []
            st.session_state.selected_results = []
            st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    💾 Results are stored in your session. Export to save permanently.
</div>
""", unsafe_allow_html=True)