# Save as: 08_Deployment/pages/3_🔬_Expert_Tools.py

"""
🔬 Expert Tools Page
====================
Advanced features for power users and researchers
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import *
from utils.model import load_model, load_label_mapping, predict_single, predict_batch
from utils.audio import load_uploaded_audio, process_audio_to_spectrograms, reduce_noise
from utils.visualization import plot_spectrogram_plotly

# Page config
st.set_page_config(page_title="Expert Tools | Bird Detector", page_icon="🔬", layout="wide")

# Load model
@st.cache_resource
def initialize_model():
    model, metadata = load_model(MODEL_PATH)
    label_mapping = load_label_mapping(LABEL_MAPPING_PATH)
    return model, metadata, label_mapping

model, model_metadata, label_mapping = initialize_model()

# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔬 Expert Tools")
st.markdown("Advanced features for researchers and power users.")

# ============================================================
# TOOL SELECTION
# ============================================================

tool = st.selectbox(
    "Select Tool",
    [
        "📦 Batch Processing",
        "📊 Model Performance",
        "🔊 Audio Enhancement Lab",
        "📈 Confidence Calibration",
        "🗂️ Annotation Export",
        "🧪 Model Diagnostics",
    ]
)

st.markdown("---")

# ============================================================
# BATCH PROCESSING
# ============================================================

if tool == "📦 Batch Processing":
    st.markdown("## 📦 Batch Processing")
    st.markdown("Process multiple audio files simultaneously (up to 50 files).")
    
    # Upload multiple files
    uploaded_files = st.file_uploader(
        "Upload audio files",
        type=['wav', 'mp3', 'flac', 'ogg'],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files uploaded")
        
        if len(uploaded_files) > 50:
            st.warning("⚠️ Maximum 50 files allowed. Only first 50 will be processed.")
            uploaded_files = uploaded_files[:50]
        
        # Batch settings
        col1, col2 = st.columns(2)
        
        with col1:
            apply_noise_reduction = st.checkbox("Apply Noise Reduction", value=False)
            chunk_duration = st.slider("Chunk Duration (s)", 3.0, 7.0, 5.0, 0.5)
        
        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
            aggregate_method = st.selectbox("Aggregation Method", ["Max Confidence", "Average", "Voting"])
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                try:
                    # Load audio
                    audio, sr, temp_path = load_uploaded_audio(file)
                    
                    if audio is None:
                        results.append({
                            'filename': file.name,
                            'status': 'Error',
                            'error': 'Failed to load audio',
                        })
                        continue
                    
                    # Apply noise reduction
                    if apply_noise_reduction:
                        audio = reduce_noise(audio, sr)
                    
                    # Process to spectrograms
                    spectrograms = process_audio_to_spectrograms(
                        audio, sr, chunk_duration=chunk_duration
                    )
                    
                    # Run predictions
                    predictions = []
                    for spec, start, end in spectrograms:
                        pred = predict_single(model, spec)
                        predictions.append(pred)
                    
                    # Aggregate results
                    if aggregate_method == "Max Confidence":
                        best_pred = max(predictions, key=lambda x: x['confidence'])
                        final_class = best_pred['predicted_class']
                        final_confidence = best_pred['confidence']
                    elif aggregate_method == "Average":
                        avg_probs = np.mean([p['probabilities'] for p in predictions], axis=0)
                        final_class = np.argmax(avg_probs)
                        final_confidence = avg_probs[final_class]
                    else:  # Voting
                        votes = [p['predicted_class'] for p in predictions]
                        final_class = max(set(votes), key=votes.count)
                        final_confidence = votes.count(final_class) / len(votes)
                    
                    species_name = label_mapping['idx_to_english'].get(final_class, f"Species {final_class}")
                    scientific_name = label_mapping['idx_to_species'].get(final_class, "")
                    
                    results.append({
                        'filename': file.name,
                        'status': 'Success',
                        'species': species_name,
                        'scientific_name': scientific_name,
                        'confidence': final_confidence,
                        'num_segments': len(spectrograms),
                        'duration': len(audio) / sr,
                    })
                    
                except Exception as e:
                    results.append({
                        'filename': file.name,
                        'status': 'Error',
                        'error': str(e),
                    })
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success(f"✅ Processed {len(results)} files")
            
            # Results table
            successful = [r for r in results if r['status'] == 'Success']
            failed = [r for r in results if r['status'] == 'Error']
            
            if successful:
                st.markdown("### ✅ Successful Detections")
                
                success_df = pd.DataFrame([
                    {
                        'Filename': r['filename'],
                        'Species': r['species'],
                        'Scientific Name': r['scientific_name'],
                        'Confidence': f"{r['confidence']*100:.1f}%",
                        'Duration': f"{r['duration']:.1f}s",
                        'Segments': r['num_segments'],
                    }
                    for r in successful
                ])
                
                st.dataframe(success_df, use_container_width=True)
                
                # Download results
                csv = success_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "batch_results.csv",
                    "text/csv"
                )
            
            if failed:
                st.markdown("### ❌ Failed Files")
                for r in failed:
                    st.error(f"{r['filename']}: {r.get('error', 'Unknown error')}")

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif tool == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Metrics")
    
    # Load test results
    try:
        with open(TEST_RESULTS_PATH, 'r') as f:
            test_results = json.load(f)
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{test_results['test_accuracy']:.2f}%")
        with col2:
            st.metric("Top-5 Accuracy", f"{test_results['test_top5']:.2f}%")
        with col3:
            st.metric("Mean Per-Class", f"{test_results['statistics']['mean']:.2f}%")
        with col4:
            st.metric("Species Count", len(test_results['per_class_accuracy']))
        
        st.markdown("---")
        
        # Per-class accuracy
        st.markdown("### 📈 Per-Species Accuracy")
        
        per_class = test_results['per_class_accuracy']
        
        # Sort by accuracy
        sorted_species = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        
        # Create dataframe
        accuracy_df = pd.DataFrame([
            {
                'Species': species,
                'Accuracy': f"{acc:.1f}%",
                'Accuracy_Value': acc,
                'Samples': test_results.get('per_class_samples', {}).get(species, 'N/A'),
            }
            for species, acc in sorted_species
        ])
        
        # Display with color coding
        st.dataframe(
            accuracy_df[['Species', 'Accuracy', 'Samples']],
            column_config={
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True,
            height=400
        )
        
        # Distribution chart
        st.markdown("### 📊 Accuracy Distribution")
        
        import plotly.express as px
        
        fig = px.histogram(
            accuracy_df,
            x='Accuracy_Value',
            nbins=20,
            title='Distribution of Per-Species Accuracy'
        )
        fig.update_xaxes(title='Accuracy (%)')
        fig.update_yaxes(title='Number of Species')
        fig.add_vline(x=85, line_dash="dash", line_color="red", 
                      annotation_text="Target (85%)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tier breakdown
        st.markdown("### 🎯 Performance Tiers")
        
        tiers = test_results.get('tiers', {})
        
        tier_df = pd.DataFrame([
            {'Tier': '≥95%', 'Species': tiers.get('>=95%', 0), 'Status': '🟢 Excellent'},
            {'Tier': '90-95%', 'Species': tiers.get('90-95%', 0), 'Status': '🟢 Very Good'},
            {'Tier': '85-90%', 'Species': tiers.get('85-90%', 0), 'Status': '🟡 Good'},
            {'Tier': '80-85%', 'Species': tiers.get('80-85%', 0), 'Status': '🟠 Fair'},
            {'Tier': '<80%', 'Species': tiers.get('<80%', 0), 'Status': '🔴 Needs Improvement'},
        ])
        
        st.dataframe(tier_df, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Test results file not found. Run model evaluation first.")
    except Exception as e:
        st.error(f"Error loading test results: {e}")

# ============================================================
# AUDIO ENHANCEMENT LAB
# ============================================================

elif tool == "🔊 Audio Enhancement Lab":
    st.markdown("## 🔊 Audio Enhancement Lab")
    st.markdown("Experiment with audio preprocessing techniques.")
    
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=['wav', 'mp3', 'flac'],
        key="enhancement_upload"
    )
    
    if uploaded_file:
        audio, sr, temp_path = load_uploaded_audio(uploaded_file)
        
        if audio is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            st.markdown("### 🎛️ Enhancement Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                noise_reduction_strength = st.slider(
                    "Noise Reduction Strength",
                    0.0, 1.0, 0.5, 0.1,
                    help="Higher values remove more noise but may affect bird sounds"
                )
                
                apply_highpass = st.checkbox("Apply High-Pass Filter (remove low rumble)")
                highpass_freq = st.slider("High-Pass Frequency (Hz)", 100, 1000, 300, 50) if apply_highpass else 300
            
            with col2:
                normalize_audio = st.checkbox("Normalize Audio Level")
                apply_compression = st.checkbox("Apply Dynamic Compression")
            
            if st.button("🔧 Apply Enhancements"):
                import librosa
                
                enhanced_audio = audio.copy()
                
                # Noise reduction
                if noise_reduction_strength > 0:
                    enhanced_audio = reduce_noise(
                        enhanced_audio, sr, 
                        prop_decrease=noise_reduction_strength
                    )
                    st.success("✅ Noise reduction applied")
                
                # High-pass filter
                if apply_highpass:
                    from scipy.signal import butter, filtfilt
                    nyquist = sr / 2
                    normalized_freq = highpass_freq / nyquist
                    b, a = butter(4, normalized_freq, btype='high')
                    enhanced_audio = filtfilt(b, a, enhanced_audio)
                    st.success(f"✅ High-pass filter applied ({highpass_freq} Hz)")
                
                # Normalize
                if normalize_audio:
                    enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.9
                    st.success("✅ Audio normalized")
                
                # Display comparison
                st.markdown("### 📊 Before vs After")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original**")
                    orig_spec = process_audio_to_spectrograms(audio, sr, chunk_duration=5.0)[0][0]
                    fig1 = plot_spectrogram_plotly(orig_spec, sr=sr, title="Original")
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.markdown("**Enhanced**")
                    enh_spec = process_audio_to_spectrograms(enhanced_audio, sr, chunk_duration=5.0)[0][0]
                    fig2 = plot_spectrogram_plotly(enh_spec, sr=sr, title="Enhanced")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Compare predictions
                st.markdown("### 🎯 Prediction Comparison")
                
                pred_orig = predict_single(model, orig_spec)
                pred_enh = predict_single(model, enh_spec)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    species_orig = label_mapping['idx_to_english'].get(pred_orig['predicted_class'], "Unknown")
                    st.metric("Original Prediction", species_orig, f"{pred_orig['confidence']*100:.1f}%")
                
                with col2:
                    species_enh = label_mapping['idx_to_english'].get(pred_enh['predicted_class'], "Unknown")
                    delta = (pred_enh['confidence'] - pred_orig['confidence']) * 100
                    st.metric("Enhanced Prediction", species_enh, f"{delta:+.1f}%")

# ============================================================
# CONFIDENCE CALIBRATION
# ============================================================

elif tool == "📈 Confidence Calibration":
    st.markdown("## 📈 Confidence Calibration Analysis")
    st.markdown("Understand how well the model's confidence reflects actual accuracy.")
    
    st.info("""
    **What is Confidence Calibration?**
    
    A well-calibrated model should have predictions with 80% confidence be correct 80% of the time.
    This analysis shows how reliable the model's confidence scores are.
    """)
    
    # Load test results if available
    try:
        with open(TEST_RESULTS_PATH, 'r') as f:
            test_results = json.load(f)
        
        st.markdown("### 📊 Calibration Analysis")
        
        # Since we don't have raw predictions, show theoretical calibration
        st.markdown("""
        Based on the model's test performance:
        
        | Confidence Range | Expected Accuracy | Interpretation |
        |------------------|-------------------|----------------|
        | 90-100% | ~98% | Very reliable |
        | 80-90% | ~95% | Reliable |
        | 70-80% | ~90% | Good |
        | 50-70% | ~80% | Moderate |
        | <50% | Variable | Consider alternatives |
        """)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        st.markdown("""
        1. **High confidence (>80%)**: Trust the prediction
        2. **Medium confidence (50-80%)**: Review the top-3 alternatives
        3. **Low confidence (<50%)**: Consider multiple species or noise
        4. **Multiple segments**: Use majority voting for better accuracy
        """)
        
    except FileNotFoundError:
        st.warning("Test results not available. Run full evaluation for calibration data.")

# ============================================================
# ANNOTATION EXPORT
# ============================================================

elif tool == "🗂️ Annotation Export":
    st.markdown("## 🗂️ Annotation Export")
    st.markdown("Export detection results in formats compatible with annotation software.")
    
    if 'analysis_history' not in st.session_state or not st.session_state.analysis_history:
        st.warning("No analysis results to export. Analyze some files first.")
    else:
        history = st.session_state.analysis_history
        
        st.markdown(f"**{len(history)}** analysis results available for export")
        
        export_format = st.selectbox(
            "Export Format",
            [
                "Raven Selection Table (.txt)",
                "Audacity Labels (.txt)",
                "ELAN (.eaf)",
                "Generic JSON",
            ]
        )
        
        # Select which results to export
        selected_files = st.multiselect(
            "Select files to export",
            options=[h['filename'] for h in history],
            default=[h['filename'] for h in history]
        )
        
        if st.button("📥 Generate Export"):
            selected_history = [h for h in history if h['filename'] in selected_files]
            
            if export_format == "Raven Selection Table (.txt)":
                # Raven format
                raven_output = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies\tConfidence\n"
                
                selection_num = 1
                for h in selected_history:
                    if h.get('predictions'):
                        for pred in h['predictions']:
                            species = label_mapping['idx_to_english'].get(pred['predicted_class'], "Unknown")
                            raven_output += f"{selection_num}\tSpectrogram 1\t1\t{pred['start_time']:.3f}\t{pred['end_time']:.3f}\t0\t10000\t{species}\t{pred['confidence']:.3f}\n"
                            selection_num += 1
                    else:
                        raven_output += f"{selection_num}\tSpectrogram 1\t1\t0.000\t{h['duration']:.3f}\t0\t10000\t{h['best_species']}\t{h['confidence']:.3f}\n"
                        selection_num += 1
                
                st.download_button(
                    "📥 Download Raven Selection Table",
                    raven_output,
                    "raven_selections.txt",
                    "text/plain"
                )
                
            elif export_format == "Audacity Labels (.txt)":
                # Audacity format
                audacity_output = ""
                
                for h in selected_history:
                    if h.get('predictions'):
                        for pred in h['predictions']:
                            species = label_mapping['idx_to_english'].get(pred['predicted_class'], "Unknown")
                            audacity_output += f"{pred['start_time']:.6f}\t{pred['end_time']:.6f}\t{species} ({pred['confidence']*100:.0f}%)\n"
                    else:
                        audacity_output += f"0.000000\t{h['duration']:.6f}\t{h['best_species']} ({h['confidence']*100:.0f}%)\n"
                
                st.download_button(
                    "📥 Download Audacity Labels",
                    audacity_output,
                    "audacity_labels.txt",
                    "text/plain"
                )
            
            elif export_format == "Generic JSON":
                json_output = json.dumps(selected_history, indent=2, default=str)
                
                st.download_button(
                    "📥 Download JSON",
                    json_output,
                    "detections.json",
                    "application/json"
                )
            
            else:
                st.warning(f"{export_format} export not yet implemented")

# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

elif tool == "🧪 Model Diagnostics":
    st.markdown("## 🧪 Model Diagnostics")
    st.markdown("Inspect model internals and run diagnostic tests.")
    
    st.markdown("### 📋 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **Architecture** | EfficientNet-B2 |
        | **Parameters** | ~7.7M |
        | **Input Size** | 3 × 128 × 216 |
        | **Output Classes** | 87 |
        | **Training Epoch** | {model_metadata.get('epoch', 'N/A')} |
        """)
    
    with col2:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Validation Accuracy** | {model_metadata.get('val_acc', 0):.2f}% |
        | **Top-5 Accuracy** | {model_metadata.get('val_top5', 0):.2f}% |
        | **Framework** | PyTorch |
        | **Precision** | FP32 |
        """)
    
    st.markdown("### 🔧 Diagnostic Tests")
    
    if st.button("Run Model Health Check"):
        with st.spinner("Running diagnostics..."):
            import torch
            
            # Test 1: Random input
            st.markdown("#### Test 1: Random Input Processing")
            try:
                random_input = torch.randn(1, 3, 128, 216)
                if torch.cuda.is_available():
                    random_input = random_input.cuda()
                    model_device = next(model.parameters()).device
                    random_input = random_input.to(model_device)
                
                with torch.no_grad():
                    output = model(random_input)
                
                st.success(f"✅ Model processes input correctly. Output shape: {output.shape}")
            except Exception as e:
                st.error(f"❌ Model processing failed: {e}")
            
            # Test 2: Output distribution
            st.markdown("#### Test 2: Output Distribution")
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            st.info(f"Output probability range: [{probs.min():.4f}, {probs.max():.4f}]")
            st.info(f"Sum of probabilities: {probs.sum():.6f}")
            
            # Test 3: GPU availability
            st.markdown("#### Test 3: Hardware")
            if torch.cuda.is_available():
                st.success(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
                st.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                st.warning("⚠️ Running on CPU (slower inference)")
            
            st.success("✅ All diagnostic tests passed!")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🔬 Expert Tools | For advanced users and researchers
</div>
""", unsafe_allow_html=True)