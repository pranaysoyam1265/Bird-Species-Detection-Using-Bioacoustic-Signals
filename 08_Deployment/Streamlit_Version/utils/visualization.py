# Save as: 08_Deployment/utils/visualization.py

"""
Visualization utilities for spectrograms and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa.display
import streamlit as st

# ============================================================
# COLOR SCHEMES
# ============================================================

CONFIDENCE_COLORS = {
    'high': '#28a745',      # Green
    'medium': '#ffc107',    # Yellow
    'low': '#fd7e14',       # Orange
    'very_low': '#dc3545',  # Red
}

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return CONFIDENCE_COLORS['high']
    elif confidence >= 0.5:
        return CONFIDENCE_COLORS['medium']
    elif confidence >= 0.2:
        return CONFIDENCE_COLORS['low']
    else:
        return CONFIDENCE_COLORS['very_low']

# ============================================================
# SPECTROGRAM VISUALIZATION
# ============================================================

def plot_spectrogram_matplotlib(spec, sr=22050, hop_length=512, 
                                 title="Mel Spectrogram", figsize=(12, 4)):
    """
    Plot spectrogram using matplotlib.
    
    Args:
        spec: Mel spectrogram array
        sr: Sample rate
        hop_length: Hop length used in spectrogram
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    img = librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='magma'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')
    
    plt.tight_layout()
    
    return fig

def plot_spectrogram_plotly(spec, sr=22050, hop_length=512, title="Mel Spectrogram"):
    """
    Plot interactive spectrogram using Plotly.
    
    Args:
        spec: Mel spectrogram array
        sr: Sample rate
        hop_length: Hop length
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Calculate time and frequency axes
    times = librosa.times_like(spec, sr=sr, hop_length=hop_length)
    freqs = librosa.mel_frequencies(n_mels=spec.shape[0])
    
    fig = go.Figure(data=go.Heatmap(
        z=spec,
        x=times,
        y=freqs,
        colorscale='Magma',
        colorbar=dict(title='Power (dB)'),
        hoverongaps=False,
        hovertemplate='Time: %{x:.2f}s<br>Freq: %{y:.0f}Hz<br>Power: %{z:.1f}dB<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=400,
        yaxis_type='log',
        template='plotly_white'
    )
    
    return fig

# ============================================================
# PREDICTION VISUALIZATION
# ============================================================

def plot_top_predictions(predictions, label_mapping, top_k=5):
    """
    Plot horizontal bar chart of top predictions.
    
    Args:
        predictions: dict with 'top_indices' and 'top_probs'
        label_mapping: dict with 'idx_to_english' mapping
        top_k: Number of predictions to show
    
    Returns:
        Plotly figure
    """
    idx_to_english = label_mapping['idx_to_english']
    idx_to_species = label_mapping['idx_to_species']
    
    top_indices = predictions['top_indices'][:top_k]
    top_probs = predictions['top_probs'][:top_k]
    
    # Prepare data
    species_names = [idx_to_english.get(idx, f"Species {idx}") for idx in top_indices]
    scientific_names = [idx_to_species.get(idx, "") for idx in top_indices]
    colors = [get_confidence_color(p) for p in top_probs]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_probs * 100,
        y=species_names,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in top_probs],
        textposition='inside',
        textfont=dict(color='white', size=14),
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Top Predictions', font=dict(size=18)),
        xaxis_title='Confidence (%)',
        xaxis=dict(range=[0, 105]),
        yaxis=dict(autorange='reversed'),
        height=max(200, top_k * 50),
        margin=dict(l=10, r=10, t=50, b=10),
        template='plotly_white'
    )
    
    # Add threshold line
    fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                  annotation_text="50% threshold", annotation_position="top")
    
    return fig

def plot_confidence_gauge(confidence, title="Confidence"):
    """
    Plot a gauge chart for confidence level.
    
    Args:
        confidence: Confidence value (0-1)
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': get_confidence_color(confidence)},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 20], 'color': '#ffebee'},
                {'range': [20, 50], 'color': '#fff3e0'},
                {'range': [50, 80], 'color': '#fffde7'},
                {'range': [80, 100], 'color': '#e8f5e9'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# ============================================================
# TIMELINE VISUALIZATION
# ============================================================

def plot_detection_timeline(detections, audio_duration, label_mapping):
    """
    Plot timeline of detections across audio.
    
    Args:
        detections: list of (prediction, start_time, end_time)
        audio_duration: Total audio duration in seconds
        label_mapping: Label mapping dict
    
    Returns:
        Plotly figure
    """
    idx_to_english = label_mapping['idx_to_english']
    
    fig = go.Figure()
    
    # Unique species detected
    unique_species = set()
    for pred, _, _ in detections:
        unique_species.add(pred['predicted_class'])
    
    species_colors = px.colors.qualitative.Set3[:len(unique_species)]
    species_color_map = {s: c for s, c in zip(sorted(unique_species), species_colors)}
    
    # Add bars for each detection
    for pred, start, end in detections:
        species_idx = pred['predicted_class']
        species_name = idx_to_english.get(species_idx, f"Species {species_idx}")
        confidence = pred['confidence']
        
        fig.add_trace(go.Bar(
            x=[end - start],
            y=[species_name],
            base=[start],
            orientation='h',
            marker_color=species_color_map[species_idx],
            opacity=min(0.3 + confidence * 0.7, 1.0),
            text=f"{confidence*100:.0f}%",
            textposition='inside',
            hovertemplate=f'<b>{species_name}</b><br>Time: {start:.1f}s - {end:.1f}s<br>Confidence: {confidence*100:.1f}%<extra></extra>',
            showlegend=False
        ))
    
    fig.update_layout(
        title='Detection Timeline',
        xaxis_title='Time (seconds)',
        xaxis=dict(range=[0, audio_duration]),
        barmode='stack',
        height=max(200, len(unique_species) * 40 + 100),
        template='plotly_white'
    )
    
    return fig

# ============================================================
# WAVEFORM VISUALIZATION
# ============================================================

def plot_waveform(audio, sr, title="Audio Waveform"):
    """
    Plot audio waveform using Plotly.
    
    Args:
        audio: Audio array
        sr: Sample rate
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Downsample for plotting if too long
    max_points = 10000
    if len(audio) > max_points:
        step = len(audio) // max_points
        audio_plot = audio[::step]
        times = np.arange(len(audio_plot)) * step / sr
    else:
        audio_plot = audio
        times = np.arange(len(audio)) / sr
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=audio_plot,
        mode='lines',
        line=dict(color='#1f77b4', width=1),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=200,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# ============================================================
# COMPARISON VISUALIZATIONS
# ============================================================

def plot_species_comparison(predictions_list, label_mapping, titles=None):
    """
    Compare predictions across multiple recordings.
    
    Args:
        predictions_list: list of prediction dicts
        label_mapping: Label mapping dict
        titles: list of titles for each prediction
    
    Returns:
        Plotly figure
    """
    idx_to_english = label_mapping['idx_to_english']
    
    n = len(predictions_list)
    if titles is None:
        titles = [f"Recording {i+1}" for i in range(n)]
    
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=titles,
        shared_yaxes=True
    )
    
    # Get all unique species in top-5 across all predictions
    all_species = set()
    for pred in predictions_list:
        for idx in pred['top_indices'][:5]:
            all_species.add(idx)
    
    species_list = sorted(all_species)
    species_names = [idx_to_english.get(s, f"Species {s}") for s in species_list]
    
    for i, pred in enumerate(predictions_list):
        probs = [pred['probabilities'][s] * 100 for s in species_list]
        colors = [get_confidence_color(p/100) for p in probs]
        
        fig.add_trace(
            go.Bar(
                x=probs,
                y=species_names,
                orientation='h',
                marker_color=colors,
                name=titles[i],
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title='Species Comparison',
        height=max(300, len(species_list) * 30),
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Confidence (%)', range=[0, 105])
    
    return fig