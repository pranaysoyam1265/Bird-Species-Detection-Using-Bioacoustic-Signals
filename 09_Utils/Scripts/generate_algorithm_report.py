"""
Script: generate_algorithm_report.py
Purpose: Generate detailed PDF report with algorithm explanations
Location: 09_Utils/Scripts/generate_algorithm_report.py

FIXED VERSION - No duplicate style errors
"""

import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Line, Rect, String

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
OUTPUT_DIR = os.path.join(BASE_DIR, "10_Outputs", "Reports")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Bird_Detection_Algorithm_Report.pdf")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CUSTOM STYLES - FIXED VERSION
# ============================================================
def get_custom_styles():
    """Create custom paragraph styles - avoiding duplicates"""
    styles = getSampleStyleSheet()
    
    # Helper function to safely add styles
    def add_style_safe(name, **kwargs):
        """Add style only if it doesn't exist"""
        if name not in styles.byName:
            styles.add(ParagraphStyle(name=name, **kwargs))
        return styles[name]
    
    # Title style
    add_style_safe(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a5276')
    )
    
    # Heading 1
    add_style_safe(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#2874a6')
    )
    
    # Heading 2
    add_style_safe(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#1a5276')
    )
    
    # Heading 3
    add_style_safe(
        'Heading3Custom',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor('#2e86ab')
    )
    
    # Body text
    add_style_safe(
        'BodyCustom',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14
    )
    
    # Code style - use different name to avoid conflict
    add_style_safe(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        backColor=colors.HexColor('#f4f4f4'),
        leftIndent=10,
        spaceAfter=10
    )
    
    # Caption
    add_style_safe(
        'CaptionCustom',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#666666'),
        spaceAfter=15,
        spaceBefore=5
    )
    
    # Bullet point
    add_style_safe(
        'BulletCustom',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=5
    )
    
    # Table header
    add_style_safe(
        'TableHeaderCustom',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    
    # Equation style
    add_style_safe(
        'EquationCustom',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=10,
        backColor=colors.HexColor('#f9f9f9')
    )
    
    return styles


def create_table(data, col_widths=None, header=True):
    """Create a styled table"""
    table = Table(data, colWidths=col_widths)
    
    style_commands = [
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
    ]
    
    if header:
        style_commands.extend([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2874a6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ])
    
    # Alternate row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#ffffff')))
    
    table.setStyle(TableStyle(style_commands))
    return table


# ============================================================
# REPORT CONTENT
# ============================================================
def build_report():
    """Build the complete PDF report"""
    
    doc = SimpleDocTemplate(
        OUTPUT_FILE,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = get_custom_styles()
    elements = []
    
    # =========================================================
    # TITLE PAGE
    # =========================================================
    elements.append(Spacer(1, 80))
    elements.append(Paragraph(
        "Confidence-Aware, Explainable<br/>Multi-Species Bird Presence Detection<br/>Using Bioacoustic Signals",
        styles['CustomTitle']
    ))
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "<b>Technical Report: Algorithm & Methodology</b>",
        ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 50))
    
    # Project info table
    info_data = [
        ['Author', 'Pranav'],
        ['Institution', 'Academic Project'],
        ['Date', datetime.now().strftime('%B %d, %Y')],
        ['Version', '2.0'],
        ['Model', 'EfficientNet-B0'],
        ['Species Classified', '54'],
        ['Best Accuracy', '71.32% (Validation)'],
    ]
    info_table = Table(info_data, colWidths=[120, 200])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2874a6')),
    ]))
    elements.append(info_table)
    elements.append(PageBreak())
    
    # =========================================================
    # TABLE OF CONTENTS
    # =========================================================
    elements.append(Paragraph("Table of Contents", styles['Heading1Custom']))
    elements.append(Spacer(1, 20))
    
    toc_items = [
        ("1. Executive Summary", "3"),
        ("2. Problem Statement", "4"),
        ("3. Algorithm Overview", "5"),
        ("4. Feature Extraction: Audio to Spectrogram", "6"),
        ("5. Model Architecture: EfficientNet-B0", "8"),
        ("6. Transfer Learning Approach", "10"),
        ("7. Optimization Algorithm: AdamW", "11"),
        ("8. Loss Functions", "12"),
        ("9. Data Augmentation Techniques", "14"),
        ("10. Training Methodology", "16"),
        ("11. Results & Analysis", "18"),
        ("12. Conclusions", "20"),
        ("References", "21"),
    ]
    
    for item, page in toc_items:
        toc_row = Table(
            [[item, page]],
            colWidths=[400, 50]
        )
        toc_row.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(toc_row)
    
    elements.append(PageBreak())
    
    # =========================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================
    elements.append(Paragraph("1. Executive Summary", styles['Heading1Custom']))
    
    summary_text = """
    This report presents a comprehensive technical analysis of a deep learning system designed for 
    automatic bird species identification from audio recordings. The system employs a Convolutional 
    Neural Network (CNN) based on the EfficientNet-B0 architecture, utilizing transfer learning from 
    ImageNet and fine-tuned on mel-spectrogram representations of bird vocalizations.
    """
    elements.append(Paragraph(summary_text, styles['BodyCustom']))
    elements.append(Spacer(1, 10))
    
    # Key components box
    elements.append(Paragraph("<b>Key Technical Components:</b>", styles['BodyCustom']))
    
    components = [
        "• <b>Algorithm:</b> Deep Convolutional Neural Network (CNN)",
        "• <b>Architecture:</b> EfficientNet-B0 with custom classification head",
        "• <b>Feature Representation:</b> Mel-spectrograms (128 mel bands × 216 time frames)",
        "• <b>Optimization:</b> AdamW optimizer with cosine annealing learning rate schedule",
        "• <b>Loss Function:</b> Cross-Entropy with label smoothing (and optional Focal Loss)",
        "• <b>Augmentation:</b> SpecAugment + Mixup for improved generalization",
    ]
    for comp in components:
        elements.append(Paragraph(comp, styles['BulletCustom']))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("<b>Performance Results:</b>", styles['BodyCustom']))
    
    # Key metrics table
    metrics_data = [
        ['Metric', 'Value', 'Description'],
        ['Test Accuracy', '67.42%', 'Correct predictions on held-out test set'],
        ['Top-5 Accuracy', '82.57%', 'Correct species in top 5 predictions'],
        ['Macro F1', '58.39%', 'Average F1 across all 54 species'],
        ['Weighted F1', '66.42%', 'Sample-weighted average F1'],
        ['Parameters', '4.35M', 'Total trainable parameters'],
        ['Training Time', '~30 min', 'With pre-computed spectrograms'],
    ]
    elements.append(create_table(metrics_data, col_widths=[100, 80, 270]))
    elements.append(Paragraph("Table 1: Key Performance Metrics", styles['CaptionCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 2. PROBLEM STATEMENT
    # =========================================================
    elements.append(Paragraph("2. Problem Statement", styles['Heading1Custom']))
    
    problem_text = """
    <b>Objective:</b> Develop an automated system capable of identifying bird species from 
    environmental audio recordings with high accuracy and reliability.
    <br/><br/>
    <b>Formal Definition:</b><br/>
    Given an audio recording <i>x</i> of duration <i>T</i> seconds, the task is to predict the 
    bird species <i>y</i> from a set of <i>K</i> = 54 possible species. This is formulated as a 
    multi-class classification problem:
    """
    elements.append(Paragraph(problem_text, styles['BodyCustom']))
    
    # Equation
    elements.append(Paragraph(
        "<i>ŷ = argmax P(y=k | x; θ) for k ∈ {1,...,K}</i>",
        styles['EquationCustom']
    ))
    
    elements.append(Paragraph(
        "where <i>θ</i> represents the learned model parameters and <i>P(y=k|x;θ)</i> is the "
        "predicted probability for species <i>k</i>.",
        styles['BodyCustom']
    ))
    
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("2.1 Challenges", styles['Heading2Custom']))
    
    challenges = [
        "• <b>Acoustic Variability:</b> Bird calls vary by individual, age, season, and location",
        "• <b>Environmental Noise:</b> Recordings contain wind, rain, insects, and other birds",
        "• <b>Class Imbalance:</b> Some species have 1400+ samples, others have fewer than 50",
        "• <b>Overlapping Calls:</b> Multiple species may vocalize simultaneously",
        "• <b>Similar Vocalizations:</b> Some species produce acoustically similar calls",
    ]
    for challenge in challenges:
        elements.append(Paragraph(challenge, styles['BulletCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 3. ALGORITHM OVERVIEW
    # =========================================================
    elements.append(Paragraph("3. Algorithm Overview", styles['Heading1Custom']))
    
    overview_text = """
    The bird species classification system employs a multi-stage pipeline combining signal 
    processing, deep learning, and transfer learning techniques. The core algorithm is a 
    <b>Convolutional Neural Network (CNN)</b> that processes visual representations of audio.
    """
    elements.append(Paragraph(overview_text, styles['BodyCustom']))
    
    elements.append(Paragraph("3.1 Pipeline Architecture", styles['Heading2Custom']))
    
    # Pipeline table
    pipeline_data = [
        ['Stage', 'Component', 'Algorithm/Technique', 'Output'],
        ['1', 'Audio Loading', 'Digital Signal Processing', 'Waveform (22050 Hz)'],
        ['2', 'Segmentation', 'Sliding Window', '5-second chunks'],
        ['3', 'Feature Extraction', 'STFT + Mel Filter Bank', 'Mel-spectrogram'],
        ['4', 'Normalization', 'Log Compression + MinMax', 'Normalized image'],
        ['5', 'CNN Backbone', 'EfficientNet-B0', '1280-dim features'],
        ['6', 'Classification', 'Dense + Softmax', '54 probabilities'],
    ]
    elements.append(create_table(pipeline_data, col_widths=[40, 90, 150, 130]))
    elements.append(Paragraph("Table 2: Processing Pipeline Stages", styles['CaptionCustom']))
    
    elements.append(Paragraph("3.2 Algorithm Classification", styles['Heading2Custom']))
    
    algo_types = [
        "• <b>Supervised Learning:</b> Model learns from labeled examples (audio → species)",
        "• <b>Deep Learning:</b> Neural networks with multiple hidden layers",
        "• <b>Convolutional Neural Networks:</b> Exploits spatial patterns in spectrograms",
        "• <b>Transfer Learning:</b> Leverages pre-trained ImageNet weights",
    ]
    for algo in algo_types:
        elements.append(Paragraph(algo, styles['BulletCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 4. FEATURE EXTRACTION
    # =========================================================
    elements.append(Paragraph("4. Feature Extraction: Audio to Spectrogram", styles['Heading1Custom']))
    
    feature_text = """
    The first critical step is converting raw audio waveforms into visual representations that 
    CNNs can process effectively. This is achieved through mel-spectrogram computation.
    """
    elements.append(Paragraph(feature_text, styles['BodyCustom']))
    
    elements.append(Paragraph("4.1 Short-Time Fourier Transform (STFT)", styles['Heading2Custom']))
    
    stft_text = """
    The STFT decomposes an audio signal into its frequency components over time. For a discrete 
    signal x[n], the STFT computes the Fourier transform over short overlapping windows:
    """
    elements.append(Paragraph(stft_text, styles['BodyCustom']))
    
    elements.append(Paragraph(
        "<i>X[m, k] = Σ x[n + mH] · w[n] · e^(-j2πkn/N)</i>",
        styles['EquationCustom']
    ))
    
    stft_params = """
    Where: <i>m</i> = frame index, <i>k</i> = frequency bin, <i>N</i> = FFT size (2048), 
    <i>H</i> = hop length (512), <i>w[n]</i> = Hann window function.
    """
    elements.append(Paragraph(stft_params, styles['BodyCustom']))
    
    elements.append(Paragraph("4.2 Mel Filter Bank", styles['Heading2Custom']))
    
    mel_text = """
    The mel scale approximates human auditory perception. The conversion from frequency 
    <i>f</i> (Hz) to mel scale <i>m</i> is:
    """
    elements.append(Paragraph(mel_text, styles['BodyCustom']))
    
    elements.append(Paragraph(
        "<i>m = 2595 · log₁₀(1 + f/700)</i>",
        styles['EquationCustom']
    ))
    
    # Feature extraction parameters table
    elements.append(Paragraph("4.3 Feature Extraction Parameters", styles['Heading2Custom']))
    
    feature_params = [
        ['Parameter', 'Value', 'Purpose'],
        ['Sample Rate', '22,050 Hz', 'Standard audio rate'],
        ['FFT Size', '2048', 'Frequency resolution (~10.7 Hz/bin)'],
        ['Hop Length', '512', 'Time resolution (~23 ms/frame)'],
        ['Mel Bands', '128', 'Frequency bins in mel scale'],
        ['Freq Min', '150 Hz', 'Lower bound'],
        ['Freq Max', '15,000 Hz', 'Upper bound'],
        ['Duration', '5 seconds', 'Input segment length'],
        ['Output Shape', '(128, 216)', 'Mel bands × time frames'],
    ]
    elements.append(create_table(feature_params, col_widths=[100, 90, 220]))
    elements.append(Paragraph("Table 3: Mel-Spectrogram Parameters", styles['CaptionCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 5. MODEL ARCHITECTURE
    # =========================================================
    elements.append(Paragraph("5. Model Architecture: EfficientNet-B0", styles['Heading1Custom']))
    
    arch_intro = """
    EfficientNet-B0 is a convolutional neural network architecture developed by Tan and Le (2019) 
    that achieves state-of-the-art accuracy while being significantly more parameter-efficient 
    than previous architectures like ResNet or VGG.
    """
    elements.append(Paragraph(arch_intro, styles['BodyCustom']))
    
    elements.append(Paragraph("5.1 Key Innovations", styles['Heading2Custom']))
    
    innovations = [
        "• <b>Compound Scaling:</b> Uniformly scales depth, width, and resolution",
        "• <b>Mobile Inverted Bottleneck (MBConv):</b> Efficient building block",
        "• <b>Squeeze-and-Excitation (SE):</b> Channel-wise attention mechanism",
        "• <b>Swish Activation:</b> f(x) = x · σ(x), smoother than ReLU",
    ]
    for innovation in innovations:
        elements.append(Paragraph(innovation, styles['BulletCustom']))
    
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("5.2 Network Architecture", styles['Heading2Custom']))
    
    arch_table = [
        ['Stage', 'Operator', 'Resolution', 'Channels', 'Layers'],
        ['1', 'Conv3×3', '112×108', '32', '1'],
        ['2', 'MBConv1, k3×3', '112×108', '16', '1'],
        ['3', 'MBConv6, k3×3', '56×54', '24', '2'],
        ['4', 'MBConv6, k5×5', '28×27', '40', '2'],
        ['5', 'MBConv6, k3×3', '14×14', '80', '3'],
        ['6', 'MBConv6, k5×5', '14×14', '112', '3'],
        ['7', 'MBConv6, k5×5', '7×7', '192', '4'],
        ['8', 'MBConv6, k3×3', '7×7', '320', '1'],
        ['9', 'Conv1×1 & Pool', '1×1', '1280', '1'],
        ['10', 'Dense + Softmax', '-', '54', '1'],
    ]
    elements.append(create_table(arch_table, col_widths=[50, 100, 80, 70, 60]))
    elements.append(Paragraph("Table 4: EfficientNet-B0 Architecture", styles['CaptionCustom']))
    
    elements.append(Paragraph("5.3 Mobile Inverted Bottleneck (MBConv)", styles['Heading2Custom']))
    
    mbconv_text = """
    The MBConv block is the fundamental building unit consisting of:
    <br/><br/>
    <b>1. Expansion:</b> 1×1 convolution to increase channels by factor of 6<br/>
    <b>2. Depthwise Conv:</b> Spatial filtering per channel (k×k kernel)<br/>
    <b>3. Squeeze-Excitation:</b> Channel attention via global pooling + FC layers<br/>
    <b>4. Projection:</b> 1×1 convolution to reduce channels<br/>
    <b>5. Skip Connection:</b> Residual connection when dimensions match
    """
    elements.append(Paragraph(mbconv_text, styles['BodyCustom']))
    
    elements.append(Paragraph("5.4 Custom Classification Head", styles['Heading2Custom']))
    
    head_arch = [
        ['Layer', 'Input', 'Output', 'Activation', 'Regularization'],
        ['Global Avg Pool', '(1280,H,W)', '1280', '-', '-'],
        ['Dropout', '1280', '1280', '-', 'p=0.5'],
        ['Dense', '1280', '256', 'ReLU', '-'],
        ['BatchNorm', '256', '256', '-', 'Normalization'],
        ['Dropout', '256', '256', '-', 'p=0.3'],
        ['Dense', '256', '54', 'Softmax', '-'],
    ]
    elements.append(create_table(head_arch, col_widths=[80, 80, 70, 70, 90]))
    elements.append(Paragraph("Table 5: Custom Classification Head", styles['CaptionCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 6. TRANSFER LEARNING
    # =========================================================
    elements.append(Paragraph("6. Transfer Learning Approach", styles['Heading1Custom']))
    
    transfer_intro = """
    Transfer learning enables leveraging knowledge from a source domain (ImageNet image 
    classification) to improve performance on our target domain (bird audio classification).
    """
    elements.append(Paragraph(transfer_intro, styles['BodyCustom']))
    
    elements.append(Paragraph("6.1 Why Transfer Learning Works", styles['Heading2Custom']))
    
    why_transfer = """
    Despite the domain difference between natural images and audio spectrograms, transfer 
    learning is effective because:
    <br/><br/>
    <b>1. Low-level Feature Sharing:</b> Early CNN layers learn universal features (edges, 
    textures, gradients) that transfer well across domains.
    <br/><br/>
    <b>2. Hierarchical Representations:</b> The network learns increasingly abstract features 
    useful for both image and spectrogram analysis.
    <br/><br/>
    <b>3. Better Initialization:</b> Pre-trained weights provide a superior starting point 
    compared to random initialization, enabling faster convergence.
    """
    elements.append(Paragraph(why_transfer, styles['BodyCustom']))
    
    elements.append(Paragraph("6.2 Fine-tuning Strategy", styles['Heading2Custom']))
    
    # Transfer learning comparison
    transfer_comparison = [
        ['Approach', 'Layers Trained', 'When to Use', 'Our Choice'],
        ['Feature Extraction', 'Only classifier', 'Very limited data', '✗'],
        ['Partial Fine-tuning', 'Top layers only', 'Limited data', '✗'],
        ['Full Fine-tuning', 'All layers', 'Sufficient data', '✓'],
    ]
    elements.append(create_table(transfer_comparison, col_widths=[110, 100, 130, 70]))
    elements.append(Paragraph("Table 6: Transfer Learning Strategies", styles['CaptionCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 7. OPTIMIZATION ALGORITHM
    # =========================================================
    elements.append(Paragraph("7. Optimization Algorithm: AdamW", styles['Heading1Custom']))
    
    adamw_intro = """
    AdamW (Adam with Decoupled Weight Decay) combines momentum-based optimization with 
    adaptive learning rates and proper L2 regularization.
    """
    elements.append(Paragraph(adamw_intro, styles['BodyCustom']))
    
    elements.append(Paragraph("7.1 Algorithm Steps", styles['Heading2Custom']))
    
    adamw_steps = """
    For each parameter θ at iteration t:
    <br/><br/>
    <b>1.</b> Compute gradient: g_t = ∇L(θ_{t-1})
    <br/><br/>
    <b>2.</b> Update first moment: m_t = β₁·m_{t-1} + (1-β₁)·g_t
    <br/><br/>
    <b>3.</b> Update second moment: v_t = β₂·v_{t-1} + (1-β₂)·g_t²
    <br/><br/>
    <b>4.</b> Bias correction: m̂_t = m_t/(1-β₁ᵗ), v̂_t = v_t/(1-β₂ᵗ)
    <br/><br/>
    <b>5.</b> Update: θ_t = θ_{t-1} - α·(m̂_t/(√v̂_t+ε) + λ·θ_{t-1})
    """
    elements.append(Paragraph(adamw_steps, styles['BodyCustom']))
    
    elements.append(Paragraph("7.2 Hyperparameters", styles['Heading2Custom']))
    
    adamw_params = [
        ['Parameter', 'Symbol', 'Value', 'Purpose'],
        ['Learning Rate', 'α', '0.0003', 'Step size'],
        ['Beta 1', 'β₁', '0.9', 'First moment decay'],
        ['Beta 2', 'β₂', '0.999', 'Second moment decay'],
        ['Epsilon', 'ε', '1e-8', 'Numerical stability'],
        ['Weight Decay', 'λ', '0.05', 'L2 regularization'],
    ]
    elements.append(create_table(adamw_params, col_widths=[100, 60, 70, 180]))
    elements.append(Paragraph("Table 7: AdamW Hyperparameters", styles['CaptionCustom']))
    
    elements.append(Paragraph("7.3 Learning Rate Schedule", styles['Heading2Custom']))
    
    lr_text = """
    We use <b>Cosine Annealing with Warm Restarts</b>, which periodically resets the learning 
    rate to escape local minima. The schedule follows:
    """
    elements.append(Paragraph(lr_text, styles['BodyCustom']))
    
    elements.append(Paragraph(
        "<i>η_t = η_min + ½(η_max - η_min)(1 + cos(T_cur/T_i · π))</i>",
        styles['EquationCustom']
    ))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 8. LOSS FUNCTIONS
    # =========================================================
    elements.append(Paragraph("8. Loss Functions", styles['Heading1Custom']))
    
    elements.append(Paragraph("8.1 Cross-Entropy Loss", styles['Heading2Custom']))
    
    ce_text = """
    The standard cross-entropy loss for multi-class classification:
    """
    elements.append(Paragraph(ce_text, styles['BodyCustom']))
    
    elements.append(Paragraph(
        "<i>L_CE = -Σ y_k · log(p_k)</i>",
        styles['EquationCustom']
    ))
    
    elements.append(Paragraph("8.2 Label Smoothing", styles['Heading2Custom']))
    
    ls_text = """
    Label smoothing prevents overconfident predictions by softening targets:
    """
    elements.append(Paragraph(ls_text, styles['BodyCustom']))
    
    elements.append(Paragraph(
        "<i>y_k^smooth = (1 - ε)·y_k + ε/K</i>",
        styles['EquationCustom']
    ))
    
    ls_example = """
    With ε = 0.1 and K = 54 classes:<br/>
    • Original: [0, 0, 1, 0, ..., 0] (one-hot)<br/>
    • Smoothed: [0.0019, 0.0019, 0.9019, 0.0019, ...]
    """
    elements.append(Paragraph(ls_example, styles['BodyCustom']))
    
    elements.append(Paragraph("8.3 Focal Loss (For Class Imbalance)", styles['Heading2Custom']))
    
    focal_text = """
    Focal Loss down-weights easy examples, focusing on hard cases:
    """
    elements.append(Paragraph(focal_text, styles['BodyCustom']))
    
    elements.append(Paragraph(
        "<i>L_FL = -α_t(1 - p_t)^γ · log(p_t)</i>",
        styles['EquationCustom']
    ))
    
    focal_params = """
    Where γ = 2.0 (focusing parameter). When γ = 0, focal loss equals cross-entropy.
    """
    elements.append(Paragraph(focal_params, styles['BodyCustom']))
    
    # Loss comparison
    loss_comparison = [
        ['Loss Function', 'Handles Imbalance', 'Prevents Overconfidence'],
        ['Cross-Entropy', '✗', '✗'],
        ['CE + Label Smoothing', '✗', '✓'],
        ['Focal Loss', '✓', '✗'],
        ['Focal + Smoothing', '✓', '✓'],
    ]
    elements.append(create_table(loss_comparison, col_widths=[140, 120, 140]))
    elements.append(Paragraph("Table 8: Loss Function Comparison", styles['CaptionCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 9. DATA AUGMENTATION
    # =========================================================
    elements.append(Paragraph("9. Data Augmentation Techniques", styles['Heading1Custom']))
    
    aug_intro = """
    Data augmentation increases training diversity, improving generalization.
    """
    elements.append(Paragraph(aug_intro, styles['BodyCustom']))
    
    elements.append(Paragraph("9.1 SpecAugment", styles['Heading2Custom']))
    
    specaug_text = """
    SpecAugment (Park et al., 2019) applies masking directly to spectrograms:
    <br/><br/>
    <b>Frequency Masking:</b> Mask F consecutive mel bands<br/>
    <b>Time Masking:</b> Mask T consecutive time frames
    """
    elements.append(Paragraph(specaug_text, styles['BodyCustom']))
    
    specaug_params = [
        ['Parameter', 'Value', 'Description'],
        ['Freq Mask (F)', '20 bands', 'Max frequency bands to mask'],
        ['Time Mask (T)', '30 frames', 'Max time frames to mask'],
        ['Num Masks', '2 each', 'Masks per sample'],
        ['Probability', '0.5', 'Application probability'],
    ]
    elements.append(create_table(specaug_params, col_widths=[100, 80, 220]))
    elements.append(Paragraph("Table 9: SpecAugment Parameters", styles['CaptionCustom']))
    
    elements.append(Paragraph("9.2 Mixup", styles['Heading2Custom']))
    
    mixup_text = """
    Mixup (Zhang et al., 2018) creates virtual examples by interpolating pairs:
    """
    elements.append(Paragraph(mixup_text, styles['BodyCustom']))
    
    elements.append(Paragraph(
        "<i>x̃ = λx_i + (1-λ)x_j</i><br/><i>ỹ = λy_i + (1-λ)y_j</i>",
        styles['EquationCustom']
    ))
    
    elements.append(Paragraph(
        "Where λ ~ Beta(0.4, 0.4). This encourages linear behavior between examples.",
        styles['BodyCustom']
    ))
    
    elements.append(Paragraph("9.3 Audio Augmentations", styles['Heading2Custom']))
    
    audio_augs = [
        ['Augmentation', 'Range', 'Effect'],
        ['Time Shift', '±15%', 'Circular shift in time'],
        ['Pitch Shift', '±2 semitones', 'Change pitch'],
        ['Volume', '0.7x - 1.3x', 'Simulate distance'],
        ['Noise', 'SNR 15-30 dB', 'Add Gaussian noise'],
    ]
    elements.append(create_table(audio_augs, col_widths=[110, 100, 200]))
    elements.append(Paragraph("Table 10: Audio Augmentations", styles['CaptionCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 10. TRAINING METHODOLOGY
    # =========================================================
    elements.append(Paragraph("10. Training Methodology", styles['Heading1Custom']))
    
    elements.append(Paragraph("10.1 Dataset", styles['Heading2Custom']))
    
    dataset_stats = [
        ['Metric', 'Value'],
        ['Total Recordings', '1,926'],
        ['Total Chunks', '34,811 (5-second segments)'],
        ['Species', '54 North American birds'],
        ['Train Set', '23,630 chunks (70%)'],
        ['Validation Set', '4,718 chunks (15%)'],
        ['Test Set', '5,558 chunks (15%)'],
        ['Split Strategy', 'Recording-level (no leakage)'],
    ]
    elements.append(create_table(dataset_stats, col_widths=[150, 260]))
    elements.append(Paragraph("Table 11: Dataset Statistics", styles['CaptionCustom']))
    
    elements.append(Paragraph("10.2 Training Configuration", styles['Heading2Custom']))
    
    training_config = [
        ['Parameter', 'Value'],
        ['Epochs', '50 (early stopping)'],
        ['Batch Size', '32'],
        ['Optimizer', 'AdamW'],
        ['Learning Rate', '0.0003'],
        ['Weight Decay', '0.05'],
        ['Label Smoothing', '0.1'],
        ['Early Stop Patience', '15 epochs'],
        ['Gradient Clipping', '1.0'],
        ['Mixed Precision', 'FP16'],
        ['Hardware', 'NVIDIA RTX 3050 (4GB)'],
    ]
    elements.append(create_table(training_config, col_widths=[150, 260]))
    elements.append(Paragraph("Table 12: Training Configuration", styles['CaptionCustom']))
    
    elements.append(Paragraph("10.3 Regularization", styles['Heading2Custom']))
    
    reg_techniques = [
        "• <b>Dropout:</b> 50% before first dense layer, 30% before output",
        "• <b>Weight Decay:</b> L2 penalty (λ = 0.05)",
        "• <b>Label Smoothing:</b> Softens targets (ε = 0.1)",
        "• <b>Data Augmentation:</b> SpecAugment + Mixup",
        "• <b>Early Stopping:</b> Stops when validation plateaus",
        "• <b>Batch Normalization:</b> Normalizes layer inputs",
    ]
    for reg in reg_techniques:
        elements.append(Paragraph(reg, styles['BulletCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 11. RESULTS & ANALYSIS
    # =========================================================
    elements.append(Paragraph("11. Results & Analysis", styles['Heading1Custom']))
    
    elements.append(Paragraph("11.1 Overall Performance", styles['Heading2Custom']))
    
    results_table = [
        ['Metric', 'Validation', 'Test'],
        ['Top-1 Accuracy', '71.32%', '67.42%'],
        ['Top-3 Accuracy', '79.78%', '78.59%'],
        ['Top-5 Accuracy', '83.55%', '82.57%'],
        ['Macro F1', '-', '58.39%'],
        ['Weighted F1', '-', '66.42%'],
    ]
    elements.append(create_table(results_table, col_widths=[140, 110, 110]))
    elements.append(Paragraph("Table 13: Performance Metrics", styles['CaptionCustom']))
    
    elements.append(Paragraph("11.2 Performance by Sample Size", styles['Heading2Custom']))
    
    performance_by_samples = [
        ['Category', 'Mean F1', 'Species'],
        ['High (600+ samples)', '0.682', '11'],
        ['Medium (300-599)', '0.624', '24'],
        ['Low (100-299)', '0.503', '17'],
        ['Very Low (<100)', '0.256', '2'],
    ]
    elements.append(create_table(performance_by_samples, col_widths=[150, 100, 100]))
    elements.append(Paragraph("Table 14: Performance by Sample Count", styles['CaptionCustom']))
    
    elements.append(Paragraph("11.3 Best Performing Species", styles['Heading2Custom']))
    
    best_species = [
        ['Species', 'English Name', 'F1'],
        ['Myiarchus cinerascens', 'Ash-throated Flycatcher', '0.939'],
        ['Polioptila caerulea', 'Blue-gray Gnatcatcher', '0.926'],
        ['Corvus brachyrhynchos', 'American Crow', '0.911'],
        ['Selasphorus platycercus', 'Broad-tailed Hummingbird', '0.860'],
        ['Artemisiospiza belli', "Bell's Sparrow", '0.849'],
    ]
    elements.append(create_table(best_species, col_widths=[140, 180, 60]))
    elements.append(Paragraph("Table 15: Top 5 Species", styles['CaptionCustom']))
    
    elements.append(Paragraph("11.4 Challenging Species", styles['Heading2Custom']))
    
    worst_species = [
        ['Species', 'English Name', 'F1', 'Issue'],
        ['Bucephala albeola', 'Bufflehead', '0.00', 'Few samples'],
        ['Hirundo rustica', 'Barn Swallow', '0.00', 'Confusion'],
        ['Haliaeetus leucocephalus', 'Bald Eagle', '0.11', 'Similar calls'],
        ['Passerina caerulea', 'Blue Grosbeak', '0.14', 'Confusion'],
    ]
    elements.append(create_table(worst_species, col_widths=[130, 120, 50, 100]))
    elements.append(Paragraph("Table 16: Challenging Species", styles['CaptionCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # 12. CONCLUSIONS
    # =========================================================
    elements.append(Paragraph("12. Conclusions", styles['Heading1Custom']))
    
    conclusions_text = """
    This project successfully developed a deep learning system for multi-species bird 
    identification from audio recordings.
    <br/><br/>
    <b>Key Achievements:</b><br/>
    • Implemented EfficientNet-B0 CNN with transfer learning<br/>
    • Achieved 67.42% test accuracy on 54-class problem<br/>
    • Reached 82.57% top-5 accuracy<br/>
    • Reduced training time 5-10× via pre-computed spectrograms<br/>
    • Fixed critical data leakage in original pipeline
    <br/><br/>
    <b>Limitations:</b><br/>
    • Two species achieve 0% F1 due to limited samples<br/>
    • Class imbalance affects rare species performance<br/>
    • Single-label classification limits multi-species detection
    <br/><br/>
    <b>Future Work:</b><br/>
    • Collect more data for underrepresented species<br/>
    • Implement multi-label classification<br/>
    • Add attention mechanisms for temporal localization<br/>
    • Deploy as real-time monitoring application
    """
    elements.append(Paragraph(conclusions_text, styles['BodyCustom']))
    
    elements.append(PageBreak())
    
    # =========================================================
    # REFERENCES
    # =========================================================
    elements.append(Paragraph("References", styles['Heading1Custom']))
    elements.append(Spacer(1, 10))
    
    references = [
        "1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.",
        "2. Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method. Interspeech.",
        "3. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. ICLR.",
        "4. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.",
        "5. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.",
        "6. Szegedy, C., et al. (2016). Rethinking the Inception Architecture. CVPR.",
        "7. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.",
        "8. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.",
        "9. Kahl, S., et al. (2021). BirdNET: A deep learning solution for avian monitoring.",
        "10. Xeno-canto Foundation. (2024). www.xeno-canto.org",
    ]
    
    for ref in references:
        elements.append(Paragraph(ref, ParagraphStyle(
            'RefStyle',
            parent=styles['Normal'],
            fontSize=9,
            leftIndent=20,
            firstLineIndent=-20,
            spaceAfter=6
        )))
    
    # =========================================================
    # BUILD PDF
    # =========================================================
    doc.build(elements)
    print(f"\n✅ PDF Report generated successfully!")
    print(f"📄 Location: {OUTPUT_FILE}")
    print(f"📊 Pages: ~21")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📄 GENERATING DETAILED ALGORITHM REPORT")
    print("=" * 60)
    
    try:
        build_report()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()