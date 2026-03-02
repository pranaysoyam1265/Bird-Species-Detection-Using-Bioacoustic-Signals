# Bird Detection Technical Report

This folder contains the generated technical reports for the Bird Species Detection project.

## Reports Generated

### 1. Bird_Detection_Technical_Report.docx
**Status**: ✅ Successfully generated
**Format**: Microsoft Word (.docx)
**Usage**: 
- Open in Microsoft Word or compatible editor
- Edit text, formatting, and layout freely
- Replace image placeholders with actual images
- Generate table of contents automatically in Word

### 2. Bird_Detection_Technical_Report.pdf
**Format**: PDF (.pdf)
**Features**:
- Professional formatting with proper page structure
- Automatically embeds images from `Report_images/` folder
- Optimized for printing and sharing
- Cannot be edited (use DOCX for edits, then export as PDF)

## Report Contents

### Title & Abstract
- Report title with author and date
- Abstract with keywords
- Professional cover page

### 1. Introduction
- Project overview
- Objectives and motivations
- Research questions

### 2. Methodology
- **System Architecture**: End-to-end pipeline diagram
  - Audio input → Preprocessing → Feature extraction → Model → Predictions + Grad-CAM
- **Dataset**: 4,521 bird recordings, 54 species, 34,811 audio chunks
- **Audio Preprocessing**: 5-second chunks, 50% overlap, standardization
- **Feature Extraction**: 128 mel-spectrogram, 22,050 Hz sampling
- **Model Architecture**: EfficientNet-B0 (4.07M parameters)
- **Training Configuration**: AdamW optimizer, label smoothing, mixed precision
- **Explainability**: Grad-CAM visualization methodology

### 3. Results
- **Training Performance**: 99.83% train accuracy, 68.79% val accuracy (epoch 10)
- **Test Evaluation**: 72.26% top-1, 82.83% top-3, 85.55% top-5 accuracy
- **Per-Species Performance**: Individual metrics for each species
- **Grad-CAM Results**: Visualization of model attention mechanisms
- **Real-World Validation**: Test on unseen recordings
- **Application**: Streamlit web interface for end-users

### 4. Discussion
- Model strengths and limitations
- Performance analysis
- Future improvements
- Deployment considerations

### 5. Future Work & Conclusion
- Potential enhancements
- Deployment roadmap
- Final remarks

## How to Use Images

The script automatically manages images:

1. **Place images** in `Report_images/` folder
2. **Name them**: `Figure_1.png`, `Figure_2.jpg`, etc. (matching Figure numbers in report)
3. **Supported formats**: PNG, JPG, JPEG, GIF
4. **Run generator** to embed images
5. **Missing images**: Orange placeholders shown with instructions

See `Report_images/README.md` for detailed instructions.

## Professional Formatting Features

### Text Formatting
✅ **Proper Indentation**
- First-line indent: 0.5 inches
- Justified alignment for body text
- Consistent line spacing (16pt)

✅ **Typography**
- Headers: Times Bold (sizes 18pt, 14pt, 12pt)
- Body: Times Roman (11pt)
- Captions: Times Italic (10pt)

✅ **Spacing**
- Sections properly spaced with breaks
- Paragraphs separated for readability
- Spacers before/after figures and tables

### Tables
✅ **Professional Styling**
- Dark blue headers (#203040) with white text
- Alternating row colors (white, light blue)
- 8pt padding on all cells
- Grid lines for clarity
- Proper header formatting (bold, 10pt)

✅ **Table Types**
1. **Dataset Statistics** - Recording counts and splits
2. **Spectrogram Parameters** - Audio processing settings
3. **Training Configuration** - Model hyperparameters
4. **Training History** - Epoch-by-epoch progression
5. **Test Performance** - Accuracy metrics
6. **Real-World Tests** - Validation results

### Figures
✅ **Automatic Image Embedding**
- Loads PNG, JPG, JPEG, GIF files
- Scales images to 5" × 3.75"
- Adds professional captions
- Shows placeholders if images missing

## Printing Guidelines

For best results when printing:
1. Use 11" × 8.5" paper (standard letter)
2. Set margins to 1 inch (0.72" in PDF)
3. Use serif font (Times Roman) for readability
4. Ensure images are high resolution (300 DPI minimum)
5. Color is recommended (for tables and figures)
6. Print on standard white paper for professional appearance

## Customization

### Edit DOCX
1. Open `Bird_Detection_Technical_Report.docx` in Word
2. Edit text, formatting, colors as needed
3. Replace image placeholders by:
   - Right-click placeholder image
   - Select "Change Picture" (Word 2016+) or "Replace Picture"
   - Choose actual image file
4. Generate table of contents: **References** → **Table of Contents**
5. Save and export as PDF

### Regenerate PDF
1. Update `Report_images/` with new images
2. Run `python 09_Utils/Scripts/generate_report.py`
3. Close any open PDF files first (Windows lock issue)
4. New PDF will be created with latest images

## Technical Details

- **Generator Script**: `09_Utils/Scripts/generate_report.py`
- **PDF Library**: ReportLab (Platypus)
- **DOCX Library**: python-docx
- **Output Directory**: `10_Outputs/Reports/`
- **Image Directory**: `10_Outputs/Report_images/`
- **Report Metadata**: Configured in script (author, institution, date)

## Troubleshooting

**PDF won't generate**
→ Close any open PDF if Windows says "permission denied"
→ Run script again (it should work now)

**Images not showing in PDF**
→ Check filenames match exactly: `Figure_1.png`, `Figure_2.png`, etc.
→ Verify files are in `Report_images/` folder
→ Ensure image files are not corrupted
→ Run generator again

**Text looks unformatted**
→ Check if styles were applied correctly
→ For DOCX: Manually apply "Normal", "Heading 1", "Heading 2" styles
→ For PDF: Regenerate (should auto-format)

**Tables misaligned**
→ Column widths auto-calculated based on content
→ For DOCX: Manually adjust in Word
→ For PDF: Regenerate with different data

## Contact & Support

For issues with report generation:
1. Check this README
2. Verify image files and naming
3. Check script output messages
4. Ensure Python packages are installed: `python-docx`, `reportlab`

---

**Last Updated**: February 9, 2026
**Report Version**: 1.0 (Improved Formatting & Image Support)
