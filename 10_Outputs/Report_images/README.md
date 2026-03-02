# Report Images Directory

This folder is for storing images to be included in the technical report PDF.

## How to Use

1. **Prepare your images** in PNG, JPG, JPEG, or GIF format
2. **Name them according to figure numbers**: `Figure_1.png`, `Figure_2.jpg`, etc.
3. **Place them in this directory** (Report_images)
4. **Run the report generator** to automatically embed them

## Expected Image Locations

The report generation script will look for:

- `Figure_1.png` - System Architecture diagram
- `Figure_2.png` - Mel-Spectrogram Conversion example
- `Figure_3.png` - Training Curves (loss and accuracy)
- `Figure_4.png` - Confusion Matrix heatmap
- `Figure_5.png` - Grad-CAM: Correct Prediction
- `Figure_6.png` - Grad-CAM: Incorrect Prediction
- `Figure_7.png` - Streamlit Application screenshot
- `Figure_8.png` - PDF Report Output example

## Image Specifications

For best results with PDF reports:
- **Resolution**: 300 DPI or higher
- **Size**: ~5 inches wide × 3.75 inches tall
- **Format**: PNG (best for diagrams), JPG (best for screenshots)
- **Color**: RGB or Grayscale

## What Happens

- ✅ **Images found**: Automatically embedded in the PDF report at 5" × 3.75"
- ⚠️ **Images missing**: Orange placeholder boxes shown with instructions
- ❌ **Loading fails**: Falls back to placeholder with error message

## Generated Report Features

✅ **Professional Formatting**
- Proper paragraph indentation (0.5 inch first line)
- Justified text alignment for body paragraphs
- Consistent spacing between sections

✅ **Enhanced Tables**
- Dark blue headers with white text
- Alternating row colors (white/light blue)
- Proper cell padding and grid lines
- Professional typography

✅ **Figure Integration**
- Automatically loads and embeds your images
- Professional captions below each figure
- Falls back to placeholders if images missing

## Creating Images

Recommended tools:
1. **System Architecture**: draw.io, Lucidchart, or PowerPoint
2. **Training Curves**: Matplotlib, TensorBoard exports, or Excel charts
3. **Confusion Matrix**: Scikit-learn plotting, Matplotlib, or online generators
4. **Grad-CAM Visualizations**: From model output, matplotlib, or Jupyter notebooks
5. **Screenshots**: Built-in OS tools or SnagIt

## Troubleshooting

- If images don't appear, check the filename matches exactly (case-sensitive)
- Verify image format is supported (.png, .jpg, .jpeg, .gif)
- Ensure image files are not corrupted
- Close any open PDF file before regenerating the report
