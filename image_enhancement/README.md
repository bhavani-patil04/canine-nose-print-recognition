# 🐕 Canine Nose Image Analysis & Enhancement

An image-processing and deep-feature evaluation pipeline for improving the quality of canine nose images before recognition and identification.

The project focuses on handling common image-quality challenges such as low contrast, blur, uneven illumination, wet/shiny surfaces, glare, and loss of fine nose texture.

---

## 📌 Project Overview

The canine nose contains unique visual characteristics such as:

- Nose texture
- Ridge patterns
- Pores
- Creases
- Fine edges
- Surface patterns

However, these features can become difficult to analyze when images contain poor lighting, blur, glare, or low contrast.

This project develops and evaluates an image preprocessing pipeline to improve the visibility and quality of canine nose images while preserving useful visual information for downstream recognition systems.

### 🔄 Processing Pipeline

```text
Canine Nose Dataset
        ↓
Image Quality Analysis
        ↓
Glare Detection
        ↓
Glare Correction
        ↓
CLAHE Enhancement
        ↓
Sharpening
        ↓
Nose ROI Analysis
        ↓
Final Preprocessing Pipeline
        ↓
ResNet50 Feature Extraction
        ↓
Feature Similarity Analysis
        ↓
Final Validation

```
🎯 Objectives

The main objectives of this project are:

Analyze the quality of canine nose images.
Detect glare caused by wet or shiny nose surfaces.
Correct excessive glare while preserving image details.
Improve local contrast using CLAHE.
Enhance fine details using image sharpening.
Optimize CLAHE parameters experimentally.
Optimize sharpening parameters experimentally.
Compare original and enhanced images quantitatively.
Extract deep visual features using ResNet50.
Evaluate whether preprocessing preserves useful visual information.
Establish an optimized preprocessing pipeline for future canine nose recognition.

📊 Dataset

The experiments were performed on a canine image dataset containing:

3,005 images

The dataset was used for:

Image quality analysis
Image enhancement
Glare detection
Glare correction
Parameter optimization
Deep feature evaluation

Dataset structure:

dataset/
└── train/
    └── images/
    
🛠️ Technologies Used
Technology	Purpose
Python	Core development
OpenCV	Image processing
NumPy	Numerical computation
Pandas	Data analysis
Matplotlib	Data visualization
Scikit-image	Image analysis
Pillow	Image handling
PyTorch	Deep learning
Torchvision	ResNet50
Scikit-learn	Feature similarity analysis

🔬 Image Enhancement
1. CLAHE Enhancement

Contrast Limited Adaptive Histogram Equalization (CLAHE) is used to improve local image contrast.

Instead of applying global histogram equalization, CLAHE operates on local regions of the image and limits excessive contrast amplification.

Processing
Input Image
     ↓
BGR → LAB
     ↓
Extract Luminance Channel
     ↓
CLAHE
     ↓
Merge L + A + B
     ↓
LAB → BGR
     ↓
Enhanced Image

CLAHE is useful for improving the visibility of texture and local details in canine nose images.

🔍 2. Image Sharpening

Gaussian-blur-based unsharp masking is used to enhance fine details and edges.

Original Image
      ↓
Gaussian Blur
      ↓
Unsharp Masking
      ↓
Sharpened Image

Different sharpening strengths and Gaussian sigma values were tested to identify effective configurations.

💡 3. Glare Detection

Wet or shiny canine noses can produce bright reflections that may interfere with image analysis.

A brightness-based glare detection method was developed to identify highly illuminated regions.

Images were categorized into:

Low Glare
Moderate Glare
High Glare
Results

A total of:

3,005 images

were analyzed.

Glare Level	Number of Images
Low	2,390
Moderate	448
High	167

Average detected glare:

1.05%

🛠️ 4. Glare Correction

After detecting glare regions, a correction stage was developed to reduce excessive bright reflections.

Pipeline
Input Image
      ↓
Brightness Analysis
      ↓
Glare Mask Generation
      ↓
Mask Refinement
      ↓
Glare Correction
      ↓
Corrected Image

Results
Metric	Value
Average Original Glare	1.55%
Average Corrected Glare	0.02%
Average Glare Reduction	1.54 percentage points

The bright-pixel percentage decreased by approximately:

98.98%

during glare quality validation.

⚙️ 5. CLAHE Parameter Optimization

A total of 12 CLAHE configurations were evaluated.

Clip Limits Tested
1.0
2.0
3.0
4.0
Tile Grid Sizes Tested
(4, 4)
(8, 8)
(16, 16)
Best Configuration Based on Sharpness
Clip Limit = 4.0
Tile Grid Size = (8, 8)
Result
Metric	Value
Brightness	98.63
Contrast	56.13
Sharpness	2228.98
Edge Density	0.2067
Dark Pixels	18.14%
Bright Pixels	3.34%

This configuration produced the highest measured sharpness among the tested CLAHE configurations.

🔧 6. Sharpening Parameter Optimization

A total of 12 sharpening configurations were evaluated.

Strength Values
1.1
1.3
1.5
1.7
Sigma Values
1
2
3
Best Configuration Based on Sharpness
Strength = 1.7
Sigma = 3
Result
Metric	Value
Brightness	71.30
Contrast	47.32
Sharpness	1101.35
Edge Density	0.1128
Dark Pixels	36.19%
Bright Pixels	1.68%

📈 7. Enhancement Comparison

The preprocessing methods were compared against the original images.

Percentage Change from Original
Method	Brightness	Contrast	Sharpness	Edge Density
CLAHE	+24.03%	+16.34%	+222.99%	+146.60%
Sharpened	+0.02%	+4.49%	+123.51%	+61.97%
CLAHE + Sharpening	+23.99%	+27.08%	+595.68%	+227.18%

The combined CLAHE + Sharpening approach produced the largest measured increase in contrast, sharpness, and edge density.

📊 8. Image Quality Metrics

The project evaluates multiple image-quality indicators.

Brightness

Measures the average intensity of the image.

Contrast

Measures the variation in pixel intensity.

Sharpness

Estimated using Laplacian variance.

Higher values generally indicate stronger high-frequency image details.

Edge Density

Calculated using Canny edge detection.

This provides an estimate of detected edge information.

Dark Pixel Percentage

Measures the percentage of pixels below a defined intensity threshold.

Bright Pixel Percentage

Measures highly illuminated pixels and is especially useful for evaluating glare.

🐕 9. Nose ROI Analysis

A Region of Interest (ROI) stage is used to focus analysis on the canine nose instead of the entire image.

Full Image
     ↓
Nose Region
     ↓
ROI Extraction
     ↓
Image Enhancement
     ↓
Feature Extraction

This helps reduce the influence of irrelevant background regions and allows future experiments to focus specifically on the visual characteristics of the canine nose.

🧠 10. ResNet50 Feature Evaluation

A pretrained ResNet50 network was used as a deep feature extractor.

The classification layer was removed so that each image could be represented by a:

2048-dimensional feature vector

Processing
Canine Nose Image
       ↓
Preprocessing
       ↓
ResNet50
       ↓
2048-D Feature Vector
       ↓
Feature Comparison

Original and enhanced images were compared using deep feature representations.

📌 ResNet50 Results

Total images evaluated:

3,005

Metric	Result
Average Cosine Similarity	0.656387
Minimum Cosine Similarity	0.303443
Maximum Cosine Similarity	0.969096
Average Feature Difference	0.111889
Original Feature Norm	14.226354
Enhanced Feature Norm	14.225070

The original and enhanced feature norms remained very close:

Original  = 14.226354
Enhanced  = 14.225070

This indicates that the overall magnitude of the ResNet50 feature representation remained stable after preprocessing.

📊 11. ResNet50 Visualizations

The project generates several visualizations for deep-feature analysis:

Cosine similarity distribution
Feature difference distribution
Feature norm comparison
Feature mean comparison
Similarity vs feature difference

Generated visualizations are stored in:

outputs/resnet50/

🔮 Future Work

The next phase will focus on:

Final quantitative validation of the complete preprocessing pipeline.
Selection of the most suitable preprocessing configuration.
Improved canine nose ROI localization.
Comparison of full-image and nose-ROI preprocessing.
Testing preprocessing performance on different image-quality conditions.
ResNet50-based recognition and similarity experiments.
Evaluation of whether image enhancement improves canine nose identification accuracy.
Integration with the complete canine nose recognition system.


The pipeline combines:

Glare Analysis
      +
Glare Correction
      +
CLAHE Optimization
      +
Sharpening Optimization
      +
Nose ROI Analysis
      +
ResNet50 Feature Evaluation

The resulting analysis provides a quantitative foundation for selecting an effective preprocessing pipeline before canine nose recognition.
