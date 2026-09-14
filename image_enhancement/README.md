# Canine Nose Image Analysis

A staged computer-vision workflow for assessing and enhancing canine nose images.

## Setup

```powershell
cd .\canine_nose_image_analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place source images in `dataset/`, then run:

```powershell
python 01_quality_analysis.py
python 02_clahe.py
python 03_sharpening.py
python 04_denoising.py
python 05_edge_analysis.py
python 06_resnet50.py
```

The preprocessing stages write to `outputs/`; metrics go to `results/`, and edge visualizations go to `plots/`. The ResNet-50 stage downloads pretrained weights on its first run and provides baseline ImageNet predictions, not canine disease diagnosis.
