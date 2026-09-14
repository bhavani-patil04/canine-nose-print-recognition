# Pet Biometric QA & Feature Audit Pipeline

An automated quality assurance and field-testing framework designed to quantify the impact of specular glare-reduction pre-processing on canine biometric nose-print identification. 

This suite evaluates image feature match quality using **SIFT (Scale-Invariant Feature Transform)** keypoint descriptors, pair-matching matrices, and **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** metrics to ensure enhancement algorithms retain biometric integrity without introducing destructive artifacts.

---

## 📌 Features & Capabilities

* **Automated Pair Matrix Generation:** Automatically builds positive (same-identity) and negative (different-identity) evaluation pairs (`gt.csv`).
* **SIFT Keypoint Analysis:** Measures keypoint density and matching accuracy (`pred_orig.csv` vs. `pred_enh.csv`).
* **ROC-AUC Evaluation Engine:** Quantifies identity discrimination score performance before and after pre-processing.
* **Position-Based File Alignment:** Self-healing mapping logic that handles filename mismatches and extension variations between raw and enhanced output sets.
* **Dynamic Enhancement Execution:** Runs image enhancement routines dynamically over selected benchmark sets without altering core algorithm repositories.

---

## 🛠 Project Structure

```text
QA_Testing_Suite/
├── benchmark_set/
│   ├── original/             # Raw input benchmark images (high glare)
│   └── enhanced/             # Enhanced outputs (CLAHE + Sharpening)
├── outputs/
│   └── final_preprocessed/   # Staging directory for enhanced image outputs
├── results/                  # Generated CSV metrics and parameter logs
├── select_benchmark.py       # Utility to sample edge-case benchmark images
├── run_prep.py               # Wrapper runner to execute image enhancement
├── run_qa_audit.py           # Main SIFT feature audit & verification script
├── metric_calculation.py     # ROC-AUC metric evaluation module
├── .gitignore                # Workspace boundary rules
└── README.md                 # Technical documentation

```

---

## 🚀 Environment Requirements

* **Python Version:** `3.14` (or `3.10+`)
* **Dependencies:**
```bash
pip install opencv-python numpy pandas matplotlib

```



---

## 💻 Usage & Execution Commands

Run all commands from the root of the `QA_Testing_Suite` directory using your Python terminal.

### Step 1: Populate Benchmark Dataset

Sample high-glare edge-case images from the primary dataset archive:

```cmd
py -3.14 select_benchmark.py --count 30

```

### Step 2: Run Enhancement Pipeline

Execute the pre-processing pipeline over the benchmark set (applies CLAHE, unsharp masking, and specular glare reduction):

```cmd
py -3.14 run_prep.py

```

### Step 3: Synchronize Processed Images

Copy the generated enhancement outputs into the dedicated audit folder:

```cmd
py -3.14 -c "import shutil, glob, os; os.makedirs(r'.\benchmark_set\enhanced', exist_ok=True); [shutil.copy(f, r'.\benchmark_set\enhanced') for f in glob.glob(r'.\outputs\final_preprocessed\*.*')]; print('Enhanced set synchronized!')"

```

### Step 4: Execute SIFT QA Metric Audit

Run the main evaluation script to generate keypoint density reports, pair prediction matrices, and ROC-AUC verification scores:

```cmd
py -3.14 run_qa_audit.py

```

---

## 📊 Output Artifacts & Metrics

After running the audit script, the following evaluation files are generated in the root workspace:

| File Name | Description |
| --- | --- |
| `gt.csv` | Ground-truth binary labels (`1` for same-dog pairs, `0` for different dogs). |
| `pred_orig.csv` | SIFT match ratio similarity scores computed on original raw images. |
| `pred_enh.csv` | SIFT match ratio similarity scores computed on enhanced images. |

### Sample Terminal Audit Results

```text
--- SIFT QA METRIC AUDIT RESULTS ---
Original Dataset ROC-AUC Score : 1.0000
Enhanced Dataset ROC-AUC Score : 1.0000
ROC-AUC Score Variance         : +0.0000

--- FEATURE DETECTOR DETAILED ANALYSIS ---
Avg Keypoint Matches (Original): 5213.5
Avg Keypoint Matches (Enhanced): 8852.8
Keypoint Match Variance         : +3639.2 (+69.80%)

Verdict: PASS - Specular glare removal significantly improved keypoint retention.

```

---

## ⚙ Core Modules Explained

* **`run_qa_audit.py`:** Standardizes image loading via position-based array indexing, extracts SIFT descriptors, calculates bidirectional matches via `cv2.BFMatcher`, and determines keypoint retention percentage.
* **`metric_calculation.py`:** Evaluates prediction outputs against ground truth data to derive ROC-AUC scores.
* **`select_benchmark.py`:** Interfaces with external raw datasets (`Pet Biometric Challenge 2022`) to extract consistent test subsets for rapid algorithm validation.
* **`run_prep.py`:** Bridges the QA suite with the `image_enhancement` module dynamically without requiring code modifications to the enhancement scripts.
