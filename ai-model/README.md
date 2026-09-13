## Living Bridges - AI Model

The AI Model component of the Living Bridges Stray Dog Tracking and Identification System handles canine nose-print feature extraction and similarity-based recognition.

---

## 🌟 Overview

The **Living Bridges AI Model** identifies canine nose prints by converting nose-print images into numerical feature representations and comparing them with a stored feature database.

The implemented pipeline is:

* 🐕 **Nose-print image input**
* 🔍 **Image quality assessment** using Laplacian variance
* 🖼️ **Image preprocessing** and resizing to `224 × 224`
* 🧠 **ResNet50 feature extraction** using pretrained ImageNet weights
* 📊 **2048-dimensional feature vector generation**
* 🗄️ **Feature database** containing extracted representations
* 🔗 **Cosine similarity matching**
* 🆔 **Best matching Dog ID retrieval**

---

## 🚀 Features

* **Image Quality Assessment**: Uses Laplacian variance to assess image sharpness.
* **Image Preprocessing**: Converts images to RGB, resizes them to `224 × 224`, and applies ResNet50 preprocessing.
* **Pretrained ResNet50**: Uses ResNet50 with ImageNet weights as a feature extractor.
* **2048-Dimensional Features**: Generates a 2048-dimensional feature vector for each nose-print image.
* **Feature Database**: Stores feature vectors, Dog IDs, and image names using NumPy `.npy` files.
* **Cosine Similarity Matching**: Compares a query nose-print feature against stored feature vectors.
* **Dog ID Retrieval**: Returns the Dog ID associated with the highest similarity score.
* **Dataset Cleaning**: Includes duplicate analysis and creation of a clean nose-print dataset.

---

## 🛠️ Tech Stack

* **Programming Language**: Python 3.10+
* **Deep Learning Framework**: TensorFlow / Keras
* **Feature Extractor**: ResNet50
* **Pretrained Weights**: ImageNet
* **Image Processing**: OpenCV
* **Numerical Processing**: NumPy
* **Similarity Matching**: Scikit-learn
* **Image Handling**: Pillow
* **Feature Storage**: NumPy `.npy` files

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/bhavani-patil04/canine-nose-print-recognition.git
cd canine-nose-print-recognition/ai-model
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the AI Model

All commands should be executed from the `ai-model` directory.

### Test Image Preprocessing

```bash
python src/test_preprocessing.py
```

This verifies that the input image is converted to the required format:

```text
(1, 224, 224, 3)
```

### Test ResNet50 Feature Extraction

```bash
python src/test_feature_extractor.py
```

The expected feature output is:

```text
(1, 2048)
```

### Generate the Feature Database

```bash
python src/extract_all_features.py
```

This processes the clean dataset and generates:

```text
features/
├── features.npy
├── dog_ids.npy
└── image_names.npy
```

### Test Similarity Matching

```bash
python src/test_similarity.py
```

### Test Same-Dog vs Different-Dog Similarity

```bash
python src/test_similarity_pairs.py
```

### Analyze Similarity Distribution

```bash
python src/analyze_similarity_distribution.py
```

---

## 🗄️ Feature Database

The extracted feature database contains:

```text
features/
├── features.npy
├── dog_ids.npy
└── image_names.npy
```

### `features.npy`

Contains the 2048-dimensional feature vector for each stored image.

```text
Shape: (2892, 2048)
```

### `dog_ids.npy`

Contains the Dog ID corresponding to each feature vector.

### `image_names.npy`

Contains the image filename corresponding to each feature vector.

The arrays maintain the same index relationship:

```text
features[i]
     ↓
dog_ids[i]
     ↓
image_names[i]
```

## 📈 Current Validation

The implemented pipeline has been tested at multiple stages.

### Preprocessing

```text
Input Shape:  (1, 224, 224, 3)
Output Type: float32
```

### Feature Extraction

```text
Input:
(1, 224, 224, 3)

Output:
(1, 2048)
```

### Feature Database

```text
Images processed: 2,892
Feature vectors: 2,892
Feature dimension: 2,048
Failed images: 0
```

## 📁 Project Structure

```text
ai-model/
│
├── dataset/
│   ├── pet_biometric_challenge_2022/
│   ├── pet_biometric_available/
│   └── pet_biometric_clean/
│
├── features/
│   ├── features.npy
│   ├── dog_ids.npy
│   └── image_names.npy
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extractor.py
│   ├── extract_all_features.py
│   ├── similarity.py
│   ├── test_preprocessing.py
│   ├── test_feature_extractor.py
│   ├── test_similarity.py
│   ├── test_similarity_pairs.py
│   └── analyze_similarity_distribution.py
│
├── requirements.txt
└── README.md

## **License**

This project is part of the **Living Bridges Stray Dog Tracking and Identification System**.
