# 🍾 Bottle Quality Inspection on a Production Line

## 📋 Overview
A **binary image classification** project that detects defective bottles (broken or contaminated) on a production line using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

## 🎯 Goal
Classify bottle images into:
- **`good`** — no visible defects
- **`defect`** — broken (large/small) or contaminated

## 📊 Dataset
Based on the **MVTec Bottle** dataset structure:

| Class | Count |
|-------|:---:|
| Good | 209 |
| Broken (large) | 20 |
| Broken (small) | 22 |
| Contamination | 21 |
| **Total defect** | **63** |
| **Total images** | **272** |

- Train/test split: 80/20, **stratified** (217 train / 55 test)
- Image size: resized to **224×224**, normalized to [0, 1]

## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `pandas` | Data handling |
| `opencv-python (cv2)` | Reading, color conversion (BGR→RGB), resizing images |
| `matplotlib` | Visualizing sample images, training curves, confusion matrix |
| `glob` | Collecting image file paths |
| `scikit-learn` | `train_test_split`, `confusion_matrix`, `ConfusionMatrixDisplay` |
| `tensorflow` / `keras` | Building and training the CNN, `EarlyStopping` callback |

## 🧠 Model Architecture
A CNN built with `Sequential`:

```
Conv2D(32, 3x3, relu) → MaxPooling2D
Conv2D(64, 3x3, relu) → MaxPooling2D
Conv2D(128, 3x3, relu) → MaxPooling2D
Flatten → Dense(128, relu) → Dense(1, sigmoid)
```
- **Total parameters:** 11,169,089
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Callback:** EarlyStopping (`patience=3`, monitors `val_loss`, restores best weights)
- **Training:** up to 20 epochs, batch size 32

## 🚀 Pipeline
1. Load and visualize sample "good" and defect images with OpenCV
2. Collect image paths for all classes via `glob`
3. Label images (`0` = good, `1` = defect) and combine into one dataset
4. Stratified train/test split
5. Preprocess: resize to 224×224, normalize pixel values
6. Build and train the CNN with early stopping
7. Evaluate on the test set
8. Visualize accuracy/loss curves, confusion matrix, and sample predictions

## 📈 Results
| Metric | Value |
|--------|:---:|
| **Test Accuracy** | **94.55%** |
| **Test Loss** | 0.189 |

Training stabilized quickly thanks to early stopping, and the confusion matrix + sample prediction grid confirm the model reliably separates `good` vs `defect` bottles.

## 📁 Files
- `bottle-quality-inspection-on-a-production-line.ipynb` — full notebook (data loading → CNN training → evaluation)

## ▶️ How to Run
```bash
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow
```
Update the dataset paths (`/kaggle/input/...`) to match your local/Colab environment, then run the notebook top to bottom.

## ✅ Status
**Completed** — CNN trained and evaluated, achieving 94.55% test accuracy.
