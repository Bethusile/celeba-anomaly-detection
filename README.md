# CelebA Anomaly Detection

WDWV401 Assignment 4 — Deanna Smith & Bethusile Mafumana

Anomaly detection on the CelebA Bald vs Non-Bald dataset using two methods:

- **Method 1:** Parametric Gaussian (Deanna Smith)
- **Method 2:** kNN Distance-Based (Bethusile Mafumana)

---

## Folder structure

```
celeba-anomaly-detection/
├── celeba_baldvsnonbald_normalised.csv   ← dataset (add this yourself)
├── requirements.txt
├── src/
│   ├── gaussian.py       ← Method 1: runs Gaussian pipeline, saves results
│   ├── knn.py            ← Method 2: runs kNN pipeline, saves results
│   └── plot_figures.py   ← generates all figures from saved results
└── figures/              ← all outputs saved here (created automatically)
```

---

## Setup

```bash
python -m venv venv --upgrade-deps
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

pip install -r requirements.txt
```

---

## Running the pipeline

Run the two analysis scripts first, then the plotting script.
All three must be run from the project root folder.

```bash
python src/gaussian.py       # Method 1
python src/knn.py            # Method 2
python src/plot_figures.py   # generates all 12 figures
```

---

## Output figures

| File | Description |
|------|-------------|
| fig1_gaussian_scores.png | Gaussian score distribution by class |
| fig2_gaussian_cm.png | Gaussian confusion matrix |
| fig3_gaussian_roc.png | Gaussian ROC curve |
| fig4_gaussian_pr.png | Gaussian Precision-Recall curve |
| fig5_gaussian_box.png | Gaussian score boxplot |
| fig7_knn_scores.png | kNN score distribution by class |
| fig8_knn_cm.png | kNN confusion matrix |
| fig9_knn_roc.png | kNN ROC curve |
| fig10_knn_pr.png | kNN Precision-Recall curve |
| fig11_knn_box.png | kNN score boxplot |
| fig12_comparison.png | Side-by-side comparison of both methods |

---

## Dataset

The dataset was sourced from: 


