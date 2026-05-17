"""
knn.py
Runs the kNN distance-based anomaly detection pipeline on the CelebA dataset.
Saves scores and metrics to figures/ for use by plot_figures.py.

Usage:
    python src/knn.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix, average_precision_score,
    precision_recall_curve, roc_auc_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'celeba_baldvsnonbald_normalised.csv'
K         = 20
REF_SIZE  = 10_000
TEST_SIZE = 0.3
SEED      = 42


def load_and_split(path):
    df = pd.read_csv(path)
    X  = df.iloc[:, :-1]
    y  = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train anomalies: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Test  anomalies: {y_test.sum()} ({y_test.mean():.2%})")
    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test):
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def compute_scores(X_train_scaled, X_test_scaled, y_train):
    X_normal = X_train_scaled[y_train.values == 0]
    print(f"Normal training instances: {X_normal.shape[0]}")

    np.random.seed(SEED)
    ref_idx = np.random.choice(len(X_normal), size=REF_SIZE, replace=False)
    X_ref   = X_normal[ref_idx]
    print(f"Reference set size: {X_ref.shape[0]}")

    nn = NearestNeighbors(n_neighbors=K, metric='euclidean', n_jobs=-1)
    nn.fit(X_ref)
    print(f"kNN model fitted (k={K}).")

    scores = np.zeros(len(X_test_scaled))
    batch  = 5000
    for i in range(0, len(X_test_scaled), batch):
        chunk = X_test_scaled[i:i + batch]
        dists, _ = nn.kneighbors(chunk)
        scores[i:i + batch] = dists.mean(axis=1)
        print(f"  Scored {min(i + batch, len(X_test_scaled))}/{len(X_test_scaled)}")

    print(f"\nScore range:  {scores.min():.4f} to {scores.max():.4f}")
    return scores


def evaluate(scores, y_test):
    roc = roc_auc_score(y_test, scores)
    ap  = average_precision_score(y_test, scores)

    prec_v, rec_v, thresholds = precision_recall_curve(y_test, scores)
    f1_v        = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-9)
    best_idx    = f1_v.argmax()
    best_thresh = thresholds[best_idx]
    y_pred      = (scores >= best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    spec = tn / (tn + fp)
    fpr  = fp / (fp + tn)
    acc  = (tp + tn) / (tp + tn + fp + fn)

    print(f"\nROC-AUC:        {roc:.4f}")
    print(f"Avg Precision:  {ap:.4f}")
    print(f"Best threshold: {best_thresh:.4f}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"FPR:         {fpr:.4f}")
    print(f"F1:          {f1:.4f}")
    print(f"\nNormal mean:  {scores[y_test == 0].mean():.4f}")
    print(f"Anomaly mean: {scores[y_test == 1].mean():.4f}")
    print(f"Separation:   {scores[y_test==1].mean()/scores[y_test==0].mean():.2f}x")

    return {
        'scores': scores, 'y_pred': y_pred, 'y_test': y_test,
        'best_thresh': best_thresh, 'roc': roc, 'ap': ap,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'prec': prec, 'rec': rec, 'f1': f1,
        'spec': spec, 'fpr': fpr, 'acc': acc
    }


def main():
    print("=== Loading data ===")
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)

    print("\n=== Preprocessing ===")
    X_train_s, X_test_s = preprocess(X_train, X_test)

    print("\n=== Computing kNN anomaly scores (k=20) ===")
    scores = compute_scores(X_train_s, X_test_s, y_train)

    print("\n=== Evaluation ===")
    results = evaluate(scores, y_test)

    np.save('figures/knn_scores.npy', results['scores'])
    np.save('figures/knn_ytest.npy',
            results['y_test'].values
            if hasattr(results['y_test'], 'values') else results['y_test'])
    np.save('figures/knn_ypred.npy',  results['y_pred'])
    np.save('figures/knn_metrics.npy', np.array([
        results['best_thresh'], results['roc'], results['ap'],
        results['tp'], results['fp'], results['tn'], results['fn'],
        results['acc'], results['prec'], results['rec'],
        results['spec'], results['fpr'], results['f1']
    ]))
    print("\nResults saved to figures/. Run src/plot_figures.py to generate images.")


if __name__ == '__main__':
    main()
