"""
plot_figures.py
Loads saved results from gaussian.py and knn.py and generates all figures.

Run gaussian.py and knn.py first, then run this script.

Usage:
    python src/plot_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# ── colour palette ────────────────────────────────────────────────────
BLUE  = '#185FA5'
AMBER = '#BA7517'
RED   = '#A32D2D'
GRAY  = '#5F5E5A'
LGRAY = '#D3D1C7'
BG    = '#F8F7F4'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.facecolor':    BG,
    'figure.facecolor':  'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        LGRAY,
    'grid.linewidth':    0.5,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})

OUT = 'figures/'


# ── loaders ───────────────────────────────────────────────────────────

def load_gaussian():
    scores  = np.load(OUT + 'gaussian_scores.npy')
    y_test  = np.load(OUT + 'gaussian_ytest.npy')
    y_pred  = np.load(OUT + 'gaussian_ypred.npy')
    m       = np.load(OUT + 'gaussian_metrics.npy')
    return scores, y_test, y_pred, m


def load_knn():
    scores  = np.load(OUT + 'knn_scores.npy')
    y_test  = np.load(OUT + 'knn_ytest.npy')
    y_pred  = np.load(OUT + 'knn_ypred.npy')
    m       = np.load(OUT + 'knn_metrics.npy')
    return scores, y_test, y_pred, m


# ── figure functions ──────────────────────────────────────────────────

def plot_score_distribution(scores, y_test, threshold, bins,
                             xlabel, title, filename):
    """Histogram of anomaly scores split by class."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(scores[y_test == 0], bins=bins, density=True,
            color=BLUE,  alpha=0.65, edgecolor='none', label='Normal (class 0)')
    ax.hist(scores[y_test == 1], bins=bins, density=True,
            color=AMBER, alpha=0.75, edgecolor='none', label='Anomaly / Bald (class 1)')
    ax.axvline(threshold, color=RED, lw=1.6, ls='--',
               label=f'Best threshold = {threshold:.2f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT + filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')


def plot_confusion_matrix(tp, fp, tn, fn, threshold, title, filename):
    """Colour-coded confusion matrix."""
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap='Blues', aspect='auto')
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{labels[i][j]}\n{cm[i, j]:,}',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else GRAY)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Normal', 'Predicted Anomaly'], fontsize=9)
    ax.set_yticklabels(['Actual Normal', 'Actual Anomaly'], fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT + filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')


def plot_roc(scores, y_test, roc_val, rec, fpr_op, title, filename, color):
    """ROC curve with operating point marked."""
    fpr_arr, tpr_arr, _ = roc_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr_arr, tpr_arr, color=color, lw=2,
            label=f'AUC = {roc_val:.4f}')
    ax.plot([0, 1], [0, 1], color=LGRAY, lw=1, ls='--',
            label='Random (AUC = 0.50)')
    ax.scatter([fpr_op], [rec], s=70, color=RED, zorder=5,
               label=f'Operating point (recall = {rec:.3f})')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT + filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')


def plot_pr(scores, y_test, rec, prec, title, filename, color):
    """Precision-Recall curve with best F1 point marked."""
    p_arr, r_arr, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(r_arr, p_arr, color=color, lw=2)
    ax.axhline(y_test.mean(), color=LGRAY, ls='--', lw=1,
               label=f'Baseline (prevalence = {y_test.mean():.3f})')
    ax.scatter([rec], [prec], s=70, color=RED, zorder=5,
               label='Best F1 point')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT + filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')


def plot_boxplot(scores, y_test, threshold, ylabel, title, filename):
    """Boxplot of anomaly scores by class."""
    fig, ax = plt.subplots(figsize=(5, 3.8))
    bp = ax.boxplot(
        [scores[y_test == 0], scores[y_test == 1]],
        labels=['Normal\n(class 0)', 'Anomaly / Bald\n(class 1)'],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='white', linewidth=2)
    )
    for patch, col in zip(bp['boxes'], [BLUE, AMBER]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    for elem in ['whiskers', 'caps', 'fliers']:
        for item in bp[elem]:
            item.set(color=GRAY, linewidth=0.8)
    ax.axhline(threshold, color=RED, ls='--', lw=1.2,
               label=f'Threshold = {threshold:.2f}')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT + filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')


def plot_comparison(g_metrics, k_metrics):
    """Side-by-side bar chart comparing both methods across all metrics."""
    labels = ['ROC-AUC', 'Recall\n(Sensitivity)', 'Specificity',
              'Precision', 'FPR', 'F1']

    # gaussian_metrics indices: 0=thresh,1=roc,2=ap,3=tp,4=fp,5=tn,6=fn,
    #                           7=acc,8=prec,9=rec,10=spec,11=fpr,12=f1
    gauss_vals = [g_metrics[1], g_metrics[9], g_metrics[10],
                  g_metrics[8], g_metrics[11], g_metrics[12]]
    knn_vals   = [k_metrics[1], k_metrics[9], k_metrics[10],
                  k_metrics[8], k_metrics[11], k_metrics[12]]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.2))
    b1 = ax.bar(x - w / 2, gauss_vals, w, color=BLUE,  alpha=0.85,
                label='Parametric Gaussian (Deanna)', edgecolor='none')
    b2 = ax.bar(x + w / 2, knn_vals,   w, color=AMBER, alpha=0.85,
                label='kNN Distance-Based (Bethusile)', edgecolor='none')

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7.5, color=GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title(
        'Figure 12: Performance comparison — Parametric Gaussian vs kNN Distance-Based',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT + 'fig12_comparison.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('Saved fig12_comparison.png')


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=== Loading Gaussian results ===")
    g_scores, g_ytest, g_ypred, g_m = load_gaussian()

    print("=== Loading kNN results ===")
    k_scores, k_ytest, k_ypred, k_m = load_knn()

    print("\n=== Generating Gaussian figures ===")

    plot_score_distribution(
        scores=g_scores, y_test=g_ytest,
        threshold=g_m[0],
        bins=np.linspace(g_scores.min(), g_scores.max(), 80),
        xlabel='Negative log-likelihood (anomaly score)',
        title='Figure 1: Gaussian anomaly score distribution by class',
        filename='fig1_gaussian_scores.png'
    )

    plot_confusion_matrix(
        tp=int(g_m[3]), fp=int(g_m[4]), tn=int(g_m[5]), fn=int(g_m[6]),
        threshold=g_m[0],
        title=f'Figure 2: Confusion matrix (Gaussian, threshold = {g_m[0]:.2f})',
        filename='fig2_gaussian_cm.png'
    )

    plot_roc(
        scores=g_scores, y_test=g_ytest,
        roc_val=g_m[1], rec=g_m[9], fpr_op=g_m[11],
        title='Figure 3: ROC curve (Parametric Gaussian)',
        filename='fig3_gaussian_roc.png',
        color=BLUE
    )

    plot_pr(
        scores=g_scores, y_test=g_ytest,
        rec=g_m[9], prec=g_m[8],
        title='Figure 4: Precision-Recall curve (Parametric Gaussian)',
        filename='fig4_gaussian_pr.png',
        color=BLUE
    )

    plot_boxplot(
        scores=g_scores, y_test=g_ytest,
        threshold=g_m[0],
        ylabel='Negative log-likelihood (anomaly score)',
        title='Figure 5: Gaussian score distribution by class',
        filename='fig5_gaussian_box.png'
    )

    print("\n=== Generating kNN figures ===")

    plot_score_distribution(
        scores=k_scores, y_test=k_ytest,
        threshold=k_m[0],
        bins=np.linspace(0, 10, 80),
        xlabel='Average distance to k=20 nearest neighbours (anomaly score)',
        title='Figure 7: kNN anomaly score distribution by class',
        filename='fig7_knn_scores.png'
    )

    plot_confusion_matrix(
        tp=int(k_m[3]), fp=int(k_m[4]), tn=int(k_m[5]), fn=int(k_m[6]),
        threshold=k_m[0],
        title=f'Figure 8: Confusion matrix (kNN, threshold = {k_m[0]:.2f})',
        filename='fig8_knn_cm.png'
    )

    plot_roc(
        scores=k_scores, y_test=k_ytest,
        roc_val=k_m[1], rec=k_m[9], fpr_op=k_m[11],
        title='Figure 9: ROC curve (kNN Distance-Based)',
        filename='fig9_knn_roc.png',
        color=AMBER
    )

    plot_pr(
        scores=k_scores, y_test=k_ytest,
        rec=k_m[9], prec=k_m[8],
        title='Figure 10: Precision-Recall curve (kNN Distance-Based)',
        filename='fig10_knn_pr.png',
        color=AMBER
    )

    plot_boxplot(
        scores=k_scores, y_test=k_ytest,
        threshold=k_m[0],
        ylabel='Average kNN distance (anomaly score)',
        title='Figure 11: kNN score distribution by class',
        filename='fig11_knn_box.png'
    )

    print("\n=== Generating comparison figure ===")
    plot_comparison(g_m, k_m)

    print("\nAll figures saved to figures/")


if __name__ == '__main__':
    main()
