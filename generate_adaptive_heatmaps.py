"""
This file generates cross-method AUROC heatmaps
"""


import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

QWEN  = "scripts/matrix_supervised_qwen.json"
LLAMA = "scripts/matrix_supervised_llama.json"
OUT = Path("figures_adaptive")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 14, 'axes.labelsize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 200, 'savefig.bbox': 'tight',
})

NICE = {
    'ANCESTRAL': 'Ancestral', 'TEMP_09': 'Temp 0.9', 'TEMP_13': 'Temp 1.3',
    'REP_105': 'Rep 1.05', 'REP_130': 'Rep 1.30', 'TOP_K_50': 'Top-k 50',
    'TOP_K_1000': 'Top-k 1000', 'TOP_P_03': 'Top-p 0.3', 'TOP_P_095': 'Top-p 0.95',
    'TYPICAL_03': 'Typical 0.3', 'TYPICAL_095': 'Typical 0.95', 'ETA_1E4': 'Eta 1e-4',
    'ETA_10': 'Eta 0.1', 'CONTRASTIVE_K8_A06': 'Contrastive', 'CFG_20': 'CFG 2.0',
    'P_LESS': 'P-less', 'TOP_H_05': 'Top-H 0.5', 'TOP_H_07': 'Top-H 0.7',
    'MBR_16_BERTSCORE': 'MBR-16', 'mixture': 'Mixture',
    'ADAPTIVE_BINOCULARS': 'Adaptive (Bino)', 'ADAPTIVE_FASTDETECT': 'Adaptive (FDGPT)',
}

# Train rows order: base methods, then mixture, then adaptive
TRAIN_ORDER = [
    'ANCESTRAL', 'TEMP_09', 'TEMP_13', 'REP_105', 'REP_130',
    'TOP_K_50', 'TOP_K_1000', 'TOP_P_095', 'TOP_P_03',
    'TYPICAL_095', 'TYPICAL_03', 'ETA_1E4', 'ETA_10',
    'CONTRASTIVE_K8_A06', 'CFG_20', 'P_LESS', 'TOP_H_05', 'TOP_H_07', 'MBR_16_BERTSCORE',
    'mixture', 'ADAPTIVE_BINOCULARS', 'ADAPTIVE_FASTDETECT',
]
# Test cols order: same as train but without mixture (not a test method)
TEST_ORDER = [m for m in TRAIN_ORDER if m != 'mixture']


def build_matrix(rows, detector):
    m = np.full((len(TRAIN_ORDER), len(TEST_ORDER)), np.nan)
    train_idx = {t: i for i, t in enumerate(TRAIN_ORDER)}
    test_idx = {t: i for i, t in enumerate(TEST_ORDER)}
    for r in rows:
        if r['detector_name'] != detector:
            continue
        if r['train_method'] not in train_idx or r['test_method'] not in test_idx:
            continue
        m[train_idx[r['train_method']], test_idx[r['test_method']]] = r['auroc']
    return m


def plot_heatmap(mat, title, outpath):
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(mat, cmap='viridis', vmin=0.0, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(TEST_ORDER)))
    ax.set_xticklabels([NICE[t] for t in TEST_ORDER], rotation=55, ha='right')
    ax.set_yticks(range(len(TRAIN_ORDER)))
    ax.set_yticklabels([NICE[t] for t in TRAIN_ORDER])
    ax.set_xlabel('Test method')
    ax.set_ylabel('Train method')
    ax.set_title(title)
    # Visual separators: between base methods / mixture / adaptive rows
    for y in [18.5, 19.5]:
        ax.axhline(y, color='white', linewidth=1.5)
    # And between base and adaptive test cols
    ax.axvline(18.5, color='white', linewidth=1.5)
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('AUROC')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"  wrote {outpath}")


for model, path in [('Qwen', QWEN), ('Llama', LLAMA)]:
    rows = json.load(open(path))
    for arch in ['roberta-base', 'deberta-v3-base', 'mdeberta-v3-base']:
        mat = build_matrix(rows, arch)
        title = f"{arch}: cross-method AUROC ({model} generator)"
        plot_heatmap(mat, title, OUT / f"heatmap_{model.lower()}_{arch}.png")

print(f"\nAll heatmaps saved to {OUT}/")
