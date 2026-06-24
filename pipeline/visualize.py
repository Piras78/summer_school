"""
visualize.py — Tutte le visualizzazioni del progetto.

Plot 1 : Confronto denoising (griglia noisy | pred | clean | residual)
Plot 2 : Curve di training (loss, PSNR, SSIM)
Plot 3 : Ring plot 2D - UMAP colorato per conformazione (Plotly interattivo)
Plot 4 : Ring plot 3D - UMAP 3D interattivo (Plotly)
Plot 5 : Clustering K-Means sul ring 2D
Plot 6 : Distribuzione metriche per split

Tutti i plot vengono salvati come PNG (Matplotlib) e HTML (Plotly).
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless; sostituisci con "TkAgg" o rimuovi su Colab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG

plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="deep")
SAVE_DIR = CFG["results_dir"]


# ── Utility ───────────────────────────────────────────────────────────────────

def _savefig(fig, name: str, dpi: int = 150):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Salvato: {path}")
    plt.close(fig)


def _cyclic_cmap():
    """Colormap ciclica per la conformazione 0-99 (ring continuo)."""
    return plt.get_cmap("hsv")


# ── Plot 1: Confronto Denoising ───────────────────────────────────────────────

def plot_denoising(noisy_batch: np.ndarray,
                   pred_batch:  np.ndarray,
                   clean_batch: np.ndarray,
                   n_samples:   int = 6):
    """
    Griglia N x 4: [Noisy | Pred | Clean | Residual]
    noisy_batch, pred_batch, clean_batch: (N, 1, H, W) numpy
    """
    n = min(n_samples, len(noisy_batch))
    fig = plt.figure(figsize=(16, 3.2 * n), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(n, 4, hspace=0.08, wspace=0.04)

    col_titles = ["Input Rumoroso", "U-Net Output", "Ground Truth", "Residual |GT-Pred|"]
    cmaps      = ["gray", "gray", "gray", "hot"]

    for row in range(n):
        imgs = [
            noisy_batch[row, 0],
            pred_batch[row, 0],
            clean_batch[row, 0],
            np.abs(clean_batch[row, 0] - pred_batch[row, 0]),
        ]
        for col, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cmap, aspect="equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], color="white", fontsize=11,
                             fontweight="bold", pad=8)

    fig.suptitle("Confronto Denoising — U-Net Cryo-EM", color="white",
                 fontsize=14, y=1.01)
    _savefig(fig, "01_denoising_comparison.png")


# ── Plot 2: Curve di Training ─────────────────────────────────────────────────

def plot_training_curves(history_path: str = None):
    """Carica history.json e plotta loss, PSNR, SSIM per train/val."""
    if history_path is None:
        history_path = os.path.join(CFG["results_dir"], "history.json")
    with open(history_path) as f:
        history = json.load(f)

    epochs = [m["epoch"] + 1 for m in history["train"]]
    metrics_to_plot = [
        ("loss", "Loss (MSE+SSIM)", True),   # True = lower is better
        ("psnr", "PSNR (dB)",        False),
        ("ssim", "SSIM",             False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    for ax, (key, ylabel, lower_better) in zip(axes, metrics_to_plot):
        tr = [m[key] for m in history["train"]]
        vl = [m[key] for m in history["val"]]

        ax.plot(epochs, tr, label="Train", color="#4fc3f7", linewidth=2)
        ax.plot(epochs, vl, label="Val",   color="#ff8a65", linewidth=2)

        # Evidenzia best val
        best_idx = int(np.argmin(vl) if lower_better else np.argmax(vl))
        ax.axvline(epochs[best_idx], color="#a5d6a7", linestyle="--",
                   alpha=0.7, label=f"Best val (ep.{epochs[best_idx]})")

        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(ylabel, color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Training Curves — U-Net Cryo-EM", color="white",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "02_training_curves.png")


# ── Plot 3: Ring 2D (Plotly) ──────────────────────────────────────────────────

def plot_ring_2d(emb_2d: np.ndarray, true_labels: np.ndarray,
                 split: str = "test"):
    """UMAP 2D colorato per conformazione. Salva HTML interattivo + PNG statico."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly non installato: pip install plotly")

    df_dict = {
        "UMAP-1":       emb_2d[:, 0],
        "UMAP-2":       emb_2d[:, 1],
        "Conformazione": true_labels.astype(int),
    }

    fig = px.scatter(
        df_dict,
        x="UMAP-1", y="UMAP-2",
        color="Conformazione",
        color_continuous_scale="HSV",
        range_color=[0, 99],
        opacity=0.5,
        title=f"UMAP 2D — Spazio Latente Bottleneck ({split})",
        labels={"Conformazione": "Conf. (0-99)"},
        width=900, height=700,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        plot_bgcolor="#0d0d0d",
        paper_bgcolor="#0d0d0d",
        font=dict(color="white"),
        coloraxis_colorbar=dict(title="Conf.", tickfont=dict(color="white")),
    )

    html_path = os.path.join(SAVE_DIR, f"03_ring_2d_{split}.html")
    fig.write_html(html_path)
    print(f"  Salvato: {html_path}  (apri nel browser)")

    # PNG statico con Matplotlib
    fig2, ax = plt.subplots(figsize=(9, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                    c=true_labels, cmap="hsv", s=2, alpha=0.5,
                    vmin=0, vmax=99)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Conformazione (0-99)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xlabel("UMAP-1", color="white")
    ax.set_ylabel("UMAP-2", color="white")
    ax.set_title(f"Ring Plot 2D — {split}", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    plt.tight_layout()
    _savefig(fig2, f"03_ring_2d_{split}.png")


# ── Plot 4: Ring 3D (Plotly) ──────────────────────────────────────────────────

def plot_ring_3d(emb_3d: np.ndarray, true_labels: np.ndarray,
                 split: str = "test"):
    """UMAP 3D interattivo con Plotly. Solo HTML."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly non installato: pip install plotly")

    fig = go.Figure(data=[go.Scatter3d(
        x=emb_3d[:, 0],
        y=emb_3d[:, 1],
        z=emb_3d[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=true_labels,
            colorscale="HSV",
            cmin=0, cmax=99,
            opacity=0.6,
            colorbar=dict(title="Conf.", x=1.0),
        ),
        hovertemplate="Conf: %{marker.color:.0f}<extra></extra>",
    )])
    fig.update_layout(
        title=f"Ring 3D — UMAP Bottleneck ({split})",
        scene=dict(
            bgcolor="#0d0d0d",
            xaxis=dict(color="white", gridcolor="#333"),
            yaxis=dict(color="white", gridcolor="#333"),
            zaxis=dict(color="white", gridcolor="#333"),
        ),
        paper_bgcolor="#0d0d0d",
        font=dict(color="white"),
        width=900, height=750,
    )

    html_path = os.path.join(SAVE_DIR, f"04_ring_3d_{split}.html")
    fig.write_html(html_path)
    print(f"  Salvato: {html_path}  (apri nel browser)")


# ── Plot 5: Clustering K-Means ────────────────────────────────────────────────

def plot_clustering(emb_2d: np.ndarray, true_labels: np.ndarray,
                    cluster_labels: np.ndarray, metrics: dict,
                    split: str = "test"):
    """Confronto side-by-side: conformazioni vere vs cluster K-Means."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), facecolor="#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    titles = [
        f"Ground Truth Conformazioni ({split})",
        f"K-Means (k={CFG['n_clusters']})   ARI={metrics['ARI']:.3f}",
    ]
    label_sets  = [true_labels, cluster_labels]
    cmap_names  = ["hsv", "tab20"]
    vmaxes      = [99, CFG["n_clusters"] - 1]

    for ax, title, labs, cmap_name, vmax in zip(
            axes, titles, label_sets, cmap_names, vmaxes):
        ax.set_facecolor("#0d0d0d")
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                        c=labs, cmap=cmap_name, s=2, alpha=0.5,
                        vmin=0, vmax=vmax)
        plt.colorbar(sc, ax=ax).ax.yaxis.set_tick_params(color="white")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_xlabel("UMAP-1", color="white")
        ax.set_ylabel("UMAP-2", color="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    metric_str = (f"ARI={metrics['ARI']:.4f}   "
                  f"NMI={metrics['NMI']:.4f}   "
                  f"Silhouette={metrics['Silhouette']:.4f}")
    fig.suptitle(f"Clustering — {metric_str}", color="#a5d6a7",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, f"05_clustering_{split}.png")


# ── Orchestratore ─────────────────────────────────────────────────────────────

def plot_all(split: str = "test",
             noisy_batch=None, pred_batch=None, clean_batch=None,
             history_path=None):
    """
    Genera tutti i plot.
    Parametri opzionali: noisy/pred/clean servono per Plot 1 (denoising).
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  VISUALIZZAZIONI  (split={split})")
    print(f"{'='*55}")

    # Plot 1: denoising (solo se i batch sono forniti)
    if noisy_batch is not None:
        print("\n[1/5] Denoising comparison...")
        plot_denoising(noisy_batch, pred_batch, clean_batch)

    # Plot 2: training curves
    hp = history_path or os.path.join(CFG["results_dir"], "history.json")
    if os.path.exists(hp):
        print("\n[2/5] Training curves...")
        plot_training_curves(hp)
    else:
        print("\n[2/5] history.json non trovato, skip.")

    # Carica risultati analisi
    try:
        from analysis import load_analysis_results
        res = load_analysis_results(split)
    except FileNotFoundError:
        print(f"\n  Risultati analisi non trovati per split '{split}'. "
              f"Esegui analysis.py prima.")
        return

    # Plot 3: Ring 2D
    print("\n[3/5] Ring 2D...")
    plot_ring_2d(res["emb_2d"], res["true_labels"], split)

    # Plot 4: Ring 3D
    print("\n[4/5] Ring 3D...")
    plot_ring_3d(res["emb_3d"], res["true_labels"], split)

    # Plot 5: Clustering
    print("\n[5/5] Clustering...")
    plot_clustering(res["emb_2d"], res["true_labels"],
                    res["cluster_labels"], res["metrics"], split)

    print(f"\nTutti i plot salvati in: {SAVE_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_all(split="test")
