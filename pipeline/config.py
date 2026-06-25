"""
config.py — Configurazione centralizzata del progetto.
Modifica qui tutti gli iperparametri, poi i moduli li importano da qui.
"""
from pathlib import Path
import torch

# ── Radice del progetto (cartella che contiene pipeline/) ─────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CFG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "project_root":    str(_PROJECT_ROOT),
    "dataset_root":    str(_PROJECT_ROOT / "IgG-1D" / "IgG-1D"),
    "noisy_dir":       str(_PROJECT_ROOT / "IgG-1D" / "IgG-1D" / "images" / "snr0.005"),
    "clean_dir":       str(_PROJECT_ROOT / "IgG-1D" / "IgG-1D" / "clean_projections"),
    "checkpoint_dir":  str(_PROJECT_ROOT / "pipeline" / "checkpoints"),
    "results_dir":     str(_PROJECT_ROOT / "pipeline" / "results"),

    # ── Dataset ───────────────────────────────────────────────────────────────
    "n_conformations": 100,
    "n_per_conf":      1000,
    "img_size":        128,
    "snr_tag":         "snr0.005",

    # Split per conformazioni (non per singole immagini — evita data leakage)
    "train_confs": list(range(0, 80)),   # 80 000 immagini
    "val_confs":   list(range(80, 90)),  # 10 000 immagini
    "test_confs":  list(range(90, 100)), # 10 000 immagini

    # ── DataLoader ────────────────────────────────────────────────────────────
    "batch_size":  64,
    "num_workers": 2,      # 0 su Windows se si hanno problemi con multiprocessing
    "pin_memory":  True,

    # ── U-Net ─────────────────────────────────────────────────────────────────
    "base_channels":       32,   # canali al primo livello encoder
    "depth":                4,   # numero di livelli encoder/decoder
    "dropout_bottleneck": 0.1,   # dropout applicato solo al bottleneck

    # ── Training ──────────────────────────────────────────────────────────────
    "epochs":       30,
    "lr":         3e-4,
    "weight_decay": 1e-5,
    "grad_clip":    1.0,
    "seed":        42,

    # ── Loss ──────────────────────────────────────────────────────────────────
    "lambda_ssim": 0.3,   # peso SSIM; (1 - lambda_ssim) = peso MSE
    "lambda_topo": 0.1,   # peso topology loss sugli embeddings

    # ── Scheduler ─────────────────────────────────────────────────────────────
    "scheduler": "onecycle",   # "onecycle" | "cosine"
    "pct_start":  0.3,         # OneCycleLR: frazione epoch di warm-up

    # ── Mixed Precision (AMP) ─────────────────────────────────────────────────
    "mixed_precision": True,

    # ── Embedding & Analisi ───────────────────────────────────────────────────
    "embedding_batch_size": 256,   # batch più grande in inference (no grad)
    "pca_components":        50,
    "n_clusters":            10,
    "umap_neighbors":        30,
    "umap_min_dist":        0.1,
    "umap_metric":       "cosine",

    # ── Device ────────────────────────────────────────────────────────────────
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ── Checkpoint paths (derivati) ───────────────────────────────────────────────
CFG["ckpt_best"]   = str(Path(CFG["checkpoint_dir"]) / "best.pth")
CFG["ckpt_last"]   = str(Path(CFG["checkpoint_dir"]) / "last.pth")
CFG["ckpt_resume"] = str(Path(CFG["checkpoint_dir"]) / "training_state.pth")

# ── Crea directory necessarie ─────────────────────────────────────────────────
for _d in [CFG["checkpoint_dir"], CFG["results_dir"]]:
    Path(_d).mkdir(parents=True, exist_ok=True)


def print_cfg():
    print("\n" + "=" * 60)
    print("  CONFIGURAZIONE PROGETTO")
    print("=" * 60)
    for k, v in CFG.items():
        if not isinstance(v, list):
            print(f"  {k:<25} {v}")
        else:
            print(f"  {k:<25} {v[0]}..{v[-1]} ({len(v)} elem.)")
    print("=" * 60 + "\n")
