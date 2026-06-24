"""
test.py — Valutazione finale sul test set (conformazioni 90-99).

Produce:
  - Metriche aggregate: Loss, PSNR, SSIM stampate a console
  - results/test_examples.png  : griglia N x 4 (Noisy | Denoised | GT | Residuo)
  - results/test_metrics.json  : metriche numeriche salvate
"""
import sys
import os
import math
import json
import numpy as np
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG
from dataset import CryoEMDataset
from model import UNet
from loss import CompositeLoss
from torch.utils.data import DataLoader


# ── Metrica PSNR ─────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(data_range ** 2 / mse)


# ── Valutazione ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(n_examples: int = 8):
    device = torch.device(CFG["device"])

    # Carica modello
    if not os.path.exists(CFG["ckpt_best"]):
        raise FileNotFoundError(
            f"Checkpoint non trovato: {CFG['ckpt_best']}\nEsegui prima train.py"
        )
    model = UNet(
        base_channels=CFG["base_channels"],
        depth=CFG["depth"],
        dropout_p=0.0,
    ).to(device)
    model.load_state_dict(torch.load(CFG["ckpt_best"], map_location=device))
    model.eval()
    print(f"Modello caricato: {CFG['ckpt_best']}")

    # Test dataset
    test_ds = CryoEMDataset(
        confs=CFG["test_confs"],
        noisy_dir=CFG["noisy_dir"],
        clean_dir=CFG["clean_dir"],
        snr_tag=CFG["snr_tag"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
    )
    print(f"Test set: {len(test_ds)} immagini  ({len(CFG['test_confs'])} conformazioni)")

    criterion = CompositeLoss(CFG["lambda_ssim"])

    # Accumula metriche e salva qualche batch per la visualizzazione
    total_loss = total_mse = total_ssim = total_psnr = 0.0
    n_batches  = len(test_loader)

    saved_noisy  = []   # tensori CPU per la griglia visiva
    saved_pred   = []
    saved_clean  = []
    collected    = 0

    for batch_idx, (noisy, clean, conf_label) in enumerate(test_loader):
        noisy_dev = noisy.to(device)
        clean_dev = clean.to(device)

        pred, _ = model(noisy_dev)
        loss     = criterion(pred, clean_dev)

        total_loss += loss.item()
        total_mse  += criterion.last_mse
        total_ssim += criterion.last_ssim
        total_psnr += psnr(pred, clean_dev)

        # Salva i primi n_examples campioni per la griglia
        if collected < n_examples:
            need = n_examples - collected
            saved_noisy.append(noisy[:need].cpu())
            saved_pred.append(pred[:need].cpu())
            saved_clean.append(clean[:need].cpu())
            collected += min(need, len(noisy))

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            print(f"  {(batch_idx+1)/n_batches:5.1%}  "
                  f"loss={loss.item():.4f}  ssim={criterion.last_ssim:.4f}", end="\r")

    print()

    # Metriche finali
    metrics = {
        "loss":  round(total_loss / n_batches, 6),
        "mse":   round(total_mse  / n_batches, 6),
        "ssim":  round(total_ssim / n_batches, 6),
        "psnr":  round(total_psnr / n_batches, 4),
        "n_images": len(test_ds),
        "conformations": f"{CFG['test_confs'][0]}-{CFG['test_confs'][-1]}",
    }

    print("\n" + "="*50)
    print("  RISULTATI TEST SET")
    print("="*50)
    print(f"  Loss (MSE+SSIM) : {metrics['loss']:.6f}")
    print(f"  MSE             : {metrics['mse']:.6f}")
    print(f"  SSIM            : {metrics['ssim']:.6f}  (0=pessimo, 1=perfetto)")
    print(f"  PSNR            : {metrics['psnr']:.2f} dB")
    print("="*50)

    # Salva metriche JSON
    os.makedirs(CFG["results_dir"], exist_ok=True)
    metrics_path = os.path.join(CFG["results_dir"], "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetriche salvate: {metrics_path}")

    # Costruisce griglia visiva
    noisy_all = torch.cat(saved_noisy, dim=0)[:n_examples]  # (N,1,H,W)
    pred_all  = torch.cat(saved_pred,  dim=0)[:n_examples]
    clean_all = torch.cat(saved_clean, dim=0)[:n_examples]

    save_example_grid(noisy_all, pred_all, clean_all, n_examples)


# ── Griglia visiva ────────────────────────────────────────────────────────────

def save_example_grid(noisy: torch.Tensor,
                      pred:  torch.Tensor,
                      clean: torch.Tensor,
                      n:     int):
    """
    Salva una griglia PNG: ogni riga = 1 campione, 4 colonne:
      Noisy | Denoised (U-Net) | Ground Truth | Residuo assoluto
    """
    noisy = noisy.numpy()
    pred  = pred.numpy()
    clean = clean.numpy()

    col_titles = ["Input Rumoroso", "U-Net Denoised", "Ground Truth", "Residuo |GT−Pred|"]
    cmaps      = ["gray", "gray", "gray", "gray"]

    fig = plt.figure(figsize=(16, 3.5 * n), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(n, 4, hspace=0.06, wspace=0.04)

    for row in range(n):
        residual = np.abs(clean[row, 0] - pred[row, 0])

        imgs = [noisy[row, 0], pred[row, 0], clean[row, 0], residual]

        for col, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = fig.add_subplot(gs[row, col])

            vmin, vmax = img.min(), img.max()
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

            # Titolo colonna solo alla prima riga
            if row == 0:
                ax.set_title(col_titles[col], color="white",
                             fontsize=11, fontweight="bold", pad=8)

            # Etichetta riga: numero conformazione
            if col == 0:
                ax.set_ylabel(f"Sample {row}", color="#aaa",
                              fontsize=8, rotation=90, labelpad=4)

            ax.axis("off")

    fig.suptitle(
        f"Test Set Denoising — conformazioni {CFG['test_confs'][0]}–{CFG['test_confs'][-1]}  "
        f"(snr={CFG['snr_tag']})",
        color="white", fontsize=13, fontweight="bold", y=1.005,
    )

    out_path = os.path.join(CFG["results_dir"], "test_examples.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Griglia esempi salvata: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Numero di esempi da visualizzare nella griglia (modificabile)
    N_EXAMPLES = 8
    evaluate_test(n_examples=N_EXAMPLES)
