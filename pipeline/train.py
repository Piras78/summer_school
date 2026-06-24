"""
train.py — Loop di addestramento con:
  - AMP (Automatic Mixed Precision) per GPU T4 / RTX 4060
  - Salvataggio checkpoint migliore (val loss) e ultimo
  - Ripresa automatica dal crash (resume da training_state.pth)
  - Scheduler OneCycleLR o CosineAnnealingLR
  - Gradient clipping
  - Logging completo: loss, PSNR, SSIM per ogni epoch

Checkpoint salvati:
  checkpoints/best.pth          -> miglior val loss (usato per inference)
  checkpoints/last.pth          -> fine ultima epoch (backup)
  checkpoints/training_state.pth -> stato completo per resume
"""
import os
import sys
import math
import time
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, print_cfg
from dataset import make_loaders
from model import UNet, count_parameters
from loss import CompositeLoss


# ── Metriche ──────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(data_range ** 2 / mse)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, scaler, scheduler,
                    epoch: int, best_val_loss: float, history: dict):
    torch.save({
        "epoch":         epoch,
        "model_state":   model.state_dict(),
        "optim_state":   optimizer.state_dict(),
        "scaler_state":  scaler.state_dict(),
        "sched_state":   scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "history":       history,
    }, path)


def load_checkpoint(path: str, model, optimizer, scaler, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler and ckpt["sched_state"] is not None:
        scheduler.load_state_dict(ckpt["sched_state"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["history"]


# ── Epoch singola ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, scheduler,
              device, is_train: bool):
    model.train() if is_train else model.eval()

    total_loss = total_mse = total_ssim = total_psnr = 0.0
    n_batches  = len(loader)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (noisy, clean, _) in enumerate(loader):
            noisy, clean = noisy.to(device), clean.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=CFG["mixed_precision"]):
                pred, _ = model(noisy)
                loss     = criterion(pred, clean)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and CFG["scheduler"] == "onecycle":
                    scheduler.step()

            total_loss += loss.item()
            total_mse  += criterion.last_mse
            total_ssim += criterion.last_ssim
            total_psnr += psnr(pred.detach(), clean)

            # Progresso ogni 10% degli step
            if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                frac = (batch_idx + 1) / n_batches
                tag  = "TRAIN" if is_train else "  VAL"
                print(f"    [{tag}] {frac:5.1%}  loss={loss.item():.4f}  "
                      f"ssim={criterion.last_ssim:.4f}", end="\r")

    if is_train and scheduler is not None and CFG["scheduler"] == "cosine":
        scheduler.step()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "mse":  total_mse  / n,
        "ssim": total_ssim / n,
        "psnr": total_psnr / n,
    }


# ── Training principale ───────────────────────────────────────────────────────

def train():
    print_cfg()
    torch.manual_seed(CFG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG["seed"])

    device = torch.device(CFG["device"])
    print(f"Device: {device}")

    # DataLoaders
    train_loader, val_loader, _ = make_loaders(CFG)

    # Modello
    model = UNet(
        base_channels=CFG["base_channels"],
        depth=CFG["depth"],
        dropout_p=CFG["dropout_bottleneck"],
    ).to(device)
    print(f"Parametri U-Net : {count_parameters(model):,}")

    # Loss, Optimizer
    criterion = CompositeLoss(lambda_ssim=CFG["lambda_ssim"])
    optimizer = optim.AdamW(model.parameters(),
                            lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])

    # Scaler AMP
    scaler = GradScaler("cuda", enabled=CFG["mixed_precision"])

    # Scheduler
    steps_per_epoch = len(train_loader)
    if CFG["scheduler"] == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CFG["lr"],
            total_steps=CFG["epochs"] * steps_per_epoch,
            pct_start=CFG["pct_start"],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["epochs"]
        )

    # ── Resume da checkpoint se esiste ────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    history       = {"train": [], "val": []}

    resume_path = CFG["ckpt_resume"]
    if os.path.exists(resume_path):
        print(f"\nResume da {resume_path}")
        start_epoch, best_val_loss, history = load_checkpoint(
            resume_path, model, optimizer, scaler, scheduler, device
        )
        start_epoch += 1
        print(f"  Riprendo da epoch {start_epoch}  (best val loss: {best_val_loss:.6f})")
    else:
        print("\nNessun checkpoint trovato — partenza da zero.")

    # ── Loop epoche ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["epochs"]):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}  lr={lr_now:.2e}")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                                  scaler, scheduler, device, is_train=True)
        val_metrics   = run_epoch(model, val_loader,   criterion, None,
                                  None,  None,      device, is_train=False)

        elapsed = time.time() - t0
        print(f"\n  Train  loss={train_metrics['loss']:.4f}  "
              f"mse={train_metrics['mse']:.4f}  "
              f"ssim={train_metrics['ssim']:.4f}  "
              f"psnr={train_metrics['psnr']:.2f}dB")
        print(f"  Val    loss={val_metrics['loss']:.4f}  "
              f"mse={val_metrics['mse']:.4f}  "
              f"ssim={val_metrics['ssim']:.4f}  "
              f"psnr={val_metrics['psnr']:.2f}dB  "
              f"[{elapsed:.0f}s]")

        # Aggiorna history
        history["train"].append({"epoch": epoch, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})

        # ── Checkpoint ultimo (sempre) ────────────────────────────────────────
        save_checkpoint(CFG["ckpt_last"], model, optimizer, scaler,
                        scheduler, epoch, best_val_loss, history)

        # ── Checkpoint resume (uguale all'ultimo) ─────────────────────────────
        save_checkpoint(CFG["ckpt_resume"], model, optimizer, scaler,
                        scheduler, epoch, best_val_loss, history)

        # ── Checkpoint migliore ───────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), CFG["ckpt_best"])
            print(f"  ** Nuovo best val loss: {best_val_loss:.6f} -> salvato best.pth")

    # Salva history come JSON per le visualizzazioni
    hist_path = os.path.join(CFG["results_dir"], "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory salvata in {hist_path}")
    print(f"Training completato. Best val loss: {best_val_loss:.6f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
