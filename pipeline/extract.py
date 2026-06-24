"""
extract.py — Estrazione degli embedding dal bottleneck della U-Net.

Logica:
  Il bottleneck (512, 8, 8) rappresenta la struttura molecolare compressa.
  Global Average Pooling (GAP) collassa le dimensioni spaziali -> vettore (512,).
  Si usa il modello con i pesi di best.pth (niente gradient, batch grande).

Output salvati in results/:
  embeddings_train.npy  shape (80000, 512)
  embeddings_val.npy    shape (10000, 512)
  embeddings_test.npy   shape (10000, 512)
  labels_train.npy      shape (80000,)  int, indice conformazione 0-99
  labels_val.npy        shape (10000,)
  labels_test.npy       shape (10000,)
"""
import sys
import os
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG
from dataset import make_loaders
from model import UNet


# ── Estrazione ────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_split(model: torch.nn.Module,
                  loader,
                  device: torch.device) -> tuple:
    """
    Ritorna (embeddings, labels).
      embeddings : np.ndarray  (N, embedding_dim)
      labels     : np.ndarray  (N,)  int
    """
    model.eval()
    embs, labs = [], []

    for batch_idx, (noisy, _, conf_label) in enumerate(loader):
        noisy = noisy.to(device)

        _, bottleneck = model(noisy)
        # bottleneck: (B, 512, H', W') -> GAP -> (B, 512)
        emb = bottleneck.mean(dim=[2, 3])

        embs.append(emb.cpu().numpy())
        labs.append(conf_label.numpy())

        if (batch_idx + 1) % 20 == 0:
            done = (batch_idx + 1) * loader.batch_size
            print(f"    estratti {done:>6} campioni", end="\r")

    print()
    return np.concatenate(embs, axis=0), np.concatenate(labs, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_all():
    device = torch.device(CFG["device"])

    # Carica il modello con i pesi migliori
    if not os.path.exists(CFG["ckpt_best"]):
        raise FileNotFoundError(
            f"Checkpoint non trovato: {CFG['ckpt_best']}\n"
            "Esegui prima train.py"
        )

    model = UNet(
        base_channels=CFG["base_channels"],
        depth=CFG["depth"],
        dropout_p=0.0,   # niente dropout in inference
    ).to(device)
    model.load_state_dict(torch.load(CFG["ckpt_best"], map_location=device))
    model.eval()
    print(f"Modello caricato da {CFG['ckpt_best']}")

    # DataLoaders (batch piu' grande per velocita')
    cfg_inf = {**CFG, "batch_size": CFG["embedding_batch_size"]}
    train_loader, val_loader, test_loader = make_loaders(cfg_inf)

    out = CFG["results_dir"]
    os.makedirs(out, exist_ok=True)

    for split_name, loader in [("train", train_loader),
                                ("val",   val_loader),
                                ("test",  test_loader)]:
        print(f"\nEstraendo split '{split_name}' ({len(loader.dataset)} img)...")
        embs, labs = extract_split(model, loader, device)

        emb_path = os.path.join(out, f"embeddings_{split_name}.npy")
        lab_path = os.path.join(out, f"labels_{split_name}.npy")
        np.save(emb_path, embs)
        np.save(lab_path, labs)
        print(f"  Salvato  {emb_path}  shape={embs.shape}")
        print(f"  Salvato  {lab_path}  shape={labs.shape}")

    print("\nEstrazione completata.")


def load_embeddings(split: str = "train") -> tuple:
    """
    Utility per caricare gli embedding gia' estratti.
    Ritorna (embeddings, labels).
    """
    out = CFG["results_dir"]
    embs = np.load(os.path.join(out, f"embeddings_{split}.npy"))
    labs = np.load(os.path.join(out, f"labels_{split}.npy"))
    return embs, labs


# ── Verifica ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_all()

    # Verifica forme
    for split in ["train", "val", "test"]:
        embs, labs = load_embeddings(split)
        n_conf = len(set(labs.tolist()))
        print(f"  {split:5s}  embs={embs.shape}  labs={labs.shape}  "
              f"conformazioni={n_conf}  label_range=[{labs.min()},{labs.max()}]")
