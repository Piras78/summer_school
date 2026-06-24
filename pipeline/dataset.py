"""
dataset.py — Caricamento e preprocessing delle immagini Cryo-EM.

Split per conformazioni (non per immagini singole) per evitare data leakage:
  Train : conformazioni 0-79   -> 80 000 immagini
  Val   : conformazioni 80-89  -> 10 000 immagini
  Test  : conformazioni 90-99  -> 10 000 immagini
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import mrcfile


# ── Preprocessing ─────────────────────────────────────────────────────────────

def percentile_normalize(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile clip + z-score. Gestisce i valori estremi tipici dei file .mrc."""
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    arr = np.clip(arr, p_lo, p_hi)
    mean, std = arr.mean(), arr.std()
    return (arr - mean) / (std + 1e-8)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CryoEMDataset(Dataset):
    """
    Ogni campione: (noisy_tensor, clean_tensor, conf_label)
      - noisy_tensor : float32 (1, H, W) in [-1, 1] circa
      - clean_tensor : float32 (1, H, W) in [-1, 1] circa
      - conf_label   : int, indice conformazione 0-99 (usato per ARI finale)

    I file .mrcs vengono aperti in modalita' memory-mapped per non saturare la RAM.
    I file .npy vengono caricati interamente in RAM (circa 64 MB per conformazione).
    """

    def __init__(self, confs: list, noisy_dir: str, clean_dir: str, snr_tag: str):
        self.confs     = confs
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.snr_tag   = snr_tag

        # Verifica esistenza file e costruisce indice lineare
        self._index = []   # lista di (conf_idx, particle_idx)
        for c in confs:
            npath = self._noisy_path(c)
            cpath = self._clean_path(c)
            if not os.path.exists(npath):
                raise FileNotFoundError(f"Noisy non trovato: {npath}")
            if not os.path.exists(cpath):
                raise FileNotFoundError(f"Clean non trovato: {cpath}")
            with mrcfile.mmap(npath, mode='r', permissive=True) as mrc:
                n_particles = mrc.data.shape[0]
            for p in range(n_particles):
                self._index.append((c, p))

        # Cache .npy in RAM (clean targets, ~640 MB totali per 100 conf)
        self._clean_cache: dict = {}

    def _noisy_path(self, conf: int) -> str:
        return os.path.join(self.noisy_dir, f"{conf:03d}_particles_128.mrcs")

    def _clean_path(self, conf: int) -> str:
        return os.path.join(self.clean_dir, f"clean_2d_targets_{conf:03d}.npy")

    def _get_clean(self, conf: int) -> np.ndarray:
        if conf not in self._clean_cache:
            self._clean_cache[conf] = np.load(self._clean_path(conf))  # (1000,1,128,128)
        return self._clean_cache[conf]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        conf, particle = self._index[idx]

        # Noisy: memory-mapped read
        npath = self._noisy_path(conf)
        with mrcfile.mmap(npath, mode='r', permissive=True) as mrc:
            noisy = mrc.data[particle].copy().astype(np.float32)   # (H, W)

        # Clean: da cache
        clean = self._get_clean(conf)[particle].copy().astype(np.float32)  # (1, H, W)

        # Preprocessing noisy — cast esplicito a float32 dopo normalize
        # (np.percentile e divisione restituiscono float64 di default)
        noisy = percentile_normalize(noisy).astype(np.float32)
        noisy = torch.from_numpy(noisy).unsqueeze(0)   # (1, H, W) float32

        # Preprocessing clean
        clean = percentile_normalize(clean.squeeze()).astype(np.float32)
        clean = torch.from_numpy(clean).unsqueeze(0)   # (1, H, W) float32

        return noisy, clean, conf


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loaders(cfg: dict):
    """
    Ritorna (train_loader, val_loader, test_loader).
    Usa i range di conformazioni definiti in CFG.
    """
    common = dict(
        noisy_dir=cfg["noisy_dir"],
        clean_dir=cfg["clean_dir"],
        snr_tag=cfg["snr_tag"],
    )

    train_ds = CryoEMDataset(cfg["train_confs"], **common)
    val_ds   = CryoEMDataset(cfg["val_confs"],   **common)
    test_ds  = CryoEMDataset(cfg["test_confs"],  **common)

    loader_kw = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    print(f"Dataset  train={len(train_ds):>6}  val={len(val_ds):>6}  test={len(test_ds):>6} immagini")
    return train_loader, val_loader, test_loader


# ── Verifica rapida ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from config import CFG

    print("Verifica dataset...")
    train_loader, val_loader, test_loader = make_loaders(CFG)

    noisy, clean, label = next(iter(train_loader))
    print(f"  Batch noisy : {noisy.shape}  dtype={noisy.dtype}  "
          f"min={noisy.min():.2f}  max={noisy.max():.2f}")
    print(f"  Batch clean : {clean.shape}  dtype={clean.dtype}  "
          f"min={clean.min():.2f}  max={clean.max():.2f}")
    print(f"  Labels      : {label[:8].tolist()}")
    print("Dataset OK")
