"""
loss.py — Loss composta MSE + SSIM per regressione immagine->immagine.

Motivazione:
  MSE sola tende a produrre output sfocati (minimizza la media delle differenze).
  SSIM misura la similarita' strutturale in patch locali (luminanza, contrasto,
  struttura) e penalizza la perdita di bordi e dettagli fini della molecola.

  L_total = (1 - lambda_ssim) * L_mse + lambda_ssim * (1 - SSIM)

  Con lambda_ssim=0.3: 70% MSE (stabilita') + 30% SSIM (qualita' strutturale).
"""
import torch
import torch.nn as nn


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Kernel gaussiano 1D normalizzato."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,   # range tipico dopo z-score in [-1,1] circa
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    SSIM implementato natively in PyTorch (no dipendenze esterne).
    Ritorna il valore medio di (1 - SSIM) sul batch: da minimizzare.

    Nota: data_range=2.0 perche' i valori dopo z-score stanno circa in [-1, 1].
    """
    device = pred.device
    B, C, H, W = pred.shape

    # Kernel gaussiano 2D separabile
    k1d = _gaussian_kernel(window_size, sigma, device)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)                # (W, W)
    kernel = k2d.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)  # (C, 1, W, W)

    pad = window_size // 2

    def conv(x):
        return torch.nn.functional.conv2d(x, kernel, padding=pad, groups=C)

    mu_p  = conv(pred)
    mu_t  = conv(target)
    mu_pp = conv(pred * pred)
    mu_tt = conv(target * target)
    mu_pt = conv(pred * target)

    sigma_p  = mu_pp - mu_p * mu_p
    sigma_t  = mu_tt - mu_t * mu_t
    sigma_pt = mu_pt - mu_p * mu_t

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator   = (2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p ** 2 + mu_t ** 2 + c1) * (sigma_p + sigma_t + c2)

    ssim_map = numerator / (denominator + eps)
    return 1.0 - ssim_map.mean()


def _topology_loss(emb: torch.Tensor, labels: torch.Tensor, n_confs: int = 100) -> torch.Tensor:
    """
    Calcola la loss topologica forzando gli embedding a organizzarsi ad anello.
    """
    B = emb.size(0)
    if B < 2:
        return torch.tensor(0.0, device=emb.device)
        
    # Distanza logica ad anello
    labels_i = labels.unsqueeze(1)
    labels_j = labels.unsqueeze(0)
    diff = torch.abs(labels_i - labels_j)
    dist_label = torch.min(diff, n_confs - diff)
    
    # Normalizza in [0, 1]
    norm_dist_label = dist_label.float() / (n_confs / 2.0)
    
    # Target distance in [0, 2] (compatibile con distanza coseno)
    target_d = norm_dist_label * 2.0
    
    # Distanza coseno degli embedding
    emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
    cos_sim = torch.mm(emb_norm, emb_norm.t())
    d_emb = 1.0 - cos_sim
    
    return torch.nn.functional.mse_loss(d_emb, target_d)


class CompositeLoss(nn.Module):
    """
    L_total = (1 - lambda_ssim) * MSE + lambda_ssim * (1 - SSIM) + lambda_topo * TopoLoss

    Attributi pubblici per il logging:
      .last_mse   : valore MSE dell'ultimo forward
      .last_ssim  : valore SSIM dell'ultimo forward (0-1, piu' alto e' meglio)
      .last_topo  : valore TopoLoss dell'ultimo forward
    """

    def __init__(self, lambda_ssim: float = 0.3, lambda_topo: float = 0.0, n_confs: int = 100):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_topo = lambda_topo
        self.n_confs     = n_confs
        self.mse         = nn.MSELoss()
        self.last_mse    = 0.0
        self.last_ssim   = 0.0
        self.last_topo   = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor, emb: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        l_mse  = self.mse(pred, target)
        l_ssim = _ssim_loss(pred, target)

        # Salva per il logging
        self.last_mse  = l_mse.item()
        self.last_ssim = 1.0 - l_ssim.item()

        l_total = (1.0 - self.lambda_ssim) * l_mse + self.lambda_ssim * l_ssim
        
        if self.lambda_topo > 0 and emb is not None and labels is not None:
            l_topo = _topology_loss(emb, labels, self.n_confs)
            self.last_topo = l_topo.item()
            l_total = l_total + self.lambda_topo * l_topo
        else:
            self.last_topo = 0.0

        return l_total


# ── Verifica ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from config import CFG

    device = torch.device(CFG["device"])
    crit   = CompositeLoss(lambda_ssim=CFG["lambda_ssim"]).to(device)

    pred   = torch.randn(8, 1, 128, 128, device=device)
    target = torch.randn(8, 1, 128, 128, device=device)
    loss   = crit(pred, target)

    print(f"Loss totale : {loss.item():.6f}")
    print(f"MSE         : {crit.last_mse:.6f}")
    print(f"SSIM        : {crit.last_ssim:.6f}  (0=pessimo, 1=perfetto)")

    # Con pred == target la loss deve essere ~0 e SSIM ~1
    perfect = crit(target, target)
    print(f"\nLoss (pred==target) : {perfect.item():.6f}  (atteso ~0)")
    print(f"SSIM (pred==target) : {crit.last_ssim:.6f}  (atteso ~1)")
    print("Loss OK")
