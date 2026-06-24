"""
analysis.py — Riduzione dimensionalita' e clustering sugli embedding.

Pipeline:
  1. Carica embeddings (512-dim) dal disco
  2. PCA(50)   : compressione rapida, mantiene varianza principale
  3. UMAP(2D)  : proiezione non lineare che preserva la topologia (ring detection)
  4. UMAP(3D)  : per la visualizzazione 3D interattiva
  5. K-Means   : clustering nel spazio 2D ridotto
  6. Metriche  : ARI, Silhouette, NMI

Tutti i risultati vengono salvati in results/ come .npy e .json.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG
from extract import load_embeddings


# ── Riduzione dimensionalita' ─────────────────────────────────────────────────

def run_pca(embeddings: np.ndarray, n_components: int) -> tuple:
    """Ritorna (embeddings_ridotti, pca_object)."""
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=n_components, random_state=CFG["seed"])
    emb_pca = pca.fit_transform(emb_scaled)

    var_cum = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA({n_components})  varianza spiegata: {var_cum[-1]:.2%}")
    return emb_pca, pca, scaler


def run_umap(emb_pca: np.ndarray, n_components: int, label: str = "") -> np.ndarray:
    """Proiezione UMAP. Ritorna array ridotto."""
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn non installato: pip install umap-learn")

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=CFG["umap_neighbors"],
        min_dist=CFG["umap_min_dist"],
        metric=CFG["umap_metric"],
        random_state=CFG["seed"],
        verbose=False,
    )
    emb_umap = reducer.fit_transform(emb_pca)
    print(f"  UMAP({n_components}D){label}  shape={emb_umap.shape}")
    return emb_umap


# ── Clustering ────────────────────────────────────────────────────────────────

def run_kmeans(embeddings_2d: np.ndarray, n_clusters: int) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=CFG["seed"])
    labels = km.fit_predict(embeddings_2d)
    return labels


def compute_metrics(true_labels: np.ndarray,
                    pred_labels: np.ndarray,
                    embeddings_2d: np.ndarray) -> dict:
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    sil = silhouette_score(embeddings_2d, pred_labels) if len(set(pred_labels)) > 1 else 0.0
    return {"ARI": round(ari, 4), "NMI": round(nmi, 4), "Silhouette": round(sil, 4)}


# ── Pipeline completa ─────────────────────────────────────────────────────────

def run_analysis(split: str = "test") -> dict:
    """
    Esegue PCA -> UMAP 2D/3D -> K-Means sul split scelto.
    Ritorna un dict con tutti i risultati per le visualizzazioni.
    """
    out = CFG["results_dir"]
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  ANALISI su split: {split.upper()}")
    print(f"{'='*55}")

    # Carica embeddings
    embs, labels = load_embeddings(split)
    print(f"  Embedding caricati: {embs.shape}  label_range=[{labels.min()},{labels.max()}]")

    # PCA
    print("\n[1/4] PCA...")
    emb_pca, pca, scaler = run_pca(embs, CFG["pca_components"])

    # UMAP 2D
    print("\n[2/4] UMAP 2D...")
    emb_2d = run_umap(emb_pca, n_components=2, label=" (per clustering)")

    # UMAP 3D
    print("\n[3/4] UMAP 3D...")
    emb_3d = run_umap(emb_pca, n_components=3, label=" (per visualizzazione)")

    # K-Means sul 2D
    print(f"\n[4/4] K-Means (k={CFG['n_clusters']})...")
    cluster_labels = run_kmeans(emb_2d, CFG["n_clusters"])
    metrics = compute_metrics(labels, cluster_labels, emb_2d)

    print(f"\n  Metriche clustering:")
    print(f"    ARI        = {metrics['ARI']:.4f}  (1=perfetto, 0=casuale)")
    print(f"    NMI        = {metrics['NMI']:.4f}  (mutual info normalizzata)")
    print(f"    Silhouette = {metrics['Silhouette']:.4f}  (1=cluster compatti)")

    # Salva risultati
    results = {
        "split":          split,
        "emb_2d":         emb_2d,
        "emb_3d":         emb_3d,
        "true_labels":    labels,
        "cluster_labels": cluster_labels,
        "metrics":        metrics,
    }

    np.save(os.path.join(out, f"umap_2d_{split}.npy"),         emb_2d)
    np.save(os.path.join(out, f"umap_3d_{split}.npy"),         emb_3d)
    np.save(os.path.join(out, f"cluster_labels_{split}.npy"),  cluster_labels)

    with open(os.path.join(out, f"metrics_{split}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Risultati salvati in {out}/")
    return results


def load_analysis_results(split: str = "test") -> dict:
    """Carica i risultati gia' calcolati dal disco."""
    out = CFG["results_dir"]
    embs, labels = load_embeddings(split)
    return {
        "split":          split,
        "emb_2d":         np.load(os.path.join(out, f"umap_2d_{split}.npy")),
        "emb_3d":         np.load(os.path.join(out, f"umap_3d_{split}.npy")),
        "true_labels":    labels,
        "cluster_labels": np.load(os.path.join(out, f"cluster_labels_{split}.npy")),
        "metrics":        json.load(open(os.path.join(out, f"metrics_{split}.json"))),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Analisi su tutti e tre gli split
    for split in ["train", "val", "test"]:
        try:
            run_analysis(split)
        except FileNotFoundError as e:
            print(f"  Skip {split}: {e}")
