# Implementazione Topology Loss

Il modello U-Net originale, addestrato unicamente tramite una combinazione di `MSE` e `SSIM`, risultava capace di applicare il denoising alle immagini ma ignorava la struttura sottostante delle conformazioni. I campioni finivano ammassati casualmente all'interno dello spazio latente.

Con le recenti modifiche ho introdotto una metrica "topologica" nel bottleneck per costringere il modello a posizionare gli embedding secondo la loro vera forma ad anello.

## Modifiche effettuate

### 1. `config.py`
Aggiunto un nuovo iperparametro per regolare l'intensità della loss sul bottleneck:
```python
"lambda_topo": 0.1,   # peso topology loss sugli embeddings
```

### 2. `loss.py`
Ho introdotto la funzione `_topology_loss`:
- **Distanza logica:** Dato l'indice di conformazione (0-99), ho calcolato la distanza ad anello tra le label nel batch: `min(|i - j|, 100 - |i - j|)`.
- **Distanza latente:** Sugli embedding derivati dal bottleneck è stata calcolata la distanza del coseno `(1 - cos_sim(emb_i, emb_j))`.
- **Topologia:** La Topological Loss ora non è altro che un *Mean Squared Error* tra la distanza logica (normalizzata) e la distanza latente, in modo che quest'ultima si adatti proporzionalmente per ricreare la corretta geometria ad anello nel tensore compresso a 512 dimensioni.
Ho quindi aggiornato `CompositeLoss` per includere questa loss e calcolarla se gli array `emb` e `labels` vengono passati dal loop di training.

### 3. `train.py`
Ho aggiornato il loop di estrazione per prelevare il terzo output del Dataset (cioè `conf_labels`):
- Il GAP (Global Average Pooling) viene calcolato e inviato, con le corrispondenti labels, al costruttore `criterion()`.
- Ho aggiunto un counter nei log in tempo reale per misurare come performa `topo_loss` a ogni epoca, così potrai monitorare se il modello sta imparando la topologia oltre che a de-rumorizzare l'immagine.

## Come testare

Puoi avviare il training regolarmente:
```powershell
python pipeline/train.py
```

Vedrai comparire nella barra di stato un nuovo valore `topo` accanto alla `ssim` e alla `loss`. Al termine del training, estrai gli embedding ed esegui la PCA/UMAP. Ti aspetterai che ora `umap_2d` mostri distintamente la struttura ad anello delle conformazioni al centro!
