# BoVW Retrieval App

Two variants of the same app live side-by-side:

- app (v1): metadata-only UI (no images), clean dark theme.
- app2 (v2): shows real images from your local archive in all cards.

Both versions provide:

- Search by existing dataset item (filename/title), ranked by cosine similarity
- Similarity bars, full-dataset cosine histogram with clickable bins
- Browse with filters (author, genres) and quick jump to search
- Explain similarity via top contributing visual words (q[i]*x[i])

## Setup

Activate your venv and install requirements:

```zsh
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

- v1 (metadata-only):

```zsh
uvicorn app.main:app --reload --port 8000
```

Open: <http://127.0.0.1:8000>

- v2 (with images):

```zsh
uvicorn app2.main:app --reload --port 8001
```

Open: <http://127.0.0.1:8001>

Notes for v2:

- Images are served directly from your local archive mounted at `/archive`.
- Archive root path is hard-coded to `/Users/rail/Downloads/archive` and files are resolved as `/archive/{filename}` where `filename` comes from `dataset2.csv` (e.g., `Abstract_Expressionism/aaron-siskind_*.jpg`). The archive is not moved or copied.

## Data paths

- CSV: `data/dataset2.csv`
- NPZ: `data/bovw_vectors_kmeans_1000.npz` (contains `bovw_vectors`, `kmeans_centers`, `filenames`)
- IPCA artifacts for upload search: `data/ipca_mean.npy`, `data/ipca_components.npy`, `data/global_pca_scale.npz`

Ensure these files exist relative to the project root.
