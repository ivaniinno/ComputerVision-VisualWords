from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import cv2
import uuid
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "data").resolve()

CSV_PATH = DATA_DIR / "dataset2.csv"
NPZ_BOW_PATH = DATA_DIR / "bovw_vectors_kmeans_1000.npz"
IPCA_MEAN_PATH = DATA_DIR / 'ipca_mean.npy'
IPCA_COMP_PATH = DATA_DIR / 'ipca_components.npy'
GLOB_SCALE_PATH = DATA_DIR / 'global_pca_scale.npz'

app = FastAPI(title="BoVW Retrieval App")

# Mount static and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory data
DF: Optional[pd.DataFrame] = None
FILENAMES: Optional[np.ndarray] = None
AUTHORS: List[str] = []
GENRE_COLS: List[str] = []
X_BOW: Optional[np.ndarray] = None
X_NORM: Optional[np.ndarray] = None
CENTERS: Optional[np.ndarray] = None
FILENAME_TO_IDX: Dict[str, int] = {}
IPCA_MEAN: Optional[np.ndarray] = None
IPCA_COMP: Optional[np.ndarray] = None
GMIN: Optional[np.ndarray] = None
GMAX: Optional[np.ndarray] = None

# Cache for last uploads
LAST_UPLOADS: Dict[str, Dict[str, Any]] = {}


def df_to_records_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Convert NaN to None
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def load_data() -> None:
    global DF, FILENAMES, AUTHORS, GENRE_COLS, X_BOW, X_NORM, CENTERS, FILENAME_TO_IDX, IPCA_MEAN, IPCA_COMP, GMIN, GMAX
    # Load CSV
    DF = pd.read_csv(CSV_PATH)

    # Normalize filename to str
    DF["filename"] = DF["filename"].astype(str)

    # Authors list
    AUTHORS = sorted([a for a in DF["author"].dropna().unique().tolist() if a != "Unknown" ])

    # Detect genre columns: everything except these base cols
    base_cols = {
        "Unnamed: 0", "filename", "width", "height", "genre_count", "subset",
        "author", "pic_name"
    }

    GENRE_COLS = [c for c in DF.columns if c not in base_cols]

    # Load NPZ with BoVW
    npz = np.load(NPZ_BOW_PATH, allow_pickle=True)
    X_BOW = npz["bovw_vectors"].astype(np.float32)
    CENTERS = npz["kmeans_centers"].astype(np.float32)
    FILENAMES = np.array(npz["filenames"], dtype=object)

    # Build index mapping
    FILENAME_TO_IDX = {str(fn): i for i, fn in enumerate(FILENAMES)}

    # Precompute L2-normalized matrix for cosine
    X_NORM = X_BOW / (np.linalg.norm(X_BOW, axis=1, keepdims=True) + 1e-12)

    # Load IPCA and global scale for query image workflow
    if IPCA_MEAN_PATH.exists() and IPCA_COMP_PATH.exists():
        IPCA_MEAN = np.load(IPCA_MEAN_PATH).astype(np.float32)
        IPCA_COMP = np.load(IPCA_COMP_PATH).astype(np.float32)

    if GLOB_SCALE_PATH.exists():
        _s = np.load(GLOB_SCALE_PATH)
        GMIN = _s['gmin'].astype(np.float32)
        GMAX = _s['gmax'].astype(np.float32)


def _require_sift():
    if not hasattr(cv2, 'SIFT_create'):
        raise RuntimeError("SIFT not available.")
    
    return cv2.SIFT_create()


def compute_sift_descriptors_image_bytes(img_bytes: bytes, max_desc: int = 1800) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Cannot decode image bytes")
    
    sift = _require_sift()
    kps, desc = sift.detectAndCompute(img, None)
    if desc is None or desc.size == 0:
        return np.empty((0, 128), dtype=np.float32)
    desc = desc.astype(np.float32)
    norms = np.linalg.norm(desc, axis=1)
    keep = np.argsort(-norms)[:max_desc]

    return desc[keep]


def ipca_project(desc128: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    if desc128.size == 0:
        return np.empty((0, comps.shape[0]), dtype=np.float32)
    
    Xc = desc128 - mean.reshape(1, -1)

    return (Xc @ comps.T).astype(np.float32)


def clamp_to_global_range(desc_proj: np.ndarray, gmin: Optional[np.ndarray], gmax: Optional[np.ndarray]) -> np.ndarray:
    if desc_proj.size == 0 or gmin is None or gmax is None:
        return desc_proj
    
    return np.clip(desc_proj, gmin.reshape(1, -1), gmax.reshape(1, -1))


def bovw_from_projected(desc_proj: np.ndarray, centers: np.ndarray) -> np.ndarray:
    K = centers.shape[0]
    if desc_proj.size == 0:
        return np.zeros(K, dtype=np.float32)
    
    d2 = ((desc_proj[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    a = np.argmin(d2, axis=1)
    hist = np.bincount(a, minlength=K).astype(np.float32)

    return hist


@app.on_event("startup")
def on_startup():
    load_data()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    total = int(len(FILENAMES)) if FILENAMES is not None else 0

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "total": total,
            "authors": AUTHORS,
            "genres": GENRE_COLS,
        },
    )


@app.get("/api/suggest")
async def suggest(query: str, limit: int = 10):
    q = query.strip().lower()
    if not q:
        return {"items": []}
    
    # simple substring match on filename or pic_name
    rows = DF[(DF["filename"].str.lower().str.contains(q)) | (DF["pic_name"].fillna("").str.lower().str.contains(q))]
    rows = rows.head(limit)
    items = [
        {
            "filename": r["filename"],
            "pic_name": r.get("pic_name", None),
            "author": r.get("author", None),
        }
        for _, r in rows.iterrows()
    ]

    return {"items": items}


def _search_idx_by_filename(filename: str, top_k: int = 10) -> Dict[str, Any]:
    if filename not in FILENAME_TO_IDX:
        return {"error": f"filename not found: {filename}"}
    
    qi = FILENAME_TO_IDX[filename]
    q = X_NORM[qi]
    sims = (X_NORM @ q).astype(np.float32)
    order = np.argsort(-sims)
    idx = order[: int(top_k)]

    # Histogram for plotting on frontend
    hist_counts, hist_edges = np.histogram(sims, bins=40, range=(-0.1, 1.0))

    return {
        "query_index": int(qi),
        "hits": [
            {
                "index": int(j),
                "filename": str(FILENAMES[j]),
                "similarity": float(sims[j]),
            }
            for j in idx
        ],
        "hist": {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        },
    }


@app.post("/api/search/by-filename")
async def search_by_filename(payload: Dict[str, Any]):
    filename = str(payload.get("filename", ""))
    top_k = int(payload.get("top_k", 12))
    res = _search_idx_by_filename(filename, top_k)

    if "hits" in res:
        df_hits = pd.DataFrame(res["hits"])
        meta = DF[["filename", "author", "pic_name"]]
        merged = df_hits.merge(meta, on="filename", how="left")
        res["hits"] = df_to_records_safe(merged)

    return JSONResponse(res)


@app.post("/api/search/by-upload")
async def search_by_upload(file: UploadFile = File(...), top_k: int = 12):
    if IPCA_MEAN is None or IPCA_COMP is None or CENTERS is None:
        return JSONResponse({"error": "IPCA or centers not available"}, status_code=400)
    
    content = await file.read()

    d128 = compute_sift_descriptors_image_bytes(content, max_desc=1800)
    d64 = ipca_project(d128, IPCA_MEAN, IPCA_COMP)
    d64 = clamp_to_global_range(d64, GMIN, GMAX)
    q_hist = bovw_from_projected(d64, CENTERS)
    q = q_hist / (np.linalg.norm(q_hist) + 1e-12)
    sims = (X_NORM @ q).astype(np.float32)
    order = np.argsort(-sims)
    idx = order[: int(top_k)]
    hist_counts, hist_edges = np.histogram(sims, bins=40, range=(-0.1, 1.0))

    # cache
    uid = str(uuid.uuid4())
    LAST_UPLOADS[uid] = {
        "sims": sims,  # numpy array
        "edges": hist_edges,
        "q": q.astype(np.float32),  # store unit query vector for explanations
    }

    # trim cache size
    if len(LAST_UPLOADS) > 5:
        for k in list(LAST_UPLOADS.keys())[:-5]:
            LAST_UPLOADS.pop(k, None)
    hits = [
        {
            "index": int(j),
            "filename": str(FILENAMES[j]),
            "similarity": float(sims[j]),
        }
        for j in idx
    ]

    df_hits = pd.DataFrame(hits)
    meta = DF[["filename", "author", "pic_name"]]
    merged = df_to_records_safe(df_hits.merge(meta, on="filename", how="left"))

    return {
        "upload_id": uid,
        "hits": merged,
        "hist": {"counts": hist_counts.tolist(), "edges": hist_edges.tolist()},
    }


@app.post("/api/search/bin-items")
async def bin_items(payload: Dict[str, Any]):
    bin_index = int(payload.get("bin_index", -1))
    limit = int(payload.get("limit", 60))
    q_filename = payload.get("query_filename")
    upload_id = payload.get("upload_id")

    if bin_index < 0:
        return {"items": []}

    if q_filename:
        if q_filename not in FILENAME_TO_IDX:
            return {"items": [], "error": "filename not found"}
        qi = FILENAME_TO_IDX[q_filename]
        sims = (X_NORM @ X_NORM[qi]).astype(np.float32)
        edges = np.histogram_bin_edges(sims, bins=40, range=(-0.1, 1.0))
    elif upload_id and upload_id in LAST_UPLOADS:
        sims = LAST_UPLOADS[upload_id]["sims"]
        edges = LAST_UPLOADS[upload_id]["edges"]
    else:
        return {"items": [], "error": "no query context"}

    lo = edges[bin_index]
    hi = edges[bin_index + 1]
    mask = (sims >= lo) & (sims < hi)
    idxs = np.where(mask)[0]

    # sort by similarity desc
    idxs = idxs[np.argsort(-sims[idxs])]
    idxs = idxs[:limit]
    items = [
        {
            "index": int(j),
            "filename": str(FILENAMES[j]),
            "similarity": float(sims[j]),
        }
        for j in idxs
    ]
    df_items = pd.DataFrame(items).merge(DF[["filename", "author", "pic_name"]], on="filename", how="left")

    return {"items": df_to_records_safe(df_items), "range": [float(lo), float(hi)]}


def _explain_pair_contributions_from_vec(q_vec: np.ndarray, ji: int, top: int = 15) -> Dict[str, Any]:
    # q_vec and X_NORM[ji] expected to be L2-normalized
    x = X_NORM[ji]
    contrib = q_vec * x
    order = np.argsort(-contrib)
    take = order[:top]

    return {
        "pair": {"ji": int(ji), "filename": str(FILENAMES[ji])},
        "total_similarity": float((q_vec @ x)),
        "top_contributions": [
            {"word": int(i), "q": float(q_vec[i]), "x": float(x[i]), "product": float(contrib[i])}
            for i in take
        ],
    }


@app.get("/api/dataset")
async def dataset_list(author: Optional[str] = None, genres: Optional[str] = None, page: int = 1, size: int = 24, query: Optional[str] = None):
    dd = DF

    if query:
        q = query.strip().lower()
        dd = dd[(dd["filename"].str.lower().str.contains(q)) | (dd["pic_name"].fillna("").str.lower().str.contains(q))]
    if author:
        dd = dd[dd["author"] == author]
    if genres:
        # genres is comma-separated list of genre column names
        selected = [g for g in genres.split(",") if g in GENRE_COLS]
        for g in selected:
            dd = dd[dd[g] == 1]

    total = len(dd)
    page = max(1, page)
    size = max(1, min(100, size))

    start = (page - 1) * size
    end = start + size
    rows = dd.iloc[start:end][["filename", "author", "pic_name"]].copy()
    items = rows.to_dict(orient="records")

    return {"items": items, "page": page, "size": size, "total": int(total)}


@app.get("/api/metadata")
async def metadata(filename: str):
    row = DF[DF["filename"] == filename]

    if row.empty:
        return {"error": f"filename not found: {filename}"}
    
    rr = row.iloc[0].to_dict()

    return rr


def _explain_pair_contributions(qi: int, ji: int, top: int = 15) -> Dict[str, Any]:
    # cosine(q, x) with L2-normalized vectors equals sum_i q[i]*x[i]
    q = X_NORM[qi]
    x = X_NORM[ji]

    contrib = q * x
    order = np.argsort(-contrib)
    take = order[:top]

    return {
        "pair": {"qi": int(qi), "ji": int(ji), "filename": str(FILENAMES[ji])},
        "total_similarity": float((q @ x)),
        "top_contributions": [
            {"word": int(i), "q": float(q[i]), "x": float(x[i]), "product": float(contrib[i])}
            for i in take
        ],
    }


@app.post("/api/explain")
async def explain(payload: Dict[str, Any]):
    j_filename = str(payload.get("candidate_filename", ""))
    upload_id = payload.get("upload_id")
    q_filename = payload.get("query_filename")
    
    if j_filename not in FILENAME_TO_IDX:
        return {"error": "candidate filename not found"}
    ji = FILENAME_TO_IDX[j_filename]

    # If upload_id provided, prefer upload-based explanation
    if upload_id and upload_id in LAST_UPLOADS and LAST_UPLOADS[upload_id].get("q") is not None:
        q_vec = LAST_UPLOADS[upload_id]["q"]
        return _explain_pair_contributions_from_vec(q_vec, ji, top=int(payload.get("top", 15)))

    # Else fallback to filename-based query
    if q_filename and q_filename in FILENAME_TO_IDX:
        qi = FILENAME_TO_IDX[q_filename]
        return _explain_pair_contributions(qi, ji, top=int(payload.get("top", 15)))

    return {"error": "query context not found (upload_id or query_filename)"}
