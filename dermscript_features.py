"""
dermscript_features.py  (v9.0)
==============================
SINGLE SOURCE OF TRUTH for how DermScript turns an image (and optional
metadata text) into a feature vector.  The training notebook AND the
Streamlit app must both import THIS file -- that is what removes the
train/serve skew that made the deployed app disagree with the notebook
(different backbone weights, no color correction, flip-averaging,
different metadata text).

Put one copy next to streamlit_app.py in the GitHub repo and one copy in
Google Drive at  DermScript/code/dermscript_features.py.

Only numpy / PIL are imported at module level, so the pure-numpy helpers
(shades_of_gray, canonical_text, parity_report, conformal helpers) work
without torch.  torch / torchvision / transformers are imported lazily.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# The ONE feature configuration.  If the notebook's parity test (Cell 1.2)
# tells you the cached training embeddings were made a different way, change
# it HERE (once) and both notebook and app follow.
# --------------------------------------------------------------------------
FEATURE_CONFIG = {
    "backbone": "mobilenet_v3_large",
    "weights": "IMAGENET1K_V2",                    # torchvision file mobilenet_v3_large-5c1a4163.pth
    "weights_file": "mobilenet_v3_large-5c1a4163.pth",
    # v9.1: Cell 1.2's parity test (run on the real bundle) found ISIC2020/2017/2016 caches
    # match "V2, no Shades-of-Gray" at 92-100%, NOT "V2 + SoG" as this file assumed before.
    # Cell 4.8's SoG re-extraction only ever touched DDI as a one-off diagnostic; it was
    # never applied to the ISIC caches that make up most of the training pool. Off by default.
    "shades_of_gray": False,
    "sog_power": 6,
    "flip_tta": False,                             # training never flip-averaged
    "image_size": 224,
    "text_model": "emilyalsentzer/Bio_ClinicalBERT",
    "text_max_length": 64,
    "vis_dim": 960,
    "nlp_dim": 768,
    "placeholder_text": "Dermoscopy image.",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
def shades_of_gray(img_pil: Image.Image, p: int = 6) -> Image.Image:
    """Shades-of-Gray color constancy.  Line-for-line the same arithmetic as
    the notebook's `_shades_of_gray` (Cell 2 generic extractor / Cell 4.8)."""
    img_pil = img_pil.convert("RGB")
    arr = np.asarray(img_pil).astype(np.float32)
    illum = np.power(np.mean(np.power(arr, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-6) * np.sqrt(3)
    out = arr / (illum[None, None, :] + 1e-6)
    out = np.clip(out * (255.0 / max(out.max(), 1e-6)) if out.max() > 255 else out, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def make_image_transform(sog: bool | None = None, size: int | None = None):
    """torchvision transform: [Shades-of-Gray] -> Resize -> ToTensor -> Normalize."""
    from torchvision import transforms

    sog = FEATURE_CONFIG["shades_of_gray"] if sog is None else sog
    size = FEATURE_CONFIG["image_size"] if size is None else size
    steps = []
    if sog:
        p = FEATURE_CONFIG["sog_power"]
        steps.append(transforms.Lambda(lambda im: shades_of_gray(im, p)))
    steps += [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------
def load_vision_model(device: str = "cpu", weights: str = "V2", checkpoint_dir=None):
    """MobileNetV3-Large with classifier=Identity -> 960-d embedding.

    weights="V2" (default, what the notebook trained with) or "V1" (only used
    by the parity test to prove/disprove a weights mismatch).  The app must
    NEVER fall back to V1 -- that was the deployed bug (log showed
    mobilenet_v3_large-8738ca79.pth, the V1 file)."""
    import torch
    import torch.nn as nn
    from torchvision.models import (
        MobileNet_V3_Large_Weights,
        mobilenet_v3_large,
    )

    if weights not in ("V1", "V2"):
        raise ValueError("weights must be 'V1' or 'V2'")
    enum = (MobileNet_V3_Large_Weights.IMAGENET1K_V2 if weights == "V2"
            else MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    model = None
    if weights == "V2":
        fname = FEATURE_CONFIG["weights_file"]
        candidates = []
        if checkpoint_dir:
            candidates.append(Path(checkpoint_dir) / fname)
        if os.environ.get("TORCH_HOME"):
            candidates.append(Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / fname)
        for c in candidates:
            if c.exists():
                model = mobilenet_v3_large(weights=None)
                model.load_state_dict(torch.load(c, map_location="cpu", weights_only=True))
                break
    if model is None:
        model = mobilenet_v3_large(weights=enum)   # downloads (or reads TORCH_HOME cache)

    model.classifier = nn.Identity()
    model.eval().to(device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


def weights_fingerprint(model) -> float:
    """Sum of the first conv layer's weights.  Print it in Colab and in the
    app's debug panel -- if the two numbers differ, the weights differ."""
    return float(model.features[0][0].weight.detach().double().sum().cpu())


def embed_pil_images(pils, model, transform, device: str = "cpu",
                     batch_size: int = 32, autocast: bool = False) -> np.ndarray:
    """(N, 960) float32.  autocast=True mimics the Colab extractor's fp16 path
    (only matters for tiny numeric differences, ~1e-3 cosine)."""
    import torch

    out = []
    for i in range(0, len(pils), batch_size):
        t = torch.stack([transform(im.convert("RGB")) for im in pils[i:i + batch_size]]).to(device)
        with torch.no_grad():
            if autocast and str(device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    v = model(t)
            else:
                v = model(t)
        out.append(v.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, FEATURE_CONFIG["vis_dim"]), np.float32)


def load_text_model(device: str = "cpu", local_files_only: bool = False):
    from transformers import AutoModel, AutoTokenizer

    name = FEATURE_CONFIG["text_model"]
    tok = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    bert = AutoModel.from_pretrained(name, local_files_only=local_files_only).eval().to(device)
    for prm in bert.parameters():
        prm.requires_grad_(False)
    return tok, bert


def embed_texts(texts, tok, bert, device: str = "cpu") -> np.ndarray:
    import torch

    with torch.no_grad():
        enc = tok(list(texts), padding=True, truncation=True,
                  max_length=FEATURE_CONFIG["text_max_length"], return_tensors="pt").to(device)
        return bert(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()


# --------------------------------------------------------------------------
# Metadata text (only used by legacy FUSION bundles; VISION-ONLY bundles ignore it)
# --------------------------------------------------------------------------
_SITE_TO_TRAINING_VOCAB = {
    "scalp": "head/neck", "face": "head/neck", "neck": "head/neck",
    "trunk": "torso",
    "upper extremity": "upper extremity",
    "lower extremity": "lower extremity",
    "palms/soles": "palms/soles",
}


def canonical_text(age=None, sex=None, site=None) -> str:
    """Same template as the ISIC2019/2020/BCN extractors in the notebook:
    'Patient age 55.0 years. Sex: male. Lesion location: torso.'
    Unknown fields are omitted; nothing known -> the placeholder string.
    NOTE: free-text clinical notes are deliberately NOT embedded (the model
    never saw free text in training)."""
    parts = []
    if age is not None:
        parts.append(f"Patient age {float(age):.1f} years.")
    if sex is not None and str(sex).strip().lower() in ("male", "female"):
        parts.append(f"Sex: {str(sex).strip().lower()}.")
    if site is not None:
        mapped = _SITE_TO_TRAINING_VOCAB.get(str(site).strip().lower())
        if mapped:
            parts.append(f"Lesion location: {mapped}.")
    return " ".join(parts) or FEATURE_CONFIG["placeholder_text"]


# --------------------------------------------------------------------------
# Parity test helper (numpy only)
# --------------------------------------------------------------------------
def parity_report(new_vecs: np.ndarray, cached_mat: np.ndarray, match_cos: float = 0.995) -> dict:
    """For each freshly computed vector, find its nearest neighbour (cosine) in
    the cached training matrix.  If the pipeline that made the cache equals the
    pipeline that made `new_vecs`, the nearest neighbour is the very same image
    (cosine ~ 1.0)."""
    a = np.asarray(new_vecs, np.float64)
    b = np.asarray(cached_mat, np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    best = (a @ b.T).max(axis=1)
    return {
        "n": int(len(best)),
        "median_max_cos": float(np.median(best)),
        "min_max_cos": float(best.min()),
        "frac_match": float((best >= match_cos).mean()),
    }


# --------------------------------------------------------------------------
# Class-conditional split conformal (numpy only)
# --------------------------------------------------------------------------
def fit_classconditional_conformal(p, y, alpha: float = 0.05) -> dict:
    """Calibrate on OUT-OF-FOLD probabilities p (P(malignant)) and labels y.
    Nonconformity of class k = 1 - P(class k).  One threshold per class, so the
    ~8% malignant class gets its own ~(1-alpha) coverage instead of being
    swamped by the benign majority (marginal conformal can hit 95% overall
    while missing many melanomas)."""
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    out = {"alpha": float(alpha), "source": "oof"}
    for k, name in ((1, "pos"), (0, "neg")):
        pk = p[y == k]
        s = (1.0 - pk) if k == 1 else pk
        n = len(s)
        if n == 0:
            raise ValueError(f"no calibration examples for class {k}")
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        out[f"q_hat_{name}"] = float(np.quantile(s, level, method="higher"))
        out[f"n_{name}"] = int(n)
    return out


def conformal_set(p: float, cp: dict) -> list:
    """Prediction set for one score.  [] can happen; callers treat it as uncertain."""
    s = []
    if p <= cp["q_hat_neg"]:
        s.append("benign")
    if (1.0 - p) <= cp["q_hat_pos"]:
        s.append("malignant")
    return s


# --------------------------------------------------------------------------
# Environment record (saved in the bundle; the app compares against it)
# --------------------------------------------------------------------------
def describe_environment() -> dict:
    env = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("sklearn", "lightgbm", "scipy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env
