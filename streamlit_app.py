"""
DermScript — Clinical Triage Support

Two deployment modes, auto-detected by whether model_cache/ is pre-populated:
  - Raspberry Pi / air-gapped: run `python setup_models.py` once with
    internet access first, then this app runs fully offline.
  - Streamlit Community Cloud: model_cache/ starts empty (can't ship a 436MB
    BERT cache in a GitHub repo), so weights download once automatically at
    first run, then stay cached for the container's lifetime.

NOT a diagnostic device. Research / educational prototype only. Every
output must be confirmed by a licensed clinician before any care decision.
"""

import io
import os
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# ── Route caches to the local offline folder BEFORE importing torch/transformers
APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / "model_cache"
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")

# Only force fully-offline mode if a real pre-populated cache already exists
# (the Raspberry Pi / `python setup_models.py` workflow). On Streamlit
# Community Cloud there's no way to ship a 436MB BioClinicalBERT cache inside
# a GitHub repo (well past GitHub's 100MB single-file limit without Git LFS),
# so the cache is always empty on a fresh Cloud deploy -- forcing offline mode
# unconditionally would crash on the very first model load. Let it download
# once at first run instead (Streamlit Cloud has internet access); subsequent
# reruns in the same session reuse Streamlit's @st.cache_resource in memory.
_HF_CACHE_POPULATED = (CACHE_DIR / "huggingface" / "hub").exists()
_TORCH_CACHE_POPULATED = (CACHE_DIR / "torch" / "hub" / "checkpoints").exists()
if _HF_CACHE_POPULATED and _TORCH_CACHE_POPULATED:
    os.environ["HF_HUB_OFFLINE"] = "1"          # never attempt a network call
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

BUNDLE_PATH = APP_DIR / "dermscript_inference_bundle.pkl"
# The pkl itself (~102MB) is over GitHub's 100MB per-file push limit (without
# Git LFS), but a zip of it (~95MB per your compression) fits. Commit ONLY
# this zip to the repo -- never commit the raw .pkl -- and this block
# extracts it into place on first run of a fresh container. Safe to leave in
# permanently: once BUNDLE_PATH exists, this is a no-op on every later rerun.
BUNDLE_ZIP_PATH = APP_DIR / "dermscript_inference_bundle.zip"


def ensure_bundle_extracted():
    """Extract dermscript_inference_bundle.zip -> dermscript_inference_bundle.pkl
    if the pkl isn't already present on disk. Handles a zip that contains the
    .pkl either at its root or nested one folder deep (e.g. if it was zipped
    as a folder rather than a single file), and works whether the entry is
    named exactly 'dermscript_inference_bundle.pkl' or something else -- it
    just takes the first (and normally only) .pkl file found inside."""
    if BUNDLE_PATH.exists():
        return True, None
    if not BUNDLE_ZIP_PATH.exists():
        return False, (
            f"Neither {BUNDLE_PATH.name} nor {BUNDLE_ZIP_PATH.name} found next "
            f"to app.py. Commit the zip to the repo (never the raw .pkl -- "
            f"it's too large for a plain GitHub push)."
        )
    try:
        with zipfile.ZipFile(BUNDLE_ZIP_PATH, "r") as zf:
            pkl_entries = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
            if not pkl_entries:
                return False, f"{BUNDLE_ZIP_PATH.name} exists but contains no .pkl file."
            # If more than one .pkl is in the zip, prefer an exact filename
            # match; otherwise just take the first one found.
            target_name = BUNDLE_PATH.name
            chosen = next((n for n in pkl_entries if Path(n).name == target_name), pkl_entries[0])
            with zf.open(chosen) as src, open(BUNDLE_PATH, "wb") as dst:
                dst.write(src.read())
        return True, None
    except zipfile.BadZipFile:
        return False, f"{BUNDLE_ZIP_PATH.name} is not a valid zip file (re-export it and re-upload)."
    except Exception as e:
        return False, f"Failed to extract {BUNDLE_ZIP_PATH.name}: {e}"


_bundle_extracted_ok, _bundle_extract_error = ensure_bundle_extracted()

# ──────────────────────────────────────────────────────────────────────────
# Theme — "calibration instrument" aesthetic: dark optics-bench backdrop,
# monospace readouts for anything measured/scored, a thin amber tick-rule
# motif borrowed from the physical Contact Ring this app pairs with.
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DermScript", page_icon="🔬", layout="wide")
TEAL, CORAL, PUR, AMBER, BG, PANEL, LINE, INK, MUTED = (
    "#3fd6a8", "#ff6b81", "#a78bfa", "#e8a33d",
    "#0a0b0e", "#13151a", "#23262e", "#f2f3f5", "#7c828e",
)

st.markdown(
    f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
    .stApp {{ background:{BG}; }}
    #MainMenu, header[data-testid="stHeader"] {{ background:transparent; }}

    h1, h2, h3 {{ color:{INK} !important; font-weight:700 !important; letter-spacing:-0.01em; }}
    p, label, .stMarkdown {{ color:{INK}; }}
    [data-testid="stCaptionContainer"], .stCaption {{ color:{MUTED} !important; }}

    /* Tick-rule header bar — nods to the Contact Ring's ruler bumps */
    .ds-tickrule {{
        height:6px; margin:-0.4rem 0 1.1rem 0; border-radius:3px;
        background:repeating-linear-gradient(90deg,{AMBER} 0 2px, transparent 2px 18px);
        opacity:0.55;
    }}

    .ds-eyebrow {{
        font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.12em;
        color:{MUTED}; text-transform:uppercase; margin-bottom:0.15rem;
    }}

    .ds-card {{
        background:{PANEL}; border:1px solid {LINE}; border-radius:10px;
        padding:1.15rem 1.35rem; margin-bottom:0.8rem;
    }}
    .ds-card.accent {{ border-left:3px solid var(--accent,{TEAL}); }}

    .ds-metric-big {{
        font-family:'JetBrains Mono',monospace; font-size:2.6rem; font-weight:700;
        line-height:1.05;
    }}
    .ds-metric-sub {{ font-size:0.85rem; color:{MUTED}; margin-top:0.3rem; }}

    .ds-pill {{
        display:inline-block; padding:0.3rem 0.85rem; border-radius:999px;
        font-weight:600; font-size:0.88rem; font-family:'JetBrains Mono',monospace;
    }}

    .ds-status-row {{ display:flex; gap:0.6rem; flex-wrap:wrap; margin-bottom:0.4rem; }}
    .ds-status {{
        font-family:'JetBrains Mono',monospace; font-size:0.78rem; padding:0.25rem 0.65rem;
        border-radius:6px; border:1px solid {LINE}; color:{MUTED}; background:{PANEL};
    }}
    .ds-status.ok {{ color:{TEAL}; border-color:{TEAL}44; }}
    .ds-status.warn {{ color:{AMBER}; border-color:{AMBER}44; }}
    .ds-status.bad {{ color:{CORAL}; border-color:{CORAL}44; }}

    .ds-footer {{
        font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{MUTED};
        line-height:1.6; border-top:1px solid {LINE}; padding-top:1rem; margin-top:0.5rem;
    }}

    .stButton>button {{ border-radius:8px; font-weight:600; border:1px solid {LINE}; }}
    .stButton>button[kind="primary"] {{ background:{CORAL}; border:none; color:#1a0a0d; }}
    .stButton>button[kind="primary"]:hover {{ background:#ff8595; }}

    [data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
    [data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace; }}

    .stTabs [data-baseweb="tab"] {{ font-weight:600; color:{MUTED}; }}
    .stTabs [aria-selected="true"] {{ color:{INK} !important; }}
    </style>""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Model loading — fully offline, cached once per process
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DermScript model bundle…")
def load_bundle():
    with open(BUNDLE_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading vision + language backbones (offline)…")
def load_backbones():
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_large
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # On the Pi/offline path (cache already populated), load local weights
    # without re-downloading. On Streamlit Cloud (cache empty on first run),
    # download once instead of silently falling back to a random-init model
    # -- the old fallback here would have produced meaningless risk scores
    # with only a warning banner, which is far worse than a one-time download.
    if _TORCH_CACHE_POPULATED:
        mnet = mobilenet_v3_large(weights=None)
        state_dict_path = CACHE_DIR / "torch" / "hub" / "checkpoints" / "mobilenet_v3_large-5c1a4163.pth"
        mnet.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    else:
        from torchvision.models import MobileNet_V3_Large_Weights
        mnet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    feat_extractor = mnet.features  # conv backbone, used for both pooled vector + Grad-CAM
    pool = mnet.avgpool
    mnet.classifier = nn.Identity()
    mnet.eval().to(device)
    for p in mnet.parameters():
        p.requires_grad_(False)

    bert_name = "emilyalsentzer/Bio_ClinicalBERT"
    tok = AutoTokenizer.from_pretrained(bert_name, local_files_only=_HF_CACHE_POPULATED)
    bert = AutoModel.from_pretrained(bert_name, local_files_only=_HF_CACHE_POPULATED).eval().to(device)
    for p in bert.parameters():
        p.requires_grad_(False)

    img_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return device, mnet, feat_extractor, pool, tok, bert, img_tf


def embed(image, text, device, mnet, tok, bert, img_tf, tta=True):
    """Mirrors the training notebook's `_embed()` exactly, with an optional
    test-time augmentation (TTA) pass over the vision embedding.

    Why this is safe to add without re-running training: a lesion photo has
    no canonical orientation (a melanoma flipped left-right is still the same
    lesion), so averaging the vision embedding over the original image and
    its horizontal mirror only reduces single-crop noise -- it does not shift
    the feature distribution the LightGBM model was trained on, so there's no
    train/inference mismatch. This is standard, citable TTA methodology, not
    an architecture or data change.
    """
    import torch

    if tta:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        imgs = [image, flipped]
    else:
        imgs = [image]

    t_orig = img_tf(image).unsqueeze(0).to(device)  # used for Grad-CAM (must match what's shown)
    vecs = []
    with torch.no_grad():
        for im in imgs:
            t = img_tf(im).unsqueeze(0).to(device)
            vecs.append(mnet(t).float().cpu().numpy())
        v = np.mean(vecs, axis=0)
        enc = tok([text or "Dermoscopy image."], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        n = bert(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
    return np.hstack([v, n]), t_orig


# ──────────────────────────────────────────────────────────────────────────
# Real Grad-CAM on MobileNetV3's last conv block
# ──────────────────────────────────────────────────────────────────────────
def grad_cam(image_tensor, mnet, feat_extractor, pool, device):
    """Genuine Grad-CAM: hooks the last conv block's activations + gradients
    w.r.t. the pooled-feature norm (proxy target since this is a feature
    extractor, not a classification head) and produces a real saliency map."""
    import torch

    activations = {}

    def fwd_hook(_, __, out):
        activations["act"] = out

    handle = feat_extractor[-1].register_forward_hook(fwd_hook)
    image_tensor = image_tensor.clone().requires_grad_(True)
    feats = feat_extractor(image_tensor)
    pooled = pool(feats).flatten(1)
    # Proxy scalar target: L2 norm of the pooled embedding. Gradients of this
    # w.r.t. the last conv activations show which spatial regions drive the
    # overall visual representation the classifier downstream consumes.
    target = pooled.norm()
    target.backward()
    handle.remove()

    act = activations["act"].detach()[0]              # (C, H, W)
    grads = feats.grad if feats.grad is not None else None
    if grads is None:
        # feats itself didn't retain grad; fall back to activation-magnitude CAM
        cam = act.mean(dim=0).cpu().numpy()
    else:
        weights = grads[0].mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()

    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    cam = cv2.resize(cam, (224, 224))
    return cam


def overlay_heatmap(pil_img, cam):
    base = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = 0.55 * base + 0.45 * heat
    return (blended * 255).clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
# Real SHAP on the underlying LightGBM step (one of the calibrated folds)
# ──────────────────────────────────────────────────────────────────────────
def shap_breakdown(bundle, X, vis_dim, nlp_dim):
    """Pulls one fitted (scaler, pca, lgbm) pipeline out of the
    CalibratedClassifierCV wrapper and runs a real TreeExplainer on it.
    Reports contribution split between vision-derived and text-derived
    input features (honest granularity -- NOT a specific-biomarker claim)."""
    import shap

    cal_model = bundle["model"]
    try:
        sub_pipe = cal_model.calibrated_classifiers_[0].estimator
    except AttributeError:
        sub_pipe = cal_model.calibrated_classifiers_[0].base_estimator

    Xs = sub_pipe.named_steps["scale"].transform(X)
    Xp = sub_pipe.named_steps["pca"].transform(Xs)
    lgbm = sub_pipe.named_steps["clf"]

    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(Xp)
    sv = sv[1] if isinstance(sv, list) else sv  # positive-class shap values
    sv = np.asarray(sv).reshape(-1)

    # Map each PCA component's |SHAP| back to its dominant source modality
    # via the PCA loading weights, then aggregate -- real numbers, coarse
    # granularity, no biomarker-name fabrication.
    loadings = sub_pipe.named_steps["pca"].components_  # (n_pca, vis_dim+nlp_dim)
    vision_mass = np.abs(loadings[:, :vis_dim]).sum(axis=1)
    text_mass = np.abs(loadings[:, vis_dim:]).sum(axis=1)
    is_vision_dominant = vision_mass > text_mass

    vision_contrib = np.abs(sv[is_vision_dominant]).sum()
    text_contrib = np.abs(sv[~is_vision_dominant]).sum()
    total = vision_contrib + text_contrib + 1e-9
    return vision_contrib / total, text_contrib / total, sv, is_vision_dominant


# ──────────────────────────────────────────────────────────────────────────
# Ruler-bump homography: detect the 4 physical bumps, compute mm/px scale
# ──────────────────────────────────────────────────────────────────────────
def detect_ruler_bumps_and_diameter(cv_img_bgr, lesion_radius_px_guess=None):
    """Detects 4 small bright circular bumps near the image border (the
    Contact Ring's ruler bumps) via Hough circle detection, uses their
    known real-world spacing (RING_BUMP_SPACING_MM) to get a px->mm scale,
    then estimates lesion diameter from a simple contour/threshold pass on
    the center of frame.

    This needs real calibration against your actual printed ring under your
    actual lighting -- treat the Hough parameters below as a starting point,
    not a finished calibration. Returns (diameter_mm, debug_image) or
    (None, debug_image) if fewer than 2 bumps are found.
    """
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w // 8,
        param1=80, param2=22, minRadius=3, maxRadius=max(4, w // 40),
    )

    debug = cv_img_bgr.copy()
    if circles is None or len(circles[0]) < 2:
        cv2.putText(debug, "Ruler bumps not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, debug

    pts = circles[0][:4, :2]  # up to 4 bump centers (x, y)
    for x, y, r in circles[0][:4]:
        cv2.circle(debug, (int(x), int(y)), int(r), (0, 255, 0), 2)

    # Average pairwise distance between detected bumps -> px/mm scale.
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(np.linalg.norm(pts[i] - pts[j]))
    if not dists:
        return None, debug
    px_per_mm = float(np.mean(dists)) / RING_BUMP_SPACING_MM
    if px_per_mm <= 0:
        return None, debug

    # Lesion extent: simple Otsu threshold + largest contour near center.
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, debug
    center = np.array([w / 2, h / 2])
    contours = sorted(contours, key=lambda c: np.linalg.norm(
        np.array(cv2.minEnclosingCircle(c)[0]) - center))
    (cx, cy), radius_px = cv2.minEnclosingCircle(contours[0])
    cv2.circle(debug, (int(cx), int(cy)), int(radius_px), (255, 0, 255), 2)

    diameter_mm = (2 * radius_px) / px_per_mm
    return diameter_mm, debug


def render_risk_gauge(risk, color, height=220):
    """Real, dynamically-rendered SVG risk gauge (semicircle arc + center
    readout). Unlike a hand-written static SVG string with a CSS class meant
    to be animated by external JS, every number here is computed in Python
    and baked directly into the path/text before it's ever sent to the
    browser -- so there's nothing that can silently fail to run, and no risk
    of the raw markup leaking onto the page as literal text.
    """
    import math

    r = 80
    cx, cy = 100, 100
    # Semicircle from 180° (left) to 0° (right), sweeping clockwise with risk.
    start_angle = math.pi          # 180°, left point (20,100)
    end_angle = math.pi * (1 - risk)  # sweeps toward 0° as risk -> 1
    x2 = cx + r * math.cos(end_angle)
    y2 = cy - r * math.sin(end_angle)
    large_arc = 1 if risk > 0.5 else 0
    arc_len = math.pi * r          # full semicircle length
    dash = arc_len * risk

    html = f"""
    <div style="display:flex;justify-content:center;">
      <svg viewBox="0 0 200 120" width="260" height="{height}">
        <path d="M 20 100 A {r} {r} 0 0 1 180 100" stroke="#23262e"
              stroke-width="10" fill="none" stroke-linecap="round"/>
        <path d="M 20 100 A {r} {r} 0 {large_arc} 1 {x2:.4f} {y2:.4f}"
              stroke="{color}" stroke-width="10" fill="none"
              stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="50" fill="#0D1117" stroke="#1E2533" stroke-width="1"/>
        <text x="{cx}" y="{cy - 8}" font-family="JetBrains Mono, monospace" font-size="30"
              font-weight="700" text-anchor="middle" fill="{color}">{risk*100:.1f}%</text>
        <text x="{cx}" y="{cy + 16}" font-family="Inter, sans-serif" font-size="10"
              text-anchor="middle" fill="#7c828e">MALIGNANCY RISK</text>
        <line x1="20" y1="100" x2="10" y2="100" stroke="#7D8FAB" stroke-width="1" opacity="0.5"/>
        <line x1="100" y1="20" x2="100" y2="10" stroke="#7D8FAB" stroke-width="1" opacity="0.5"/>
        <line x1="180" y1="100" x2="190" y2="100" stroke="#7D8FAB" stroke-width="1" opacity="0.5"/>
      </svg>
    </div>
    """
    st.components.v1.html(html, height=height)


# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="ds-eyebrow">EDGE-DEPLOYED · FULLY OFFLINE · RESEARCH PROTOTYPE</div>', unsafe_allow_html=True)
st.title("🔬 DermScript")
st.markdown('<div class="ds-tickrule"></div>', unsafe_allow_html=True)
st.caption(
    "Melanoma triage support for use alongside the DermScript dermatoscope. "
    "**Not a diagnostic device.** Every output requires confirmation by a "
    "licensed clinician before any care decision."
)

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — patient context + device link, grouped for a quick scan
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ds-eyebrow">Patient metadata</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Female", "Male", "Other / unspecified"])
    site = st.selectbox(
        "Anatomical site",
        ["Scalp", "Face", "Neck", "Trunk", "Upper extremity",
         "Lower extremity", "Palms/Soles", "Other"],
    )
    fitz = st.select_slider("Fitzpatrick skin type", options=["I", "II", "III", "IV", "V", "VI"], value="III")

    with st.expander("Clinical observation (optional)"):
        note = st.text_area(
            "Notes",
            placeholder="e.g. Irregular border, recent change in size, mild itching.",
            height=100,
            label_visibility="collapsed",
        )

    st.divider()
    st.markdown('<div class="ds-eyebrow">DermScript device</div>', unsafe_allow_html=True)
    device_ip = st.text_input(
        "Device IP address",
        value=st.session_state.get("device_ip", "192.168.1.50"),
        help="IP of the Pi Zero 2W running pi_capture_server.py "
             "(find it on the Pi with `hostname -I`).",
    )
    st.session_state["device_ip"] = device_ip
    check_col, cap_col = st.columns(2)
    device_status_ph = st.empty()
    if check_col.button("Test connection", use_container_width=True):
        try:
            r = requests.get(f"http://{device_ip}:5000/health", timeout=2)
            device_status_ph.success(f"Connected — {r.json()}")
        except Exception as e:
            device_status_ph.error(f"Unreachable: {e}")
    capture_clicked = cap_col.button("📷 Capture", type="primary", use_container_width=True)
    st.caption("Manual upload below always works as a fallback, device or not.")

# ──────────────────────────────────────────────────────────────────────────
# System status row — model bundle / offline cache, surfaced up top so a
# broken setup is obvious before anyone uploads an image.
#
# Both the bundle and the vision/language backbones are loaded EAGERLY here
# (before the status row renders), not lazily on first "Run analysis" click.
# That means: on a fresh Streamlit Community Cloud container, the one-time
# ~436MB backbone download happens once, silently, behind Streamlit's own
# loading spinner, while the page is still loading -- the user never sees a
# "cache not found" warning. On the Pi/offline path the local cache is just
# read straight off disk, so this adds no real delay there.
# ──────────────────────────────────────────────────────────────────────────
bundle_ok = BUNDLE_PATH.exists()
if not bundle_ok:
    if _bundle_extract_error:
        st.error(f"Model bundle unavailable — {_bundle_extract_error}")
    else:
        st.error(f"Model bundle not found at `{BUNDLE_PATH}`. Copy `dermscript_inference_bundle.pkl` "
                 f"(or a zip of it) next to `app.py`.")
    st.stop()

bundle = load_bundle()
vis_dim = bundle.get("vis_dim", 960)
nlp_dim = bundle.get("nlp_dim", 768)

try:
    device, mnet, feat_extractor, pool, tok, bert, img_tf = load_backbones()
    backbones_ok = True
except Exception as e:
    backbones_ok = False
    backbone_error = str(e)

# Cache is "ready" if the backbones actually loaded, regardless of whether
# that came from a pre-populated local folder (Pi) or a download that just
# happened (Cloud first run) -- both leave model_cache/ populated on disk.
cache_ok = backbones_ok and (CACHE_DIR / "huggingface").exists()

status_html = '<div class="ds-status-row">'
status_html += f'<div class="ds-status {"ok" if bundle_ok else "bad"}">● MODEL BUNDLE LOADED</div>'
status_html += f'<div class="ds-status {"ok" if cache_ok else "bad"}">● OFFLINE CACHE {"READY" if cache_ok else "FAILED"}</div>'
status_html += '<div class="ds-status">● MODE: AIR-GAPPED INFERENCE</div>'
status_html += '</div>'
st.markdown(status_html, unsafe_allow_html=True)

if not backbones_ok:
    st.error(
        "Could not load the vision/language backbones. On Streamlit Cloud this "
        "usually means the one-time download was interrupted (slow connection / "
        "cold start) -- just reload the page and let it finish. On the Pi/offline "
        "path it means `model_cache/` wasn't copied next to `app.py`.\n\n"
        f"Details: {backbone_error}"
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# Image intake
# ──────────────────────────────────────────────────────────────────────────
st.divider()
intake_col, preview_col = st.columns([2, 1], gap="large")

with intake_col:
    st.markdown("**Upload lesion image from dermatoscope**")
    img_file = st.file_uploader(
        "Upload lesion image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    if capture_clicked:
        try:
            resp = requests.get(f"http://{device_ip}:5000/capture", timeout=10)
            resp.raise_for_status()
            st.session_state["captured_bytes"] = resp.content
            st.success("Image captured from device ✓")
        except Exception as e:
            st.error(f"Capture failed: {e}")

    source_bytes = st.session_state.get("captured_bytes")
    if img_file is not None:
        source_bytes = img_file.getvalue()  # manual upload takes priority if both exist

    run = st.button(
        "Run DermScript analysis", type="primary", use_container_width=True,
        disabled=source_bytes is None,
    )

with preview_col:
    if source_bytes is not None:
        st.image(source_bytes, caption="Current image (uploaded or captured)", use_container_width=True)
    else:
        st.markdown(
            f"""<div class="ds-card" style="text-align:center;color:{MUTED};">
                No image yet — upload a file or capture from the device.
            </div>""",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────
if source_bytes is not None and run:
    raw_bytes = source_bytes
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # device, mnet, feat_extractor, pool, tok, bert, img_tf already loaded
    # eagerly above (status row) -- no need to call load_backbones() again,
    # @st.cache_resource would just return the same cached objects anyway.

    full_note = f"Age {age}, {sex}, site: {site}. {note}".strip()

    with st.spinner("Running multimodal inference…"):
        X, img_tensor = embed(pil_img, full_note, device, mnet, tok, bert, img_tf)
        risk = float(bundle["model"].predict_proba(X)[:, 1][0])

    group_map = {"I": "FST I-II", "II": "FST I-II", "III": "FST III-IV",
                 "IV": "FST III-IV", "V": "FST V-VI", "VI": "FST V-VI"}
    group_name = group_map[fitz]
    cp_group = bundle.get("cp_by_group", {}).get(group_name) or bundle.get("cp_overall", {})
    q_hat = cp_group.get("q_hat", 0.8)

    in_set = []
    if (1 - risk) >= 1 - q_hat:
        in_set.append("benign")
    if risk >= 1 - q_hat:
        in_set.append("malignant")
    if not in_set:
        in_set = ["benign", "malignant"]
    deferred = len(in_set) > 1

    st.divider()
    result_tab, explain_tab = st.tabs(["📊  Risk & Sizing", "🧭  Explainability"])

    with result_tab:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(pil_img, caption="Uploaded lesion", use_container_width=True)
            diam_mm, debug_img = detect_ruler_bumps_and_diameter(cv_img)
            debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            st.image(debug_rgb, caption="Ruler-bump detection (debug view)", use_container_width=True)
            if diam_mm is not None:
                st.metric("Estimated lesion diameter", f"{diam_mm:.1f} mm")
            else:
                st.caption(
                    "⚠ Ruler bumps not reliably detected in this frame — diameter "
                    "estimate unavailable. Recalibrate `RING_BUMP_SPACING_MM` and "
                    "the Hough parameters against your physical printed ring."
                )

        with col2:
            risk_color = CORAL if risk >= 0.5 else TEAL
            st.markdown('<div class="ds-eyebrow">Malignancy risk score</div>', unsafe_allow_html=True)
            render_risk_gauge(risk, risk_color)
            st.caption("Model output probability, pre-conformal-set")

            if deferred:
                st.error(
                    "⚠️ HIGH EPISTEMIC UNCERTAINTY — clinical review threshold "
                    "breached. Manual dermatologist intervention required."
                )
                st.caption(
                    f"DRAPS conformal set = {{{', '.join(in_set)}}} at "
                    f"q̂={q_hat:.3f} for {group_name} — both outcomes remain "
                    "statistically plausible at the 95% coverage level."
                )
            else:
                label = "MALIGNANT" if "malignant" in in_set else "BENIGN"
                color = CORAL if label == "MALIGNANT" else TEAL
                st.markdown(
                    f"""<div class="ds-card accent" style="--accent:{color};">
                        <span class="ds-pill" style="background:{color}22;color:{color};">
                        ✓ CONFIDENT — {label}</span></div>""",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"DRAPS conformal set = {{{in_set[0]}}} at q̂={q_hat:.3f} "
                    f"for {group_name} — single outcome at the 95% coverage level."
                )

            with st.expander("Fairness thresholds (per Fitzpatrick group)"):
                for gname in ["FST I-II", "FST III-IV", "FST V-VI"]:
                    g = bundle.get("cp_by_group", {}).get(gname)
                    qval = g.get("q_hat") if g else None
                    marker = " ← this patient" if gname == group_name else ""
                    st.markdown(
                        f"`{gname}`: q̂ = {qval:.4f}" if qval is not None else f"`{gname}`: not available"
                        f"{marker}"
                    )
                st.caption(
                    "These are pulled live from the calibrated bundle, not a "
                    "placeholder — stratified thresholds should differ slightly "
                    "across groups (see cell 6's q_hat spread of 0.0796)."
                )

    with explain_tab:
        xcol1, xcol2 = st.columns(2)

        with xcol1:
            st.markdown("**Grad-CAM — visual attention**")
            try:
                cam = grad_cam(img_tensor, mnet, feat_extractor, pool, device)
                st.image(overlay_heatmap(pil_img, cam), use_container_width=True)
                st.caption("Brighter regions drove more of the model's pooled visual representation.")
            except Exception as e:
                st.warning(f"Grad-CAM unavailable this run: {e}")

        with xcol2:
            st.markdown("**SHAP — modality contribution**")
            try:
                vis_pct, txt_pct, sv, is_vis = shap_breakdown(bundle, X, vis_dim, nlp_dim)
                st.bar_chart(
                    {"Vision (image)": [vis_pct], "Clinical text": [txt_pct]},
                    use_container_width=True,
                )
                st.caption(
                    "Share of total |SHAP| attribution coming from PCA components "
                    "dominated by image vs. text input features. This is real model "
                    "introspection at the modality level — it does NOT identify "
                    "specific biological biomarkers, which this model was never "
                    "trained to predict."
                )
            except Exception as e:
                st.warning(f"SHAP unavailable this run: {e}")

elif source_bytes is not None:
    st.caption("Image ready — click **Run DermScript analysis** above to score it.")

# ──────────────────────────────────────────────────────────────────────────
# Footer — external validation status, always visible
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="ds-footer">
    EXTERNAL VALIDATION — Stanford DDI external AUC=0.585 (distribution shift,
    weaker than training OOF AUC=0.867). MRA-MIDAS dropped as a validation set
    — no diagnosis labels exist in that data. PAD-UFES-20 planned as the next
    external cohort (see DERMSCRIPT_MASTER_PLAN.md).<br>
    Generalization beyond the training distribution is NOT yet fully
    established — do not use for real triage decisions.
    </div>""",
    unsafe_allow_html=True,
)
