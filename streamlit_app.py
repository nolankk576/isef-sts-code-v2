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

v8.7 CHANGELOG vs. the version this replaces:
  1. cp_by_group_ddi key fix: bundle stores per-skin-tone DRAPS thresholds
     under '12'/'34'/'56' (Fitzpatrick type-pair codes), not "FST I-II" etc.
     The old code's literal-string lookup never matched, silently falling
     back to cp_overall every time.
  2. Sidebar patient fields are now fully optional (checkbox-gated age,
     "Unknown" default for sex/site/skin type) instead of forcing defaults
     like age=45 into every prediction's text embedding.
  3. Device connection error handling distinguishes timeout / connection-
     refused / not-an-image instead of a generic exception dump, and the
     buttons disable themselves until an IP is entered.
  4. NEW: routes to bundle['clinical_camera_specialist'] when the modality
     detector flags an image as clinical-camera, instead of only ever
     showing the dermatoscope model's near-chance score on those images.
     This specialist predicts "malignant lesion" broadly (not melanoma
     specifically) due to its SCIN-sourced label scope -- the UI says so
     explicitly wherever this branch fires.
  5. Footer corrected: MED-NODE is clinical-camera (not dermatoscope --
     its own source paper is titled "...using non-dermoscopic images"),
     PH2 is the only genuinely dermatoscopic external cohort.
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

try:
    from referral_note import generate_referral_note
    REFERRAL_NOTE_AVAILABLE = True
except ImportError:
    REFERRAL_NOTE_AVAILABLE = False  # app still runs fully without this feature

# -- Route caches to the local offline folder BEFORE importing torch/transformers
APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / "model_cache"
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")

_HF_CACHE_POPULATED = (CACHE_DIR / "huggingface" / "hub").exists()
_TORCH_CACHE_POPULATED = (CACHE_DIR / "torch" / "hub" / "checkpoints").exists()
if _HF_CACHE_POPULATED and _TORCH_CACHE_POPULATED:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

BUNDLE_PATH = APP_DIR / "dermscript_inference_bundle_v8.pkl"
BUNDLE_ZIP_PATH = APP_DIR / "dermscript_inference_bundle_v8.zip"

RING_BUMP_SPACING_MM = 10.0  # <-- replace with your ring's measured spacing
MIN_BUMPS_FOR_SCALE = 3


def ensure_bundle_extracted():
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

    if _TORCH_CACHE_POPULATED:
        mnet = mobilenet_v3_large(weights=None)
        state_dict_path = CACHE_DIR / "torch" / "hub" / "checkpoints" / "mobilenet_v3_large-5c1a4163.pth"
        mnet.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    else:
        from torchvision.models import MobileNet_V3_Large_Weights
        mnet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    feat_extractor = mnet.features
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
    import torch

    if tta:
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        imgs = [image, flipped]
    else:
        imgs = [image]

    t_orig = img_tf(image).unsqueeze(0).to(device)

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


def grad_cam(image_tensor, mnet, feat_extractor, pool, device):
    import torch

    activations = {}

    def fwd_hook(_, __, out):
        activations["act"] = out

    handle = feat_extractor[-1].register_forward_hook(fwd_hook)

    image_tensor = image_tensor.clone().requires_grad_(True)
    feats = feat_extractor(image_tensor)
    pooled = pool(feats).flatten(1)

    target = pooled.norm()
    target.backward()
    handle.remove()

    act = activations["act"].detach()[0]
    grads = feats.grad if feats.grad is not None else None
    if grads is None:
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


def shap_breakdown(cal_model, X, vis_dim, nlp_dim, max_display=12):
    """Takes an explicit cal_model argument (main model OR specialist) rather
    than always reading bundle['model'], so this works for either branch."""
    import shap

    try:
        sub_pipe = cal_model.calibrated_classifiers_[0].estimator
    except AttributeError:
        sub_pipe = cal_model.calibrated_classifiers_[0].base_estimator

    Xs = sub_pipe.named_steps["scale"].transform(X)
    Xp = sub_pipe.named_steps["pca"].transform(Xs)
    lgbm = sub_pipe.named_steps["clf"]

    explainer = shap.TreeExplainer(lgbm)
    explanation = explainer(Xp)

    if explanation.values.ndim == 3:
        sv = explanation.values[0, :, 1]
        base_value = explanation.base_values[0, 1]
    else:
        sv = explanation.values[0]
        base_value = explanation.base_values[0]
        base_value = float(np.asarray(base_value).reshape(-1)[0])

    sv = np.asarray(sv).reshape(-1)
    feature_values = np.asarray(Xp[0]).reshape(-1)

    loadings = sub_pipe.named_steps["pca"].components_
    vision_mass = np.abs(loadings[:, :vis_dim]).sum(axis=1)
    text_mass = np.abs(loadings[:, vis_dim:]).sum(axis=1)
    is_vision_dominant = vision_mass > text_mass

    vision_contrib = np.abs(sv[is_vision_dominant]).sum()
    text_contrib = np.abs(sv[~is_vision_dominant]).sum()
    total = vision_contrib + text_contrib + 1e-9

    feature_names = [
        f"PC{i+1} ({'vision' if is_vision_dominant[i] else 'text'})"
        for i in range(len(sv))
    ]

    single_explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names,
    )

    return (
        vision_contrib / total,
        text_contrib / total,
        single_explanation,
        is_vision_dominant,
    )


def render_shap_waterfall(explanation, max_display=12, width=760):
    values = np.asarray(explanation.values).reshape(-1)
    names = list(explanation.feature_names)
    base_value = float(explanation.base_values)

    order = np.argsort(-np.abs(values))
    top_idx = order[:max_display]
    rest_idx = order[max_display:]

    rows = [(names[i], values[i]) for i in top_idx]
    if len(rest_idx) > 0:
        rows.append((f"{len(rest_idx)} other components", float(values[rest_idx].sum())))

    rows = rows[::-1]

    final_value = base_value + sum(v for _, v in rows)
    all_vals = [base_value] + [base_value + sum(v for _, v in rows[:i + 1]) for i in range(len(rows))]
    lo, hi = min(all_vals), max(all_vals)
    span = max(hi - lo, 1e-6)
    pad = span * 0.15
    lo, hi = lo - pad, hi + pad

    row_h = 34
    top_margin = 40
    bottom_margin = 50
    height = top_margin + row_h * len(rows) + bottom_margin
    label_w = 230
    plot_w = width - label_w - 90

    def x_of(v):
        return label_w + (v - lo) / (hi - lo) * plot_w

    bars = []
    running = base_value
    for i, (name, val) in enumerate(rows):
        y = top_margin + i * row_h
        start = running
        end = running + val
        running = end
        x1, x2 = x_of(start), x_of(end)
        left, right = min(x1, x2), max(x1, x2)
        color = CORAL if val >= 0 else TEAL
        sign = "+" if val >= 0 else ""
        bars.append(f"""
          <text x="{label_w - 10}" y="{y + row_h/2 + 4}" text-anchor="end"
                font-family="JetBrains Mono, monospace" font-size="11" fill="{INK}">{name}</text>
          <rect x="{left:.1f}" y="{y + 6}" width="{max(right-left, 2):.1f}" height="{row_h - 14}"
                fill="{color}" opacity="0.85" rx="2"/>
          <text x="{right + 8:.1f}" y="{y + row_h/2 + 4}"
                font-family="JetBrains Mono, monospace" font-size="10.5" fill="{color}">{sign}{val:.3f}</text>
        """)

    base_x = x_of(base_value)
    final_x = x_of(final_value)

    svg = f"""
    <div style="display:flex;justify-content:center;">
      <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">
        <line x1="{base_x:.1f}" y1="{top_margin - 10}" x2="{base_x:.1f}" y2="{height - bottom_margin + 10}"
              stroke="{MUTED}" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>
        <text x="{base_x:.1f}" y="{top_margin - 16}" text-anchor="middle"
              font-family="JetBrains Mono, monospace" font-size="10" fill="{MUTED}">base {base_value:.3f}</text>
        {"".join(bars)}
        <line x1="{final_x:.1f}" y1="{top_margin - 10}" x2="{final_x:.1f}" y2="{height - bottom_margin + 10}"
              stroke="{AMBER}" stroke-width="1.5" opacity="0.8"/>
        <text x="{final_x:.1f}" y="{height - bottom_margin + 26}" text-anchor="middle"
              font-family="JetBrains Mono, monospace" font-size="11" font-weight="700" fill="{AMBER}">f(x) = {final_value:.3f}</text>
      </svg>
    </div>
    """
    return svg


def detect_ruler_bumps_and_diameter(cv_img_bgr, lesion_radius_px_guess=None):
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w // 8,
        param1=80, param2=22, minRadius=3, maxRadius=max(4, w // 40),
    )

    debug = cv_img_bgr.copy()

    if circles is None or len(circles[0]) < MIN_BUMPS_FOR_SCALE:
        cv2.putText(debug, "Ruler bumps not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, debug

    pts = circles[0][:4, :2]
    for x, y, r in circles[0][:4]:
        cv2.circle(debug, (int(x), int(y)), int(r), (0, 255, 0), 2)

    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(np.linalg.norm(pts[i] - pts[j]))

    if not dists:
        return None, debug

    px_per_mm = float(np.mean(dists)) / RING_BUMP_SPACING_MM
    if px_per_mm <= 0:
        return None, debug

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


def render_risk_gauge(risk, color, height=220, label="MALIGNANCY RISK"):
    import math

    r = 80
    cx, cy = 100, 100

    start_angle = math.pi
    end_angle = math.pi * (1 - risk)
    x2 = cx + r * math.cos(end_angle)
    y2 = cy - r * math.sin(end_angle)
    large_arc = 1 if risk > 0.5 else 0

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
        <text x="{cx}" y="{cy + 16}" font-family="Inter, sans-serif" font-size="9"
              text-anchor="middle" fill="#7c828e">{label}</text>
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
    "This device is validated for dermatoscope images (AUC ~0.87 on the "
    "genuinely dermatoscopic held-out cohort). For clinical-camera photos, "
    "a separate specialist model routes in automatically, with reduced "
    "confidence and a broader 'malignant lesion' label. "
    "**Not a diagnostic device.** Every output requires confirmation by a "
    "licensed clinician before any care decision."
)

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — patient context (fully optional) + device link
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ds-eyebrow">Patient metadata (optional)</div>', unsafe_allow_html=True)
    st.caption(
        "Nothing here is required — the model runs fine with everything "
        "left as \"Unknown.\" Fill in what you actually have; anything "
        "unknown is simply omitted, not guessed at."
    )

    have_age = st.checkbox("Patient age known", value=False)
    age = (
        st.number_input("Age", min_value=0, max_value=120, value=45)
        if have_age else None
    )

    sex_choice = st.selectbox("Sex", ["Unknown", "Female", "Male", "Other / unspecified"])
    sex = None if sex_choice == "Unknown" else sex_choice

    site_choice = st.selectbox(
        "Anatomical site",
        ["Unknown", "Scalp", "Face", "Neck", "Trunk", "Upper extremity",
         "Lower extremity", "Palms/Soles", "Other"],
    )
    site = None if site_choice == "Unknown" else site_choice

    fitz_choice = st.select_slider(
        "Fitzpatrick skin type",
        options=["Unknown", "I", "II", "III", "IV", "V", "VI"],
        value="Unknown",
        help="Only affects which per-skin-tone DRAPS threshold is used. "
             "Leave as Unknown to use the overall (non-stratified) threshold.",
    )
    fitz = None if fitz_choice == "Unknown" else fitz_choice

    with st.expander("Clinical observation (optional)"):
        note = st.text_area(
            "Notes",
            placeholder="e.g. Irregular border, recent change in size, mild itching.",
            height=100,
            label_visibility="collapsed",
        )
    st.divider()
    st.markdown('<div class="ds-eyebrow">DermScript device (optional)</div>', unsafe_allow_html=True)
    st.caption(
        "Only needed if you're capturing from the physical dermatoscope "
        "over Wi-Fi. Manual upload below always works with no device at all."
    )
    device_ip = st.text_input(
        "Device IP address",
        value=st.session_state.get("device_ip", ""),
        placeholder="e.g. 192.168.1.50",
        help="IP of the Pi running pi_capture_server.py on your local "
             "network (find it on the Pi with `hostname -I`). Leave blank "
             "if you're not using the physical device this session.",
    )
    st.session_state["device_ip"] = device_ip
    check_col, cap_col = st.columns(2)
    device_status_ph = st.empty()
    device_ready = bool(device_ip.strip())
    if check_col.button("Test connection", use_container_width=True, disabled=not device_ready):
        try:
            r = requests.get(f"http://{device_ip}:5000/health", timeout=3)
            r.raise_for_status()
            device_status_ph.success(f"Connected — {r.json()}")
        except requests.exceptions.ConnectTimeout:
            device_status_ph.error("Timed out — device IP unreachable on this network.")
        except requests.exceptions.ConnectionError:
            device_status_ph.error("Connection refused — check the IP and that pi_capture_server.py is running.")
        except Exception as e:
            device_status_ph.error(f"Unreachable: {e}")
    capture_clicked = cap_col.button(
        "📷 Capture", type="primary", use_container_width=True, disabled=not device_ready
    )
    if not device_ready:
        st.caption("Enter a device IP above to enable capture, or just upload a file below.")
    else:
        st.caption("Manual upload below always works as a fallback too.")

# ──────────────────────────────────────────────────────────────────────────
# System status row
# ──────────────────────────────────────────────────────────────────────────
bundle_ok = BUNDLE_PATH.exists()
if not bundle_ok:
    if _bundle_extract_error:
        st.error(f"Model bundle unavailable — {_bundle_extract_error}")
    else:
        st.error(f"Model bundle not found at `{BUNDLE_PATH}`. Copy `dermscript_inference_bundle_v8.pkl` "
                 f"(or a zip of it) next to `app.py`.")
    st.stop()

bundle = load_bundle()
vis_dim = bundle.get("vis_dim", 960)
nlp_dim = bundle.get("nlp_dim", 768)
specialist_bundle = bundle.get("clinical_camera_specialist")

try:
    device, mnet, feat_extractor, pool, tok, bert, img_tf = load_backbones()
    backbones_ok = True
except Exception as e:
    backbones_ok = False
    backbone_error = str(e)

cache_ok = backbones_ok and (CACHE_DIR / "huggingface").exists()

status_html = '<div class="ds-status-row">'
status_html += f'<div class="ds-status {"ok" if bundle_ok else "bad"}">● MODEL BUNDLE LOADED</div>'
status_html += f'<div class="ds-status {"ok" if cache_ok else "bad"}">● OFFLINE CACHE {"READY" if cache_ok else "FAILED"}</div>'
status_html += f'<div class="ds-status {"ok" if specialist_bundle else "warn"}">● CLINICAL-CAMERA SPECIALIST {"LOADED" if specialist_bundle else "NOT IN BUNDLE"}</div>'
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
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                st.error(
                    f"Device responded but didn't return an image "
                    f"(Content-Type: '{content_type or 'missing'}'). "
                    f"Check pi_capture_server.py's /capture route."
                )
            else:
                st.session_state["captured_bytes"] = resp.content
                st.success("Image captured from device ✓")
        except requests.exceptions.ConnectTimeout:
            st.error("Capture timed out — device unreachable on this network.")
        except requests.exceptions.ConnectionError:
            st.error("Connection refused — check the IP and that pi_capture_server.py is running.")
        except Exception as e:
            st.error(f"Capture failed: {e}")

    source_bytes = st.session_state.get("captured_bytes")
    if img_file is not None:
        source_bytes = img_file.getvalue()

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

    note_parts = []
    if age is not None:
        note_parts.append(f"Age {age}")
    if sex is not None:
        note_parts.append(sex)
    if site is not None:
        note_parts.append(f"site: {site}")
    context_str = ", ".join(note_parts)
    full_note = f"{context_str}. {note}".strip(" .") if (context_str or note) else ""

    with st.spinner("Running multimodal inference…"):
        X, img_tensor = embed(pil_img, full_note, device, mnet, tok, bert, img_tf)
        main_risk = float(bundle["model"].predict_proba(X)[:, 1][0])

    # ── Modality detection decides which model's score is the HEADLINE one.
    modality_det = bundle.get("modality_detector")
    p_clinical = None
    if modality_det is not None:
        try:
            X_pca_for_modality = modality_det["pca_step"].transform(X)
            p_clinical = float(modality_det["model"].predict_proba(X_pca_for_modality)[:, 1][0])
        except Exception:
            pass

    use_specialist = (p_clinical is not None and p_clinical > 0.5 and specialist_bundle is not None)

    if use_specialist:
        risk = float(specialist_bundle["model"].predict_proba(X)[:, 1][0])
        active_model = specialist_bundle["model"]
        risk_label = "MALIGNANT LESION RISK"

        # v8.7: percentile rank + Youden's J threshold, if this bundle has
        # them (older bundles won't until the notebook is rerun -- falls
        # back gracefully to the plain caption below if missing).
        oof_dist = specialist_bundle.get("oof_score_distribution_sorted")
        op_thresh = specialist_bundle.get("operating_threshold_youden")
        percentile_rank = None
        if oof_dist is not None:
            percentile_rank = float(np.searchsorted(oof_dist, risk) / len(oof_dist) * 100)

        extra_bits = []
        if percentile_rank is not None:
            extra_bits.append(
                f"This image scores higher than {percentile_rank:.0f}% of the "
                f"specialist's own calibration images — a more informative "
                f"read than the raw percentage above, since that number is "
                f"compressed by this specialist's very low (1.55%) positive rate."
            )
        if op_thresh is not None:
            above_below = "ABOVE" if risk >= op_thresh else "below"
            extra_bits.append(
                f"This model's own best decision threshold (Youden's J) is "
                f"{op_thresh:.1%} — this image is {above_below} that threshold."
            )
        extra_caption = (" " + " ".join(extra_bits)) if extra_bits else ""

        risk_caption = (
            "Clinical-camera specialist model score (NOT the dermatoscope "
            "model). This specialist predicts a BROAD 'malignant lesion' "
            "outcome, not melanoma specifically — its training data (SCIN) "
            "labels malignant/pre-malignant conditions generally, not just "
            "melanoma. Cross-validated AUC on held-out clinical-camera "
            "images: ~0.66-0.80 depending on source. Treat as a rougher "
            "signal than the dermatoscope model's score. IMPORTANT: this "
            "specialist trained on data where only 1.55% of images are "
            "positive, so its calibrated scores rarely exceed 50% even for "
            "genuine malignant images — a LOW-looking number here is NOT "
            "evidence of benign. This branch is always treated as REFER "
            "regardless of the number shown."
        ) + extra_caption
    else:
        risk = main_risk
        active_model = bundle["model"]
        risk_label = "MALIGNANCY RISK"
        risk_caption = "Dermatoscope model output probability, pre-conformal-set."

    # ── FIX: bundle stores per-skin-tone DRAPS thresholds under '12'/'34'/'56'
    # (Fitzpatrick type-pair codes), not "FST I-II" etc. -- the old lookup
    # never matched and always silently fell back to cp_overall.
    group_map = {
        "I": ("12", "FST I-II"), "II": ("12", "FST I-II"),
        "III": ("34", "FST III-IV"), "IV": ("34", "FST III-IV"),
        "V": ("56", "FST V-VI"), "VI": ("56", "FST V-VI"),
    }
    if fitz is not None:
        cp_key, group_name = group_map[fitz]
        cp_group = bundle.get("cp_by_group_ddi", {}).get(cp_key) or bundle.get("cp_overall", {})
    else:
        cp_key, group_name = None, "Overall (Fitzpatrick not provided)"
        cp_group = bundle.get("cp_overall", {})
    q_hat = cp_group.get("q_hat", 0.8)

    # DRAPS conformal set is only meaningful for the DERMATOSCOPE model,
    # which is what it was calibrated on -- skip it on the specialist branch
    # rather than silently applying a mismatched threshold.
    if not use_specialist:
        in_set = []
        if (1 - risk) >= 1 - q_hat:
            in_set.append("benign")
        if risk >= 1 - q_hat:
            in_set.append("malignant")
        if not in_set:
            in_set = ["benign", "malignant"]
        deferred = len(in_set) > 1
    else:
        in_set = None
        deferred = None

    with st.expander("🔧 Debug — bundle & inference sanity checks"):
        try:
            classes_ = getattr(bundle["model"], "classes_", None)
            st.markdown(f"`model.classes_` = `{classes_}` "
                        f"— index 1 (used by `predict_proba(X)[:, 1]`) should "
                        f"correspond to the MALIGNANT class. If this is reversed "
                        f"or unexpected, the risk score is silently flipped.")
        except Exception as e:
            st.markdown(f"Could not read `model.classes_`: {e}")

        expected_dim = vis_dim + nlp_dim
        actual_dim = X.shape[1] if hasattr(X, "shape") else None
        dim_ok = actual_dim == expected_dim
        st.markdown(
            f"Feature vector: `X.shape` = `{getattr(X, 'shape', None)}` vs "
            f"expected `vis_dim + nlp_dim` = `{expected_dim}` "
            f"({'✓ match' if dim_ok else '✗ MISMATCH — scaler/PCA inputs are misaligned'})."
        )

        cp_by_group = bundle.get("cp_by_group_ddi", {})
        expected_keys = ["12", "34", "56"]
        missing = [g for g in expected_keys if g not in cp_by_group]
        if missing:
            st.markdown(f"`cp_by_group_ddi` is missing keys: `{missing}` — fairness panel will show 'not available' for these.")
        else:
            st.markdown("`cp_by_group_ddi` has all three expected Fitzpatrick group keys. ✓")

        st.markdown(
            f"Modality detector: P(clinical-camera) = `{p_clinical:.3f}`" if p_clinical is not None
            else "Modality detector: not available this run"
        )
        st.markdown(
            f"Routing decision: **{'CLINICAL-CAMERA SPECIALIST' if use_specialist else 'MAIN DERMATOSCOPE MODEL'}**"
        )
        if use_specialist:
            oof = specialist_bundle.get("oof_auc_by_source", {})
            st.markdown(f"Specialist model per-source cross-validated AUC: `{oof}`")
            if percentile_rank is not None:
                st.markdown(f"Percentile rank vs. specialist's own calibration set: `{percentile_rank:.1f}%`")
            if op_thresh is not None:
                st.markdown(f"Specialist operating threshold (Youden's J): `{op_thresh:.4f}` "
                            f"(this score is {'ABOVE' if risk >= op_thresh else 'below'} it)")

        st.caption(
            "This panel is here so a mis-keyed or stale bundle shows up immediately "
            "in the UI, instead of only being caught later by an unexpected score."
        )

    if use_specialist:
        st.warning(
            f"📷 This image scored {p_clinical:.0%} likely to be a clinical-camera "
            f"photo, not a dermatoscope capture. Showing the clinical-camera "
            f"specialist model's score instead of the dermatoscope model's — "
            f"see the debug panel and caption below for what that means."
        )

    st.divider()
    result_tab, explain_tab, referral_tab = st.tabs(
        ["📊  Risk & Sizing", "🧭  Explainability", "📋  Referral Note"])

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
            # FIX: the specialist model's positive rate in training is only
            # 1.55% (609/39,301) -- calibrated probabilities from a model
            # this imbalanced almost never exceed 50% even for genuine
            # malignant images (same compression effect documented for the
            # main model in the footer, worse here due to the lower base
            # rate). The backend ALREADY treats every specialist-routed
            # prediction as REFER-tier regardless of the number -- showing
            # a green "LOW" pill next to that same score contradicted the
            # REFER banner. Specialist branch now always renders amber,
            # never green, matching what the REFER tier is already saying.
            if use_specialist:
                risk_color = AMBER
            else:
                risk_color = CORAL if risk >= 0.5 else TEAL
            st.markdown(f'<div class="ds-eyebrow">{risk_label}</div>', unsafe_allow_html=True)
            render_risk_gauge(risk, risk_color, label=risk_label)
            st.caption(risk_caption)

            # v8.7: LOW / MEDIUM / HIGH tier, using the model's own DRAPS
            # thresholds as the boundaries rather than an arbitrary 50/50
            # split. For the main model this maps exactly onto the
            # conformal logic already computed above: below q_hat's lower
            # bound = LOW, the DRAPS-uncertain middle band = MEDIUM, above
            # the upper bound = HIGH. For the specialist branch (no DRAPS
            # calibration exists for it), the percentile rank against its
            # own OOF distribution is used instead, in thirds.
            # v8.7 FIX: use the sensitivity-targeted screening threshold
            # (Cell 4/4.10) as the primary HIGH-risk trigger instead of the
            # DRAPS/0.5 boundary alone. Class-imbalance compression means
            # true positives can calibrate well under 50% even with
            # balanced training -- sensitivity_threshold_90 is the
            # properly-derived operating point for "catch real melanomas,"
            # and is expected to sit well below 0.5.
            if use_specialist:
                sens_thr = specialist_bundle.get("sensitivity_threshold_90")
                if sens_thr is not None and risk >= sens_thr:
                    tier3, tier3_color = "HIGH", CORAL
                elif percentile_rank is None:
                    tier3, tier3_color = "MEDIUM", AMBER
                elif percentile_rank >= 66:
                    tier3, tier3_color = "HIGH", CORAL
                elif percentile_rank >= 33:
                    tier3, tier3_color = "MEDIUM", AMBER
                else:
                    tier3, tier3_color = "LOW", TEAL
            else:
                sens_thr = bundle.get("sensitivity_threshold_90")
                if sens_thr is not None and risk >= sens_thr:
                    tier3, tier3_color = "HIGH", CORAL
                elif deferred:
                    tier3, tier3_color = "MEDIUM", AMBER
                elif "malignant" in in_set:
                    tier3, tier3_color = "HIGH", CORAL
                else:
                    tier3, tier3_color = "LOW", TEAL
            st.markdown(
                f"""<div style="text-align:center;margin:0.6rem 0 1rem 0;">
                    <span class="ds-pill" style="background:{tier3_color}22;
                        color:{tier3_color};font-size:1.05rem;padding:0.5rem 1.4rem;">
                    ● {tier3} RISK</span></div>""",
                unsafe_allow_html=True,
            )
            if sens_thr is not None:
                st.caption(
                    f"HIGH triggers at risk ≥ {sens_thr:.1%} — a sensitivity-"
                    f"targeted screening threshold (catches ~90% of true "
                    f"positives in calibration data), not an arbitrary 50% cutoff."
                )

            if use_specialist:
                st.markdown(
                    f"""<div class="ds-card accent" style="--accent:{AMBER};">
                        <span class="ds-pill" style="background:{AMBER}22;color:{AMBER};">
                        ⚠ SPECIALIST MODEL — REDUCED CONFIDENCE</span></div>""",
                    unsafe_allow_html=True,
                )
            elif deferred:
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

            if not use_specialist:
                with st.expander("Fairness thresholds (per Fitzpatrick group)"):
                    for gkey, gname in [("12", "FST I-II"), ("34", "FST III-IV"), ("56", "FST V-VI")]:
                        g = bundle.get("cp_by_group_ddi", {}).get(gkey)
                        qval = g.get("q_hat") if g else None
                        marker = " ← this patient" if gname == group_name else ""
                        if qval is not None:
                            st.markdown(f"`{gname}`: q̂ = {qval:.4f}{marker}")
                        else:
                            st.markdown(f"`{gname}`: not available{marker}")
                    st.caption(
                        "Pulled live from the calibrated bundle — stratified "
                        "thresholds differ across groups by design."
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
            st.markdown("**SHAP — this lesion's prediction**")
            try:
                vis_pct, txt_pct, explanation, is_vis = shap_breakdown(active_model, X, vis_dim, nlp_dim)
                waterfall_svg = render_shap_waterfall(explanation)
                st.components.v1.html(waterfall_svg, height=520)
                st.caption(
                    f"Waterfall for this single lesion (from the "
                    f"{'clinical-camera specialist' if use_specialist else 'dermatoscope'} model): "
                    f"starts at the model's average output and shows how each PCA "
                    f"component pushed the prediction up or down. Roughly "
                    f"{vis_pct*100:.0f}% of total attribution came from vision-dominated "
                    f"components vs. {txt_pct*100:.0f}% from text-dominated ones. This "
                    f"does NOT identify specific biological biomarkers."
                )
            except Exception as e:
                st.warning(f"SHAP unavailable this run: {e}")

    with referral_tab:
        if not REFERRAL_NOTE_AVAILABLE:
            st.warning(
                "`referral_note.py` not found next to `app.py` — copy it over to "
                "enable this tab. The rest of the app works fine without it."
            )
        else:
            st.markdown(
                "Structured handoff note combining this capture's risk score, "
                "confidence signals, measured diameter, and patient context — "
                "for a teledermatology referral or printed record, not a diagnosis."
            )

            modality_warn = None
            if p_clinical is not None and p_clinical > 0.5:
                modality_warn = (
                    f"This image scored {p_clinical:.0%} likely to be a "
                    f"clinical-camera photo rather than a genuine dermatoscope "
                    f"capture. " +
                    ("The clinical-camera specialist model's score is shown "
                     "above (broad 'malignant lesion' label, not melanoma-"
                     "specific)." if use_specialist else
                     "No specialist model was available this run — treat the "
                     "dermatoscope model's score with substantially reduced "
                     "confidence.")
                )

            novelty_score = None
            novelty_det = bundle.get("novelty_detector")
            if novelty_det is not None:
                try:
                    X_pca_for_novelty = novelty_det["pca_step"].transform(X)
                    raw_maha = novelty_det["covariance_model"].mahalanobis(X_pca_for_novelty)[0]
                    novelty_score = float(raw_maha / novelty_det["novelty_median"])
                except Exception:
                    pass

            confidence_reasons = []
            if deferred:
                confidence_reasons.append("DRAPS conformal set includes both outcomes (model genuinely uncertain)")
            if use_specialist:
                confidence_reasons.append(f"image likely clinical-camera, routed to specialist model ({p_clinical:.0%})")
            if novelty_score is not None and novelty_score > 2.5:
                confidence_reasons.append(f"image is {novelty_score:.1f}x more unusual than a typical training image")

            if use_specialist or len(confidence_reasons) >= 2:
                confidence_tier = "REFER"
            elif len(confidence_reasons) == 1:
                confidence_tier = "VERIFY"
            else:
                confidence_tier = "TRUST"

            recapture_hint = None
            if use_specialist:
                recapture_hint = ("This looks like a clinical-camera photo. If you have the "
                                   "DermScript dermatoscope attachment, recapture with it for "
                                   "a validated melanoma-specific risk score.")
            elif novelty_score is not None and novelty_score > 2.5:
                recapture_hint = ("This image is unusually different from anything the model "
                                   "was trained on. Check lighting, focus, and that the lesion "
                                   "fills the frame, then recapture.")

            tier_colors = {"TRUST": TEAL, "VERIFY": AMBER, "REFER": CORAL}
            st.markdown(
                f"""<div class="ds-card accent" style="--accent:{tier_colors[confidence_tier]};">
                    <div class="ds-eyebrow">Triage Confidence</div>
                    <span class="ds-pill" style="background:{tier_colors[confidence_tier]}22;
                        color:{tier_colors[confidence_tier]};">{confidence_tier}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            if confidence_reasons:
                st.caption("Contributing factors: " + "; ".join(confidence_reasons) + ".")
            if recapture_hint:
                st.info(f"📷 {recapture_hint}")

            note_obj = generate_referral_note(
                risk_score=risk,
                draps_conformal_set=in_set,
                draps_q_hat=q_hat if not use_specialist else None,
                patient_age=(int(age) if age is not None else None), patient_sex=sex, anatomical_site=site,
                fitzpatrick_skin_type=fitz, clinical_notes=note or None,
                estimated_diameter_mm=diam_mm,
                diameter_measurement_reliable=(diam_mm is not None),
                modality_warning=modality_warn,
                novelty_score=novelty_score,
                confidence_tier=confidence_tier,
                recapture_hint=recapture_hint,
            )
            note_obj.model_uncertain = bool(deferred) if not use_specialist else True

            st.components.v1.html(note_obj.to_html(), height=520, scrolling=True)

            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                "Download as text", note_obj.to_text(),
                file_name="dermscript_referral_note.txt", use_container_width=True,
            )
            dl_col2.download_button(
                "Download as JSON (teledermatology API)", note_obj.to_json(),
                file_name="dermscript_referral_note.json", use_container_width=True,
            )

            st.caption(
                "Note: this is a single-timepoint screening note. Lesion "
                "change-tracking is available via `lesion_tracker.py` but not "
                "yet wired into this app session."
            )

elif source_bytes is not None:
    st.caption("Image ready — click **Run DermScript analysis** above to score it.")

# ──────────────────────────────────────────────────────────────────────────
# Footer — external validation status, corrected to match the current bundle
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="ds-footer">
    THE BLOCKER — public benchmarks for AI melanoma detection are almost
    entirely dermatoscope-only; nothing in this space measures what happens
    when a patient photographs their own skin with a phone instead. This
    device quantifies that gap directly: <b>AUC≈0.87 on the genuinely
    dermatoscopic held-out cohort (PH2) vs. AUC≈0.51-0.55 (near chance) on
    heterogeneous real-world clinical-camera photos (DDI, MRA-MIDAS) —
    same model, same patients, different camera.</b><br><br>
    TRAINING — N=90,953 images, N_pos=7,178 (7.9%), 14 deduplicated cohorts
    (ISIC 2016-2024, HAM10000, BCN20000, PAD-UFES-20, Derm7pt, MILK10K).
    In-distribution OOF AUC=0.882.<br><br>
    EXTERNAL VALIDATION:<br>
    &nbsp;&nbsp;• PH2 (Porto) — the ONLY genuinely dermatoscopic external
    cohort: AUC≈0.87<br>
    &nbsp;&nbsp;• MED-NODE (Groningen) — clinical-camera, but controlled
    close-up photography: AUC≈0.80<br>
    &nbsp;&nbsp;• DDI (Stanford), MRA-MIDAS — clinical-camera, heterogeneous
    real-world photos: AUC≈0.51-0.55, near chance<br>
    The pattern suggests performance on clinical-camera photos tracks
    acquisition CONSISTENCY (controlled vs. heterogeneous), not camera type
    alone — a more nuanced finding than a simple dermatoscope-vs-phone
    story.<br><br>
    CLINICAL-CAMERA SPECIALIST — a separate model trained on PAD-UFES-20,
    DDI, and SCIN routes in automatically for images the modality detector
    flags as clinical-camera. It predicts a broader "malignant lesion"
    outcome (not melanoma-specifically) and should be treated as a rougher
    signal than the dermatoscope model.<br><br>
    A cohort-identity classifier can distinguish source datasets with ~94%
    accuracy, indicating the model partly keys off acquisition artifacts —
    the mechanism behind the gap above, reported honestly as unresolved
    rather than corrected for.<br><br>
    NOT a diagnostic device. Generalization beyond dermatoscope-modality
    images is NOT established. Every output requires confirmation by a
    licensed clinician before any care decision.
    </div>""",
    unsafe_allow_html=True,
)
