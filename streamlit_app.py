"""
DermScript v11 -- Clinical Triage Support (Streamlit app)

Research / educational prototype. NOT a diagnostic device. Every output must be
confirmed by a licensed clinician before any care decision.

v11 replaces the v9/v10 engine. What changed and why:
  * ONE small model: DINOv2-S embedding (ONNX) -> PCA -> gradient-boosted classifier, shipped as
    dermscript_v11.py + dermscript_v11_bundle.joblib + dinov2_s_224.onnx. No torch, transformers, BERT or SHAP.
  * The decision rules were fixed on leave-one-cohort-out scores BEFORE any external cohort was scored.
  * LOW is only shown for a validated input: dermoscopy-like AND not out-of-distribution AND the verdict
    does not depend on the image border. Clinical photos are not validated (same-lesion AUC drops by ~0.16-0.20).
  * The explanation is an occlusion map of the REAL risk score (not Grad-CAM on an embedding norm).
  * Patient metadata (age, sex, site, skin type) does not change the model score; it only goes into the referral note.
  * Removed: clinical-camera specialist, DRAPS / conformal sets, per-Fitzpatrick thresholds (none were validated).
"""
import hashlib
import io
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

import dermscript_logic as L
import dermscript_v11 as dv

try:
    from referral_note import generate_referral_note
    REFERRAL_NOTE_AVAILABLE = True
except ImportError:
    REFERRAL_NOTE_AVAILABLE = False  # app still runs fully without this feature

APP_DIR = Path(__file__).parent
BUNDLE_PATH = APP_DIR / "dermscript_v11_bundle.joblib"
ONNX_PATH = APP_DIR / "dinov2_s_224.onnx"
TORCHSCRIPT_PATH = APP_DIR / "dinov2_s_224_torchscript.pt"   # only used if the ONNX file is absent (needs torch)

# "offline" only if you really run it air-gapped (set DERMSCRIPT_DEPLOY_MODE=offline).
DEPLOY_MODE = os.environ.get("DERMSCRIPT_DEPLOY_MODE", "cloud_demo")
EXPECTED_LIBS = {"scikit-learn": "1.6.1", "lightgbm": "4.6.0"}   # must equal the versions printed by notebook Cell F6

RING_BUMP_SPACING_MM = 10.0  # <-- replace with your ring's measured spacing
MIN_BUMPS_FOR_SCALE = 3

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

def render_html(html, height=None, scrolling=False):
    """st.components.v1.html is deprecated; fall back to st.html if it disappears."""
    try:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=scrolling)
    except Exception:
        st.html(html)


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

    # v9.2: a real ruler's bumps are COLLINEAR and EVENLY spaced along that line.
    # Raw all-pairwise distances are the wrong statistic to test that with --
    # 4 evenly-spaced collinear points naturally have unequal pairwise distances
    # (e.g. spacings of 10/20/30mm between different pairs), so a variance check
    # on all-pairwise distances rejects good rulers too. Instead: fit the
    # principal axis, check the points sit close to that line (collinear), then
    # check CONSECUTIVE gaps along it are consistent (evenly spaced).
    pts_c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_c)
    major, minor = vt[0], vt[1]
    along = pts_c @ major
    perp = pts_c @ minor
    collinear_ok = (np.abs(perp).max() < 0.4 * (along.max() - along.min() + 1e-6))
    order = np.argsort(along)
    gaps = np.diff(along[order])
    evenly_spaced_ok = (len(gaps) == 0) or (gaps.std() / (gaps.mean() + 1e-6) < 0.35)
    if not (collinear_ok and evenly_spaced_ok):
        cv2.putText(debug, "Ruler bumps not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
    # v9.2: sanity backstop. Melanomas presenting for screening are essentially
    # never this large or this small; treat an out-of-range result as a failed
    # calibration rather than display it with false confidence.
    if not (1.0 <= diameter_mm <= 30.0):
        cv2.putText(debug, "Diameter estimate out of plausible range", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return None, debug
    return diameter_mm, debug


def render_risk_gauge(risk, color, height=240, label="MALIGNANCY RISK",
                       tier="MEDIUM", confidence_pct=None, confidence_label="Score",
                       show_confidence=True):
    import math

    r = 80
    cx, cy = 100, 100

    # v10.2 -- FIX (round 2). v10.1's fixed-fraction fill (LOW=16%, MEDIUM=
    # 55%, HIGH=92% of the arc) still looked like it was measuring a
    # magnitude, so a HIGH result whose raw score was 40% still looked
    # "wrong" next to a bar that was 92% full. Replaced with a plain
    # 3-segment traffic light: LOW / MEDIUM / HIGH are always the same
    # three equal-sized arcs in the same three fixed colors; the ONLY thing
    # that changes is which one is lit at full opacity/thickness. It no
    # longer implies a continuous quantity at all, so it can't visually
    # disagree with a percentage shown next to it. The raw model score (or,
    # for the specialist branch, the percentile rank -- a more honest number
    # given that model's compressed output range) is shown as a small,
    # clearly separate line below, never overlapping the tier visually.
    segs = [
        ("LOW", TEAL, math.pi, math.pi * 2 / 3),
        ("MEDIUM", AMBER, math.pi * 2 / 3, math.pi * 1 / 3),
        ("HIGH", CORAL, math.pi * 1 / 3, 0.0),
    ]

    def pt(angle):
        return (cx + r * math.cos(angle), cy - r * math.sin(angle))

    arcs_svg = []
    for name, seg_color, a0, a1 in segs:
        x0, y0 = pt(a0)
        x1, y1 = pt(a1)
        active = (name == tier)
        arcs_svg.append(
            f'<path d="M {x0:.3f} {y0:.3f} A {r} {r} 0 0 1 {x1:.3f} {y1:.3f}" '
            f'stroke="{seg_color}" stroke-width="{14 if active else 9}" fill="none" '
            f'stroke-linecap="round" opacity="{1.0 if active else 0.28}"/>'
        )
    arcs_html = "\n".join(arcs_svg)

    if confidence_pct is None:
        confidence_pct = risk * 100
    conf_svg = ""
    if show_confidence:
        conf_svg = (
            f'<text x="{cx}" y="{cy + 32}" font-family="JetBrains Mono, monospace" '
            f'font-size="11" text-anchor="middle" fill="#a8adb8">'
            f'{confidence_label}: {confidence_pct:.1f}%</text>'
        )

    html = f"""
    <div style="display:flex;justify-content:center;">
      <svg viewBox="0 0 200 140" width="260" height="{height}">
        {arcs_html}
        <circle cx="{cx}" cy="{cy}" r="48" fill="#0D1117" stroke="#1E2533" stroke-width="1"/>
        <text x="{cx}" y="{cy - 6}" font-family="JetBrains Mono, monospace" font-size="26"
              font-weight="700" text-anchor="middle" fill="{color}">{tier}</text>
        <text x="{cx}" y="{cy + 14}" font-family="Inter, sans-serif" font-size="9"
              text-anchor="middle" fill="#7c828e">{label}</text>
        {conf_svg}
      </svg>
    </div>
    """
    render_html(html, height=height)


def installed_libs():
    out = {}
    for mod, key in (("sklearn", "scikit-learn"), ("lightgbm", "lightgbm")):
        try:
            out[key] = __import__(mod).__version__
        except Exception:
            out[key] = None
    return out


@st.cache_resource(show_spinner="Loading DermScript model…")
def load_engine(bundle_mtime, model_mtime):
    """The two mtime args are only cache keys: a new bundle or model file on disk always invalidates the cache."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng = dv.Engine.load(str(BUNDLE_PATH), onnx_path=str(ONNX_PATH), torchscript_path=str(TORCHSCRIPT_PATH))
    msgs = sorted({str(w.message).split("\n")[0] for w in caught if "version" in str(w.message).lower()})
    return eng, msgs


def compute(engine, raw_bytes, force_clin, want_map):
    """One full analysis. Returns everything the result tabs need (kept in session_state)."""
    res = engine.analyze(raw_bytes)
    if force_clin:
        res = L.force_clinical(res)
    border = engine.border_check(raw_bytes)
    res = L.apply_border_check(res, border)
    occ = engine.occlusion(raw_bytes, grid=6) if want_map else None
    return dict(res=res, occ=occ)


# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="ds-eyebrow">{"LOCAL / OFFLINE" if DEPLOY_MODE == "offline" else "CLOUD DEMO"} · RESEARCH PROTOTYPE · NOT CLINICALLY VALIDATED</div>',
    unsafe_allow_html=True,
)
st.title("🔬 DermScript")
st.markdown('<div class="ds-tickrule"></div>', unsafe_allow_html=True)
st.caption(
    "Research prototype that scores dermatoscope images for melanoma-like appearance. It was tested on "
    "outside dermoscopy cohorts, but it has NOT been clinically validated. It is not validated for clinical-camera "
    "or phone photos, so those never receive a LOW result. **Not a diagnostic device.** A low score is never a "
    "rule-out. Every output requires confirmation by a licensed clinician."
)

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — patient context (fully optional) + device link
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ds-eyebrow">Patient metadata (optional)</div>', unsafe_allow_html=True)
    st.caption(
        "These fields do NOT change the model score. They are only copied into the referral note. "
        "Anything left as \"Unknown\" is simply omitted."
    )
    have_age = st.checkbox("Patient age known", value=False)
    age = st.number_input("Age", min_value=0, max_value=120, value=45) if have_age else None

    sex_choice = st.selectbox("Sex", ["Unknown", "Female", "Male", "Other / unspecified"])
    sex = None if sex_choice == "Unknown" else sex_choice

    site_choice = st.selectbox(
        "Anatomical site",
        ["Unknown", "Scalp", "Face", "Neck", "Trunk", "Upper extremity",
         "Lower extremity", "Palms/Soles", "Other"],
    )
    site = None if site_choice == "Unknown" else site_choice

    fitz_choice = st.select_slider(
        "Fitzpatrick skin type", options=["Unknown", "I", "II", "III", "IV", "V", "VI"], value="Unknown",
        help="Recorded in the referral note only. The model does not use it.",
    )
    fitz = None if fitz_choice == "Unknown" else fitz_choice

    source_choice = st.selectbox(
        "Image source",
        ["Auto-detect", "This is a clinical / phone photo"],
        help="The detector flags clinical photos automatically. Choosing the second option can only make the result "
             "more cautious (LOW is withheld); it can never unlock LOW.",
    )
    force_clin = source_choice.startswith("This is")
    want_map = st.checkbox("Compute explanation map (slower, about 37 extra passes)", value=True)

    with st.expander("Clinical observation (optional)"):
        note = st.text_area(
            "Notes", placeholder="e.g. Irregular border, recent change in size, mild itching.",
            height=100, label_visibility="collapsed",
        )
    st.divider()
    st.markdown('<div class="ds-eyebrow">DermScript device (optional)</div>', unsafe_allow_html=True)
    st.caption(
        "Only needed if you're capturing from the physical dermatoscope over Wi-Fi. "
        "Manual upload below always works with no device at all."
    )
    device_ip = st.text_input(
        "Device IP address", value=st.session_state.get("device_ip", ""), placeholder="e.g. 192.168.1.50",
        help="IP of the Pi running pi_capture_server.py on your local network (find it on the Pi with `hostname -I`).",
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
    capture_clicked = cap_col.button("📷 Capture", type="primary", use_container_width=True, disabled=not device_ready)
    if not device_ready:
        st.caption("Enter a device IP above to enable capture, or just upload a file below.")
    else:
        st.caption("Manual upload below always works as a fallback too.")

# ──────────────────────────────────────────────────────────────────────────
# System status row
# ──────────────────────────────────────────────────────────────────────────
_missing = [p.name for p in (BUNDLE_PATH,) if not p.exists()]
if not (ONNX_PATH.exists() or TORCHSCRIPT_PATH.exists()):
    _missing.append(f"{ONNX_PATH.name} (or {TORCHSCRIPT_PATH.name})")
if _missing:
    st.error("Model files not found next to streamlit_app.py: " + ", ".join(_missing) +
             ". Copy them from the notebook's DermScript/v9/final/ folder into the repo.")
    st.stop()

_model_path = ONNX_PATH if ONNX_PATH.exists() else TORCHSCRIPT_PATH
try:
    engine, load_warnings = load_engine(BUNDLE_PATH.stat().st_mtime_ns, _model_path.stat().st_mtime_ns)
    engine_ok, engine_error = True, None
except Exception as e:
    engine_ok, engine_error, load_warnings = False, str(e), []

if not engine_ok:
    st.error(
        "Could not load the DermScript model. Most often this is a library-version mismatch (the bundle was saved with "
        "different scikit-learn / lightgbm versions than this app installed) or a missing `dermscript_v11.py`. "
        f"Details: {engine_error}"
    )
    st.stop()

meta = engine.b.get("meta", {}) or {}
libs = installed_libs()
lib_mismatch = {k: (v, EXPECTED_LIBS[k]) for k, v in libs.items() if v != EXPECTED_LIBS[k]}
thr = engine.b["thr"]

status_html = '<div class="ds-status-row">'
status_html += '<div class="ds-status ok">● MODEL LOADED</div>'
status_html += f'<div class="ds-status ok">● BACKBONE: DINOv2-S ({"ONNX" if _model_path == ONNX_PATH else "TorchScript"})</div>'
status_html += f'<div class="ds-status">● VERSION {meta.get("version", dv.__version__)}</div>'
status_html += f'<div class="ds-status {"ok" if DEPLOY_MODE == "offline" else "warn"}">● MODE: {"OFFLINE" if DEPLOY_MODE == "offline" else "CLOUD DEMO"}</div>'
status_html += '</div>'
st.markdown(status_html, unsafe_allow_html=True)
if DEPLOY_MODE != "offline":
    st.warning("Cloud demo: uploaded images are sent to a remote server. Do not upload real patient images.")
for _w in load_warnings:
    st.warning(f"Model bundle was built with a different library version: {_w} Predictions may be invalid.")
if lib_mismatch:
    st.warning("Installed library versions differ from the ones the model was built with (installed, expected): "
               f"{lib_mismatch}. Pin the versions in requirements.txt.")

# ──────────────────────────────────────────────────────────────────────────
# Image intake
# ──────────────────────────────────────────────────────────────────────────
st.divider()
intake_col, preview_col = st.columns([2, 1], gap="large")

with intake_col:
    st.markdown("**Upload lesion image from dermatoscope**")
    img_file = st.file_uploader("Upload lesion image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if capture_clicked:
        try:
            resp = requests.get(f"http://{device_ip}:5000/capture", timeout=10)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                st.error(f"Device responded but didn't return an image (Content-Type: '{content_type or 'missing'}'). "
                         f"Check pi_capture_server.py's /capture route.")
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

    run = st.button("Run DermScript analysis", type="primary", use_container_width=True, disabled=source_bytes is None)

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
# Analysis (computed on click, kept in session_state so other widgets don't wipe it)
# ──────────────────────────────────────────────────────────────────────────
cur_key = None
if source_bytes is not None:
    cur_key = f"{hashlib.sha1(source_bytes).hexdigest()}|{int(force_clin)}|{int(want_map)}"

if run and source_bytes is not None:
    try:
        with st.spinner("Running inference…" + (" and building the explanation map…" if want_map else "")):
            out = compute(engine, source_bytes, force_clin, want_map)
        out["key"] = cur_key
        st.session_state["analysis"] = out
    except Exception as e:
        st.session_state.pop("analysis", None)
        st.error(f"Analysis failed: {e}. If this is not an image file, upload a JPG or PNG.")

analysis = st.session_state.get("analysis")
if analysis is not None and cur_key is not None and analysis["key"] == cur_key:
    res, occ = analysis["res"], analysis["occ"]
    pil_img = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    tier = res["tier"]
    tier_color = {"HIGH": CORAL, "MEDIUM": AMBER, "LOW": TEAL}[tier]
    conf_color = {"TRUST": TEAL, "VERIFY": AMBER, "REFER": CORAL}[res["confidence"]]
    clinical = res["modality"].startswith("clinical")

    with st.expander("🔧 Debug — model & inference sanity checks"):
        st.markdown(
            f"Version `{res['version']}` · backbone `{meta.get('backbone', 'dinov2_s')}` · built `{meta.get('date', 'n/a')}` · "
            f"inference `{res['seconds']} s` · raw score `{res['score']:.4f}`"
        )
        st.markdown(f"Cut-offs: LOW below `{thr['t_low']:.4f}` · HIGH at or above `{thr['t_high']:.4f}`")
        st.markdown(f"Modality: `{res['modality']}` (P(clinical) = `{res['p_clinical']:.3f}`, cut at 0.5)")
        st.markdown(f"Novelty: `{res['novelty']:.2f}`× the training median (flag above `{res['novelty_limit']:.2f}`) · "
                    f"out-of-distribution: `{res['out_of_distribution']}`")
        b = res.get("border", {})
        if b:
            st.markdown(f"Border check: score `{b['score']:.4f}` → `{b['score_border_hidden']:.4f}` with the outer 15% hidden "
                        f"(tier {b['tier']} → {b['tier_border_hidden']})")
        st.markdown(f"Installed libs: `{libs}` · expected: `{EXPECTED_LIBS}`")
        for r_ in res["reasons"]:
            st.markdown(f"- {r_}")

    for n_ in res["notes"]:
        st.warning(n_)

    st.divider()
    result_tab, explain_tab, referral_tab = st.tabs(["📊  Risk & Sizing", "🧭  Explainability", "📋  Referral Note"])

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
                    "⚠ Ruler bumps not reliably detected in this frame — diameter estimate unavailable. "
                    "Recalibrate `RING_BUMP_SPACING_MM` and the Hough parameters against your physical printed ring."
                )

        with col2:
            st.markdown('<div class="ds-eyebrow">RISK SCORE</div>', unsafe_allow_html=True)
            render_risk_gauge(res["score"], tier_color, label="RISK TIER", tier=tier, show_confidence=False)
            st.markdown(
                f"""<div style="text-align:center;margin:0.6rem 0 0.2rem 0;">
                    <span class="ds-pill" style="background:{tier_color}22;color:{tier_color};
                        font-size:1.05rem;padding:0.5rem 1.4rem;">● {tier} RISK</span></div>
                    <div style="text-align:center;color:{MUTED};font-size:0.85rem;margin-bottom:0.6rem;">
                    model score {res['score']:.2f} (a ranking score, not a probability)</div>""",
                unsafe_allow_html=True,
            )
            st.caption(
                f"This image scores higher than {res['benign_percentile']:.0f}% of benign and "
                f"{res['melanoma_percentile']:.0f}% of melanoma lesions in the cross-validated reference set. "
                f"LOW is below {thr['t_low']:.2f} (95% of melanomas scored above this in cross-validation); "
                f"HIGH is {thr['t_high']:.2f} or above (90% of benign lesions scored below this). "
                "A LOW result still misses about 1 in 20 melanomas in cross-validation and is NOT a rule-out."
            )
            st.markdown(
                f"""<div class="ds-card accent" style="--accent:{conf_color};">
                    <div class="ds-eyebrow">Triage Confidence</div>
                    <span class="ds-pill" style="background:{conf_color}22;color:{conf_color};">{res['confidence']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.caption("Why: " + "; ".join(res["reasons"]) + ".")
            if res.get("border_downgraded"):
                st.info("LOW was withdrawn because the verdict changes when the image border is hidden.")
            if clinical:
                st.error("This is not a validated input: clinical and phone photos lose about 0.16–0.20 AUC on the same lesions. "
                         "Treat any score as REFER and recapture with the dermatoscope.")
            if tier == "LOW":
                st.success("LOW was given because the image looks like dermoscopy, is typical of the training data, "
                           "and its verdict does not depend on the border.")

    with explain_tab:
        if occ is None:
            st.info("Explanation map was switched off in the sidebar. Turn it on and run the analysis again.")
        else:
            xcol1, xcol2 = st.columns(2)
            with xcol1:
                st.markdown("**Occlusion map — what pushes this score up**")
                st.image(dv.heat_overlay(occ["image224"], occ["heat"]), use_container_width=True)
                st.caption(
                    "Each cell of a 6×6 grid was hidden in turn and the real risk score was recomputed. Red regions are "
                    "where hiding the area LOWERS the score, so those areas push the risk up. This explains the score "
                    "itself, but it shows association, not a biological cause, and it is not a diagnosis."
                )
            with xcol2:
                st.markdown("**Does the verdict depend on the border?**")
                b = res["border"]
                st.metric("Score with the outer 15% hidden", f"{b['score_border_hidden']:.2f}",
                          delta=f"{b['score_border_hidden'] - b['score']:+.2f}", delta_color="off")
                if b["depends_on_border"]:
                    st.warning(L.BORDER_NOTE)
                else:
                    st.success("The tier is the same with the border hidden, so a ruler, ink or vignette at the edge "
                               "is not what decides this result.")
                strongest = float(np.max(occ["heat"]))
                st.caption(f"Largest single-region effect on the score: {strongest:+.3f}. "
                           "If the red areas sit on the image edge, a ruler, ink or glare may be driving the result.")

    with referral_tab:
        if not REFERRAL_NOTE_AVAILABLE:
            st.warning("`referral_note.py` not found next to `streamlit_app.py` — copy it over to enable this tab.")
        else:
            st.markdown(
                "Structured handoff note combining this capture's risk tier, confidence signals, measured diameter, "
                "and patient context — for a teledermatology referral or printed record, not a diagnosis."
            )
            note_obj = generate_referral_note(
                patient_age=(int(age) if age is not None else None), patient_sex=sex, anatomical_site=site,
                fitzpatrick_skin_type=fitz, clinical_notes=note or None,
                estimated_diameter_mm=diam_mm, diameter_measurement_reliable=(diam_mm is not None),
                **L.referral_kwargs(res),
            )
            note_obj.model_uncertain = False
            render_html(note_obj.to_html(), height=520, scrolling=True)
            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button("Download as text", note_obj.to_text(),
                                    file_name="dermscript_referral_note.txt", use_container_width=True)
            dl_col2.download_button("Download as JSON (teledermatology API)", note_obj.to_json(),
                                    file_name="dermscript_referral_note.json", use_container_width=True)
            st.caption("Note: this is a single-timepoint screening note. Lesion change-tracking is available via "
                       "`lesion_tracker.py` but not yet wired into this app session.")

elif source_bytes is not None:
    st.caption("Image ready — click **Run DermScript analysis** above to score it.")

# ──────────────────────────────────────────────────────────────────────────
# Footer — frozen validation numbers (notebook v9.1, cells F2-F4)
# ──────────────────────────────────────────────────────────────────────────
_loco = meta.get("mean_loco_auc")
_loco_txt = f"{_loco:.3f}" if isinstance(_loco, (int, float)) else "0.776"
st.markdown(
    f"""<div class="ds-footer">
    WHAT THIS IS — a research prototype that scores a dermoscopy image for melanoma-like appearance. It has not been
    clinically validated and must not guide care.<br><br>
    TRAINING — six dermoscopy collections (HAM10000, ISIC 2016, 2017, 2019, 2020, MILK10K dermoscopy), exact duplicates
    removed, 15,301 images with 3,781 melanomas. Leave-one-collection-out mean AUC: {_loco_txt}. The cut-offs were fixed on
    those held-out scores before any outside cohort was scored.<br><br>
    OUTSIDE COHORTS (AUC, frozen cut-offs):<br>
    &nbsp;&nbsp;• PH2, dermoscopy, N=200 (40 melanoma): 0.903<br>
    &nbsp;&nbsp;• DERM12345, dermoscopy, N=12,287 (400 melanoma; benign labels are not biopsy-confirmed): 0.947<br>
    &nbsp;&nbsp;• MIDAS dermatoscope views, N=913 (79 melanoma): 0.700<br>
    &nbsp;&nbsp;• MED-NODE, clinical photos, N=170: 0.827 &nbsp;|&nbsp; DDI, clinical photos, N=656 (any malignancy): 0.632<br>
    &nbsp;&nbsp;• MIDAS clinical views, N=1,787 (155 melanoma): 0.497, which is chance level.<br>
    On the same lesions, clinical photos score 0.16–0.20 AUC lower than dermoscopy (MIDAS and MILK10K). Because the
    modality detector misses roughly 5–12% of clinical photos, about 0.3–4% of clinical photos could still reach LOW in
    testing.<br><br>
    NOT a diagnostic device. A low score is not a rule-out. Every output requires confirmation by a licensed clinician
    before any care decision.
    </div>""",
    unsafe_allow_html=True,
)
