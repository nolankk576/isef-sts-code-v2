"""DermScript v11 -- single source of truth shared by the training notebook (Part F) and the Streamlit app.

What lives here: image preprocessing (must match the notebook byte-for-byte), the classifier pipeline class (`Pipe`, so a
pickled bundle can be loaded anywhere), the embedder wrappers (ONNX or TorchScript), and `Engine`, which turns one image
into: score -> tier, modality check, novelty (out-of-distribution) flag, percentile context, and a faithful occlusion map.

Design rules (from the validation results):
  * LOW is only ever shown for a validated input: dermoscopy-like AND not out-of-distribution.
  * Clinical-camera photos are NOT validated (same-lesion AUC drops ~0.2) -> never LOW, always flagged.
  * The explanation is an occlusion map on the real risk score (not Grad-CAM on an embedding norm).
"""
import io
import os
import time

import numpy as np
from PIL import Image

__version__ = "11.0"
SEED = 42
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 1, 3)


# ----------------------------------------------------------------------------- image preprocessing
def load_image(src):
    """bytes | path | file-like | PIL.Image -> RGB PIL.Image (same JPEG draft decode the notebook used)."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as fh:
            data = fh.read()
    elif hasattr(src, "read"):
        data = src.read()
    else:
        data = bytes(src)
    im = Image.open(io.BytesIO(data))
    im.draft("RGB", (640, 640))
    return im.convert("RGB")


def to_224(im):
    """shorter side -> 224 (bilinear), centre crop 224x224, uint8 HxWx3"""
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def normalize(batch_uint8):
    """N,H,W,3 uint8 -> N,3,H,W float32 (ImageNet mean/std)"""
    x = (batch_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# ----------------------------------------------------------------------------- classifier pipeline
def make_clf():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                                  random_state=SEED, n_jobs=-1, verbose=-1)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, random_state=SEED)


def balanced_weights(cohort, y, power=0.5):
    """each cohort -> 50% positive / 50% negative; cohort total weight ~ N_c ** power"""
    cohort = np.asarray(cohort)
    y = np.asarray(y)
    w = np.zeros(len(y))
    for c in np.unique(cohort):
        m = cohort == c
        p = m & (y == 1)
        n = m & (y == 0)
        if p.sum() == 0 or n.sum() == 0:
            w[m] = 1.0 / max(m.sum(), 1)
            continue
        tot = m.sum() ** power
        w[p] = 0.5 * tot / p.sum()
        w[n] = 0.5 * tot / n.sum()
    return w * len(w) / w.sum()


class Pipe:
    """standardize -> PCA -> (optional: delete cohort-identifying directions) -> gradient-boosted classifier"""

    def __init__(self, pca_dim=128, proj_k=0, balanced=True):
        self.pca_dim, self.proj_k, self.balanced = pca_dim, proj_k, balanced

    def fit(self, X, y, cohort):
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(X, dtype=np.float32)
        self.sc = StandardScaler().fit(X)
        self.pca = PCA(n_components=min(self.pca_dim, X.shape[1], X.shape[0] - 1), random_state=SEED).fit(self.sc.transform(X))
        Z = self.pca.transform(self.sc.transform(X))
        self.Q = None
        if self.proj_k > 0 and len(np.unique(cohort)) > 2:
            k = min(self.proj_k, len(np.unique(cohort)) - 1)
            W = LinearDiscriminantAnalysis(solver="svd").fit(Z, cohort).scalings_[:, :k]
            self.Q, _ = np.linalg.qr(W)
            Z = Z - Z @ self.Q @ self.Q.T
        w = balanced_weights(cohort, y) if self.balanced else None
        self.clf = make_clf()
        self.clf.fit(Z, y, sample_weight=w)
        return self

    def latent(self, X):
        Z = self.pca.transform(self.sc.transform(np.asarray(X, dtype=np.float32)))
        if self.Q is not None:
            Z = Z - Z @ self.Q @ self.Q.T
        return Z

    def score(self, X):
        return self.clf.predict_proba(self.latent(X))[:, 1]


# ----------------------------------------------------------------------------- embedders
class OnnxEmbedder:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, batch_uint8, chunk=16):
        out = []
        for i in range(0, len(batch_uint8), chunk):
            out.append(self.sess.run(None, {self.inp: normalize(batch_uint8[i:i + chunk])})[0])
        return np.concatenate(out).astype(np.float32)


class TorchScriptEmbedder:
    def __init__(self, path):
        import torch
        self.torch = torch
        self.m = torch.jit.load(path, map_location="cpu").eval()

    def __call__(self, batch_uint8, chunk=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(batch_uint8), chunk):
                x = self.torch.from_numpy(normalize(batch_uint8[i:i + chunk]))
                out.append(self.m(x).numpy())
        return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------- decision logic (pure python, unit-testable)
def decide(score, thr, validated_input, ood):
    """returns (tier, confidence, reasons)"""
    reasons = []
    if score >= thr["t_high"]:
        tier = "HIGH"
        reasons.append("score is above the high-risk cut-off")
    elif score < thr["t_low"]:
        if validated_input and not ood:
            tier = "LOW"
            reasons.append("score is below the 95%-sensitivity cut-off on a validated dermoscopy input")
        else:
            tier = "MEDIUM"
            reasons.append("score is low, but LOW is only given for validated dermoscopy inputs")
    else:
        tier = "MEDIUM"
        reasons.append("score is between the two cut-offs")
    if not validated_input:
        confidence = "REFER"
    elif ood:
        confidence = "VERIFY"
    elif tier == "MEDIUM":
        confidence = "VERIFY"
    else:
        confidence = "TRUST"
    return tier, confidence, reasons


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, bundle, embedder):
        self.b = bundle
        self.embed = embedder
        self.pipe = bundle["pipe"]

    @classmethod
    def load(cls, bundle_path, onnx_path=None, torchscript_path=None):
        import joblib
        bundle = joblib.load(bundle_path)
        if onnx_path and os.path.exists(onnx_path):
            emb = OnnxEmbedder(onnx_path)
        elif torchscript_path and os.path.exists(torchscript_path):
            emb = TorchScriptEmbedder(torchscript_path)
        else:
            raise FileNotFoundError("need the exported backbone (.onnx or TorchScript .pt) next to the bundle")
        return cls(bundle, emb)

    # --- scoring
    def _features(self, batch_uint8):
        return self.embed(batch_uint8)

    def score_batch(self, batch_uint8):
        f = self._features(batch_uint8)
        return self.pipe.score(f), f

    def _novelty(self, feats):
        nv = self.b["novelty"]
        Z = self.pipe.latent(feats)[:, : nv["dim"]]
        d = np.sqrt(np.einsum("ij,jk,ik->i", Z - nv["mu"], nv["P"], Z - nv["mu"]))
        return d / nv["median"]

    def analyze(self, src):
        t0 = time.time()
        arr = to_224(load_image(src))[None]
        s, f = self.score_batch(arr)
        s = float(s[0])
        p_clin = float(self.b["modality"].predict_proba(f)[0, 1])
        clinical = p_clin >= 0.5
        nov = float(self._novelty(f)[0])
        ood = nov > self.b["novelty"]["thr"]
        validated = (not clinical)
        tier, conf, reasons = decide(s, self.b["thr"], validated, ood)
        ref_b, ref_m = self.b["ref_benign"], self.b["ref_mel"]
        out = dict(
            version=__version__, score=s, tier=tier, confidence=conf, reasons=reasons,
            benign_percentile=float(100.0 * np.searchsorted(ref_b, s) / len(ref_b)),
            melanoma_percentile=float(100.0 * np.searchsorted(ref_m, s) / len(ref_m)),
            modality="clinical-camera photo" if clinical else "dermoscopy", p_clinical=p_clin,
            novelty=nov, novelty_limit=float(self.b["novelty"]["thr"]), out_of_distribution=bool(ood),
            validated_input=bool(validated and not ood), thresholds=self.b["thr"], seconds=None,
        )
        notes = []
        if clinical:
            notes.append("Clinical-camera photo: this model is validated on dermoscopy. On the same lesions, accuracy drops sharply for "
                         "clinical photos, so LOW is never shown. Re-capture with the dermatoscope.")
        if ood:
            notes.append("Unusual image compared with the training data (possible ruler, ink, glare, blur or non-skin content).")
        out["notes"] = notes
        out["seconds"] = round(time.time() - t0, 2)
        return out

    # --- faithful explanation: occlusion of the REAL risk score
    def occlusion(self, src, grid=6):
        arr = to_224(load_image(src))
        base_s, _ = self.score_batch(arr[None])
        base = float(base_s[0])
        mean_col = arr.reshape(-1, 3).mean(0).astype(np.uint8)
        step = 224 // grid
        batch, cells = [], []
        for i in range(grid):
            for j in range(grid):
                a = arr.copy()
                a[i * step:(i + 1) * step if i < grid - 1 else 224, j * step:(j + 1) * step if j < grid - 1 else 224] = mean_col
                batch.append(a)
                cells.append((i, j))
        s, _ = self.score_batch(np.stack(batch))
        heat = np.zeros((grid, grid), np.float32)
        for (i, j), v in zip(cells, s):
            heat[i, j] = base - float(v)          # >0: hiding this region LOWERS the risk, so the region pushes risk up
        return dict(base_score=base, heat=heat, image224=arr)

    def border_check(self, src, frac=0.15):
        """does the verdict depend on the image border (rulers, ink, vignette, gel)?"""
        arr = to_224(load_image(src))
        m = int(224 * frac)
        a = arr.copy()
        col = arr[m:-m, m:-m].reshape(-1, 3).mean(0).astype(np.uint8)
        a[:m] = col
        a[-m:] = col
        a[:, :m] = col
        a[:, -m:] = col
        s, _ = self.score_batch(np.stack([arr, a]))
        t0, _, _ = decide(float(s[0]), self.b["thr"], True, False)
        t1, _, _ = decide(float(s[1]), self.b["thr"], True, False)
        return dict(score=float(s[0]), score_border_hidden=float(s[1]), tier=t0, tier_border_hidden=t1, depends_on_border=(t0 != t1))


def heat_overlay(arr224, heat, alpha=0.45):
    """red overlay of regions that push the risk UP (positive heat), returned as uint8 HxWx3"""
    h = np.clip(heat, 0, None)
    h = h / (h.max() + 1e-9)
    up = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    red = np.zeros_like(arr224, dtype=np.float32)
    red[..., 0] = 255
    out = arr224.astype(np.float32) * (1 - alpha * up[..., None]) + red * (alpha * up[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)
