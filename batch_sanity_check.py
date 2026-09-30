"""
DermScript — batch_sanity_check.py
===================================
The task PATCH_NOTES flagged as "still not done": run PH2 / MED-NODE (or any
folder of test images) through the ACTUAL APP CODE PATH -- dermscript_features.py
+ dermscript_inference_bundle_v9.pkl, the same two things streamlit_app.py
loads -- rather than trusting the notebook's own numbers. If this script's
AUC/sensitivity disagree with the notebook's, something differs between
"trained" and "deployed" that the parity test didn't catch (feature order,
a stale bundle, a version mismatch).

Run this in the SAME environment as the app (same requirements.txt versions --
that's the whole point). On the Pi or your laptop with model_cache/ already
populated, or in Colab where torch/transformers can download on the fly.

WHERE TO RUN THIS
------------------
Put this file next to streamlit_app.py, dermscript_features.py, and
dermscript_inference_bundle_v9.pkl (the same repo folder). It imports
dermscript_features.py directly, so it exercises the exact same embedding
code the deployed app uses -- not a re-implementation of it.

USAGE
-----
    # PH2 (zip has an .xlsx metadata table -- see PH2_METADATA_HINT below)
    python batch_sanity_check.py --dataset ph2 --zip /path/to/PH2Dataset.zip

    # MED-NODE (labels come from folder names inside the zip: melanoma/ or naevus/)
    python batch_sanity_check.py --dataset mednode --zip /path/to/complete_mednode_dataset.zip

    # Any folder of images with NO known labels (e.g. your 7 test images) --
    # just prints what the app would show for each one, no AUC/sensitivity.
    python batch_sanity_check.py --dataset adhoc --folder /path/to/images/

    # KEY DIAGNOSTIC: does routing MED-NODE to the specialist (what the live
    # app does) actually score worse than just using the main model directly
    # (what Cell 5's reported MED-NODE AUC=0.84 actually measured -- Cell 5
    # calls cal_clf, i.e. the main model, NOT the specialist)? Run all three
    # and compare the AUCs the script prints:
    python batch_sanity_check.py --dataset mednode --zip complete_mednode_dataset.zip --branch auto
    python batch_sanity_check.py --dataset mednode --zip complete_mednode_dataset.zip --branch main
    python batch_sanity_check.py --dataset mednode --zip complete_mednode_dataset.zip --branch specialist

    # Any folder of images WITH known labels via a CSV (columns: filename,label
    # where label is 0=benign/1=malignant)
    python batch_sanity_check.py --dataset labeled_folder --folder /path/to/images/ --labels_csv labels.csv

OUTPUT
------
  - <dataset>_predictions.csv   one row per image: filename, true_label (if
    known), predicted_prob, tier (LOW/MEDIUM/HIGH), conformal_set,
    novelty_score, p_clinical, flagged_reason (if any)
  - A printed summary: N, AUC (if labels known), sensitivity at the bundle's
    real referral threshold, and how many images got flagged.
  - FLAGS (the actual point of this script) -- any image where:
      * a known melanoma (label=1) lands in tier "LOW"           <- must never happen
      * a known benign (label=0) lands in tier "HIGH"            <- expected sometimes, but rate matters
      * novelty_score > NOVELTY_UNRELIABLE (2.5)                 <- input the model can't trust
      * the modality detector's p_clinical disagrees with what you told it the dataset is
        (e.g. PH2/dermatoscope images scoring >50% "clinical-camera")
    are written to <dataset>_flagged.csv for you to eyeball.

WHAT "GOOD" LOOKS LIKE (per PATCH_NOTES' own numbers, for comparison)
-----------------------------------------------------------------------
  PH2:      AUC ~0.86  (genuinely dermatoscopic external cohort)
  MED-NODE: AUC ~0.85  (clinical-camera, but controlled/close-up -- should
            ideally route through ANY specialist-branch logic you compare
            against, since it is NOT dermatoscopic despite the historical
            mislabeling PATCH_NOTES already corrected)
  LOCO AUC: ~0.76 is the "honest" pooled number, not ~0.88 -- don't be
            alarmed if per-cohort numbers here run a bit under the pooled
            OOF figure quoted elsewhere; that gap is expected and already
            disclosed.
If this script's numbers land noticeably BELOW even the LOCO figures, that's
the real signal something is wrong in the deployed path specifically.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# --- import the SAME single-source-of-truth file the app uses -------------
sys.path.insert(0, str(Path(__file__).parent))
import dermscript_features as DF  # noqa: E402

NOVELTY_UNRELIABLE = 2.5  # must match streamlit_app.py's constant

PH2_METADATA_HINT = (
    "PH2Dataset.xlsx must be inside the zip (it usually is, in the official "
    "release). Header row is auto-detected by finding the 'Image Name' cell."
)


# ---------------------------------------------------------------------------
# Bundle loading (identical semantics to streamlit_app.py's load_bundle())
# ---------------------------------------------------------------------------
def load_bundle(bundle_path: Path):
    if not bundle_path.exists():
        zpath = bundle_path.with_suffix(".zip")
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                pkl_entries = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
                target = next((n for n in pkl_entries if Path(n).name == bundle_path.name), pkl_entries[0])
                with zf.open(target) as src, open(bundle_path, "wb") as dst:
                    dst.write(src.read())
        else:
            raise FileNotFoundError(f"No bundle at {bundle_path} or {zpath}")
    with open(bundle_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Dataset loaders -- return list of (image_id, PIL.Image, true_label_or_None)
# ---------------------------------------------------------------------------
def load_ph2(zip_path: Path):
    import openpyxl

    records = {}  # IMD_id (upper, no ext) -> 0/1
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        xlsx_candidates = [n for n in names if n.lower().endswith(".xlsx") and "__macosx" not in n.lower()]
        if not xlsx_candidates:
            raise RuntimeError(f"No .xlsx metadata found in {zip_path}. {PH2_METADATA_HINT}")
        with zf.open(xlsx_candidates[0]) as f:
            wb = openpyxl.load_workbook(io.BytesIO(f.read()), data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = list(ws.iter_rows(values_only=True))
        header_idx = next((i for i, r in enumerate(rows) if r and r[0] == "Image Name"), None)
        if header_idx is None:
            raise RuntimeError("Could not find 'Image Name' header row in PH2 xlsx.")
        header = rows[header_idx]
        col_mel = next((i for i, h in enumerate(header) if h == "Melanoma"), None)
        col_atyp = next((i for i, h in enumerate(header) if h == "Atypical Nevus"), None)
        col_common = next((i for i, h in enumerate(header) if h == "Common Nevus"), None)
        for r in rows[header_idx + 1:]:
            if not r or not r[0]:
                continue
            key = str(r[0]).strip().upper()
            if col_mel is not None and r[col_mel] == "X":
                records[key] = 1
            elif col_atyp is not None and r[col_atyp] == "X":
                records[key] = 0
            elif col_common is not None and r[col_common] == "X":
                records[key] = 0

        print(f"PH2: parsed {len(records)} labeled records from xlsx "
              f"(expected 200: 80 common + 80 atypical + 40 melanoma).")

        img_names = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                     and "__macosx" not in n.lower() and "_dermoscopic_image" not in n.lower()]
        # PH2's real image folders look like .../IMD002/IMD002_Dermoscopic_Image/IMD002.bmp
        # -- prefer files whose immediate parent folder is literally "<ID>_Dermoscopic_Image".
        preferred = [n for n in img_names if Path(n).parent.name.endswith("_Dermoscopic_Image")]
        pool = preferred if preferred else img_names

        out = []
        with zipfile.ZipFile(zip_path) as zf:
            for n in pool:
                stem = Path(n).stem.upper()
                key = next((k for k in records if k in stem or stem in k), None)
                if key is None:
                    continue
                with zf.open(n) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                out.append((Path(n).name, img, records[key]))
        return out


def load_mednode(zip_path: Path):
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        img_names = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))
                     and "__macosx" not in n.lower()]
        for n in img_names:
            low = n.lower()
            if "melanoma" in low:
                label = 1
            elif "naevus" in low or "nevus" in low:
                label = 0
            else:
                continue  # unlabeled path, same as the notebook's behavior
            with zf.open(n) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            out.append((Path(n).name, img, label))
    n_mel = sum(1 for _, _, y in out if y == 1)
    print(f"MED-NODE: parsed {len(out)} labeled images (melanoma={n_mel}, "
          f"naevus={len(out) - n_mel}). Expected 170 total, 70 melanoma.")
    return out


def load_adhoc_folder(folder: Path, labels_csv: Path | None):
    labels = {}
    if labels_csv is not None:
        with open(labels_csv, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = int(row["label"])
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        img = Image.open(p).convert("RGB")
        out.append((p.name, img, labels.get(p.name)))
    return out


# ---------------------------------------------------------------------------
# Core: exactly what streamlit_app.py does per-image, minus the UI
# ---------------------------------------------------------------------------
def run_one(image, bundle, bb, vis_dim, nlp_dim, feature_mode, branch="auto"):
    model_text = DF.FEATURE_CONFIG["placeholder_text"]  # same default the app uses with no patient fields
    X, _ = embed(image, model_text, bb, feature_mode, nlp_dim)

    p_clinical = None
    modality_det = bundle.get("modality_detector")
    if modality_det is not None:
        try:
            X_pca = modality_det["pca_step"].transform(X)
            p_clinical = float(modality_det["model"].predict_proba(X_pca)[:, 1][0])
        except Exception:
            pass

    # branch="auto" reproduces the app's real routing (modality_detector decides).
    # branch="main"/"specialist" FORCE that model regardless of modality_detector,
    # so you can directly compare -- e.g. does the notebook's cal_clf (Cell 5's
    # MED-NODE AUC=0.84, main model, routing bypassed) actually beat the
    # clinical-camera specialist (which was never validated on MED-NODE at all --
    # Cell 4.10's per-source breakdown only covers DDI/PAD-UFES-20/SCIN) on the
    # SAME images the live app would route to the specialist?
    use_specialist = (branch == "specialist") or (
        branch == "auto" and p_clinical is not None and p_clinical > 0.5
    )

    specialist_bundle = bundle.get("clinical_camera_specialist")
    if use_specialist and specialist_bundle is not None:
        risk = float(specialist_bundle["model"].predict_proba(X)[:, 1][0])
        model_used = "specialist"
    else:
        risk = float(bundle["model"].predict_proba(X)[:, 1][0])
        model_used = "main"

    novelty_score = None
    det = bundle.get("novelty_detector")
    if det is not None:
        try:
            Xp = det["pca_step"].transform(X)
            raw = det["covariance_model"].mahalanobis(Xp)[0]
            novelty_score = float(raw / det["novelty_median"])
        except Exception:
            pass

    input_unreliable = (novelty_score is not None and novelty_score > NOVELTY_UNRELIABLE)

    # v2 -- FIX: tier logic now branches on which model actually produced
    # `risk`, matching the app's real (v10.3-fixed) logic. Previously this
    # always compared `risk` against the MAIN model's threshold (~6.6%)
    # even for specialist-scored rows, where a compressed score like 1-2%
    # would almost NEVER cross that bar -- silently making every specialist
    # row look falsely reassuring in past batch runs. v3 update (matches app
    # v10.5): HIGH stays fully capped out for the specialist branch (no
    # single high cut on a near-chance model -- DDI AUC 0.53 -- is
    # trustworthy). LOW is restored using the model's own most permissive
    # threshold (sensitivity_threshold_90, tuned for 90% sensitivity): a
    # score below it sits in the ~10% risk band the model would exclude
    # even at its most sensitive setting, and only counts if the input
    # isn't already flagged unreliable (high novelty).
    conformal_set_str = None
    if model_used == "specialist":
        sens_thr = (specialist_bundle or {}).get("sensitivity_threshold_90")
        confident_low = (sens_thr is not None and risk < sens_thr and not input_unreliable)
        tier = "LOW" if confident_low else "MEDIUM"
    else:
        sens_thr = None
        cp_cc = bundle.get("cp_classcond")
        if cp_cc is not None:
            in_set = DF.conformal_set(risk, cp_cc)
        else:
            cp_overall = bundle.get("cp_overall", {})
            q_hat = cp_overall.get("q_hat", 0.8)
            in_set = []
            if (1 - risk) >= 1 - q_hat:
                in_set.append("benign")
            if risk >= 1 - q_hat:
                in_set.append("malignant")
        if not in_set:
            in_set = ["benign", "malignant"]
        conformal_set_str = "+".join(in_set)

        sens_thr = bundle.get("sensitivity_threshold_90")
        if input_unreliable:
            tier = "MEDIUM"
        elif in_set == ["malignant"]:
            tier = "HIGH"
        elif in_set == ["benign"]:
            tier = "LOW"
        elif sens_thr is not None and risk >= sens_thr:
            tier = "HIGH"
        else:
            tier = "MEDIUM"

    return {
        "predicted_prob": risk,
        "tier": tier,
        "model_used": model_used,
        "p_clinical": p_clinical,
        "conformal_set": conformal_set_str,
        "novelty_score": novelty_score,
        "sensitivity_threshold_90": sens_thr,
    }


def embed(image, text, bb, feature_mode, nlp_dim):
    """Copied verbatim from streamlit_app.py -- keep in sync if that file changes."""
    import torch

    x = bb.img_tf(image).unsqueeze(0).to(bb.device)
    with torch.no_grad():
        v = bb.mnet(x).float().cpu().numpy()
    if feature_mode == "vision_only" or bb.bert is None:
        n = np.zeros((1, nlp_dim), dtype=np.float32)
    else:
        n = DF.embed_texts([text], bb.tok, bb.bert, bb.device)
    return np.hstack([v, n]), x


def load_backbones(need_text, cache_dir: Path):
    import torch
    from types import SimpleNamespace

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mnet = DF.load_vision_model(device, weights="V2",
                                 checkpoint_dir=cache_dir / "torch" / "hub" / "checkpoints")
    tok = bert = None
    if need_text:
        tok, bert = DF.load_text_model(device, local_files_only=False)
    return SimpleNamespace(device=device, mnet=mnet, tok=tok, bert=bert,
                            img_tf=DF.make_image_transform())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     choices=["ph2", "mednode", "adhoc", "labeled_folder"])
    ap.add_argument("--zip", type=Path, help="PH2 or MED-NODE zip path")
    ap.add_argument("--folder", type=Path, help="folder of images (adhoc / labeled_folder)")
    ap.add_argument("--labels_csv", type=Path, default=None,
                     help="for labeled_folder: CSV with columns filename,label")
    ap.add_argument("--bundle", type=Path,
                     default=Path("dermscript_inference_bundle_v9.pkl"))
    ap.add_argument("--branch", choices=["auto", "main", "specialist"], default="auto",
                     help="auto=app's real routing (modality_detector decides). "
                          "main/specialist=FORCE that model on every image, to compare "
                          "e.g. whether the specialist genuinely underperforms the main "
                          "model on a cohort like MED-NODE that the specialist was never "
                          "validated on (see script docstring).")
    ap.add_argument("--out_prefix", default=None)
    args = ap.parse_args()

    out_prefix = (args.out_prefix or args.dataset) + (
        f"_{args.branch}" if args.branch != "auto" else "")

    print(f"Loading bundle: {args.bundle}")
    bundle = load_bundle(args.bundle)
    vis_dim = bundle.get("vis_dim", 960)
    nlp_dim = bundle.get("nlp_dim", 768)
    feature_mode = bundle.get("feature_mode", "fusion")

    print("Loading backbones (this downloads weights on first run if not cached)...")
    bb = load_backbones(feature_mode != "vision_only", Path(__file__).parent / "model_cache")

    if args.dataset == "ph2":
        items = load_ph2(args.zip)
    elif args.dataset == "mednode":
        items = load_mednode(args.zip)
    elif args.dataset == "adhoc":
        items = load_adhoc_folder(args.folder, None)
    else:
        items = load_adhoc_folder(args.folder, args.labels_csv)

    print(f"\nRunning {len(items)} images through the app's exact code path...")
    rows = []
    flagged = []
    for i, (name, img, true_label) in enumerate(items):
        r = run_one(img, bundle, bb, vis_dim, nlp_dim, feature_mode, branch=args.branch)
        r["filename"] = name
        r["true_label"] = true_label
        rows.append(r)

        reasons = []
        if true_label == 1 and r["tier"] == "LOW":
            reasons.append("KNOWN MELANOMA SCORED LOW -- must never happen, investigate immediately")
        if r["novelty_score"] is not None and r["novelty_score"] > NOVELTY_UNRELIABLE:
            reasons.append(f"high novelty ({r['novelty_score']:.2f}x)")
        if args.dataset == "ph2" and r["p_clinical"] is not None and r["p_clinical"] > 0.5:
            reasons.append(f"PH2 is dermatoscopic but modality detector said "
                            f"{r['p_clinical']:.0%} clinical-camera")
        if reasons:
            r["flagged_reason"] = "; ".join(reasons)
            flagged.append(r)
        else:
            r["flagged_reason"] = ""

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)}...")

    # -- write predictions CSV --
    pred_path = f"{out_prefix}_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "true_label", "predicted_prob", "tier",
                                           "model_used", "conformal_set", "novelty_score",
                                           "p_clinical", "flagged_reason"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"\nWrote {pred_path}")

    if flagged:
        flag_path = f"{out_prefix}_flagged.csv"
        with open(flag_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "true_label", "predicted_prob", "tier",
                                               "flagged_reason"])
            w.writeheader()
            for r in flagged:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        print(f"Wrote {flag_path} ({len(flagged)} flagged images)")

    # -- summary metrics, if labels are known --
    labeled = [r for r in rows if r["true_label"] is not None]
    if labeled:
        y = np.array([r["true_label"] for r in labeled])
        p = np.array([r["predicted_prob"] for r in labeled])
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, p)
        except Exception as e:
            auc = None
            print(f"(Could not compute AUC: {e})")
        sens_thr = rows[0].get("sensitivity_threshold_90")
        sens = None
        if sens_thr is not None:
            mel = p[y == 1]
            sens = float((mel >= sens_thr).mean()) if len(mel) else None
        n_mel_scored_low = sum(1 for r in labeled if r["true_label"] == 1 and r["tier"] == "LOW")

        print("\n" + "=" * 60)
        print(f"SUMMARY — {args.dataset}")
        print("=" * 60)
        print(f"N = {len(labeled)}   N_positive = {int(y.sum())}")
        if auc is not None:
            print(f"AUC = {auc:.3f}")
        if sens is not None:
            print(f"Sensitivity at bundle's referral threshold "
                  f"({sens_thr:.1%}) = {sens:.1%}")
        print(f"Known melanomas that scored LOW tier: {n_mel_scored_low}  "
              f"{'  <-- INVESTIGATE, should be 0' if n_mel_scored_low else '(good)'}")
        print(f"Total flagged images: {len(flagged)} / {len(rows)}")
    else:
        print(f"\nNo ground-truth labels for this batch ({len(rows)} images) -- "
              f"see {pred_path} for per-image scores and tiers.")


if __name__ == "__main__":
    main()
