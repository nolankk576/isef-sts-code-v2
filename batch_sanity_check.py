"""
DermScript v11 -- batch sanity check (no Streamlit needed).

Runs the SAME engine the app uses over a folder of images and writes one CSV row per image. Use it before the
Oct 15 freeze to confirm the deployed files reproduce the notebook, and after any change to the repo.

    python batch_sanity_check.py --images my_images/ [--labels labels.csv] [--out results.csv] [--model-dir .]

labels.csv (optional): columns  filename,label   (label 1 = melanoma / malignant, 0 = benign)
With labels it also prints AUC, the share of positives that fell in LOW, and the share of benign cleared as LOW.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import dermscript_logic as L
import dermscript_v11 as dv

EXTS = {".jpg", ".jpeg", ".png"}


def load_engine(model_dir):
    model_dir = Path(model_dir)
    return dv.Engine.load(str(model_dir / "dermscript_v11_bundle.joblib"),
                          onnx_path=str(model_dir / "dinov2_s_224.onnx"),
                          torchscript_path=str(model_dir / "dinov2_s_224_torchscript.pt"))


def analyse_folder(engine, folder, force_clinical=False):
    rows = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            res = engine.analyze(p)
            if force_clinical:
                res = L.force_clinical(res)
            res = L.apply_border_check(res, engine.border_check(p))
        except Exception as e:
            rows.append(dict(filename=p.name, error=str(e)))
            continue
        rows.append(dict(filename=p.name, score=round(res["score"], 4), tier=res["tier"], confidence=res["confidence"],
                         modality=res["modality"], p_clinical=round(res["p_clinical"], 3), novelty=round(res["novelty"], 2),
                         out_of_distribution=res["out_of_distribution"], border_downgraded=res["border_downgraded"],
                         seconds=res["seconds"]))
    return rows


def read_labels(path):
    with open(path, newline="") as fh:
        return {r["filename"]: int(r["label"]) for r in csv.DictReader(fh)}


def summarise(rows, labels=None):
    ok = [r for r in rows if "error" not in r]
    print(f"\n{len(ok)} images scored, {len(rows) - len(ok)} failed")
    for t in ("LOW", "MEDIUM", "HIGH"):
        print(f"  {t:7s} {sum(r['tier'] == t for r in ok):5d}")
    print(f"  called clinical photo: {sum(r['modality'].startswith('clinical') for r in ok)}   "
          f"unusual (out-of-distribution): {sum(bool(r['out_of_distribution']) for r in ok)}   "
          f"LOW withdrawn by border check: {sum(bool(r['border_downgraded']) for r in ok)}")
    if not labels:
        return
    lab = [(r, labels[r["filename"]]) for r in ok if r["filename"] in labels]
    if not lab:
        print("no filenames in the CSV matched the images")
        return
    y = np.array([l for _, l in lab])
    s = np.array([r["score"] for r, _ in lab])
    tiers = np.array([r["tier"] for r, _ in lab])
    print(f"\nWith labels: N={len(y)}  positives={int(y.sum())}")
    if 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score
        print(f"  AUC {roc_auc_score(y, s):.3f}")
        print(f"  positives shown LOW (missed): {int(((tiers == 'LOW') & (y == 1)).sum())}/{int(y.sum())}")
        print(f"  benign cleared as LOW: {((tiers == 'LOW') & (y == 0)).sum() / max((y == 0).sum(), 1):.0%}")
        print(f"  positives reaching HIGH: {((tiers == 'HIGH') & (y == 1)).sum() / y.sum():.0%}")
    else:
        print("  need both classes for AUC")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dermscript_batch_results.csv"))
    ap.add_argument("--model-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--clinical", action="store_true", help="treat every image as a clinical/phone photo (LOW withheld)")
    a = ap.parse_args(argv)
    t0 = time.time()
    engine = load_engine(a.model_dir)
    rows = analyse_folder(engine, a.images, a.clinical)
    cols = ["filename", "score", "tier", "confidence", "modality", "p_clinical", "novelty",
            "out_of_distribution", "border_downgraded", "seconds", "error"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summarise(rows, read_labels(a.labels) if a.labels else None)
    print(f"\nwrote {a.out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
