"""
DermScript — Lesion Change-Tracking (Longitudinal Comparison)
================================================================

Compares two captures of the SAME lesion taken at different times, and
produces:
  1. An aligned overlay showing how the lesion has changed
  2. A rough quantified size-change estimate
  3. A simple visual diff highlighting new/changed regions

This is the "E" (Evolution) in the ABCDE rule that single-snapshot
classifiers (including most low-cost dermatoscope projects) don't
address at all. Because your rig captures at a FIXED distance and
fixed cross-polarization, registration between two captures is far
easier and more reliable here than with phone photos.

Dependencies:
    pip3 install opencv-python numpy

--------------------------------------------------------------------
HOW THIS FITS INTO YOUR PIPELINE
--------------------------------------------------------------------
1. When a user captures a lesion, they give it a short label/ID
   (e.g., "left-shoulder-mole-1"). Store captures under that ID.
2. When they capture the SAME lesion again later, call:

    from lesion_tracker import compare_lesion_captures
    result = compare_lesion_captures("path/to/old.jpg", "path/to/new.jpg")

3. `result` gives you an aligned overlay image, an estimated area
   change percentage, and a confidence flag on whether registration
   succeeded (it can fail on badly blurred or extremely low-contrast
   images — you should surface that to the user rather than silently
   showing a bad comparison).

This is intentionally a SKELETON / v1, not a polished module — the
alignment and lesion-segmentation steps are the two places you'll
want to iterate most once you're testing on real serial captures.
--------------------------------------------------------------------
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LesionComparisonResult:
    success: bool
    reason: Optional[str]                # populated if success is False
    aligned_overlay_path: Optional[str]   # side-by-side / blended overlay
    diff_heatmap_path: Optional[str]      # visual diff highlighting change
    estimated_area_change_pct: Optional[float]  # + = grew, - = shrank
    num_matched_features: Optional[int]   # sanity-check metric — very low
                                           # counts mean registration is
                                           # unreliable, flag it to the user


# ---------------------------------------------------------------------
# STEP 1: IMAGE ALIGNMENT (registration)
# ---------------------------------------------------------------------
# Because your device has a FIXED capture geometry (same working
# distance, same cross-polarization, same lighting rig every time),
# two captures of the same lesion should already be roughly aligned.
# ORB feature matching + homography just cleans up small differences
# from the user re-placing the device slightly differently each time.

def _align_images(img_old, img_new, min_matches=10):
    """
    Aligns img_new onto img_old's coordinate frame using ORB
    features + homography (RANSAC).

    Returns (aligned_new_img, num_good_matches) or (None, num_matches)
    if alignment isn't reliable.
    """
    orb = cv2.ORB_create(nfeatures=2000)

    gray_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray_old, None)
    kp2, des2 = orb.detectAndCompute(gray_new, None)

    if des1 is None or des2 is None:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    good_matches = matches[:min(80, len(matches))]

    if len(good_matches) < min_matches:
        return None, len(good_matches)

    src_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, len(good_matches)

    h, w = img_old.shape[:2]
    aligned_new = cv2.warpPerspective(img_new, H, (w, h))

    return aligned_new, len(good_matches)


# ---------------------------------------------------------------------
# STEP 2: ROUGH LESION SEGMENTATION
# ---------------------------------------------------------------------
# A placeholder segmentation approach: cross-polarized dermatoscope
# images typically have decent contrast between the lesion and
# surrounding skin, so simple thresholding + contour detection gets
# you a usable v1. This is the step most worth upgrading later (e.g.
# swapping in a proper segmentation model) if you have time.

def _segment_lesion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Otsu thresholding picks a threshold automatically rather than
    # you hand-tuning one brightness cutoff.
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    return largest, area


# ---------------------------------------------------------------------
# STEP 3: PUBLIC ENTRY POINT
# ---------------------------------------------------------------------

def compare_lesion_captures(old_image_path: str, new_image_path: str,
                             output_dir: str = "lesion_comparisons"
                             ) -> LesionComparisonResult:
    if not os.path.exists(old_image_path) or not os.path.exists(new_image_path):
        return LesionComparisonResult(
            success=False, reason="One or both image paths do not exist.",
            aligned_overlay_path=None, diff_heatmap_path=None,
            estimated_area_change_pct=None, num_matched_features=None,
        )

    img_old = cv2.imread(old_image_path)
    img_new = cv2.imread(new_image_path)

    aligned_new, num_matches = _align_images(img_old, img_new)

    if aligned_new is None:
        return LesionComparisonResult(
            success=False,
            reason=(
                f"Registration failed — only {num_matches} reliable "
                "feature matches found. This can happen with very "
                "blurry captures, extreme lighting differences, or if "
                "the two images aren't actually of the same lesion. "
                "Ask the user to recapture rather than showing a "
                "comparison you can't trust."
            ),
            aligned_overlay_path=None, diff_heatmap_path=None,
            estimated_area_change_pct=None, num_matched_features=num_matches,
        )

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(new_image_path))[0]

    # --- overlay: old vs. aligned-new, side by side ---
    overlay = np.hstack([img_old, aligned_new])
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    # --- segment both, compare area ---
    contour_old, area_old = _segment_lesion(img_old)
    contour_new, area_new = _segment_lesion(aligned_new)

    area_change_pct = None
    if contour_old is not None and contour_new is not None and area_old > 0:
        area_change_pct = ((area_new - area_old) / area_old) * 100.0

    # --- diff heatmap ---
    gray_old = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(aligned_new, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_old, gray_new)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(output_dir, f"{base_name}_diff.png")
    cv2.imwrite(heatmap_path, heatmap)

    return LesionComparisonResult(
        success=True,
        reason=None,
        aligned_overlay_path=overlay_path,
        diff_heatmap_path=heatmap_path,
        estimated_area_change_pct=area_change_pct,
        num_matched_features=num_matches,
    )


# ---------------------------------------------------------------------
# DEMO / MANUAL TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 lesion_tracker.py <old_image.jpg> <new_image.jpg>")
        sys.exit(1)

    result = compare_lesion_captures(sys.argv[1], sys.argv[2])

    if not result.success:
        print(f"Comparison FAILED: {result.reason}")
    else:
        print("Comparison succeeded.")
        print(f"  Matched features: {result.num_matched_features}")
        print(f"  Overlay saved to: {result.aligned_overlay_path}")
        print(f"  Diff heatmap saved to: {result.diff_heatmap_path}")
        if result.estimated_area_change_pct is not None:
            sign = "+" if result.estimated_area_change_pct >= 0 else ""
            print(f"  Estimated area change: {sign}"
                  f"{result.estimated_area_change_pct:.1f}%")
        else:
            print("  Area change could not be estimated "
                  "(segmentation didn't find a clear lesion boundary "
                  "in one or both images).")

"""
--------------------------------------------------------------------
IMPORTANT CAVEATS — be upfront about these in your writeup
--------------------------------------------------------------------
1. The Otsu-threshold segmentation here is a v1 placeholder. It works
   reasonably on high-contrast lesions against clear skin, but will
   struggle on low-contrast or irregularly-bordered lesions. If you
   have time, a lightweight trained segmentation model (or even a
   simple U-Net on ISIC segmentation masks, which exist for a subset
   of the archive) would meaningfully improve reliability. Report
   this as a known limitation rather than presenting area-change
   numbers as precise measurements.

2. "Area change %" here is in PIXELS, not real-world units (mm²). To
   report actual physical size change, you'd need a fixed known
   reference in-frame (e.g., a scale marking in your 3D-printed
   housing) to convert pixels to millimeters. Worth considering for
   the hardware design if this becomes a core feature — but you can
   ship v1 without it and just report relative % change.

3. num_matched_features is your reliability signal — always show it
   or gate on it in the UI (e.g., "not enough visual similarity to
   confidently compare these captures") rather than silently
   presenting a shaky comparison as fact. This matters more here than
   almost anywhere else in the pipeline, because a wrong "this mole
   grew 40%" is a genuinely alarming and potentially harmful false
   signal to give someone.
--------------------------------------------------------------------
"""
