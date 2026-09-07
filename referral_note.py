"""
DermScript — Structured Teledermatology Referral Note Generator
=================================================================

Closes the loop from "detect" to "gets a patient to actual care." Every
competing project in this space stops at a risk score on a screen. This
takes everything DermScript already computes for one patient encounter —
model risk score, DRAPS conformal interval, estimated lesion diameter
(from the ruler-bump homography), skin-tone context, and an OPTIONAL
lesion-tracking comparison if a prior capture exists — and assembles it
into one structured note a receiving dermatologist or teledermatology
platform can actually act on, instead of a bare probability.

This module has NO hardware dependencies (no rpi_ws281x, no camera) —
it's pure text/data assembly, so it runs identically on the Pi or your
laptop. Call generate_referral_note(...) right after your model produces
a prediction (same moment you'd call led_feedback.show_result()).

NOT a diagnostic document. This note reports what the DEVICE measured and
computed — it explicitly does NOT include a diagnosis, and says so in its
own text, because a screening tool asserting a diagnosis is exactly the
kind of overclaim this project has deliberately avoided everywhere else.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

try:
    from lesion_tracker import LesionComparisonResult
except ImportError:
    LesionComparisonResult = None  # referral notes still work without a prior capture


# ---------------------------------------------------------------------
# RISK TIER LABELING — mirrors led_feedback.py's thresholds so the LED
# color a clinician saw during capture matches the tier written in the
# note they receive. If you retune RISK_HIGH_THRESHOLD/RISK_LOW_THRESHOLD
# in led_feedback.py, update these two constants to match — kept as
# separate constants (not imported) so this module has zero hardware
# dependency and still runs on a machine without rpi_ws281x installed.
# ---------------------------------------------------------------------
RISK_HIGH_THRESHOLD = 0.6
RISK_LOW_THRESHOLD = 0.3
UNCERTAINTY_THRESHOLD = 0.35


def _risk_tier(risk_score: float) -> str:
    if risk_score >= RISK_HIGH_THRESHOLD:
        return "HIGH"
    elif risk_score <= RISK_LOW_THRESHOLD:
        return "LOW"
    return "MODERATE"


@dataclass
class ReferralNote:
    """Everything a receiving clinician needs, assembled in one place.
    to_text()/to_json()/to_html() below render this for different handoff
    channels (printed slip, teledermatology API payload, app display)."""

    # --- identifiers / context (fill in from your app's session state) ---
    patient_age: Optional[int]
    patient_sex: Optional[str]
    anatomical_site: Optional[str]
    fitzpatrick_skin_type: Optional[str]
    clinical_notes: Optional[str]
    capture_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- model output ---
    risk_score: float = 0.0
    risk_tier: str = "UNKNOWN"
    draps_interval_width: Optional[float] = None
    draps_conformal_set: Optional[list] = None  # e.g. ["benign", "malignant"] if deferred
    draps_q_hat: Optional[float] = None
    model_uncertain: bool = False

    # --- physical measurement (from ruler-bump homography) ---
    estimated_diameter_mm: Optional[float] = None
    diameter_measurement_reliable: bool = True

    # --- modality safety check (from Cell 4.7's modality detector, if
    # wired into the app — None if that feature isn't in use yet) ---
    modality_warning: Optional[str] = None  # e.g. "Image may not be a genuine
                                              # dermatoscope capture; model
                                              # validated performance does not
                                              # cover this modality."

    # --- out-of-distribution novelty score (Cell 4.9's detector) and the
    # unified Triage Confidence Score combining DRAPS + modality + novelty
    # into one actionable tier, plus an optional guided-recapture hint ---
    novelty_score: Optional[float] = None  # 1.0 = typical training image
    confidence_tier: Optional[str] = None  # "TRUST" / "VERIFY" / "REFER"
    recapture_hint: Optional[str] = None

    # --- optional: change over time, if a prior capture of this same
    # lesion exists (see lesion_tracker.compare_lesion_captures) ---
    change_tracking_available: bool = False
    change_tracking_reliable: bool = False
    change_tracking_reason: Optional[str] = None
    area_change_pct: Optional[float] = None
    days_since_prior_capture: Optional[int] = None

    device_id: str = "DermScript-v1"
    software_version: str = "v8.7"

    def recommended_action(self) -> str:
        """Plain-language line for the top of the note — this is the one
        sentence a busy clinician reads first. Leads with the unified
        Triage Confidence Score if available (it already accounts for
        DRAPS uncertainty, modality mismatch, and novelty together), then
        falls back to the risk-tier-only logic if that score isn't set."""
        if self.confidence_tier == "REFER":
            return ("REFER FOR IN-PERSON DERMATOLOGIST EVALUATION — multiple "
                     "independent uncertainty signals triggered (see Triage "
                     "Confidence above). Do not rely on the risk score alone "
                     "for this capture.")
        if self.confidence_tier == "VERIFY" and self.recapture_hint:
            return (f"VERIFY BEFORE TRUSTING THIS RESULT — {self.recapture_hint} "
                     f"If recapture isn't possible, treat this risk score with "
                     f"reduced confidence.")
        if self.model_uncertain:
            return ("REFER FOR IN-PERSON DERMATOLOGIST EVALUATION — model "
                     "confidence insufficient for either outcome at the "
                     "95% coverage level (see DRAPS interval below).")
        if self.risk_tier == "HIGH":
            urgent = ""
            if self.area_change_pct is not None and self.area_change_pct > 20:
                urgent = " Lesion also shows measurable growth since prior capture — expedite."
            return f"REFER FOR PROMPT DERMATOLOGIST EVALUATION.{urgent}"
        if self.risk_tier == "MODERATE":
            return "CONSIDER DERMATOLOGIST EVALUATION — routine timeframe reasonable absent other concerning signs."
        return "LOW MODEL-ESTIMATED RISK — routine monitoring; re-screen if lesion changes (size, color, symptoms)."

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 62)
        lines.append("DERMSCRIPT — SCREENING REFERRAL NOTE")
        lines.append("=" * 62)
        lines.append("NOT A DIAGNOSIS. Screening tool output only — for use")
        lines.append("alongside, not in place of, clinical judgment.")
        lines.append("")
        lines.append(f"Captured: {self.capture_timestamp}")
        lines.append(f"Device: {self.device_id}  Software: {self.software_version}")
        lines.append("")
        lines.append("PATIENT CONTEXT")
        lines.append("-" * 62)
        lines.append(f"  Age: {self.patient_age or 'not provided'}")
        lines.append(f"  Sex: {self.patient_sex or 'not provided'}")
        lines.append(f"  Anatomical site: {self.anatomical_site or 'not provided'}")
        lines.append(f"  Fitzpatrick skin type: {self.fitzpatrick_skin_type or 'not provided'}")
        if self.clinical_notes:
            lines.append(f"  Clinical observation notes: {self.clinical_notes}")
        lines.append("")
        lines.append("MODEL OUTPUT")
        lines.append("-" * 62)
        lines.append(f"  Risk score: {self.risk_score:.1%}  (tier: {self.risk_tier})")
        if self.draps_conformal_set is not None:
            lines.append(f"  DRAPS conformal set: {{{', '.join(self.draps_conformal_set)}}}"
                         f"  (q_hat={self.draps_q_hat:.3f})" if self.draps_q_hat is not None else "")
        if self.draps_interval_width is not None:
            lines.append(f"  DRAPS interval width: {self.draps_interval_width:.3f}"
                         f"  {'(UNCERTAIN — exceeds threshold)' if self.model_uncertain else '(within confident range)'}")
        if self.modality_warning:
            lines.append(f"  [!] MODALITY WARNING: {self.modality_warning}")
        lines.append("")
        lines.append("PHYSICAL MEASUREMENT")
        lines.append("-" * 62)
        if self.estimated_diameter_mm is not None:
            rel = "" if self.diameter_measurement_reliable else "  [LOW CONFIDENCE — ruler bumps not reliably detected]"
            lines.append(f"  Estimated lesion diameter: {self.estimated_diameter_mm:.1f} mm{rel}")
        else:
            lines.append("  Estimated lesion diameter: not available this capture")
        lines.append("")
        lines.append("CHANGE OVER TIME")
        lines.append("-" * 62)
        if not self.change_tracking_available:
            lines.append("  No prior capture of this lesion on file — single-timepoint")
            lines.append("  screening only. The 'E' (evolution) criterion in ABCDE")
            lines.append("  cannot be assessed without a follow-up capture.")
        elif not self.change_tracking_reliable:
            lines.append(f"  Prior capture exists but comparison was NOT performed:")
            lines.append(f"  {self.change_tracking_reason}")
        else:
            direction = "grew" if (self.area_change_pct or 0) > 0 else "shrank"
            lines.append(f"  Estimated area change since prior capture "
                         f"({self.days_since_prior_capture} days ago): "
                         f"{abs(self.area_change_pct):.1f}% {direction}")
        lines.append("")
        lines.append("=" * 62)
        lines.append("RECOMMENDED ACTION")
        lines.append("=" * 62)
        lines.append(f"  {self.recommended_action()}")
        lines.append("")
        lines.append("This note reports device measurements and model output only.")
        lines.append("Final diagnostic and treatment decisions rest with the")
        lines.append("evaluating licensed clinician.")
        return "\n".join(l for l in lines if l != "")

    def to_json(self) -> str:
        """Structured payload suitable for a teledermatology API handoff."""
        d = asdict(self)
        d["recommended_action"] = self.recommended_action()
        return json.dumps(d, indent=2)

    def to_html(self) -> str:
        """Lightweight HTML for embedding in the Streamlit app or printing."""
        tier_color = {"HIGH": "#ff6b81", "MODERATE": "#e8a33d", "LOW": "#3fd6a8"}.get(self.risk_tier, "#7c828e")
        rows = [
            ("Age", self.patient_age), ("Sex", self.patient_sex),
            ("Anatomical site", self.anatomical_site),
            ("Fitzpatrick skin type", self.fitzpatrick_skin_type),
        ]
        patient_rows = "".join(
            f"<tr><td style='color:#7c828e;padding:2px 12px 2px 0;'>{k}</td>"
            f"<td>{v if v is not None else 'not provided'}</td></tr>" for k, v in rows
        )
        change_html = "No prior capture on file — single-timepoint screening only."
        if self.change_tracking_available and self.change_tracking_reliable:
            direction = "grew" if (self.area_change_pct or 0) > 0 else "shrank"
            change_html = (f"Estimated area change ({self.days_since_prior_capture} days ago): "
                          f"<b>{abs(self.area_change_pct):.1f}% {direction}</b>")
        elif self.change_tracking_available:
            change_html = f"Prior capture exists but comparison skipped: {self.change_tracking_reason}"

        modality_html = (f"<div style='color:#e8a33d;margin-top:6px;'>⚠ {self.modality_warning}</div>"
                         if self.modality_warning else "")

        return f"""
        <div style="font-family:'Inter',sans-serif;background:#13151a;border:1px solid #23262e;
                    border-radius:10px;padding:1.2rem 1.4rem;color:#f2f3f5;max-width:640px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;letter-spacing:0.1em;
                      color:#7c828e;text-transform:uppercase;">DermScript Screening Referral Note</div>
          <div style="font-size:0.78rem;color:#7c828e;margin-bottom:0.8rem;">NOT A DIAGNOSIS — captured {self.capture_timestamp}</div>

          <table style="font-size:0.85rem;margin-bottom:0.8rem;">{patient_rows}</table>

          <div style="background:#0a0b0e;border-left:3px solid {tier_color};border-radius:6px;
                      padding:0.7rem 1rem;margin-bottom:0.8rem;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:{tier_color};">
              {self.risk_score:.1%} <span style="font-size:0.9rem;">({self.risk_tier})</span>
            </div>
            {"<div style='color:#e8a33d;'>DRAPS: model uncertain at 95% coverage — both outcomes plausible</div>" if self.model_uncertain else ""}
            {modality_html}
          </div>

          <div style="font-size:0.85rem;margin-bottom:0.5rem;">
            <b>Diameter:</b> {f"{self.estimated_diameter_mm:.1f} mm" if self.estimated_diameter_mm else "not available"}
          </div>
          <div style="font-size:0.85rem;margin-bottom:0.8rem;"><b>Change over time:</b> {change_html}</div>

          <div style="background:{tier_color}22;border-radius:6px;padding:0.6rem 0.9rem;
                      font-weight:600;color:{tier_color};">
            {self.recommended_action()}
          </div>
        </div>
        """


def generate_referral_note(
    risk_score: float,
    draps_interval_width: Optional[float] = None,
    draps_conformal_set: Optional[list] = None,
    draps_q_hat: Optional[float] = None,
    patient_age: Optional[int] = None,
    patient_sex: Optional[str] = None,
    anatomical_site: Optional[str] = None,
    fitzpatrick_skin_type: Optional[str] = None,
    clinical_notes: Optional[str] = None,
    estimated_diameter_mm: Optional[float] = None,
    diameter_measurement_reliable: bool = True,
    modality_warning: Optional[str] = None,
    novelty_score: Optional[float] = None,
    confidence_tier: Optional[str] = None,
    recapture_hint: Optional[str] = None,
    lesion_comparison: Optional["LesionComparisonResult"] = None,
    days_since_prior_capture: Optional[int] = None,
) -> ReferralNote:
    """Single entry point — call this right after your model produces a
    prediction, same moment you'd call led_feedback.show_result(). Pass a
    LesionComparisonResult from lesion_tracker.compare_lesion_captures()
    directly if a prior capture of this lesion exists; omit it otherwise."""

    note = ReferralNote(
        patient_age=patient_age, patient_sex=patient_sex,
        anatomical_site=anatomical_site, fitzpatrick_skin_type=fitzpatrick_skin_type,
        clinical_notes=clinical_notes,
        risk_score=risk_score, risk_tier=_risk_tier(risk_score),
        draps_interval_width=draps_interval_width,
        novelty_score=novelty_score, confidence_tier=confidence_tier,
        recapture_hint=recapture_hint,
        draps_conformal_set=draps_conformal_set, draps_q_hat=draps_q_hat,
        model_uncertain=(draps_interval_width is not None and draps_interval_width >= UNCERTAINTY_THRESHOLD),
        estimated_diameter_mm=estimated_diameter_mm,
        diameter_measurement_reliable=diameter_measurement_reliable,
        modality_warning=modality_warning,
    )

    if lesion_comparison is not None:
        note.change_tracking_available = True
        note.change_tracking_reliable = lesion_comparison.success
        note.change_tracking_reason = lesion_comparison.reason
        note.area_change_pct = lesion_comparison.estimated_area_change_pct
        note.days_since_prior_capture = days_since_prior_capture

    return note


if __name__ == "__main__":
    # Self-contained demo — no camera, no model, no Pi required. Run this
    # directly to see all three output formats before wiring it into app.py.
    demo_note = generate_referral_note(
        risk_score=0.71,
        draps_interval_width=0.22,
        draps_conformal_set=["malignant"],
        draps_q_hat=0.761,
        patient_age=54, patient_sex="Female", anatomical_site="Upper extremity",
        fitzpatrick_skin_type="III",
        clinical_notes="Irregular border, recent size change reported.",
        estimated_diameter_mm=7.8, diameter_measurement_reliable=True,
        modality_warning=None,
        days_since_prior_capture=41,
    )
    print(demo_note.to_text())
    print("\n\n--- JSON payload (teledermatology API handoff) ---\n")
    print(demo_note.to_json())
