"""Pure-python decision helpers for streamlit_app.py.
Kept apart from the UI so they can be unit-tested without Streamlit.
Every function here can only make a result MORE cautious, never less.
"""
import dermscript_v11 as dv

CLINICAL_NOTE = (
    "Clinical-camera photo: this model is validated on dermoscopy. On the same lesions, accuracy drops sharply for "
    "clinical photos, so LOW is withheld. Re-capture with the dermatoscope."
)
BORDER_NOTE = (
    "The verdict changes when the image border is hidden, so a ruler, ink, vignette or gel edge may be influencing the score."
)
TIER_TO_NOTE = {"HIGH": "HIGH", "MEDIUM": "MODERATE", "LOW": "BELOW THRESHOLD"}


def force_clinical(res):
    """The user says this is a clinical / phone photo. Withholds LOW; never lifts a restriction."""
    if res["modality"].startswith("clinical"):
        return res
    tier, conf, reasons = dv.decide(res["score"], res["thresholds"], False, res["out_of_distribution"])
    return dict(res, tier=tier, confidence=conf, reasons=reasons, validated_input=False,
                modality="clinical-camera photo (set by you)", notes=[CLINICAL_NOTE] + list(res["notes"]))


def apply_border_check(res, border):
    """If a LOW verdict flips when the border is hidden, LOW is withdrawn (MEDIUM / VERIFY)."""
    out = dict(res, border=border, border_downgraded=False)
    if border["depends_on_border"]:
        out["notes"] = list(res["notes"]) + [BORDER_NOTE]
        if res["tier"] == "LOW" and border["tier_border_hidden"] != "LOW":
            out.update(tier="MEDIUM", confidence="VERIFY", border_downgraded=True,
                       reasons=list(res["reasons"]) + ["LOW withdrawn: the verdict changes when the image border is hidden"])
    return out


def recapture_hint(res):
    if res["modality"].startswith("clinical"):
        return ("This looks like a clinical-camera photo. Recapture with the DermScript dermatoscope "
                "for a validated risk score.")
    if res["out_of_distribution"]:
        return ("This image is unusual compared with the training data. Check lighting, focus and framing "
                "(no ruler, ink or glare over the lesion), then recapture.")
    if res.get("border_downgraded"):
        return "Recapture with the lesion centred and nothing (ruler, ink, gel edge) near the image border."
    return None


def referral_kwargs(res):
    """Arguments for referral_note.generate_referral_note (model-related ones only)."""
    return dict(
        risk_score=res["score"], draps_conformal_set=None, draps_q_hat=None,
        modality_warning=(CLINICAL_NOTE if res["modality"].startswith("clinical") else None),
        novelty_score=res["novelty"], confidence_tier=res["confidence"],
        recapture_hint=recapture_hint(res), risk_tier=TIER_TO_NOTE[res["tier"]],
    )
