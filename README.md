# DermScript v11 (isef-sts-code-v2)

Research prototype that scores dermatoscope images for melanoma-like appearance. **Not a diagnostic device.**

## Files the app needs next to `streamlit_app.py`
| File | Where it comes from |
|---|---|
| `dermscript_v11.py` | notebook cell F1 writes it to `DermScript/v9/final/` (use that copy) |
| `dermscript_v11_bundle.joblib` | notebook cell F6 (0.6 MB) |
| `dinov2_s_224.onnx` | notebook cell F5 (88 MB; GitHub allows files under 100 MB) |
| `dermscript_logic.py`, `referral_note.py`, `lesion_tracker.py`, `led_feedback.py`, `pi_capture_server.py` | this repo |

`requirements.txt` pins scikit-learn / lightgbm / joblib / pandas to the versions printed by notebook cell F6. The model is a
pickle, so those versions must match exactly. The app shows a warning if they do not.

## Delete from the old repo
`dermscript_inference_bundle_v9.pkl`, `dermscript_features.py`, `setup_models.py` (they belong to the old MobileNet + BERT engine).

## Check the deployed files
```
python batch_sanity_check.py --images some_folder/ --labels labels.csv
```
Compare with notebook cells F4 / F4b. A single PH2 image through the app should match the notebook self-test in F6 (score 0.482 for
the image used there).

## Pi
`pi_capture_server.py` runs on the Pi. The app only needs the Pi's IP address in the sidebar.
