"""
DermScript — pi_capture_server.py
Runs ON the Raspberry Pi (with the Camera Module 3 and Contact Ring
attached). app.py's sidebar "Test connection" / "Capture" buttons talk to
this over your local Wi-Fi network -- this is the piece that was missing
from the repo, so those two buttons had nothing to connect to.

SETUP (one-time, on the Pi):
    sudo apt update
    sudo apt install -y python3-picamera2 python3-flask
    python3 pi_capture_server.py
    # then find the Pi's IP with: hostname -I
    # put that IP into the app's sidebar "Device IP address" field

Runs a tiny Flask server with two routes:
    GET /health   -> {"status": "ok", "device": "DermScript-v1"}
    GET /capture  -> raw JPEG bytes of a fresh photo from the camera

Keeping this file dependency-light and readable on purpose -- it's meant
to run unattended on a Pi Zero 2W, not be clever.
"""
import io
import time
from flask import Flask, Response, jsonify

app = Flask(__name__)

# ── Camera setup ────────────────────────────────────────────────────────
# Uses picamera2 (the current, actively-maintained library for the Pi
# Camera Module 3 -- the older `picamera` package does NOT support this
# camera). If picamera2 import fails, the server still starts so /health
# works for network testing, but /capture will return a clear error
# instead of crashing the whole process.
try:
    from picamera2 import Picamera2
    _camera_available = True
except ImportError as e:
    _camera_available = False
    _camera_import_error = str(e)

_picam2 = None


def _get_camera():
    """Lazily initializes the camera once and reuses it -- re-initializing
    Picamera2 on every single capture is slow and can leave the camera
    device locked if a prior request didn't clean up properly."""
    global _picam2
    if _picam2 is None:
        _picam2 = Picamera2()
        # Still (full-resolution) capture config -- adjust size if your
        # printed Contact Ring's field of view needs a different crop.
        # Cross-check RING_BUMP_SPACING_MM / the Hough circle parameters
        # in app.py's detect_ruler_bumps_and_diameter() against whatever
        # resolution you settle on here -- those were tuned as a STARTING
        # POINT, not a finished calibration, per that function's own
        # docstring.
        config = _picam2.create_still_configuration(
            main={"size": (2028, 1520)}  # adjust to taste / your ring's FOV
        )
        _picam2.configure(config)
        _picam2.start()
        time.sleep(1.0)  # let auto-exposure/auto-white-balance settle
    return _picam2


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": "DermScript-v1",
        "camera_available": _camera_available,
    })


@app.route("/capture")
def capture():
    if not _camera_available:
        return Response(
            f"picamera2 not installed or failed to import: {_camera_import_error}. "
            f"Run: sudo apt install -y python3-picamera2",
            status=500,
            mimetype="text/plain",
        )
    try:
        cam = _get_camera()
        buf = io.BytesIO()
        cam.capture_file(buf, format="jpeg")
        buf.seek(0)
        return Response(buf.read(), mimetype="image/jpeg")
    except Exception as e:
        return Response(f"Capture failed: {e}", status=500, mimetype="text/plain")


if __name__ == "__main__":
    # 0.0.0.0 so it's reachable from other devices on the same Wi-Fi
    # network (your laptop running the Streamlit app), not just localhost.
    # Port 5000 matches what app.py's sidebar already expects.
    app.run(host="0.0.0.0", port=5000)
