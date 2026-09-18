"""
DermScript — LED Ring Confidence/Risk Indicator
=================================================

Turns your 16-LED WS2812B ring into a live visual readout of the model's
risk score AND its conformal-prediction confidence, instead of a number
buried in the Streamlit app.

Hardware assumed: Raspberry Pi (Zero 2 W), WS2812B ring wired to GPIO18
(PWM0), powered appropriately (per-LED current draw adds up fast — see
notes at the bottom of this file before wiring all 16 at full brightness).

Library: rpi_ws281x (the standard low-level library most WS2812B Pi
tutorials use). Install with:
    pip3 install rpi_ws281x

You will likely need to run scripts that touch the LEDs with sudo,
because direct PWM/DMA access requires root on most Pi OS setups:
    sudo python3 led_feedback.py

--------------------------------------------------------------------
HOW TO WIRE THIS INTO YOUR EXISTING INFERENCE PIPELINE
--------------------------------------------------------------------
After your model produces:
    - risk_score: float in [0, 1]  (LightGBM output probability)
    - interval_width: float        (width of the DRAPS conformal
                                     prediction interval — a proxy for
                                     how uncertain the model is)

...just call:

    from led_feedback import LedFeedback
    led = LedFeedback()
    led.show_result(risk_score=0.73, interval_width=0.41)

That's the entire integration. Everything else in this file is the
mapping logic + the low-level LED driving.
--------------------------------------------------------------------
"""

import time
import threading

try:
    from rpi_ws281x import PixelStrip, Color
except ImportError:
    # Lets you develop/test the color-mapping logic on a laptop
    # before you're on the actual Pi with the ring wired up.
    PixelStrip = None
    Color = lambda r, g, b: (r, g, b)  # noqa: E731


# ---------------------------------------------------------------------
# TUNABLE THRESHOLDS
# ---------------------------------------------------------------------
# These are placeholders. Once you have real DRAPS interval widths from
# your retrained model on held-out data, replace these with values
# chosen by actually looking at the distribution of interval widths
# your model produces (e.g., "uncertain" = top 25% widest intervals
# in your validation set), rather than guessing.

RISK_HIGH_THRESHOLD = 0.6       # risk_score above this -> "high risk" color
RISK_LOW_THRESHOLD = 0.3        # risk_score below this -> "low risk" color
                                 # (anything between = "moderate")

UNCERTAINTY_THRESHOLD = 0.35    # interval_width above this -> "uncertain",
                                 # overrides the risk color with a pulse

# ---------------------------------------------------------------------
# COLOR PALETTE (matches your cyan Streamlit theme where it makes sense)
# ---------------------------------------------------------------------

COLOR_LOW_RISK = (0, 200, 200)      # steady cyan — confident, low risk
COLOR_MODERATE_RISK = (255, 165, 0)  # steady orange — confident, moderate risk
COLOR_HIGH_RISK = (255, 0, 0)        # steady red — confident, high risk
COLOR_UNCERTAIN = (255, 191, 0)      # pulsing amber — "not confident, see a
                                      # dermatologist" — this OVERRIDES risk
                                      # color when uncertainty is high, on
                                      # purpose. A confident wrong answer is
                                      # more dangerous than an honest "I
                                      # don't know."
COLOR_IDLE = (0, 0, 0)               # off / idle state
COLOR_CAPTURING = (0, 100, 255)      # blue sweep while a capture is in progress


class LedFeedback:
    def __init__(self, led_count=16, led_pin=18, led_freq_hz=800000,
                 led_dma=10, led_brightness=100, led_invert=False,
                 led_channel=0, simulate=None):
        """
        led_count: number of LEDs in your ring (16, per your BOM)
        led_pin: GPIO18 is the standard PWM pin used in nearly all
                 WS2812B Pi tutorials — matches your pre-wired SM2.54-3P
                 connector setup as long as you followed a standard guide
        led_brightness: 0-255. START LOW (e.g., 50-100) the first time
                 you power this on. 16 LEDs at full white brightness can
                 pull real current — don't run them at 255 straight off
                 your PowerBoost without checking your battery can supply
                 it comfortably.
        simulate: force simulation mode (prints instead of driving real
                 LEDs) — auto-detected if rpi_ws281x isn't installed,
                 e.g. when developing on a laptop.
        """
        self.led_count = led_count
        self.simulate = simulate if simulate is not None else (PixelStrip is None)
        self._stop_pulse = threading.Event()
        self._pulse_thread = None

        if not self.simulate:
            self.strip = PixelStrip(led_count, led_pin, led_freq_hz,
                                     led_dma, led_invert, led_brightness,
                                     led_channel)
            self.strip.begin()
        else:
            self.strip = None
            print("[LedFeedback] Running in SIMULATION mode "
                  "(no rpi_ws281x found, or simulate=True). "
                  "Color changes will print to console instead.")

    # -----------------------------------------------------------------
    def _set_all(self, rgb):
        r, g, b = rgb
        if self.simulate:
            print(f"[LedFeedback] LEDs -> RGB({r}, {g}, {b})")
            return
        color = Color(r, g, b)
        for i in range(self.led_count):
            self.strip.setPixelColor(i, color)
        self.strip.show()

    def _stop_any_pulse(self):
        if self._pulse_thread and self._pulse_thread.is_alive():
            self._stop_pulse.set()
            self._pulse_thread.join()
        self._stop_pulse.clear()

    def _pulse_loop(self, rgb, period_seconds=1.2, min_scale=0.15):
        """Smoothly fades the ring in and out (breathing effect) until
        stopped. Runs in a background thread so it doesn't block your
        main inference loop."""
        import math
        r, g, b = rgb
        start = time.time()
        while not self._stop_pulse.is_set():
            elapsed = time.time() - start
            # 0..1..0 breathing curve
            phase = (math.sin(2 * math.pi * elapsed / period_seconds) + 1) / 2
            scale = min_scale + (1 - min_scale) * phase
            self._set_all((int(r * scale), int(g * scale), int(b * scale)))
            time.sleep(0.03)

    def _start_pulse(self, rgb):
        self._stop_any_pulse()
        self._pulse_thread = threading.Thread(
            target=self._pulse_loop, args=(rgb,), daemon=True
        )
        self._pulse_thread.start()

    # -----------------------------------------------------------------
    # PUBLIC METHODS
    # -----------------------------------------------------------------

    def show_capturing(self):
        """Call this the moment a photo capture starts, so the user gets
        immediate feedback the device is working (no color logic needed
        here — it's a status indicator, not a diagnosis)."""
        self._stop_any_pulse()
        self._set_all(COLOR_CAPTURING)

    def show_idle(self):
        self._stop_any_pulse()
        self._set_all(COLOR_IDLE)

    def show_result(self, risk_score: float, interval_width: float):
        """
        The core function. Call this right after inference.

        risk_score: model's predicted melanoma-risk probability, 0-1
        interval_width: width of the DRAPS conformal prediction interval
                         for this specific prediction (wider = model is
                         less sure about this particular lesion)
        """
        self._stop_any_pulse()

        if interval_width >= UNCERTAINTY_THRESHOLD:
            # Uncertainty overrides risk color on purpose — see comment
            # on COLOR_UNCERTAIN above.
            self._start_pulse(COLOR_UNCERTAIN)
            return

        if risk_score >= RISK_HIGH_THRESHOLD:
            self._set_all(COLOR_HIGH_RISK)
        elif risk_score <= RISK_LOW_THRESHOLD:
            self._set_all(COLOR_LOW_RISK)
        else:
            self._set_all(COLOR_MODERATE_RISK)

    def cleanup(self):
        self.show_idle()
        self._stop_any_pulse()


# ---------------------------------------------------------------------
# DEMO / MANUAL TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    led = LedFeedback()  # will auto-simulate if not run on the Pi

    print("\n--- Demo sequence ---")
    print("Capturing...")
    led.show_capturing()
    time.sleep(1.5)

    print("Low risk, confident result:")
    led.show_result(risk_score=0.12, interval_width=0.10)
    time.sleep(2)

    print("Moderate risk, confident result:")
    led.show_result(risk_score=0.45, interval_width=0.15)
    time.sleep(2)

    print("High risk, confident result:")
    led.show_result(risk_score=0.85, interval_width=0.12)
    time.sleep(2)

    print("Uncertain result (pulsing amber) — watch this for a few seconds:")
    led.show_result(risk_score=0.55, interval_width=0.50)
    time.sleep(4)

    print("Returning to idle.")
    led.cleanup()

"""
--------------------------------------------------------------------
POWER / SAFETY NOTES (read before wiring, since this is your first
electronics project)
--------------------------------------------------------------------
1. WS2812B LEDs can each draw up to ~60mA at full white/full brightness.
   16 LEDs x 60mA = ~960mA worst case. Your 2000mAh LiPo + PowerBoost
   1000 can supply that, but don't run led_brightness=255 with all
   pixels white for long stretches without checking battery drain and
   the PowerBoost's continuous rated output — that's why the code
   above defaults brightness to 100/255, not 255/255.

2. Data line: WS2812B expects a specific timing signal on the data
   pin. A lot of Pi + WS2812B guides recommend a small logic-level
   shifter or at least a ~330-470 ohm resistor in series on the data
   line, and a large (~1000uF) capacitor across the LED strip's power
   and ground, to protect the first LED from voltage spikes. If your
   pre-wired SM2.54-3P connector setup didn't already account for
   this, it's worth a quick look before first power-on.

3. Always connect ground between the Pi and the LED ring's power
   supply, even if the ring is powered from a separate source than
   the Pi itself — a floating/missing common ground is a very common
   "why are my LEDs doing nothing / doing something insane" bug.

4. As with the battery: verify polarity on every connector before
   plugging in, every time, until it's second nature.
--------------------------------------------------------------------
"""
