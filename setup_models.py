"""
DermScript — ONE-TIME setup script. Run this ONCE while you have internet
(on a laptop, not the Pi). It downloads MobileNetV3-Large and Bio_ClinicalBERT
weights into ./model_cache/, which you then copy onto the Pi alongside
app.py. After that, app.py never needs internet access again.

Usage:
    python setup_models.py
    # then copy the whole "model_cache/" folder onto the Pi, next to app.py
    # and dermscript_inference_bundle.pkl
"""

import os
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "model_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Route both torch hub and HuggingFace caches into our local folder so
# everything needed ends up in one place you can copy onto the Pi.
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")

print(f"Downloading model weights into {CACHE_DIR} ...")

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from transformers import AutoTokenizer, AutoModel

print("  MobileNetV3-Large (ImageNet weights)...")
_ = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

print("  Bio_ClinicalBERT tokenizer + weights...")
BERT_NAME = "emilyalsentzer/Bio_ClinicalBERT"
_ = AutoTokenizer.from_pretrained(BERT_NAME)
_ = AutoModel.from_pretrained(BERT_NAME)

print("\nDone. Copy this whole 'model_cache/' folder onto the Pi, next to")
print("app.py and dermscript_inference_bundle.pkl. app.py will then load")
print("everything from disk with no internet connection required.")
