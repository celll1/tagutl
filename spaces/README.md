---
title: CL EVA02 LoRA ONNX Tagger
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.43.0 # requirements.txt と合わせるか確認
app_file: app.py
# license: apache-2.0 # または適切なライセンス
# Pinned Hardware: T4 small (GPU) or CPU upgrade (CPU)
# pinned: false # 必要に応じてTrueに
# hardware: cpu-upgrade # or cuda-t4-small
# hf_token: YOUR_HF_TOKEN # Use secrets instead!
---

# WD EVA02 LoRA ONNX Tagger

This Space demonstrates image tagging using a fine-tuned WD EVA02 model (converted to ONNX format).

**How to Use:**
1. Upload an image using the upload button.
2. Alternatively, paste an image URL into the browser (experimental paste handling).
3. Adjust the tag thresholds if needed.
4. Choose the output mode (Tags only or include visualization).
5. Click the "Predict" button.

**Note:**
- This Space uses a model from a **private** repository (`celstk/wd-eva02-lora-onnx`). You might need to duplicate this space and add your Hugging Face token (`HF_TOKEN`) to the Space secrets to allow downloading the model files.
- Image pasting behavior might vary across browsers. 