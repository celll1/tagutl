import gradio as gr
import numpy as np
from PIL import Image # Keep PIL for now, might be needed by helpers implicitly
# from PIL import Image, ImageDraw, ImageFont # No drawing yet
import json
import os
import io
import requests
import matplotlib.pyplot as plt # For visualization
import matplotlib # For backend setting
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import spaces # Required for @spaces.GPU
import onnxruntime as ort # Use ONNX Runtime

import torch # Keep torch for device check in Tagger
import timm # Restore timm
from safetensors.torch import load_file as safe_load_file # Restore safetensors loading

# MatplotlibのバックエンドをAggに設定 (Keep commented out for now)
# matplotlib.use('Agg')

# --- Data Classes and Helper Functions ---
@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    artist: list[np.int64]
    character: list[np.int64]
    copyright: list[np.int64]
    meta: list[np.int64]
    quality: list[np.int64]

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height: return image
    new_size = max(width, height)
    new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255)) # Use image.mode
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def load_tag_mapping(mapping_path):
    # Use the implementation from the original app.py as it was confirmed working
    with open(mapping_path, 'r', encoding='utf-8') as f: tag_mapping_data = json.load(f)
    # Check format compatibility (can be dict of dicts or dict with idx_to_tag/tag_to_category)
    if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
        idx_to_tag = {int(k): v for k, v in tag_mapping_data["idx_to_tag"].items()}
        tag_to_category = tag_mapping_data["tag_to_category"]
    elif isinstance(tag_mapping_data, dict):
        # Assuming the dict-of-dicts format from previous tests
        try:
             tag_mapping_data_int_keys = {int(k): v for k, v in tag_mapping_data.items()}
             idx_to_tag = {idx: data['tag'] for idx, data in tag_mapping_data_int_keys.items()}
             tag_to_category = {data['tag']: data['category'] for data in tag_mapping_data_int_keys.values()}
        except (KeyError, ValueError) as e:
             raise ValueError(f"Unsupported tag mapping format (dict): {e}. Expected int keys with 'tag' and 'category'.")
    else:
        raise ValueError("Unsupported tag mapping format: Expected a dictionary.")

    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating, general, artist, character, copyright, meta, quality = [], [], [], [], [], [], []
    for idx, tag in idx_to_tag.items():
        if idx >= len(names): names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        category = tag_to_category.get(tag, 'Unknown') # Handle missing category mapping gracefully
        idx_int = int(idx)
        if category == 'Rating': rating.append(idx_int)
        elif category == 'General': general.append(idx_int)
        elif category == 'Artist': artist.append(idx_int)
        elif category == 'Character': character.append(idx_int)
        elif category == 'Copyright': copyright.append(idx_int)
        elif category == 'Meta': meta.append(idx_int)
        elif category == 'Quality': quality.append(idx_int)

    return LabelData(names=names, rating=np.array(rating, dtype=np.int64), general=np.array(general, dtype=np.int64), artist=np.array(artist, dtype=np.int64),
                     character=np.array(character, dtype=np.int64), copyright=np.array(copyright, dtype=np.int64), meta=np.array(meta, dtype=np.int64), quality=np.array(quality, dtype=np.int64)), idx_to_tag, tag_to_category

def preprocess_image(image: Image.Image, target_size=(448, 448)):
    # Adapted from onnx_predict.py's version
    image = pil_ensure_rgb(image)
    image = pil_pad_square(image)
    image_resized = image.resize(target_size, Image.BICUBIC)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1) # HWC -> CHW
    # Assuming model expects RGB based on original code, no BGR conversion here
    img_array = img_array[::-1, :, :] # BGR conversion if needed - UNCOMMENTED based on user feedback
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return image, img_array

# Add get_tags function (from onnx_predict.py)
def get_tags(probs, labels: LabelData, gen_threshold, char_threshold):
    result = {
        "rating": [],
        "general": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "meta": [],
        "quality": []
    }
    # Rating (select max)
    if len(labels.rating) > 0:
        # Ensure indices are within bounds
        valid_indices = labels.rating[labels.rating < len(probs)]
        if len(valid_indices) > 0:
            rating_probs = probs[valid_indices]
            if len(rating_probs) > 0:
                rating_idx_local = np.argmax(rating_probs)
                rating_idx_global = valid_indices[rating_idx_local]
                # Check if global index is valid for names list
                if rating_idx_global < len(labels.names) and labels.names[rating_idx_global] is not None:
                    rating_name = labels.names[rating_idx_global]
                    rating_conf = float(rating_probs[rating_idx_local])
                    result["rating"].append((rating_name, rating_conf))
                else:
                    print(f"Warning: Invalid global index {rating_idx_global} for rating tag.")
            else:
                 print("Warning: rating_probs became empty after filtering.")
        else:
            print("Warning: No valid indices found for rating tags within probs length.")

    # Quality (select max)
    if len(labels.quality) > 0:
        valid_indices = labels.quality[labels.quality < len(probs)]
        if len(valid_indices) > 0:
            quality_probs = probs[valid_indices]
            if len(quality_probs) > 0:
                quality_idx_local = np.argmax(quality_probs)
                quality_idx_global = valid_indices[quality_idx_local]
                if quality_idx_global < len(labels.names) and labels.names[quality_idx_global] is not None:
                    quality_name = labels.names[quality_idx_global]
                    quality_conf = float(quality_probs[quality_idx_local])
                    result["quality"].append((quality_name, quality_conf))
                else:
                     print(f"Warning: Invalid global index {quality_idx_global} for quality tag.")
            else:
                print("Warning: quality_probs became empty after filtering.")
        else:
            print("Warning: No valid indices found for quality tags within probs length.")

    # Threshold-based categories
    category_map = {
        "general": (labels.general, gen_threshold),
        "character": (labels.character, char_threshold),
        "copyright": (labels.copyright, char_threshold),
        "artist": (labels.artist, char_threshold),
        "meta": (labels.meta, gen_threshold) # Use gen_threshold for meta as per original code
    }
    for category, (indices, threshold) in category_map.items():
        if len(indices) > 0:
            valid_indices = indices[(indices < len(probs))] # Check index bounds first
            if len(valid_indices) > 0:
                category_probs = probs[valid_indices]
                mask = category_probs >= threshold
                selected_indices_local = np.where(mask)[0]
                if len(selected_indices_local) > 0:
                    selected_indices_global = valid_indices[selected_indices_local]
                    selected_probs = category_probs[selected_indices_local]
                    for idx_global, prob_val in zip(selected_indices_global, selected_probs):
                        # Check if global index is valid for names list
                        if idx_global < len(labels.names) and labels.names[idx_global] is not None:
                             result[category].append((labels.names[idx_global], float(prob_val)))
                        else:
                             print(f"Warning: Invalid global index {idx_global} for {category} tag.")
                # else: print(f"No tags found for category '{category}' above threshold {threshold}")
            # else: print(f"No valid indices found for category '{category}' within probs length.")
        # else: print(f"No indices defined for category '{category}'")

    # Sort by probability (descending)
    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)
    return result

# Add visualize_predictions function (Adapted from onnx_predict.py and previous versions)
def visualize_predictions(image: Image.Image, predictions: Dict, threshold: float):
    # Filter out unwanted meta tags (e.g., id, commentary, request, mismatch)
    filtered_meta = []
    excluded_meta_patterns = ['id', 'commentary', 'request', 'mismatch']
    for tag, prob in predictions.get("meta", []):
        if not any(pattern in tag.lower() for pattern in excluded_meta_patterns):
            filtered_meta.append((tag, prob))
    predictions["meta"] = filtered_meta  # Use filtered list for visualization

    # --- Plotting Setup ---
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(8, 12), dpi=100)
    ax_tags = fig.add_subplot(1, 1, 1)

    all_tags, all_probs, all_colors = [], [], []
    color_map = {
        'rating': 'red', 'character': 'blue', 'copyright': 'purple',
        'artist': 'orange', 'general': 'green', 'meta': 'gray', 'quality': 'yellow'
    }

    # Aggregate tags from predictions dictionary
    for cat, prefix, color in [
        ('rating', 'R', color_map['rating']), ('quality', 'Q', color_map['quality']),
        ('character', 'C', color_map['character']), ('copyright', '©', color_map['copyright']),
        ('artist', 'A', color_map['artist']), ('general', 'G', color_map['general']),
        ('meta', 'M', color_map['meta'])
    ]:
        sorted_tags = sorted(predictions.get(cat, []), key=lambda x: x[1], reverse=True)
        for tag, prob in sorted_tags:
            all_tags.append(f"[{prefix}] {tag.replace('_', ' ')}")
            all_probs.append(prob)
            all_colors.append(color)

    if not all_tags:
        ax_tags.text(0.5, 0.5, "No tags found above threshold", ha='center', va='center')
        ax_tags.set_title(f"Tags (Threshold ≳ {threshold:.2f})")
        ax_tags.axis('off')
    else:
        sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i])
        all_tags = [all_tags[i] for i in sorted_indices]
        all_probs = [all_probs[i] for i in sorted_indices]
        all_colors = [all_colors[i] for i in sorted_indices]

        num_tags = len(all_tags)
        bar_height = min(0.8, max(0.1, 0.8 * (30 / num_tags))) if num_tags > 30 else 0.8
        y_positions = np.arange(num_tags)

        bars = ax_tags.barh(y_positions, all_probs, height=bar_height, color=all_colors)
        ax_tags.set_yticks(y_positions)
        ax_tags.set_yticklabels(all_tags)

        fontsize = 10 if num_tags <= 40 else 8 if num_tags <= 60 else 6
        for lbl in ax_tags.get_yticklabels():
            lbl.set_fontsize(fontsize)

        for i, (bar, prob) in enumerate(zip(bars, all_probs)):
            text_x = min(prob + 0.02, 0.98)
            ax_tags.text(text_x, y_positions[i], f"{prob:.3f}", va='center', fontsize=fontsize)

        ax_tags.set_xlim(0, 1)
        ax_tags.set_title(f"Tags (Threshold ≳ {threshold:.2f})")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=cat.capitalize())
            for cat, color in color_map.items()
            if any(t.startswith(f"[{cat[0].upper() if cat!='copyright' else '©'}]") for t in all_tags)
        ]
        if legend_elements:
            ax_tags.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- Constants ---
REPO_ID = "celstk/wd-eva02-lora-onnx"
# Model options
MODEL_OPTIONS = {
    "cl_eva02_tagger_v1_250426": "cl_eva02_tagger_v1_250426/model.onnx",
    "cl_eva02_tagger_v1_250502": "cl_eva02_tagger_v1_250503/model.onnx",
    "cl_eva02_tagger_v1_250509": "cl_eva02_tagger_v1_250509/model.onnx",
    "cl_eva02_tagger_v1_250511": "cl_eva02_tagger_v1_250511/model.onnx",
    "cl_eva02_tagger_v1_250512": "cl_eva02_tagger_v1_250512/model.onnx",
    "cl_eva02_tagger_v1_250513": "cl_eva02_tagger_v1_250513/model.onnx",
    "cl_eva02_tagger_v1_250516": "cl_eva02_tagger_v1_250516/model.onnx",
    "cl_eva02_tagger_v1_250517": "cl_eva02_tagger_v1_250517/model.onnx"
}
DEFAULT_MODEL = "cl_eva02_tagger_v1_250517"
CACHE_DIR = "./model_cache"

# --- Global variables for paths (initialized at startup) ---
g_onnx_model_path = None
g_tag_mapping_path = None
g_labels_data = None
g_idx_to_tag = None
g_tag_to_category = None
g_current_model = None

# --- Initialization Function ---
def initialize_onnx_paths(model_choice=DEFAULT_MODEL):
    global g_onnx_model_path, g_tag_mapping_path, g_labels_data, g_idx_to_tag, g_tag_to_category, g_current_model
    
    if not model_choice in MODEL_OPTIONS:
        print(f"Invalid model choice: {model_choice}, falling back to default: {DEFAULT_MODEL}")
        model_choice = DEFAULT_MODEL
    
    g_current_model = model_choice
    model_dir = model_choice
    onnx_filename = MODEL_OPTIONS[model_choice]
    tag_mapping_filename = f"{model_dir}/tag_mapping.json"
    
    print(f"Initializing ONNX paths and labels for model: {model_choice}...")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        print(f"Attempting to download ONNX model: {onnx_filename}")
        g_onnx_model_path = hf_hub_download(repo_id=REPO_ID, filename=onnx_filename, cache_dir=CACHE_DIR, token=hf_token, force_download=False)
        print(f"ONNX model path: {g_onnx_model_path}")

        print(f"Attempting to download Tag mapping: {tag_mapping_filename}")
        g_tag_mapping_path = hf_hub_download(repo_id=REPO_ID, filename=tag_mapping_filename, cache_dir=CACHE_DIR, token=hf_token, force_download=False)
        print(f"Tag mapping path: {g_tag_mapping_path}")

        print("Loading labels from mapping...")
        g_labels_data, g_idx_to_tag, g_tag_to_category = load_tag_mapping(g_tag_mapping_path)
        print(f"Labels loaded. Count: {len(g_labels_data.names)}")
        
        return True

    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback; traceback.print_exc()
        # Reset globals to force reinitialization
        g_onnx_model_path = None
        g_tag_mapping_path = None
        g_labels_data = None
        g_idx_to_tag = None
        g_tag_to_category = None
        g_current_model = None
        # Raise Gradio error to make it visible in the UI
        raise gr.Error(f"Initialization failed: {e}. Check logs and HF_TOKEN.")

# Function to handle model change
def change_model(model_choice):
    try:
        success = initialize_onnx_paths(model_choice)
        if success:
            return f"Model changed to: {model_choice}"
        else:
            return "Failed to change model. See logs for details."
    except Exception as e:
        return f"Error changing model: {str(e)}"
        
# --- Main Prediction Function (ONNX) ---
@spaces.GPU()
def predict_onnx(image_input, model_choice, gen_threshold, char_threshold, output_mode):
    print(f"--- predict_onnx function started (GPU worker) with model {model_choice} ---")
    
    # Ensure current model matches selected model
    global g_current_model
    if g_current_model != model_choice:
        print(f"Model mismatch! Current: {g_current_model}, Selected: {model_choice}. Reinitializing...")
        try:
            initialize_onnx_paths(model_choice)
        except Exception as e:
            return f"Error initializing model '{model_choice}': {str(e)}", None
    
    # --- 1. Ensure paths and labels are loaded ---
    if g_onnx_model_path is None or g_labels_data is None:
        message = "Error: Paths or labels not initialized. Check startup logs."
        print(message)
        # Return error message and None for the image output
        return message, None

    # --- 2. Load ONNX Session (inside worker) ---
    session = None
    try:
        print(f"Loading ONNX session from: {g_onnx_model_path}")
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        print(f"Attempting to load session with providers: {providers}")
        session = ort.InferenceSession(g_onnx_model_path, providers=providers)
        print(f"ONNX session loaded using: {session.get_providers()[0]}")
    except Exception as e:
        message = f"Error loading ONNX session in worker: {e}"
        print(message)
        import traceback; traceback.print_exc()
        return message, None

    # --- 3. Process Input Image ---
    if image_input is None:
        return "Please upload an image.", None

    print(f"Processing image with thresholds: gen={gen_threshold}, char={char_threshold}")
    try:
        # Handle different input types (PIL, numpy, URL, file path)
        if isinstance(image_input, str):
            if image_input.startswith("http"): # URL
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            elif os.path.exists(image_input): # File path
                image = Image.open(image_input)
            else:
                 raise ValueError(f"Invalid image input string: {image_input}")
        elif isinstance(image_input, np.ndarray):
             image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
             image = image_input # Already a PIL image
        else:
             raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Preprocess the PIL image
        original_pil_image, input_tensor = preprocess_image(image)

        # Ensure input tensor is float32, as expected by most ONNX models
        # (even if the model internally uses float16)
        input_tensor = input_tensor.astype(np.float32)

    except Exception as e:
        message = f"Error processing input image: {e}"
        print(message)
        return message, None

    # --- 4. Run Inference ---
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"Running inference with input '{input_name}', output '{output_name}'")
        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_tensor})[0]
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.3f} seconds")

        # Check for NaN/Inf in outputs
        if np.isnan(outputs).any() or np.isinf(outputs).any():
            print("Warning: NaN or Inf detected in model output. Clamping...")
            outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0) # Clamp to 0-1 range

        # Apply sigmoid (outputs are likely logits)
        # Use a stable sigmoid implementation
        def stable_sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -30, 30))) # Clip to avoid overflow
        probs = stable_sigmoid(outputs[0]) # Assuming batch size 1

    except Exception as e:
        message = f"Error during ONNX inference: {e}"
        print(message)
        import traceback; traceback.print_exc()
        return message, None
    finally:
        # Clean up session if needed (might reduce memory usage between clicks)
        del session

    # --- 5. Post-process and Format Output ---
    try:
        print("Post-processing results...")
        # Use the correct global variable for labels
        predictions = get_tags(probs, g_labels_data, gen_threshold, char_threshold)

        # Format output text string
        output_tags = []
        if predictions.get("rating"): output_tags.append(predictions["rating"][0][0].replace("_", " "))
        if predictions.get("quality"): output_tags.append(predictions["quality"][0][0].replace("_", " "))
        # Add other categories, respecting order and filtering meta if needed
        for category in ["artist", "character", "copyright", "general", "meta"]:
            tags_in_category = predictions.get(category, [])
            for tag, prob in tags_in_category:
                # Basic meta tag filtering for text output
                if category == "meta" and any(p in tag.lower() for p in ['id', 'commentary', 'request', 'mismatch']):
                    continue
                output_tags.append(tag.replace("_", " "))
        output_text = ", ".join(output_tags)

        # Generate visualization if requested
        viz_image = None
        if output_mode == "Tags + Visualization":
            print("Generating visualization...")
            # Pass the correct threshold for display title (can pass both if needed)
            # For simplicity, passing gen_threshold as a representative value
            viz_image = visualize_predictions(original_pil_image, predictions, gen_threshold)
            print("Visualization generated.")
        else:
            print("Visualization skipped.")

        print("Prediction complete.")
        return output_text, viz_image

    except Exception as e:
        message = f"Error during post-processing: {e}"
        print(message)
        import traceback; traceback.print_exc()
        return message, None

# --- Gradio Interface Definition (Full ONNX Version) ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
footer { display: none !important; }
.gr-prose { max-width: 100% !important; }
"""
# js = """ /* Keep existing JS */ """ # No JS needed currently

with gr.Blocks(css=css) as demo:
    gr.Markdown("# CL EVA02 ONNX Tagger")
    gr.Markdown("Upload an image or paste an image URL to predict tags using the CL EVA02 Tagger model (ONNX), fine-tuned from [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3).")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input Image", elem_id="input-image")
            model_choice = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()), 
                value=DEFAULT_MODEL, 
                label="Model Version",
                interactive=True
            )
            gen_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.55, label="General/Meta Tag Threshold")
            char_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.60, label="Character/Copyright/Artist Tag Threshold")
            output_mode = gr.Radio(choices=["Tags Only", "Tags + Visualization"], value="Tags + Visualization", label="Output Mode")
            predict_button = gr.Button("Predict", variant="primary")
        with gr.Column(scale=1):
            output_tags = gr.Textbox(label="Predicted Tags", lines=10, interactive=False)
            output_visualization = gr.Image(type="pil", label="Prediction Visualization", interactive=False)
    
    # Handle model change
    model_status = gr.Textbox(label="Model Status", interactive=False, visible=False)
    model_choice.change(
        fn=change_model,
        inputs=[model_choice],
        outputs=[model_status]
    )
    
    gr.Examples(
        examples=[
            ["https://pbs.twimg.com/media/GXBXsRvbQAAg1kp.jpg", DEFAULT_MODEL, 0.55, 0.70, "Tags + Visualization"],
            ["https://pbs.twimg.com/media/GjlX0gibcAA4EJ4.jpg", DEFAULT_MODEL, 0.55, 0.70, "Tags Only"],
            ["https://pbs.twimg.com/media/Gj4nQbjbEAATeoH.jpg", DEFAULT_MODEL, 0.55, 0.70, "Tags + Visualization"],
            ["https://pbs.twimg.com/media/GkbtX0GaoAMlUZt.jpg", DEFAULT_MODEL, 0.55, 0.70, "Tags + Visualization"]
        ],
        inputs=[image_input, model_choice, gen_threshold, char_threshold, output_mode],
        outputs=[output_tags, output_visualization],
        fn=predict_onnx, # Use the ONNX prediction function
        cache_examples=False # Disable caching for examples during testing
    )
    predict_button.click(
        fn=predict_onnx, # Use the ONNX prediction function
        inputs=[image_input, model_choice, gen_threshold, char_threshold, output_mode],
        outputs=[output_tags, output_visualization]
    )

# --- Main Block ---
if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"): print("Warning: HF_TOKEN environment variable not set.")
    # Initialize paths and labels at startup (with default model)
    initialize_onnx_paths(DEFAULT_MODEL)
    # Launch Gradio app
    demo.launch(share=True)
