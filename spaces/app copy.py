import gradio as gr
# import onnxruntime as ort # Removed
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
import io
import requests
import matplotlib.pyplot as plt
import matplotlib
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import spaces # Re-add for @spaces.GPU

import torch
import timm
from safetensors.torch import load_file as safe_load_file

# MatplotlibのバックエンドをAggに設定 (GUIなし環境用)
matplotlib.use('Agg')

# --- onnx_predict.pyからの移植 ---

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
    if width == height:
        return image
    new_size = max(width, height)
    new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    return new_image

def load_tag_mapping(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        tag_mapping_data = json.load(f)

    # 新旧フォーマット対応
    if isinstance(tag_mapping_data, dict) and "idx_to_tag" in tag_mapping_data:
        # 旧フォーマット (辞書の中にidx_to_tagとtag_to_categoryがある)
        idx_to_tag_dict = tag_mapping_data["idx_to_tag"]
        tag_to_category_dict = tag_mapping_data["tag_to_category"]
        # tag_mapping_dataが文字列キーになっている可能性があるのでintに変換
        idx_to_tag = {int(k): v for k, v in idx_to_tag_dict.items()}
        tag_to_category = tag_to_category_dict
    elif isinstance(tag_mapping_data, dict):
         # 新フォーマット (キーがインデックスの辞書)
        tag_mapping_data = {int(k): v for k, v in tag_mapping_data.items()}
        idx_to_tag = {}
        tag_to_category = {}
        for idx, data in tag_mapping_data.items():
            tag = data['tag']
            category = data['category']
            idx_to_tag[idx] = tag
            tag_to_category[tag] = category
    else:
        raise ValueError("Unsupported tag mapping format")


    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating = []
    general = []
    artist = []
    character = []
    copyright = []
    meta = []
    quality = []

    for idx, tag in idx_to_tag.items():
        if idx >= len(names): # namesリストのサイズが足りない場合拡張
             names.extend([None] * (idx - len(names) + 1))
        names[idx] = tag
        category = tag_to_category.get(tag, 'Unknown') # カテゴリが見つからない場合

        if category == 'Rating':
            rating.append(idx)
        elif category == 'General':
            general.append(idx)
        elif category == 'Artist':
            artist.append(idx)
        elif category == 'Character':
            character.append(idx)
        elif category == 'Copyright':
            copyright.append(idx)
        elif category == 'Meta':
            meta.append(idx)
        elif category == 'Quality':
            quality.append(idx)
        # Unknownカテゴリは無視

    label_data = LabelData(
        names=names,
        rating=np.array(rating, dtype=np.int64),
        general=np.array(general, dtype=np.int64),
        artist=np.array(artist, dtype=np.int64),
        character=np.array(character, dtype=np.int64),
        copyright=np.array(copyright, dtype=np.int64),
        meta=np.array(meta, dtype=np.int64),
        quality=np.array(quality, dtype=np.int64)
    )

    return label_data, idx_to_tag, tag_to_category


def preprocess_image(image: Image.Image, target_size=(448, 448)):
    image = pil_ensure_rgb(image)
    image = pil_pad_square(image)
    image_resized = image.resize(target_size, Image.BICUBIC)
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1) # HWC -> CHW
    # RGB -> BGR (モデルがBGRを期待する場合 - WD Tagger v3はBGR)
    # WD Tagger V2/V1はRGBなので注意
    img_array = img_array[::-1, :, :]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return image, img_array # Return original PIL image and processed numpy array

def get_tags(probs, labels: LabelData, gen_threshold, char_threshold):
    result = {
        "rating": [], "general": [], "character": [],
        "copyright": [], "artist": [], "meta": [], "quality": []
    }

    # Rating (select the max)
    if labels.rating.size > 0:
        rating_probs = probs[labels.rating]
        if rating_probs.size > 0:
            rating_idx = np.argmax(rating_probs)
            # Check if the index is valid for names list
            if labels.rating[rating_idx] < len(labels.names):
                 rating_name = labels.names[labels.rating[rating_idx]]
                 rating_conf = float(rating_probs[rating_idx])
                 result["rating"].append((rating_name, rating_conf))
            else:
                 print(f"Warning: Rating index {labels.rating[rating_idx]} out of bounds for names list (size {len(labels.names)}).")


    # Quality (select the max)
    if labels.quality.size > 0:
        quality_probs = probs[labels.quality]
        if quality_probs.size > 0:
             quality_idx = np.argmax(quality_probs)
             if labels.quality[quality_idx] < len(labels.names):
                  quality_name = labels.names[labels.quality[quality_idx]]
                  quality_conf = float(quality_probs[quality_idx])
                  result["quality"].append((quality_name, quality_conf))
             else:
                  print(f"Warning: Quality index {labels.quality[quality_idx]} out of bounds for names list (size {len(labels.names)}).")


    category_map = {
        "general": (labels.general, gen_threshold),
        "character": (labels.character, char_threshold),
        "copyright": (labels.copyright, char_threshold),
        "artist": (labels.artist, char_threshold),
        "meta": (labels.meta, gen_threshold)
    }

    for category, (indices, threshold) in category_map.items():
        if indices.size > 0:
            # Filter indices to be within the bounds of probs and labels.names
            valid_indices = indices[(indices < len(probs)) & (indices < len(labels.names))]
            if valid_indices.size > 0:
                 category_probs = probs[valid_indices]
                 mask = category_probs >= threshold
                 selected_indices = valid_indices[mask]
                 selected_probs = category_probs[mask]
                 for idx, prob in zip(selected_indices, selected_probs):
                      result[category].append((labels.names[idx], float(prob)))


    # Sort by probability
    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)

    return result

def visualize_predictions(image: Image.Image, predictions, threshold=0.45):
    # Filter out unwanted meta tags
    filtered_meta = []
    excluded_meta_patterns = ['id', 'commentary', 'request', 'mismatch']
    for tag, prob in predictions["meta"]:
        if not any(pattern in tag.lower() for pattern in excluded_meta_patterns):
            filtered_meta.append((tag, prob))
    predictions["meta"] = filtered_meta # Replace with filtered

    # Create plot
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title("Original Image")
    ax_img.axis('off')
    ax_tags = fig.add_subplot(gs[0, 1])

    all_tags = []
    all_probs = []
    all_colors = []
    color_map = {'rating': 'red', 'character': 'blue', 'copyright': 'purple',
                 'artist': 'orange', 'general': 'green', 'meta': 'gray', 'quality': 'yellow'}

    for cat, prefix, color in [('rating', 'R', 'red'), ('character', 'C', 'blue'),
                              ('copyright', '©', 'purple'), ('artist', 'A', 'orange'),
                              ('general', 'G', 'green'), ('meta', 'M', 'gray'), ('quality', 'Q', 'yellow')]:
        for tag, prob in predictions[cat]:
            all_tags.append(f"[{prefix}] {tag}")
            all_probs.append(prob)
            all_colors.append(color)

    if not all_tags:
        ax_tags.text(0.5, 0.5, "No tags found above threshold", ha='center', va='center')
        ax_tags.set_title(f"Tags (threshold={threshold})")
        ax_tags.axis('off')
        plt.tight_layout()
        # Save figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)


    sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)
    all_tags = [all_tags[i] for i in sorted_indices]
    all_probs = [all_probs[i] for i in sorted_indices]
    all_colors = [all_colors[i] for i in sorted_indices]

    all_tags.reverse()
    all_probs.reverse()
    all_colors.reverse()

    num_tags = len(all_tags)
    bar_height = 0.8
    if num_tags > 30: bar_height = 0.8 * (30 / num_tags)
    y_positions = np.arange(num_tags)

    bars = ax_tags.barh(y_positions, all_probs, height=bar_height, color=all_colors)
    ax_tags.set_yticks(y_positions)
    ax_tags.set_yticklabels(all_tags)

    fontsize = 10
    if num_tags > 40: fontsize = 8
    elif num_tags > 60: fontsize = 6
    for label in ax_tags.get_yticklabels(): label.set_fontsize(fontsize)

    for i, (bar, prob) in enumerate(zip(bars, all_probs)):
        ax_tags.text(min(prob + 0.02, 0.98), y_positions[i], f"{prob:.3f}",
                     va='center', fontsize=fontsize)

    ax_tags.set_xlim(0, 1)
    ax_tags.set_title(f"Tags (threshold={threshold})")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat.capitalize()) for cat, color in color_map.items()]
    ax_tags.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    # Save figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- Gradio App Logic ---

# 定数
REPO_ID = "cella110n/cl_tagger"
SAFETENSORS_FILENAME = "lora_model_0426/checkpoint_epoch_4.safetensors"
METADATA_FILENAME = "lora_model_0426/checkpoint_epoch_4_metadata.json"
TAG_MAPPING_FILENAME = "lora_model_0426/tag_mapping.json"
CACHE_DIR = "./model_cache"

safetensors_path_global = None
metadata_path_global = None
tag_mapping_path_global = None
labels_data = None
tag_to_category_map = None

def download_model_files():
    """Hugging Face Hubからモデル、メタデータ、タグマッピングをダウンロード"""
    global safetensors_path_global, metadata_path_global, tag_mapping_path_global
    # Check if files seem to be downloaded already
    if safetensors_path_global and tag_mapping_path_global and os.path.exists(safetensors_path_global) and os.path.exists(tag_mapping_path_global):
        print("Files seem already downloaded.")
        return

    print("Downloading model files...")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        safetensors_path_global = hf_hub_download(repo_id=REPO_ID, filename=SAFETENSORS_FILENAME, cache_dir=CACHE_DIR, token=hf_token, force_download=True) # Force download to ensure latest
        tag_mapping_path_global = hf_hub_download(repo_id=REPO_ID, filename=TAG_MAPPING_FILENAME, cache_dir=CACHE_DIR, token=hf_token, force_download=True)
        print(f"Safetensors downloaded to: {safetensors_path_global}")
        print(f"Tag mapping downloaded to: {tag_mapping_path_global}")
        try:
            metadata_path_global = hf_hub_download(repo_id=REPO_ID, filename=METADATA_FILENAME, cache_dir=CACHE_DIR, token=hf_token, force_download=True)
            print(f"Metadata downloaded to: {metadata_path_global}")
        except Exception:
            print(f"Metadata file ({METADATA_FILENAME}) not found or download failed. Proceeding without it.")
            metadata_path_global = None
    except Exception as e:
        print(f"Error downloading files: {e}")
        if "401 Client Error" in str(e) or "Repository not found" in str(e):
             raise gr.Error(f"Could not download files from {REPO_ID}. Check HF_TOKEN secret.")
        else:
            raise gr.Error(f"Error downloading files: {e}")

def initialize_labels_and_paths():
    """ラベルデータとファイルパスを準備（キャッシュ）"""
    global labels_data, tag_to_category_map, tag_mapping_path_global
    if labels_data is None:
        download_model_files() # Ensure files are downloaded
        print("Loading labels from tag_mapping.json...")
        if tag_mapping_path_global and os.path.exists(tag_mapping_path_global):
            try:
                 labels_data, _, tag_to_category_map = load_tag_mapping(tag_mapping_path_global)
                 print(f"Labels loaded successfully. Number of labels: {len(labels_data.names)}")
            except Exception as e:
                 print(f"Error loading tag mapping from {tag_mapping_path_global}: {e}")
                 raise gr.Error(f"Error loading tag mapping file: {e}")
        else:
             print(f"Tag mapping file not found after download attempt: {tag_mapping_path_global}")
             raise gr.Error("Tag mapping file could not be downloaded or found.")

# --- Prediction Function (PyTorch based) ---
@spaces.GPU() # Re-add decorator for ZeroGPU
def predict(image_input, gen_threshold, char_threshold, output_mode):
    print("--- predict function started (GPU worker) ---")
    """Gradioインターフェース用の予測関数 (PyTorch GPUワーカー内)"""
    initialize_labels_and_paths()
    print("Loading PyTorch model...")
    global safetensors_path_global, labels_data
    if safetensors_path_global is None or labels_data is None:
        initialize_labels_and_paths()
        if safetensors_path_global is None or labels_data is None:
            return "Error: Model/Labels paths could not be initialized.", None
    try:
        print(f"Creating base model: eva02_large_patch14_448.mim_m38m_ft_in1k")
        num_classes = len(labels_data.names)
        # Validate num_classes (should be > 0)
        if num_classes <= 0:
            raise ValueError(f"Invalid number of classes loaded from tag mapping: {num_classes}")
        print(f"Setting num_classes: {num_classes}")
        model = timm.create_model(
            'eva02_large_patch14_448.mim_m38m_ft_in1k',
            pretrained=True,
            num_classes=num_classes
        )
        print(f"Loading state dict from: {safetensors_path_global}")
        if not os.path.exists(safetensors_path_global):
             raise FileNotFoundError(f"Safetensors file not found at: {safetensors_path_global}")
        state_dict = safe_load_file(safetensors_path_global)
        adapted_state_dict = {}
        for k, v in state_dict.items():
            # Adjust key names if needed based on how lora.py saved the merged model
            # Example: If saved with 'base_model.' prefix
            # if k.startswith('base_model.'):
            #    adapted_state_dict[k[len('base_model.'):]] = v
            # else:
            adapted_state_dict[k] = v # Assuming direct key match for now

        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
        print(f"State dict loaded. Missing keys: {missing_keys}")
        print(f"State dict loaded. Unexpected keys: {unexpected_keys}")
        # Handle critical missing keys (like the head) if necessary
        if any(k.startswith('head.') for k in missing_keys):
             print("Warning: Classification head weights might be missing or mismatched!")
        # if unexpected_keys:
        #     print("Warning: Unexpected keys found in state_dict.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Moving model to device: {device}")
        model.to(device)
        model.eval()
        print("Model loaded and moved to device.")
    except Exception as e:
        print(f"(Worker) Error loading PyTorch model: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error loading PyTorch model: {e}", None

    if image_input is None:
        return "Please upload an image.", None
    print(f"(Worker) Processing image with thresholds: gen={gen_threshold}, char={char_threshold}")
    if not isinstance(image_input, Image.Image):
        try:
            if isinstance(image_input, str) and image_input.startswith("http"):
                response = requests.get(image_input); response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            elif isinstance(image_input, str) and os.path.exists(image_input):
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            else: raise ValueError("Unsupported image input type")
        except Exception as e:
            print(f"(Worker) Error loading image: {e}"); return f"Error loading image: {e}", None
    else: image = image_input

    original_pil_image, input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    try:
        print("(Worker) Running inference...")
        start_time = time.time()
        with torch.no_grad(): outputs = model(input_tensor)
        inference_time = time.time() - start_time
        print(f"(Worker) Inference completed in {inference_time:.3f} seconds")
        probs = torch.sigmoid(outputs)[0].cpu().numpy()
    except Exception as e:
        print(f"(Worker) Error during PyTorch inference: {e}"); import traceback; print(traceback.format_exc()); return f"Error during inference: {e}", None

    predictions = get_tags(probs, labels_data, gen_threshold, char_threshold)
    output_tags = []
    if predictions.get("rating"): output_tags.append(predictions["rating"][0][0].replace("_", " "))
    if predictions.get("quality"): output_tags.append(predictions["quality"][0][0].replace("_", " "))
    for category in ["artist", "character", "copyright", "general", "meta"]:
        tags = [tag.replace("_", " ") for tag, prob in predictions.get(category, [])
                 if not (category == "meta" and any(p in tag.lower() for p in ['id', 'commentary','mismatch']))]
        output_tags.extend(tags)
    output_text = ", ".join(output_tags)

    if output_mode == "Tags Only": return output_text, None
    else: viz_image = visualize_predictions(original_pil_image, predictions, gen_threshold); return output_text, viz_image

# --- Gradio Interface Definition ---
css = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
footer { display: none !important; }
.gr-prose { max-width: 100% !important; }
"""
js = """
async function paste_image(blob, gen_thresh, char_thresh, out_mode) {
    const data = await fetch(blob)
    const image_data = await data.blob()
    const file = new File([image_data], "pasted_image.png",{ type: image_data.type })
    const dt = new DataTransfer()
    dt.items.add(file)
    const element = document.querySelector('#input-image input[type="file"]')
    element.files = dt.files
    // Trigger the change event manually
    const event = new Event('change', { bubbles: true })
    element.dispatchEvent(event)
    // Wait a bit for Gradio to process the change, then trigger predict if needed
    // await new Promise(resolve => setTimeout(resolve, 100)); // Optional delay
    // You might need to manually trigger the prediction or rely on Gradio's auto-triggering
    return [file, gen_thresh, char_thresh, out_mode]; // Return input for Gradio function
}

async function paste_update(evt){
    if (!evt.clipboardData || !evt.clipboardData.items) return;
    var url = evt.clipboardData.getData('text');
    if (url) {
        // Basic check for image URL (you might want a more robust check)
        if (/\.(jpg|jpeg|png|webp|bmp)$/i.test(url)) {
            // Create a button or link to load the URL
            const url_container = document.getElementById('url-input-container');
            url_container.innerHTML = `<p>Detected URL: <button id="load-url-btn" class="gr-button gr-button-sm gr-button-secondary">${url}</button></p>`;

            document.getElementById('load-url-btn').onclick = async () => {
                // Simulate file upload from URL - Gradio's Image component handles URLs directly
                 const element = document.querySelector('#input-image input[type="file"]');
                 // Can't directly set URL to file input, so we pass it to Gradio fn
                 // Or maybe update the image display src directly if possible?

                 // Let Gradio handle the URL - user needs to click predict
                 // We can pre-fill the image component if Gradio supports it via JS,
                 // but it's simpler to just let the user click predict after pasting URL.
                 alert("URL detected. Please ensure the image input is cleared and then press 'Predict' or re-upload the image.");
                 // Clear current image preview if possible?

                 // A workaround: display the URL and let the user manually trigger prediction
                 // Or, try to use Gradio's JS API if available to update the Image component value
                 // For now, just inform the user.
            };
            return; // Don't process as image paste if URL is found
        }
    }

    var items = evt.clipboardData.items;
    for (var i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") === 0) {
            var blob = items[i].getAsFile();
            var reader = new FileReader();
            reader.onload = function(event){
                 // Update the Gradio Image component source directly
                 const imgElement = document.querySelector('#input-image img'); // Find the img tag inside the component
                 if (imgElement) {
                     imgElement.src = event.target.result;
                     // We still need to pass the blob to the Gradio function
                     // Use Gradio's JS API or hidden components if possible
                     // For now, let's use a simple alert and rely on manual trigger
                     alert("Image pasted. The preview should update. Please press 'Predict'.");
                     // Trigger paste_image function - requires Gradio JS interaction
                     // This part is tricky without official Gradio JS API for updates
                 }
            };
            reader.readAsDataURL(blob);
            // Prevent default paste handling
            evt.preventDefault();
            break;
        }
    }
}

document.addEventListener('paste', paste_update);
"""

with gr.Blocks(css=css, js=js) as demo:
    gr.Markdown("# WD EVA02 LoRA PyTorch Tagger")
    gr.Markdown("Upload an image or paste an image URL to predict tags using the fine-tuned WD EVA02 Tagger model (PyTorch/Safetensors).")
    gr.Markdown(f"Model Repository: [{REPO_ID}](https://huggingface.co/{REPO_ID})")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input Image", elem_id="input-image")
            gr.HTML("<div id='url-input-container'></div>")
            gen_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.55, label="General Tag Threshold")
            char_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.60, label="Character/Copyright/Artist Tag Threshold")
            output_mode = gr.Radio(choices=["Tags Only", "Tags + Visualization"], value="Tags + Visualization", label="Output Mode")
            predict_button = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):
            output_tags = gr.Textbox(label="Predicted Tags", lines=10)
            output_visualization = gr.Image(type="pil", label="Prediction Visualization")

    gr.Examples(
        examples=[
            ["https://pbs.twimg.com/media/GXBXsRvbQAAg1kp.jpg", 0.55, 0.5, "Tags + Visualization"],
            ["https://pbs.twimg.com/media/GjlX0gibcAA4EJ4.jpg", 0.5, 0.5, "Tags Only"],
            ["https://pbs.twimg.com/media/Gj4nQbjbEAATeoH.jpg", 0.55, 0.5, "Tags + Visualization"],
            ["https://pbs.twimg.com/media/GkbtX0GaoAMlUZt.jpg", 0.45, 0.45, "Tags + Visualization"]
        ],
        inputs=[image_input, gen_threshold, char_threshold, output_mode],
        outputs=[output_tags, output_visualization],
        fn=predict,
        cache_examples=False
    )

    predict_button.click(
        fn=predict,
        inputs=[image_input, gen_threshold, char_threshold, output_mode],
        outputs=[output_tags, output_visualization]
    )

if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN environment variable not set.")
    demo.launch(share=True) 