import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import onnxruntime as ort
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    artist: list[np.int64]
    character: list[np.int64]
    copyright: list[np.int64]
    meta: list[np.int64]

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    # パディングして正方形にする
    width, height = image.size
    if width == height:
        return image
    
    new_size = max(width, height)
    new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255))
    
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_image.paste(image, paste_position)
    
    return new_image

def normalize_tag(tag):
    # タグの正規化
    tag = tag.lower().strip()
    tag = tag.replace("_", " ")
    return tag

def read_tags_from_file(image_path, remove_special_prefix="remove"):
    # 画像ファイルからタグを読み込む
    tags = []
    
    # 画像ファイルと同じ名前のテキストファイルを探す
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    
    # URLの場合はダウンロードを試みる
    if image_path.startswith(("http://", "https://")):
        try:
            import requests
            txt_url = os.path.splitext(image_path)[0] + ".txt"
            response = requests.get(txt_url, timeout=5)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    tags.extend(line.split(","))
            else:
                print(f"Warning: Tag file not found at {txt_url}")
                return []
        except Exception as e:
            print(f"Error downloading tags: {e}")
            return []
    # ローカルファイルの場合
    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                tags.extend(line.split(","))
    else:
        return []
    
    # タグの前処理
    processed_tags = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
            
        # 特殊プレフィックスの処理
        if remove_special_prefix and tag.startswith(remove_special_prefix + ":"):
            continue
            
        processed_tags.append(normalize_tag(tag))
    
    print(f"読み込まれたタグ: {len(processed_tags)}個")
    return processed_tags

def get_tags(probs, labels, gen_threshold, char_threshold):
    # 確率からタグを取得
    result = {
        "rating": [],
        "general": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "meta": []
    }
    
    # レーティング（最大値を選択）
    if len(labels.rating) > 0:
        rating_probs = probs[labels.rating]
        rating_idx = np.argmax(rating_probs)
        rating_name = labels.names[labels.rating[rating_idx]]
        rating_conf = float(rating_probs[rating_idx])
        result["rating"].append((rating_name, rating_conf))
    
    # 一般タグ
    for i in labels.general:
        if probs[i] >= gen_threshold:
            result["general"].append((labels.names[i], float(probs[i])))
    
    # キャラクタータグ
    for i in labels.character:
        if probs[i] >= char_threshold:
            result["character"].append((labels.names[i], float(probs[i])))
    
    # 著作権タグ
    for i in labels.copyright:
        if probs[i] >= char_threshold:
            result["copyright"].append((labels.names[i], float(probs[i])))
    
    # アーティストタグ
    for i in labels.artist:
        if probs[i] >= char_threshold:
            result["artist"].append((labels.names[i], float(probs[i])))
    
    # メタタグ
    for i in labels.meta:
        if probs[i] >= gen_threshold:
            result["meta"].append((labels.names[i], float(probs[i])))
    
    # 確率の降順でソート
    for k in result:
        result[k] = sorted(result[k], key=lambda x: x[1], reverse=True)
    
    return result

def visualize_predictions(image, tags, predictions, threshold=0.35, output_path=None, max_tags_per_category=None):
    # Create figure with subplots
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Set font that supports English characters
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Filter out unwanted meta tags
    filtered_meta = []
    excluded_meta_patterns = ['id', 'commentary']
    
    for tag, prob in predictions["meta"]:
        # Skip tags containing 'id' or 'commentary'
        if not any(pattern in tag.lower() for pattern in excluded_meta_patterns):
            filtered_meta.append((tag, prob))
    
    # Replace meta tags with filtered list
    predictions["meta"] = filtered_meta

    # メタタグのコピーを保存（フィルタリング前）
    original_meta = predictions["meta"].copy()
    
    # 固定サイズのプロットを作成（1000x1200px）
    # 20インチ x 12インチ @ 100dpi = 1000x1200px
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    
    # Left side: Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    ax_img.set_title("Original Image")
    ax_img.axis('off')
    
    # Right side: All tags in one plot
    ax_tags = fig.add_subplot(gs[0, 1])
    
    # Prepare data for all categories
    all_tags = []
    all_probs = []
    all_colors = []
    all_categories = []
    
    # Color mapping for categories
    color_map = {
        'rating': 'red',
        'character': 'blue',
        'copyright': 'purple',
        'artist': 'orange',
        'general': 'green',
        'meta': 'gray'
    }
    
    # Add rating tags (all of them)
    for tag, prob in predictions["rating"]:
        all_tags.append(f"[R] {tag}")
        all_probs.append(prob)
        all_colors.append(color_map['rating'])
        all_categories.append('rating')
    
    # Add character tags (all above threshold)
    for tag, prob in predictions["character"]:
        all_tags.append(f"[C] {tag}")
        all_probs.append(prob)
        all_colors.append(color_map['character'])
        all_categories.append('character')
    
    # Add copyright tags (all above threshold)
    for tag, prob in predictions["copyright"]:
        all_tags.append(f"[©] {tag}")
        all_probs.append(prob)
        all_colors.append(color_map['copyright'])
        all_categories.append('copyright')
    
    # Add artist tags (all above threshold)
    for tag, prob in predictions["artist"]:
        all_tags.append(f"[A] {tag}")
        all_probs.append(prob)
        all_colors.append(color_map['artist'])
        all_categories.append('artist')
    
    # Add general tags (all above threshold)
    for tag, prob in predictions["general"]:
        all_tags.append(f"[G] {tag}")
        all_probs.append(prob)
        all_colors.append(color_map['general'])
        all_categories.append('general')
    
    # Add meta tags (filtered, all above threshold)
    for tag, prob in predictions["meta"]:
        all_tags.append(f"[M] {tag}")
        all_probs.append(prob)
        all_colors.append(color_map['meta'])
        all_categories.append('meta')
    
    # Sort by probability (descending)
    sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)
    all_tags = [all_tags[i] for i in sorted_indices]
    all_probs = [all_probs[i] for i in sorted_indices]
    all_colors = [all_colors[i] for i in sorted_indices]
    all_categories = [all_categories[i] for i in sorted_indices]
    
    # Reverse lists for bottom-to-top display
    all_tags.reverse()
    all_probs.reverse()
    all_colors.reverse()
    all_categories.reverse()
    
    # タグの数に応じてバーの高さを調整
    num_tags = len(all_tags)
    if num_tags > 0:
        # 固定サイズのプロット内にすべてのタグを表示するための調整
        bar_height = 1.2  # デフォルトの高さ
        if num_tags > 30:  # タグが多い場合は高さを調整
            bar_height = 1.2 * (30 / num_tags)
        
        # Y位置を計算（均等に分布）
        y_positions = np.linspace(0, num_tags-1, num_tags)
        
        # Plot horizontal bars
        bars = ax_tags.barh(y_positions, all_probs, height=bar_height, color=all_colors)
        
        # Set y-ticks and labels
        ax_tags.set_yticks(y_positions)
        ax_tags.set_yticklabels(all_tags)
        
        # フォントサイズを調整（タグが多い場合は小さく）
        fontsize = 10
        if num_tags > 40:
            fontsize = 8
        elif num_tags > 60:
            fontsize = 6
        
        # ラベルのフォントサイズを設定
        for label in ax_tags.get_yticklabels():
            label.set_fontsize(fontsize)
        
        # Add probability values at the end of each bar
        for i, (bar, prob) in enumerate(zip(bars, all_probs)):
            ax_tags.text(
                min(prob + 0.02, 0.98),  # Slightly offset from bar end
                y_positions[i],          # Y position
                f"{prob:.3f}",           # Probability text
                va='center',             # Vertical alignment
                fontsize=fontsize        # Adjusted font size
            )
    
    # Set x-axis limit
    ax_tags.set_xlim(0, 1)
    ax_tags.set_title(f"Tags (threshold={threshold})")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['rating'], label='Rating'),
        Patch(facecolor=color_map['character'], label='Character'),
        Patch(facecolor=color_map['copyright'], label='Copyright'),
        Patch(facecolor=color_map['artist'], label='Artist'),
        Patch(facecolor=color_map['general'], label='General'),
        Patch(facecolor=color_map['meta'], label='Meta')
    ]
    ax_tags.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Add original tags as text at the bottom of the figure
    if tags:
        tag_text = "Original Tags: " + ", ".join(tags[:20])
        fig.text(0.5, 0.01, tag_text, wrap=True, 
                horizontalalignment='center', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    
    # Save figure
    if output_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Add extension if needed
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path += ".png"
            
        plt.savefig(output_path, dpi=100)
        print(f"Result saved: {output_path}")
    
    plt.close(fig)
    
    # Also create a text file with all predictions (including filtered meta tags)
    if output_path:
        txt_path = os.path.splitext(output_path)[0] + ".txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=== Rating Tags ===\n")
            for tag, prob in predictions["rating"]:
                f.write(f"{tag}: {prob:.3f}\n")
                
            f.write("\n=== Character Tags ===\n")
            for tag, prob in predictions["character"]:
                f.write(f"{tag}: {prob:.3f}\n")
                
            f.write("\n=== Copyright Tags ===\n")
            for tag, prob in predictions["copyright"]:
                f.write(f"{tag}: {prob:.3f}\n")
                
            f.write("\n=== Artist Tags ===\n")
            for tag, prob in predictions["artist"]:
                f.write(f"{tag}: {prob:.3f}\n")
                
            f.write("\n=== General Tags ===\n")
            for tag, prob in predictions["general"]:
                f.write(f"{tag}: {prob:.3f}\n")
                
            # Save all meta tags to text file, but mark filtered ones
            f.write("\n=== Meta Tags ===\n")
            for tag, prob in predictions["meta"]:
                f.write(f"{tag}: {prob:.3f}\n")
            
            # Add a section for filtered meta tags
            filtered_count = 0
            f.write("\n=== Filtered Meta Tags (not displayed) ===\n")
            for tag, prob in original_meta:
                if any(pattern in tag.lower() for pattern in excluded_meta_patterns):
                    f.write(f"{tag}: {prob:.3f}\n")
                    filtered_count += 1
            
            if filtered_count == 0:
                f.write("None\n")
        
        print(f"Detailed results saved: {txt_path}")
    
    # Return a simple PIL image with just the original image
    return image

def load_tag_mapping(mapping_path):
    # タグマッピングの読み込み
    with open(mapping_path, 'r', encoding='utf-8') as f:
        tag_mapping = json.load(f)
    
    # インデックスを整数に変換
    tag_mapping = {int(k): v for k, v in tag_mapping.items()}
    
    # タグとカテゴリの辞書を作成
    idx_to_tag = {}
    tag_to_category = {}
    
    for idx, data in tag_mapping.items():
        tag = data['tag']
        category = data['category']
        idx_to_tag[idx] = tag
        tag_to_category[tag] = category
    
    # LabelDataの作成
    names = [None] * (max(idx_to_tag.keys()) + 1)
    rating = []
    general = []
    artist = []
    character = []
    copyright = []
    meta = []
    
    for idx, tag in idx_to_tag.items():
        names[idx] = tag
        category = tag_to_category[tag]
        
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
    
    label_data = LabelData(
        names=names,
        rating=np.array(rating, dtype=np.int64),
        general=np.array(general, dtype=np.int64),
        artist=np.array(artist, dtype=np.int64),
        character=np.array(character, dtype=np.int64),
        copyright=np.array(copyright, dtype=np.int64),
        meta=np.array(meta, dtype=np.int64)
    )
    
    return label_data, idx_to_tag, tag_to_category

def preprocess_image(image_path, target_size=(448, 448)):
    # 画像の前処理
    if image_path.startswith(("http://", "https://")):
        # URLの場合はダウンロード
        import requests
        from io import BytesIO
        response = requests.get(image_path, timeout=10)
        image = Image.open(BytesIO(response.content))
    else:
        # ローカルファイルの場合
        image = Image.open(image_path)
    
    # RGB形式に変換
    image = pil_ensure_rgb(image)
    
    # 正方形にパディング
    image = pil_pad_square(image)
    
    # リサイズ
    image_resized = image.resize(target_size, Image.LANCZOS)
    
    # NumPy配列に変換 - float32型を明示的に指定
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    
    # チャンネルを最初に移動 (HWC -> CHW)
    img_array = img_array.transpose(2, 0, 1)
    
    # RGB -> BGR変換（モデルがBGRを期待している場合）
    img_array = img_array[::-1, :, :]
    
    # 正規化 (mean=0.5, std=0.5)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # バッチ次元を追加
    img_array = np.expand_dims(img_array, axis=0)
    
    # データ型を確認して出力
    print(f"前処理後の入力データ型: {img_array.dtype}")
    
    return image, img_array

def predict_with_onnx(image_path, model_path, tag_mapping_path, gen_threshold=0.35, char_threshold=0.45, output_path=None, use_gpu=False):
    # ONNXモデルでの予測
    print(f"Loading model: {model_path}")
    
    # 利用可能なプロバイダーを確認
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    # GPUの使用設定
    if use_gpu:
        try:
            # セッションオプションを設定
            session_options = ort.SessionOptions()
            
            # 利用可能なGPUプロバイダーを選択
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            elif 'DmlExecutionProvider' in available_providers:
                providers.append('DmlExecutionProvider')
            elif 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
            
            # CPUはフォールバック用に常に追加
            providers.append('CPUExecutionProvider')
            
            if not providers or len(providers) == 1:  # CPUしかない場合
                print("No GPU providers available, using CPU only")
                session = ort.InferenceSession(model_path)
            else:
                print(f"Using providers: {providers}")
                session = ort.InferenceSession(model_path, providers=providers)
                print(f"Active provider: {session.get_providers()[0]}")
        except Exception as e:
            print(f"GPU acceleration failed: {e}")
            print("Falling back to CPU")
            session = ort.InferenceSession(model_path)
    else:
        # CPUのみを使用
        session = ort.InferenceSession(model_path)
        print("Using CPU for inference")
    
    # タグマッピングの読み込み
    print(f"Loading tag mapping: {tag_mapping_path}")
    labels, idx_to_tag, tag_to_category = load_tag_mapping(tag_mapping_path)
    
    # 画像の前処理
    print(f"Processing image: {image_path}")
    original_image, input_data = preprocess_image(image_path)
    
    # 元のタグを読み込む
    original_tags = read_tags_from_file(image_path)
    
    # 推論実行
    print("Running inference...")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 推論時間の計測
    import time
    start_time = time.time()
    outputs = session.run([output_name], {input_name: input_data})[0]
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.3f} seconds")
    
    # シグモイド関数を適用して確率に変換
    outputs = 1 / (1 + np.exp(-outputs))
    
    # 確率からタグを取得
    predictions = get_tags(outputs[0], labels, gen_threshold, char_threshold)
    
    # メタタグのコピーを保存（フィルタリング前）
    original_meta = predictions["meta"].copy()
    
    # 結果の表示
    print("--------")
    print("Ratings:")
    for tag, conf in predictions["rating"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"Character tags (threshold={char_threshold}):")
    for tag, conf in predictions["character"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"Copyright tags (threshold={char_threshold}):")
    for tag, conf in predictions["copyright"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"Artist tags (threshold={char_threshold}):")
    for tag, conf in predictions["artist"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"General tags (threshold={gen_threshold}):")
    for tag, conf in predictions["general"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"Meta tags (threshold={gen_threshold}):")
    # Filter out unwanted meta tags for display
    filtered_meta = []
    excluded_meta_patterns = ['id', 'commentary']
    
    for tag, conf in predictions["meta"]:
        if not any(pattern in tag.lower() for pattern in excluded_meta_patterns):
            print(f"  {tag}: {conf:.3f}")
            filtered_meta.append((tag, conf))
        else:
            print(f"  [FILTERED] {tag}: {conf:.3f}")
    
    # 結果の可視化
    if output_path is None and image_path.startswith(("http://", "https://")):
        # URLの場合はファイル名を抽出
        import urllib.parse
        filename = os.path.basename(urllib.parse.urlparse(image_path).path)
        output_path = f"result_{filename}.png"
    elif output_path is None:
        # ローカルファイルの場合
        output_path = f"result_{os.path.basename(image_path)}"
        # 拡張子がない場合は追加
        if not output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path += ".png"
    
    # Pass original_meta to visualize_predictions
    result_image = visualize_predictions(
        original_image, 
        original_tags, 
        predictions, 
        threshold=gen_threshold, 
        output_path=output_path
    )
    
    return predictions, result_image

def debug_preprocessing(image_path):
    # デバッグ用の関数
    
    # 元の画像を読み込む
    if image_path.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        response = requests.get(image_path, timeout=10)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # 前処理を適用
    original_image, processed_array = preprocess_image(image_path)
    
    # 処理後の配列を可視化用に変換
    # 正規化を元に戻す
    mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
    img_denorm = processed_array[0] * std + mean
    
    # BGR -> RGB変換（前処理でBGRに変換した場合）
    img_denorm = img_denorm[::-1, :, :]
    
    # [0,1]の範囲に収める
    img_denorm = np.clip(img_denorm, 0, 1)
    
    # CHW -> HWC
    img_denorm = img_denorm.transpose(1, 2, 0)
    
    # 表示
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(img_denorm)
    plt.savefig("preprocessing_debug.png")
    plt.close()
    
    print("前処理のデバッグ画像を保存しました: preprocessing_debug.png")

def main():
    parser = argparse.ArgumentParser(description='ONNXモデルを使用した画像タグ予測')
    parser.add_argument('--image', type=str, required=True, help='予測する画像のパスまたはURL')
    parser.add_argument('--model', type=str, required=True, help='ONNXモデルのパス')
    parser.add_argument('--tag_mapping', type=str, required=True, help='タグマッピングJSONファイルのパス')
    parser.add_argument('--output', type=str, default=None, help='結果画像の出力パス')
    parser.add_argument('--gen_threshold', type=float, default=0.35, help='一般タグの閾値')
    parser.add_argument('--char_threshold', type=float, default=0.45, help='キャラクタータグの閾値')
    parser.add_argument('--gpu', action='store_true', help='GPUを使用して推論を実行')
    
    args = parser.parse_args()
    
    predict_with_onnx(
        args.image,
        args.model,
        args.tag_mapping,
        args.gen_threshold,
        args.char_threshold,
        args.output,
        use_gpu=args.gpu
    )

# GPUが利用可能かチェックする関数
def check_gpu_availability():
    try:
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("CUDA is available for ONNX Runtime")
            return True
        else:
            print("CUDA is not available for ONNX Runtime")
            print("Available providers:", providers)
            return False
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False

if __name__ == "__main__":
    # GPUの利用可能性をチェック
    check_gpu_availability()
    main() 