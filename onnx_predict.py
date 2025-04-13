import os
import time
import json
import argparse
import numpy as np
import cv2
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
    """
    画像を確実にRGB形式に変換する
    
    Args:
        image: PIL Image オブジェクト
    Returns:
        RGB形式の PIL Image
    """
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    
    # RGBAの場合は白背景でRGBに変換
    if image.mode == "RGBA":
        # 白背景のキャンバスを作成
        background = Image.new("RGB", image.size, (255, 255, 255))
        # アルファチャンネルを考慮して合成
        background.paste(image, mask=image.split()[3])  # 3はアルファチャンネル
        image = background
    
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
    # 括弧をエスケープするケースも正規化
    tag = tag.replace('\\(', '(').replace('\\)', ')')
    return tag

def standardize_tag_format(tag):
    """
    タグ形式を標準化する (エクスポート用)
    aaaa bbbb (cccc) → aaaa_bbbb_(cccc)
    
    Args:
        tag: 変換前のタグ
    Returns:
        変換後のタグ
    """
    # スペースをアンダースコアに変換
    # エスケープされた括弧を通常の括弧に変換
    tag = tag.replace('\\(', '(').replace('\\)', ')')
    tag = tag.replace(' ', '_')
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
                    tags.extend([t.strip() for t in line.split(",")])
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
                tags.extend([t.strip() for t in line.split(",")])
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

def visualize_predictions(image, tags, predictions, threshold=0.45, output_path=None, max_tags_per_category=None):
    # Create figure with subplots
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Set font that supports English characters
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Filter out unwanted meta tags
    filtered_meta = []
    excluded_meta_patterns = ['id', 'commentary', 'request']
    
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
        bar_height = 0.8  # デフォルトの高さ
        if num_tags > 30:  # タグが多い場合は高さを調整
            bar_height = 0.8 * (30 / num_tags)
        
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
    # if tags:
    #     tag_text = "Original Tags: " + ", ".join(tags[:20])
    #     fig.text(0.5, 0.01, tag_text, wrap=True, 
    #             horizontalalignment='center', fontsize=10)
    
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
    image_resized = image.resize(target_size, Image.BICUBIC)
    
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

def convert_tag_format(tag: str) -> str:
    """
    タグの形式を変換する
    aaaa_bbbb_(cccc) -> aaaa bbbb \(cccc\)
    
    Args:
        tag: 変換前のタグ
    Returns:
        変換後のタグ
    """
    # 括弧をエスケープする
    tag = tag.replace('(', '\\(').replace(')', '\\)')
    
    # アンダースコアをスペースに変換
    tag = tag.replace('_', ' ')
    
    return tag

def save_tags_as_csv(predictions, output_path, threshold=0.45, mode="overwrite", remove_threshold=None):
    """
    予測されたタグをカンマ区切りのテキストファイルとして保存する
    
    Args:
        predictions: 予測結果の辞書
        output_path: 出力ファイルパス
        threshold: タグの閾値
        mode: 保存モード ("overwrite"=上書き, "add"=既存タグに追加)
        remove_threshold: 既存タグを除去する閾値（Noneの場合は除去しない）
    """
    # すべてのタグを収集
    new_tags = []
    new_tags_probs = {}  # タグと確率のマッピング
    
    # レーティングタグ（最も確率の高いもののみ）
    if predictions["rating"]:
        tag, prob = predictions["rating"][0]
        new_tags.append(tag)
        new_tags_probs[normalize_tag(tag)] = prob
    
    # 他のカテゴリのタグを追加
    for category in ["character", "copyright", "artist", "general", "meta"]:
        for tag, prob in predictions[category]:
            # メタタグの中でidやcommentaryを含むものは除外
            if category == "meta" and any(pattern in tag.lower() for pattern in ['id', 'commentary']):
                continue
            new_tags.append(tag)
            new_tags_probs[normalize_tag(tag)] = prob
    
    # 既存のタグを読み込む (addモードの場合)
    existing_tags = []
    if mode == "add" and os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    existing_tags.extend([t.strip() for t in line.split(",")])
            
            # 既存タグを正規化して重複判定に使用するセットを作成
            normalized_existing_tags = {normalize_tag(tag) for tag in existing_tags if tag.strip()}
            print(f"既存のタグ: {len(normalized_existing_tags)}個")
            
            # remove_thresholdが設定されている場合、低確率タグを除去
            if remove_threshold is not None:
                filtered_existing_tags = []
                removed_tags = []
                for tag in existing_tags:
                    normalized_tag = normalize_tag(tag)
                    # 新しい予測に含まれるタグの場合
                    if normalized_tag in new_tags_probs:
                        if new_tags_probs[normalized_tag] >= remove_threshold:
                            filtered_existing_tags.append(tag)
                        else:
                            removed_tags.append(tag)
                    else:
                        # 予測に含まれないタグは保持
                        filtered_existing_tags.append(tag)
                
                if removed_tags:
                    print(f"除去されたタグ（閾値{remove_threshold}未満）: {len(removed_tags)}個")
                    for tag in removed_tags:
                        prob = new_tags_probs.get(normalize_tag(tag), 0.0)
                        print(f"  {tag}: {prob:.3f}")
                
                existing_tags = filtered_existing_tags
                normalized_existing_tags = {normalize_tag(tag) for tag in existing_tags if tag.strip()}
            
            # 新しいタグのうち、正規化形式で既存タグと重複しないものだけを追加
            unique_new_tags = []
            for tag in new_tags:
                normalized_tag = normalize_tag(tag)
                if normalized_tag not in normalized_existing_tags:
                    unique_new_tags.append(tag)
                    normalized_existing_tags.add(normalized_tag)
            
            # 既存のタグに新しいタグを追加
            all_tags = existing_tags + unique_new_tags
            print(f"追加された新しいタグ: {len(unique_new_tags)}個")
        except Exception as e:
            print(f"既存タグの読み込みエラー: {e}。新しいタグのみで上書きします。")
            all_tags = new_tags
    else:
        # 上書きモードまたはファイルが存在しない場合は新しいタグのみ
        all_tags = new_tags
    
    # 出力前に全タグを標準形式にフォーマット
    formatted_tags = [standardize_tag_format(tag) for tag in all_tags]
    
    # カンマ区切りで保存（カンマの後にスペースを入れる）
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(", ".join(formatted_tags))
    
    print(f"タグをカンマ区切りで保存しました: {output_path} (モード: {mode})")
    return formatted_tags

def extract_frames_from_video(video_path, num_frames):
    """
    動画ファイルから等間隔でフレームを抽出する
    
    Args:
        video_path: 動画ファイルのパス
        num_frames: 抽出するフレーム数
    
    Returns:
        frames: 抽出されたフレームのリスト（PILイメージ）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return []
    
    # 動画の情報を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        print(f"Error: Invalid video file: {video_path}")
        cap.release()
        return []
    
    print(f"Video info: {video_path}, Total frames: {total_frames}, FPS: {fps}")
    
    # 抽出するフレームの間隔を計算
    if num_frames > total_frames:
        num_frames = total_frames
        
    interval = max(1, total_frames // num_frames)
    frame_indices = [i * interval for i in range(num_frames)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGRからRGBに変換してPIL Imageに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        else:
            print(f"Warning: Failed to read frame at index {idx}")
    
    cap.release()
    return frames

def predict_with_onnx(
        image_path, model_path, tag_mapping_path, 
        gen_threshold=0.45, char_threshold=0.45, output_path=None, 
        use_gpu=False, output_mode="visualization", tag_mode="add", 
        remove_threshold=None, batch_size=1):
    # ONNXモデルでの予測
    print(f"Loading ONNX model: {model_path}")
    
    # モデルがFP16かどうかを確認
    is_fp16_model = False
    try:
        import onnx
        model = onnx.load(model_path)
        # モデルの最初の入力のデータ型をチェック
        for tensor in model.graph.initializer:
            if tensor.data_type == 10:  # FLOAT16
                is_fp16_model = True
                break
        print(f"Model is {'FP16' if is_fp16_model else 'FP32'}")
    except Exception as e:
        print(f"Failed to check model precision: {e}")
    
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
                # FP16モデルの場合、CUDA設定を最適化
                if is_fp16_model:
                    cuda_provider_options = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }
                    providers.append(('CUDAExecutionProvider', cuda_provider_options))
                else:
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
    
    # 入力データ型を確認
    expected_input_type = session.get_inputs()[0].type
    print(f"Model expects input type: {expected_input_type}")
    
    # 入力データ型を調整（FP16モデルでも入力はFP32のまま）
    # FP16モデルを作成する際に keep_io_types=True を使用した場合、
    # 入力はFP32のままにする必要があります
    if "float16" in expected_input_type:
        input_data = input_data.astype(np.float16)
        print("Using FP16 input")
    else:
        input_data = input_data.astype(np.float32)
        print("Using FP32 input")
    
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
    
    # NaNチェック
    if np.isnan(outputs).any():
        print("警告: 推論結果にNaNが含まれています")
        # NaNを0に置き換え
        outputs = np.nan_to_num(outputs, nan=0.0)
        print("NaNを0に置き換えました")
    
    # 無限大チェック
    if np.isinf(outputs).any():
        print("警告: 推論結果に無限大が含まれています")
        # 無限大を大きな値に置き換え
        outputs = np.nan_to_num(outputs, posinf=100.0, neginf=-100.0)
        print("無限大を有限値に置き換えました")
    
    # シグモイド関数を適用して確率に変換（数値安定性を考慮）
    def stable_sigmoid(x):
        # 数値安定性のためのシグモイド実装
        # 大きな負の値に対する対策
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    outputs = stable_sigmoid(outputs)
    
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
    
    # 出力パスの設定
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    if output_mode == "visualization":
        # 可視化モードの場合は prediction ディレクトリに保存
        if output_path:
            prediction_dir = output_path
        else:
            prediction_dir = "prediction"
        os.makedirs(prediction_dir, exist_ok=True)
        viz_output_path = os.path.join(prediction_dir, f"{base_filename}.png")
    else:
        # タグのみの出力の場合
        if output_path:
            # 出力ディレクトリが指定されている場合はそこに保存
            tag_output_dir = output_path
            os.makedirs(tag_output_dir, exist_ok=True)
            tag_output_path = os.path.join(tag_output_dir, f"{base_filename}.txt")
        else:
            # 出力ディレクトリが指定されていない場合
            # バッチ推論時は画像と同じディレクトリ、単一推論時はカレントディレクトリ
            if batch_size > 1:
                tag_output_dir = os.path.dirname(image_path)
            else:
                tag_output_dir = "."
            tag_output_path = os.path.join(tag_output_dir, f"{base_filename}.txt")

    # 結果の保存
    if output_mode == "visualization":
        original_tags = read_tags_from_file(image_path)
        visualize_predictions(
            original_image,
            original_tags,
            predictions,
            threshold=gen_threshold,
            output_path=viz_output_path
        )
    else:
        save_tags_as_csv(predictions, tag_output_path, threshold=gen_threshold, mode=tag_mode, remove_threshold=remove_threshold)
    
    return predictions, output_path

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

def batch_predict(dirs, model_path, tag_mapping_path, gen_threshold=0.45, char_threshold=0.45, video_frames=3,
                 use_gpu=False, output_mode="visualization", recursive=False, batch_size=1, tag_mode="add", 
                 remove_threshold=None):  # remove_thresholdパラメータを追加
    """
    複数のディレクトリ内の画像に対してバッチ推論を実行する
    
    Args:
        dirs: 入力画像ディレクトリのパスのリスト
        model_path: ONNXモデルのパス
        tag_mapping_path: タグマッピングJSONファイルのパス
        gen_threshold: 一般タグの閾値
        char_threshold: キャラクタータグの閾値
        video_frames: 動画から抽出するフレーム数
        use_gpu: GPUを使用するかどうか
        output_mode: 出力モード
        recursive: サブディレクトリも処理するかどうか
        batch_size: バッチサイズ（デフォルト: 1）
        tag_mode: タグ保存モード ("overwrite"=上書き, "add"=既存タグに追加)
        remove_threshold: 既存タグを除去する閾値（Noneの場合は除去しない）
    """
    print(f"モデルを読み込み中: {model_path}")
    
    # モデルがFP16かどうかを確認
    is_fp16_model = False
    try:
        import onnx
        model = onnx.load(model_path)
        for tensor in model.graph.initializer:
            if tensor.data_type == 10:  # FLOAT16
                is_fp16_model = True
                break
        print(f"モデルは {'FP16' if is_fp16_model else 'FP32'} です")
    except Exception as e:
        print(f"モデル精度の確認に失敗しました: {e}")
    
    # 利用可能なプロバイダーを確認
    available_providers = ort.get_available_providers()
    
    # GPUの使用設定とセッションの作成
    if use_gpu:
        try:
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                if is_fp16_model:
                    cuda_provider_options = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }
                    providers.append(('CUDAExecutionProvider', cuda_provider_options))
                else:
                    providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            session = ort.InferenceSession(model_path, providers=providers)
            print(f"使用するプロバイダー: {session.get_providers()}")
        except Exception as e:
            print(f"GPU高速化に失敗しました: {e}")
            print("CPUにフォールバックします")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    else:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # タグマッピングの読み込み
    print(f"タグマッピングを読み込み中: {tag_mapping_path}")
    labels, idx_to_tag, tag_to_category = load_tag_mapping(tag_mapping_path)
    
    # 全ディレクトリから画像ファイルと動画ファイルのリストを取得
    image_files = []
    video_files = []
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            print(f"警告: ディレクトリが存在しません: {dir_path}")
            continue
            
        dir_images = []
        dir_videos = []
        
        if recursive:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        dir_images.append(os.path.join(root, file))
                    elif file.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv')):
                        dir_videos.append(os.path.join(root, file))
        else:
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    dir_images.append(os.path.join(dir_path, file))
                elif file.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv')):
                    dir_videos.append(os.path.join(dir_path, file))
        
        if dir_videos:
            print(f"ディレクトリ '{dir_path}' から {len(dir_videos)} 個の動画を見つけました")
            video_files.extend(dir_videos)
            
        if dir_images:
            print(f"ディレクトリ '{dir_path}' から {len(dir_images)} 個の画像を見つけました")
            image_files.extend(dir_images)
    
    if not image_files and not video_files:
        print("処理するファイルが見つかりませんでした")
        return
    
    total_images = len(image_files)
    total_videos = len(video_files)
    print(f"合計処理する画像ファイル数: {total_images}")
    print(f"合計処理する動画ファイル数: {total_videos}")
    print(f"画像処理のバッチサイズ: {batch_size}")
    
    # 画像ファイルのバッチ処理
    if total_images > 0:
        print("\n=== 画像ファイルの処理を開始 ===")
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            current_batch_size = batch_end - batch_start
            print(f"\nバッチ処理中: {batch_start+1}～{batch_end}/{total_images}")
            
            try:
                # バッチ用の入力データを準備
                batch_inputs = []
                batch_images = []
                batch_paths = []
                
                # バッチ内の各画像を前処理
                for i in range(batch_start, batch_end):
                    image_path = image_files[i]
                    try:
                        original_image, input_data = preprocess_image(image_path)
                        batch_inputs.append(input_data[0])  # バッチ次元を除去
                        batch_images.append(original_image)
                        batch_paths.append(image_path)
                    except Exception as e:
                        print(f"画像の前処理でエラー ({image_path}): {e}")
                        continue
                
                if not batch_inputs:
                    continue
                
                # バッチ入力の作成
                batch_input = np.stack(batch_inputs, axis=0)
                
                # 入力データ型の調整
                expected_input_type = session.get_inputs()[0].type
                if "float16" in expected_input_type:
                    batch_input = batch_input.astype(np.float16)
                else:
                    batch_input = batch_input.astype(np.float32)
                
                # バッチ推論の実行
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                
                start_time = time.time()
                outputs = session.run([output_name], {input_name: batch_input})[0]
                inference_time = time.time() - start_time
                print(f"バッチ推論が {inference_time:.3f} 秒で完了しました")
                print(f"画像あたりの平均推論時間: {inference_time/current_batch_size:.3f} 秒")
                
                # バッチ内の各画像の結果を処理
                for idx, (original_image, image_path, output) in enumerate(zip(batch_images, batch_paths, outputs)):
                    try:
                        # 出力の後処理
                        output = np.nan_to_num(output, nan=0.0, posinf=100.0, neginf=-100.0)
                        output = 1 / (1 + np.exp(-output))  # シグモイド関数
                        
                        # タグの取得
                        predictions = get_tags(output, labels, gen_threshold, char_threshold)
                        
                        # 出力パスの設定
                        base_filename = os.path.splitext(os.path.basename(image_path))[0]
                        
                        if output_mode == "visualization":
                            # 可視化モードの場合は prediction ディレクトリに保存
                            prediction_dir = "prediction"
                            os.makedirs(prediction_dir, exist_ok=True)
                            output_path = os.path.join(prediction_dir, f"{base_filename}.png")
                        else:
                            # タグのみの出力の場合、画像と同じディレクトリに保存
                            tag_output_dir = os.path.dirname(image_path)
                            output_path = os.path.join(tag_output_dir, f"{base_filename}.txt")
                        
                        # 結果の保存
                        if output_mode == "visualization":
                            original_tags = read_tags_from_file(image_path)
                            visualize_predictions(
                                original_image,
                                original_tags,
                                predictions,
                                threshold=gen_threshold,
                                output_path=output_path
                            )
                        else:
                            save_tags_as_csv(predictions, output_path, threshold=gen_threshold, 
                                             mode=tag_mode, remove_threshold=remove_threshold)
                        
                        print(f"処理完了 [{batch_start+idx+1}/{total_images}]: {image_path}")
                        
                    except Exception as e:
                        print(f"結果の保存でエラー ({image_path}): {e}")
                        import traceback
                        print(traceback.format_exc())
            
            except Exception as e:
                print(f"バッチ処理でエラー: {e}")
                import traceback
                print(traceback.format_exc())
    
    # 動画ファイルの処理
    if total_videos > 0:
        print("\n=== 動画ファイルの処理を開始 ===")
        for i, video_path in enumerate(video_files):
            try:
                print(f"\n動画処理中 [{i+1}/{total_videos}]: {video_path}")
                
                # 動画のフレームを抽出
                frames = extract_frames_from_video(video_path, video_frames)
                if not frames:
                    print(f"警告: 動画からフレームを抽出できませんでした: {video_path}")
                    continue
                    
                print(f"{len(frames)}フレームを抽出しました")
                
                # 各フレームの予測結果を保存するリスト
                all_frame_predictions = []
                
                # フレームごとの処理
                for j, frame in enumerate(frames):
                    try:
                        print(f"フレーム {j+1}/{len(frames)} の処理中...")
                        
                        # フレームを一時的にファイルとして保存
                        temp_frame_path = f"temp_frame_{j}.png"
                        frame.save(temp_frame_path)
                        
                        # preprocess_image関数を使用して前処理
                        try:
                            original_frame, input_data = preprocess_image(temp_frame_path)
                            
                            # 入力データ型の調整
                            expected_input_type = session.get_inputs()[0].type
                            if "float16" in expected_input_type:
                                input_data = input_data.astype(np.float16)
                            else:
                                input_data = input_data.astype(np.float32)
                            
                            # 推論実行
                            input_name = session.get_inputs()[0].name
                            output_name = session.get_outputs()[0].name
                            
                            # 推論時間の計測
                            start_time = time.time()
                            outputs = session.run([output_name], {input_name: input_data})[0]
                            inference_time = time.time() - start_time
                            print(f"フレーム {j+1} の推論が {inference_time:.3f} 秒で完了しました")
                            
                            # 出力の後処理
                            outputs = np.nan_to_num(outputs[0], nan=0.0, posinf=100.0, neginf=-100.0)
                            
                            # シグモイド関数を適用して確率に変換（数値安定性を考慮）
                            def stable_sigmoid(x):
                                return np.where(
                                    x >= 0,
                                    1 / (1 + np.exp(-x)),
                                    np.exp(x) / (1 + np.exp(x))
                                )
                            
                            outputs = stable_sigmoid(outputs)
                            
                            # タグの取得
                            frame_predictions = get_tags(outputs, labels, gen_threshold, char_threshold)
                            all_frame_predictions.append(frame_predictions)
                            
                        finally:
                            # 一時ファイルを削除
                            if os.path.exists(temp_frame_path):
                                os.remove(temp_frame_path)
                    
                    except Exception as e:
                        print(f"フレーム処理でエラー: {e}")
                        import traceback
                        print(traceback.format_exc())
                
                if not all_frame_predictions:
                    print(f"警告: 動画の全フレームの処理に失敗しました: {video_path}")
                    continue
                
                # 全フレームの予測結果を統合
                combined_predictions = combine_frame_predictions(all_frame_predictions, gen_threshold, char_threshold)
                
                # 出力パスの設定
                base_filename = os.path.splitext(os.path.basename(video_path))[0]
                
                if output_mode == "visualization":
                    # 可視化モードの場合は prediction ディレクトリに保存
                    prediction_dir = "prediction"
                    os.makedirs(prediction_dir, exist_ok=True)
                    output_path = os.path.join(prediction_dir, f"{base_filename}.png")
                else:
                    # タグのみの出力の場合、動画と同じディレクトリに保存
                    tag_output_dir = os.path.dirname(video_path)
                    output_path = os.path.join(tag_output_dir, f"{base_filename}.txt")
                
                # 結果の保存
                if output_mode == "visualization":
                    original_tags = read_tags_from_file(video_path)
                    visualize_predictions(
                        frame,  # 動画からの最初のフレーム
                        original_tags,
                        combined_predictions,
                        threshold=gen_threshold,
                        output_path=output_path
                    )
                else:
                    save_tags_as_csv(combined_predictions, output_path, threshold=gen_threshold, 
                                     mode=tag_mode, remove_threshold=remove_threshold)
                
                print(f"動画処理完了: {video_path}")
            
            except Exception as e:
                print(f"動画処理でエラー ({video_path}): {e}")
                import traceback
                print(traceback.format_exc())
    
    print(f"\nバッチ処理が完了しました。{total_images}個の画像と{total_videos}個の動画を処理しました。")

def combine_frame_predictions(frame_predictions, gen_threshold=0.45, char_threshold=0.45):
    """
    複数フレームの予測結果を統合する
    
    Args:
        frame_predictions: 各フレームの予測結果のリスト
        gen_threshold: 一般タグの閾値
        char_threshold: キャラクタータグの閾値
    
    Returns:
        combined_predictions: 統合された予測結果
    """
    if not frame_predictions:
        return None
    
    # カテゴリごとにタグの最大確率を保持する辞書
    combined = {
        "rating": {},
        "general": {},
        "character": {},
        "copyright": {},
        "artist": {},
        "meta": {}
    }
    
    # 各フレームの予測を処理
    for predictions in frame_predictions:
        for category in combined.keys():
            for tag, prob in predictions[category]:
                # タグがまだ辞書になければ追加
                if tag not in combined[category]:
                    combined[category][tag] = prob
                else:
                    # 既存のタグなら確率を最大値で更新
                    combined[category][tag] = max(combined[category][tag], prob)
    
    # 結果を元の形式に変換
    result = {
        "rating": [],
        "general": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "meta": []
    }
    
    # レーティングは最大確率のものを選択
    if combined["rating"]:
        top_rating = max(combined["rating"].items(), key=lambda x: x[1])
        result["rating"].append(top_rating)
    
    # その他のカテゴリは閾値以上のタグを含める
    for tag, prob in combined["general"].items():
        if prob >= gen_threshold:
            result["general"].append((tag, prob))
    
    for tag, prob in combined["character"].items():
        if prob >= char_threshold:
            result["character"].append((tag, prob))
    
    for tag, prob in combined["copyright"].items():
        if prob >= char_threshold:
            result["copyright"].append((tag, prob))
    
    for tag, prob in combined["artist"].items():
        if prob >= char_threshold:
            result["artist"].append((tag, prob))
    
    for tag, prob in combined["meta"].items():
        if prob >= gen_threshold:
            result["meta"].append((tag, prob))
    
    # 各カテゴリ内で確率の降順にソート
    for category in result:
        result[category] = sorted(result[category], key=lambda x: x[1], reverse=True)
    
    return result

def predict_video_with_onnx(video_path, model_path, tag_mapping_path, gen_threshold=0.45, char_threshold=0.45, 
                           output_path=None, use_gpu=False, output_mode="visualization", video_frames=3, tag_mode="add", remove_threshold=None):
    """
    単一の動画ファイルに対して推論を実行する
    
    Args:
        video_path: 入力動画のパス
        model_path: ONNXモデルのパス
        tag_mapping_path: タグマッピングJSONファイルのパス
        gen_threshold: 一般タグの閾値
        char_threshold: キャラクタータグの閾値
        output_path: 出力パス
        use_gpu: GPUを使用するかどうか
        output_mode: 出力モード
        video_frames: 動画から抽出するフレーム数
    """
    print(f"モデルを読み込み中: {model_path}")
    
    # モデルがFP16かどうかを確認
    is_fp16_model = False
    try:
        import onnx
        model = onnx.load(model_path)
        # モデルの最初の入力のデータ型をチェック
        for tensor in model.graph.initializer:
            if tensor.data_type == 10:  # FLOAT16
                is_fp16_model = True
                break
        print(f"モデルは {'FP16' if is_fp16_model else 'FP32'} です")
    except Exception as e:
        print(f"モデル精度の確認に失敗しました: {e}")
    
    # 利用可能なプロバイダーを確認
    available_providers = ort.get_available_providers()
    print(f"利用可能なプロバイダー: {available_providers}")
    
    # GPUの使用設定
    if use_gpu:
        try:
            # セッションオプションを設定
            session_options = ort.SessionOptions()
            
            # 利用可能なGPUプロバイダーを選択
            providers = []
            if 'CUDAExecutionProvider' in available_providers:
                # FP16モデルの場合、CUDA設定を最適化
                if is_fp16_model:
                    cuda_provider_options = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }
                    providers.append(('CUDAExecutionProvider', cuda_provider_options))
                else:
                    providers.append('CUDAExecutionProvider')
            elif 'DmlExecutionProvider' in available_providers:
                providers.append('DmlExecutionProvider')
            elif 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
            
            # CPUはフォールバック用に常に追加
            providers.append('CPUExecutionProvider')
            
            if not providers or len(providers) == 1:  # CPUしかない場合
                print("GPU用のプロバイダーが利用できません。CPUのみを使用します。")
                session = ort.InferenceSession(model_path)
            else:
                print(f"使用するプロバイダー: {providers}")
                session = ort.InferenceSession(model_path, providers=providers)
                print(f"アクティブなプロバイダー: {session.get_providers()[0]}")
        except Exception as e:
            print(f"GPU高速化に失敗しました: {e}")
            print("CPUにフォールバックします")
            session = ort.InferenceSession(model_path)
    else:
        # CPUのみを使用
        session = ort.InferenceSession(model_path)
        print("CPUを使用して推論を実行します")
    
    # タグマッピングの読み込み
    print(f"タグマッピングを読み込み中: {tag_mapping_path}")
    labels, idx_to_tag, tag_to_category = load_tag_mapping(tag_mapping_path)
    
    # 動画のフレームを抽出
    print(f"動画からフレームを抽出中: {video_path}")
    frames = extract_frames_from_video(video_path, video_frames)
    
    if not frames:
        print(f"エラー: 動画からフレームを抽出できませんでした: {video_path}")
        return None, None
    
    print(f"{len(frames)}フレームを抽出しました")
    
    # 各フレームの予測結果を保存するリスト
    all_frame_predictions = []
    
    # フレームごとの処理
    for i, frame in enumerate(frames):
        try:
            print(f"フレーム {i+1}/{len(frames)} の処理中...")
            
            # フレームを一時的にファイルとして保存
            temp_frame_path = f"temp_frame_{i}.png"
            frame.save(temp_frame_path)
            
            # preprocess_image関数を使用して前処理
            try:
                original_frame, input_data = preprocess_image(temp_frame_path)
                
                # 入力データ型の調整
                expected_input_type = session.get_inputs()[0].type
                if "float16" in expected_input_type:
                    input_data = input_data.astype(np.float16)
                else:
                    input_data = input_data.astype(np.float32)
                
                # 推論実行
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                
                # 推論時間の計測
                start_time = time.time()
                outputs = session.run([output_name], {input_name: input_data})[0]
                inference_time = time.time() - start_time
                print(f"フレーム {i+1} の推論が {inference_time:.3f} 秒で完了しました")
                
                # 出力の後処理
                outputs = np.nan_to_num(outputs[0], nan=0.0, posinf=100.0, neginf=-100.0)
                
                # シグモイド関数を適用して確率に変換（数値安定性を考慮）
                def stable_sigmoid(x):
                    return np.where(
                        x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x))
                    )
                
                outputs = stable_sigmoid(outputs)
                
                # タグの取得
                frame_predictions = get_tags(outputs, labels, gen_threshold, char_threshold)
                all_frame_predictions.append(frame_predictions)
                
            finally:
                # 一時ファイルを削除
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                
        except Exception as e:
            print(f"フレーム処理でエラー: {e}")
            import traceback
            print(traceback.format_exc())
    
    if not all_frame_predictions:
        print(f"エラー: 動画の全フレームの処理に失敗しました: {video_path}")
        return None, None
    
    # 全フレームの予測結果を統合
    combined_predictions = combine_frame_predictions(all_frame_predictions, gen_threshold, char_threshold)
    
    # 結果の表示
    print("--------")
    print("レーティング:")
    for tag, conf in combined_predictions["rating"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"キャラクタータグ (閾値={char_threshold}):")
    for tag, conf in combined_predictions["character"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"著作権タグ (閾値={char_threshold}):")
    for tag, conf in combined_predictions["copyright"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"アーティストタグ (閾値={char_threshold}):")
    for tag, conf in combined_predictions["artist"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"一般タグ (閾値={gen_threshold}):")
    for tag, conf in combined_predictions["general"]:
        print(f"  {tag}: {conf:.3f}")
    
    print("--------")
    print(f"メタタグ (閾値={gen_threshold}):")
    # 不要なメタタグをフィルタリング
    filtered_meta = []
    excluded_meta_patterns = ['id', 'commentary']
    
    for tag, conf in combined_predictions["meta"]:
        if not any(pattern in tag.lower() for pattern in excluded_meta_patterns):
            print(f"  {tag}: {conf:.3f}")
            filtered_meta.append((tag, conf))
        else:
            print(f"  [FILTERED] {tag}: {conf:.3f}")
    
    # 出力パスの設定
    if output_path is None:
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        prediction_dir = "prediction"
        os.makedirs(prediction_dir, exist_ok=True)
        
        if output_mode == "visualization":
            output_path = os.path.join(prediction_dir, f"{base_filename}.png")
        else:
            output_path = os.path.join(prediction_dir, f"{base_filename}.txt")
    else:
        # 出力ディレクトリが指定されている場合は、そのディレクトリが存在することを確認
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # 結果の保存
    if output_mode == "visualization":
        original_tags = read_tags_from_file(video_path)
        visualize_predictions(
            frame,  # 動画からの最初のフレーム
            original_tags,
            combined_predictions,
            threshold=gen_threshold,
            output_path=output_path
        )
    else:
        save_tags_as_csv(combined_predictions, output_path, threshold=gen_threshold, mode=tag_mode, remove_threshold=remove_threshold)
    
    print(f"動画処理完了: {video_path}")
    return combined_predictions, frame

def main():
    parser = argparse.ArgumentParser(description='ONNXモデルを使用した画像タグ予測')
    
    # 入力関連の引数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='予測する画像のパスまたはURL')
    input_group.add_argument('--video', type=str, help='予測する動画のパスまたはURL')
    input_group.add_argument('--dirs', type=str, nargs='+', help='予測する画像ディレクトリのパス（複数指定可）')
    input_group.add_argument('--video_frames', type=int, default=3, help='動画から抽出するフレーム数')
    
    # モデル関連の引数
    parser.add_argument('--model', type=str, required=True, help='ONNXモデルのパス')
    parser.add_argument('--tag_mapping', type=str, required=True, help='タグマッピングJSONファイルのパス')
    
    # 出力関連の引数
    parser.add_argument('--output', type=str, default=None, help='結果の出力パス（単一画像/動画モードのみ）')
    parser.add_argument('--output_mode', type=str, choices=['visualization', 'tags'], default='visualization',
                        help='出力モード: visualization=可視化画像, tags=カンマ区切りタグ')
    
    # 閾値関連の引数
    parser.add_argument('--gen_threshold', type=float, default=0.45, help='一般タグの閾値')
    parser.add_argument('--char_threshold', type=float, default=0.45, help='キャラクタータグの閾値')
    
    # その他のオプション
    parser.add_argument('--gpu', action='store_true', help='GPUを使用して推論を実行')
    parser.add_argument('--recursive', action='store_true', help='ディレクトリモードでサブディレクトリも処理する')
    parser.add_argument('--batch_size', type=int, default=1, help='バッチ処理時のバッチサイズ')
    
    # タグ保存モードの引数を追加
    parser.add_argument('--tag_mode', type=str, choices=['overwrite', 'add'], default='add',
                        help='タグ保存モード: overwrite=上書き, add=既存タグに追加')
    parser.add_argument('--remove_threshold', default=None, type=float,
                        help='既存タグを除去する確率の閾値（addモードでのみ有効）')
    
    args = parser.parse_args()
    
    if args.image:
        # 画像処理
        predict_with_onnx(
            args.image,
            args.model,
            args.tag_mapping,
            args.gen_threshold,
            args.char_threshold,
            args.output,
            use_gpu=args.gpu,
            output_mode=args.output_mode,
            tag_mode=args.tag_mode,
            remove_threshold=args.remove_threshold,
            batch_size=args.batch_size,
        )
    elif args.video:
        # 動画処理
        predict_video_with_onnx(
            args.video,
            args.model,
            args.tag_mapping,
            args.gen_threshold,
            args.char_threshold,
            args.output,
            use_gpu=args.gpu,
            output_mode=args.output_mode,
            video_frames=args.video_frames,
            tag_mode=args.tag_mode,
            remove_threshold=args.remove_threshold
        )
    elif args.dirs:
        # バッチ処理
        batch_predict(
            args.dirs,
            args.model,
            args.tag_mapping,
            args.gen_threshold,
            args.char_threshold,
            use_gpu=args.gpu,
            output_mode=args.output_mode,
            recursive=args.recursive,
            batch_size=args.batch_size,
            video_frames=args.video_frames,
            tag_mode=args.tag_mode,
            remove_threshold=args.remove_threshold
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