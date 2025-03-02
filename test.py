import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F
import re
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# デバイスの設定
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルリポジトリの設定
MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# LoRAモジュールの定義
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA A行列（低ランク行列）
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        # LoRA B行列（低ランク行列）
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # 重みの初期化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # LoRAの計算: x @ (A @ B) * scale
        lora_output = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) * self.scale
        return lora_output

# LoRA適用済み線形層
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # 入出力の次元を取得
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # LoRA層
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha
        )
        
        # ドロップアウト層
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = nn.Identity()
    
    def forward(self, x):
        # 元の線形層の出力
        base_output = self.base_layer(x)
        
        # LoRAの出力（ドロップアウト適用）
        lora_output = self.dropout_layer(self.lora(x))
        
        # 出力を合成
        return base_output + lora_output

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs_list = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs_list[i] for i in labels.rating])

    # General labels with all probabilities
    all_gen_labels = dict([probs_list[i] for i in labels.general])
    
    # Character labels with all probabilities
    all_char_labels = dict([probs_list[i] for i in labels.character])
    
    # Filtered general labels (above threshold)
    gen_labels = {k: v for k, v in all_gen_labels.items() if v > gen_threshold}
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Filtered character labels (above threshold)
    char_labels = {k: v for k, v in all_char_labels.items() if v > char_threshold}
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels for caption
    combined_names = list(gen_labels.keys())
    combined_names.extend(list(char_labels.keys()))

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels, all_char_labels, all_gen_labels


# EVA02モデルにモジュールごとのLoRAを適用するクラス
class EVA02WithModuleLoRA(nn.Module):
    def __init__(
        self, 
        num_classes, 
        lora_rank=4, 
        lora_alpha=1.0, 
        lora_dropout=0.0,
        target_modules=None,
        pretrained=True
    ):
        super().__init__()
        
        # LoRAのハイパーパラメータを保存
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # バックボーンを作成
        self.backbone = timm.create_model(
            'hf-hub:' + MODEL_REPO, 
            pretrained=pretrained
        )
        
        # 特徴量の次元を取得
        self.feature_dim = self.backbone.head.in_features
        
        # 元のヘッドを保存
        self.head = self.backbone.head
        
        # ターゲットモジュールが指定されていない場合は、デフォルトのリストを使用
        if target_modules is None:
            # デフォルトのターゲットモジュール（注意機構とFFN）
            self.target_modules = [
                # 自己注意機構の投影層
                r'blocks\.\d+\.attn\.q_proj',
                r'blocks\.\d+\.attn\.k_proj',
                r'blocks\.\d+\.attn\.v_proj',
                r'blocks\.\d+\.attn\.proj',
                # FFNの線形層
                r'blocks\.\d+\.mlp\.fc1',
                r'blocks\.\d+\.mlp\.fc2'
            ]
        else:
            self.target_modules = target_modules
        
        # 各モジュールにLoRAを適用
        self._apply_lora_to_modules()
        
        # LoRAパラメータのみを訓練可能に設定
        self._freeze_non_lora_parameters()
    
    def _apply_lora_to_modules(self):
        """モデルの各モジュールにLoRAを適用する"""
        # 正規表現パターンをコンパイル
        patterns = [re.compile(pattern) for pattern in self.target_modules]
        
        # 適用されたLoRA層を追跡
        self.lora_layers = {}
        
        # 各モジュールを調査
        for name, module in self.backbone.named_modules():
            # パターンに一致するか確認
            if isinstance(module, nn.Linear) and any(pattern.search(name) for pattern in patterns):
                print(f"Applying LoRA to module: {name}")
                
                # モジュールの親を取得
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.backbone
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                
                # 元のモジュールを取得
                original_module = getattr(parent, child_name)
                
                # LoRAを適用したモジュールで置き換え
                lora_module = LoRALinear(
                    base_layer=original_module,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout
                )
                setattr(parent, child_name, lora_module)
                
                # 適用されたLoRA層を記録
                self.lora_layers[name] = lora_module
        
        print(f"Applied LoRA to {len(self.lora_layers)} modules")
    
    def _freeze_non_lora_parameters(self):
        """LoRAパラメータ以外を凍結する"""
        # バックボーンのすべてのパラメータを凍結
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # LoRA層のパラメータのみを訓練可能に設定
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                for param in module.parameters():
                    param.requires_grad = True
    
    def forward(self, x):
        # モデル全体を通して推論
        return self.backbone(x)


def load_model(model_path=None, device=torch_device):
    """モデルを読み込む関数"""
    print(f"モデルを読み込んでいます...")
    
    # ラベルデータを読み込む
    labels = load_labels_hf(repo_id=MODEL_REPO)
    num_classes = len(labels.names)
    
    if model_path:
        # LoRAモデルのチェックポイントを読み込む
        print(f"LoRAチェックポイントを読み込んでいます: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # LoRA設定を取得
        lora_rank = checkpoint.get('lora_rank', 4)
        lora_alpha = checkpoint.get('lora_alpha', 1.0)
        lora_dropout = checkpoint.get('lora_dropout', 0.0)
        target_modules = checkpoint.get('target_modules', None)
        
        # モデルを作成
        model = EVA02WithModuleLoRA(
            num_classes=num_classes,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            pretrained=True
        )
        
        # 状態辞書をモデルに読み込む
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # 通常のモデルを読み込む
        model = timm.create_model('hf-hub:' + MODEL_REPO, pretrained=True)
    
    model = model.to(device)
    model.eval()
    
    return model, labels


def read_tags_from_file(image_path):
    """画像に紐づくタグファイルを読み込む関数"""
    # 画像パスからタグファイルのパスを生成
    tag_path = os.path.splitext(image_path)[0] + '.txt'
    
    # タグファイルが存在するか確認
    if not os.path.exists(tag_path):
        print(f"Warning: Tag file not found at {tag_path}")
        return []
    
    # タグファイルを読み込む
    with open(tag_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # カンマ区切りのタグを処理
    if ',' in content:
        # カンマ区切りのタグをリストに変換し、各タグの前後の空白を削除
        tags = [tag.strip() for tag in content.split(',') if tag.strip()]
    else:
        # 行ごとに読み込み
        tags = [line.strip() for line in content.splitlines() if line.strip()]
    
    print(f"Read {len(tags)} tags from {tag_path}")
    return tags


def predict_image(image_path, model, labels, device=torch_device, gen_threshold=0.35, char_threshold=0.75):
    """画像からタグを予測する関数"""
    # 画像の読み込みと前処理
    img_input = Image.open(image_path)
    img_input = pil_ensure_rgb(img_input)
    img_input = pil_pad_square(img_input)
    
    # データ変換の設定
    transform = create_transform(**resolve_data_config(model.pretrained_cfg if hasattr(model, 'pretrained_cfg') else {}, model=model))
    
    # 画像の変換
    inputs = transform(img_input).unsqueeze(0)
    # NCHW image RGB to BGR
    inputs = inputs[:, [2, 1, 0]]
    
    # 推論
    with torch.inference_mode():
        if device.type != "cpu":
            inputs = inputs.to(device)
        
        # モデルの出力を取得
        outputs = model(inputs)
        
        # シグモイド関数を適用
        outputs = F.sigmoid(outputs)
        
        if device.type != "cpu":
            outputs = outputs.to("cpu")
    
    # タグの取得
    caption, taglist, ratings, character, general, all_character, all_general = get_tags(
        probs=outputs.squeeze(0),
        labels=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
    )
    
    return img_input, caption, taglist, ratings, character, general, all_character, all_general


def visualize_predictions(image, tags, predictions, threshold=0.35, output_path=None, max_tags=50):
    """
    予測結果を可視化する関数
    
    Args:
        image: 既に開かれているPIL画像オブジェクト
        tags: 元々の画像に付与されていたタグのリスト
        predictions: (caption, taglist, ratings, character, general, all_character, all_general)の予測結果タプル
        threshold: タグ表示の閾値
        output_path: 出力ファイルパス（Noneの場合は表示のみ）
        max_tags: 表示する最大タグ数
    """
    caption, taglist, ratings, character, general, all_character, all_general = predictions
    
    # タグの正規化（スペースを_に変換）
    normalized_tags = []
    for tag in tags:
        # スペースを_に変換
        normalized_tag = tag.replace(' ', '_')
        normalized_tags.append(normalized_tag)
    
    print(f"Normalized tags: {normalized_tags[:5]}...")
    
    # すべてのタグと確率値を結合
    all_tags = {}
    all_tags.update(all_general)
    all_tags.update(all_character)
    
    # タグを3つのカテゴリに分類
    above_threshold_in_tags = []  # 閾値以上かつタグに含まれる（真陽性）- 緑
    above_threshold_not_in_tags = []  # 閾値以上だがタグに含まれない（偽陽性）- 赤
    below_threshold_in_tags = []  # 閾値以下だがタグに含まれる（偽陰性）- 青
    
    # 各タグを分類
    for tag, prob in all_tags.items():
        if prob >= threshold:
            if tag in normalized_tags:
                above_threshold_in_tags.append((tag, prob))
            else:
                above_threshold_not_in_tags.append((tag, prob))
        else:
            if tag in normalized_tags:
                below_threshold_in_tags.append((tag, prob))
    
    print(f"True positives: {len(above_threshold_in_tags)}")
    print(f"False positives: {len(above_threshold_not_in_tags)}")
    print(f"False negatives: {len(below_threshold_in_tags)}")
    
    # 確率値で降順ソート
    above_threshold_in_tags.sort(key=lambda x: x[1], reverse=True)
    above_threshold_not_in_tags.sort(key=lambda x: x[1], reverse=True)
    below_threshold_in_tags.sort(key=lambda x: x[1], reverse=True)
    
    # 表示するタグ数を制限
    total_tags = len(above_threshold_in_tags) + len(above_threshold_not_in_tags) + len(below_threshold_in_tags)
    if total_tags > max_tags:
        # 各カテゴリから均等に選択
        tags_per_category = max(1, max_tags // 3)
        above_threshold_in_tags = above_threshold_in_tags[:tags_per_category]
        above_threshold_not_in_tags = above_threshold_not_in_tags[:tags_per_category]
        below_threshold_in_tags = below_threshold_in_tags[:tags_per_category]
    
    # プロットの設定
    plt.figure(figsize=(15, 10))
    
    # 画像の表示
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # タグの確率をグラフ表示
    plt.subplot(1, 2, 2)
    
    # データの準備
    all_display_tags = above_threshold_in_tags + above_threshold_not_in_tags + below_threshold_in_tags
    tags_names = [t[0] for t in all_display_tags]
    probs = [t[1] for t in all_display_tags]
    
    # カラーマップの作成
    colors = []
    for tag, prob in all_display_tags:
        if prob >= threshold and tag in normalized_tags:
            colors.append('green')  # 閾値以上かつタグに含まれる - 緑
        elif prob >= threshold:
            colors.append('red')  # 閾値以上だがタグに含まれない - 赤
        else:
            colors.append('blue')  # 閾値以下だがタグに含まれる - 青
    
    # 水平バーチャートの作成
    y_pos = range(len(tags_names))
    plt.barh(y_pos, probs, color=colors)
    plt.yticks(y_pos, tags_names)
    plt.xlabel('Probability')
    plt.title('Tag Predictions')
    
    # 閾値ラインの表示
    plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.7)
    plt.text(threshold, len(tags_names) - 1, f'Threshold: {threshold}', 
             verticalalignment='bottom', horizontalalignment='right')
    
    # 凡例の追加
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive (threshold+ & in tags)'),
        Patch(facecolor='red', label='False Positive (threshold+ & not in tags)'),
        Patch(facecolor='blue', label='False Negative (threshold- & in tags)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def analyze_model_structure():
    """モデル構造を分析し、LoRAを適用すべき層を特定する関数"""
    import math
    import logging
    from datetime import datetime
    
    # ロガーのセットアップ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"model_analysis_{timestamp}.log"
    
    logger = logging.getLogger('model_analyzer')
    logger.setLevel(logging.INFO)
    
    # フォーマッターの作成
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 標準出力へのハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイル出力へのハンドラー
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"モデル '{MODEL_REPO}' の構造を分析しています...")
    
    # モデルをロード
    try:
        logger.info(f"Hugging Face Hub からモデルをロードしています...")
        model = timm.create_model('hf-hub:' + MODEL_REPO, pretrained=True)
    except Exception as e:
        logger.error(f"モデルのロード中にエラーが発生しました: {e}")
        return
    
    # 結果を格納する辞書
    result = {
        "model_name": MODEL_REPO,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "modules": []
    }
    
    # 各モジュールの情報を収集
    logger.info("モデルの各モジュールを分析しています...")
    for name, module in model.named_modules():
        if name == '':  # ルートモジュールはスキップ
            continue
        
        module_type = type(module).__name__
        
        # 線形層の場合は詳細情報を収集
        if isinstance(module, torch.nn.Linear):
            module_info = {
                "name": name,
                "type": module_type,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                "parameters": sum(p.numel() for p in module.parameters())
            }
            result["modules"].append(module_info)
    
    # 自己注意機構とFFNの線形層を特定
    attention_layers = [m for m in result["modules"] if m["type"] == "Linear" and any(x in m["name"] for x in ["attn", "qkv", "query", "key", "value"])]
    ffn_layers = [m for m in result["modules"] if m["type"] == "Linear" and any(x in m["name"] for x in ["mlp", "fc", "ffn"])]
    
    logger.info("\n=== LoRAを適用すべき主要な線形層 ===")
    
    logger.info("\n自己注意機構の線形層:")
    for layer in attention_layers[:10]:  # 最初の10個だけ表示
        logger.info(f"  - {layer['name']}: {layer['in_features']} → {layer['out_features']}")
    if len(attention_layers) > 10:
        logger.info(f"  - ... 他 {len(attention_layers) - 10} 個")
    
    logger.info("\nフィードフォワードネットワークの線形層:")
    for layer in ffn_layers[:10]:  # 最初の10個だけ表示
        logger.info(f"  - {layer['name']}: {layer['in_features']} → {layer['out_features']}")
    if len(ffn_layers) > 10:
        logger.info(f"  - ... 他 {len(ffn_layers) - 10} 個")
    
    # LoRAを適用すべき層のリストを作成
    lora_target_layers = []
    for layer in attention_layers + ffn_layers:
        lora_target_layers.append(layer["name"])
    
    # LoRAターゲット層をファイルに保存
    lora_targets_file = "lora_target_layers.txt"
    with open(lora_targets_file, 'w') as f:
        for layer in lora_target_layers:
            f.write(f"{layer}\n")
    logger.info(f"\nLoRAを適用すべき層のリストを '{lora_targets_file}' に保存しました。")
    
    # 結果をJSONファイルに保存
    output_file = "model_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"分析結果を '{output_file}' に保存しました。")
    
    logger.info(f"\n分析が完了しました。ログは '{log_file}' に保存されています。")
    return result


def main():
    parser = argparse.ArgumentParser(description="SmilingWolf/wd-eva02-large-tagger-v3モデルを使用して画像のタグを推論し、LoRAトレーニングのための準備を行います。")
    subparsers = parser.add_subparsers(dest='command', help='コマンド')
    
    # 分析コマンド
    analyze_parser = subparsers.add_parser('analyze', help='モデル構造を分析します')
    
    # 推論コマンド
    predict_parser = subparsers.add_parser('predict', help='画像からタグを予測します')
    predict_parser.add_argument('--image', type=str, required=True, help='予測する画像ファイルのパス')
    predict_parser.add_argument('--lora_model', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    predict_parser.add_argument('--output_dir', type=str, default='predictions', help='予測結果を保存するディレクトリ')
    predict_parser.add_argument('--gen_threshold', type=float, default=0.35, help='一般タグの閾値')
    predict_parser.add_argument('--char_threshold', type=float, default=0.75, help='キャラクタータグの閾値')
    
    # バッチ推論コマンド
    batch_parser = subparsers.add_parser('batch', help='複数の画像からタグを予測します')
    batch_parser.add_argument('--image_dir', type=str, required=True, help='予測する画像ファイルのディレクトリ')
    batch_parser.add_argument('--lora_model', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    batch_parser.add_argument('--output_dir', type=str, default='predictions', help='予測結果を保存するディレクトリ')
    batch_parser.add_argument('--gen_threshold', type=float, default=0.35, help='一般タグの閾値')
    batch_parser.add_argument('--char_threshold', type=float, default=0.75, help='キャラクタータグの閾値')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        # モデル構造の分析
        analyze_model_structure()
    
    elif args.command == 'predict':
        # 単一画像の予測
        model, labels = load_model(args.lora_model)
        
        # 画像に紐づくタグを読み込む
        actual_tags = read_tags_from_file(args.image)
        print(f"読み込まれたタグ: {len(actual_tags)}個")
        
        # 予測を実行
        img, caption, taglist, ratings, character, general, all_character, all_general = predict_image(
            args.image, 
            model, 
            labels, 
            gen_threshold=args.gen_threshold, 
            char_threshold=args.char_threshold
        )
        
        # 結果の表示
        print("--------")
        print(f"Caption: {caption}")
        print("--------")
        print(f"Tags: {taglist}")
        
        print("--------")
        print("Ratings:")
        for k, v in ratings.items():
            print(f"  {k}: {v:.3f}")
        
        print("--------")
        print(f"Character tags (threshold={args.char_threshold}):")
        for k, v in character.items():
            print(f"  {k}: {v:.3f}")
        
        print("--------")
        print(f"General tags (threshold={args.gen_threshold}):")
        for k, v in general.items():
            print(f"  {k}: {v:.3f}")
        
        # 結果の可視化と保存
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, os.path.basename(args.image).split('.')[0] + '_prediction.png')
        else:
            output_path = None
            
        visualize_predictions(
            image=img, 
            tags=actual_tags, 
            predictions=(caption, taglist, ratings, character, general, all_character, all_general),
            threshold=args.gen_threshold,
            output_path=output_path
        )
    
    elif args.command == 'batch':
        # モデルの読み込み
        model, labels = load_model(args.lora_model)
        
        # 画像ファイルのリストを取得
        import glob
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        
        print(f"{len(image_files)}枚の画像を処理します...")
        
        # 出力ディレクトリの作成
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # 各画像を処理
        results = {}
        for image_file in tqdm(image_files):
            try:
                # 画像に紐づくタグを読み込む
                actual_tags = read_tags_from_file(image_file)
                
                # 予測
                img, caption, taglist, ratings, character, general, all_character, all_general = predict_image(
                    image_file, 
                    model, 
                    labels, 
                    gen_threshold=args.gen_threshold, 
                    char_threshold=args.char_threshold
                )
                
                # 結果の可視化と保存
                if args.output_dir:
                    output_path = os.path.join(args.output_dir, os.path.basename(image_file).split('.')[0] + '_prediction.png')
                else:
                    output_path = None
                    
                visualize_predictions(
                    image=img, 
                    tags=actual_tags, 
                    predictions=(caption, taglist, ratings, character, general, all_character, all_general),
                    threshold=args.gen_threshold,
                    output_path=output_path
                )
                
                # 結果を保存
                results[os.path.basename(image_file)] = {
                    'caption': caption,
                    'ratings': ratings,
                    'character': {k: float(v) for k, v in character.items()},  # JSON化のためfloatに変換
                    'general': {k: float(v) for k, v in general.items()}  # JSON化のためfloatに変換
                }
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        # 結果をJSONファイルに保存
        if args.output_dir:
            with open(os.path.join(args.output_dir, 'batch_results.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"処理が完了しました。結果は {args.output_dir} に保存されています。")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    import math  # LoRALayer初期化に必要
    main()
