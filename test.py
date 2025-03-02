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
        num_classes=None,  # 初期化時にはNoneでも可能に
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
        
        # モデルの期待する画像サイズを取得
        self.img_size = self.backbone.patch_embed.img_size
        print(f"モデルの期待する画像サイズ: {self.img_size}")
        
        # 特徴量の次元を取得
        self.feature_dim = self.backbone.head.in_features
        
        # 元のヘッドを保存
        self.original_head = self.backbone.head
        self.original_num_classes = self.original_head.out_features
        
        # 新しいヘッドを設定（初期状態では元のヘッドと同じ）
        if num_classes is None:
            num_classes = self.original_num_classes
        
        self.extend_head(num_classes)
        
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
    
    def extend_head(self, num_classes):
        """
        モデルのヘッドを拡張して新しいタグに対応する
        
        Args:
            num_classes: 新しい総クラス数（既存タグ + 新規タグ）
        """
        # 元のヘッドの重みとバイアスを取得
        original_weight = self.original_head.weight.data
        original_bias = self.original_head.bias.data if self.original_head.bias is not None else None
        
        # 新しいヘッドを作成
        new_head = nn.Linear(self.feature_dim, num_classes)
        
        # 既存タグの重みとバイアスを新しいヘッドにコピー
        new_head.weight.data[:self.original_num_classes] = original_weight
        if original_bias is not None:
            new_head.bias.data[:self.original_num_classes] = original_bias
        
        # 新規タグの重みを初期化（Xavierの初期化）
        if num_classes > self.original_num_classes:
            nn.init.xavier_uniform_(new_head.weight.data[self.original_num_classes:])
            if new_head.bias is not None:
                nn.init.zeros_(new_head.bias.data[self.original_num_classes:])
            
            print(f"ヘッドを拡張しました: {self.original_num_classes} → {num_classes} クラス")
        
        # バックボーンのヘッドを新しいヘッドに置き換え
        self.backbone.head = new_head
    
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
        
        # ヘッドのパラメータも訓練可能に設定
        for param in self.backbone.head.parameters():
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
        
        # タグマッピングを取得
        tag_to_idx = checkpoint.get('tag_to_idx', None)
        idx_to_tag = checkpoint.get('idx_to_tag', None)
        
        # 新しいクラス数を決定
        if tag_to_idx is not None:
            num_classes = len(tag_to_idx)
            print(f"チェックポイントから読み込んだタグ数: {num_classes}")
        
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
    
    # タグの正規化（スペースを_に変換、エスケープされた括弧を通常の括弧に変換）
    normalized_tags = []
    for tag in tags:
        # スペースを_に変換
        normalized_tag = tag.replace(' ', '_')
        # エスケープされた括弧を通常の括弧に変換
        normalized_tag = normalized_tag.replace('\\(', '(').replace('\\)', ')')
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

# トレーニング用のデータセットクラス
class TagImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_paths, 
        tags_list, 
        tag_to_idx, 
        transform=None, 
        min_tag_freq=1,
        remove_special_prefix=True
    ):
        """
        画像とタグのデータセット
        
        Args:
            image_paths: 画像ファイルパスのリスト
            tags_list: 各画像に対応するタグのリスト（リストのリスト）
            tag_to_idx: タグから索引へのマッピング辞書
            transform: 画像変換関数
            min_tag_freq: タグの最小出現頻度
            remove_special_prefix: 特殊プレフィックス（例：a@、g@など）を除去するかどうか
        """
        self.image_paths = image_paths
        self.tags_list = tags_list
        self.tag_to_idx = tag_to_idx
        self.transform = transform
        self.min_tag_freq = min_tag_freq
        self.remove_special_prefix = remove_special_prefix
        
        # タグの出現頻度を計算
        self.tag_freq = {}
        for tags in tags_list:
            for tag in tags:
                if self.remove_special_prefix and re.match(r'^[a-zA-Z]@', tag):
                    continue  # 特殊プレフィックスを持つタグをスキップ
                self.tag_freq[tag] = self.tag_freq.get(tag, 0) + 1
        
        # 最小出現頻度でフィルタリング
        self.filtered_tag_to_idx = {
            tag: idx for tag, idx in self.tag_to_idx.items() 
            if tag in self.tag_freq and self.tag_freq[tag] >= self.min_tag_freq
        }
        
        print(f"元のタグ数: {len(self.tag_to_idx)}")
        print(f"フィルタリング後のタグ数: {len(self.filtered_tag_to_idx)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 画像の読み込みと前処理
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = pil_ensure_rgb(image)
        image = pil_pad_square(image)
        
        if self.transform:
            image = self.transform(image)
            # RGB to BGR for EVA02 model
            image = image[[2, 1, 0]]
        
        # タグをone-hotエンコーディング
        tags = self.tags_list[idx]
        num_classes = len(self.tag_to_idx)
        label = torch.zeros(num_classes)
        
        for tag in tags:
            if tag in self.tag_to_idx:
                label[self.tag_to_idx[tag]] = 1.0
        
        return image, label


# データセットの準備関数
def prepare_dataset(
    image_dirs, 
    existing_tags=None, 
    val_split=0.1, 
    min_tag_freq=5, 
    remove_special_prefix=True,
    seed=42
):
    """
    画像とタグのデータセットを準備する
    
    Args:
        image_dirs: 画像ディレクトリのリスト
        existing_tags: 既存のタグリスト（Noneの場合は全てのタグを使用）
        val_split: 検証データの割合
        min_tag_freq: タグの最小出現頻度
        remove_special_prefix: 特殊プレフィックスを除去するかどうか
        seed: 乱数シード
    
    Returns:
        train_dataset, val_dataset, tag_to_idx, idx_to_tag
    """
    # 乱数シードを設定
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 画像ファイルとタグの収集
    all_image_paths = []
    all_tags_list = []
    
    # 画像拡張子
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    print("画像とタグを収集しています...")
    for image_dir in image_dirs:
        image_dir = Path(image_dir)
        
        # 画像ファイルを再帰的に探索
        for ext in image_extensions:
            for img_path in image_dir.glob(f"**/*{ext}"):
                # 対応するタグファイルを探す
                tag_path = img_path.with_suffix('.txt')
                
                if tag_path.exists():
                    # タグファイルを読み込む
                    with open(tag_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # カンマ区切りのタグを処理
                    if ',' in content:
                        tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                    else:
                        tags = [line.strip() for line in content.splitlines() if line.strip()]
                    
                    # 特殊プレフィックスの除去
                    if remove_special_prefix:
                        tags = [tag for tag in tags if not re.match(r'^[a-zA-Z]@', tag)]
                    
                    # スペースをアンダースコアに変換
                    tags = [tag.replace(' ', '_') for tag in tags]
                    
                    all_image_paths.append(str(img_path))
                    all_tags_list.append(tags)
    
    print(f"収集された画像数: {len(all_image_paths)}")
    
    # タグの出現頻度を計算
    tag_freq = {}
    for tags in all_tags_list:
        for tag in tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
    
    # 最小出現頻度でフィルタリング
    filtered_tags = {tag for tag, freq in tag_freq.items() if freq >= min_tag_freq}
    
    # 既存タグと新規タグの分類
    if existing_tags is not None:
        existing_tags_set = set(existing_tags)
        new_tags = filtered_tags - existing_tags_set
        used_tags = filtered_tags & existing_tags_set
        
        print(f"既存タグ数: {len(existing_tags_set)}")
        print(f"使用される既存タグ数: {len(used_tags)}")
        print(f"新規タグ数: {len(new_tags)}")
        
        # タグから索引へのマッピング（既存タグを先に配置）
        tag_to_idx = {tag: i for i, tag in enumerate(sorted(used_tags))}
        # 新規タグを追加
        for i, tag in enumerate(sorted(new_tags), start=len(tag_to_idx)):
            tag_to_idx[tag] = i
    else:
        # 全てのタグを使用
        tag_to_idx = {tag: i for i, tag in enumerate(sorted(filtered_tags))}
    
    # 索引からタグへのマッピング
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    
    print(f"使用されるタグの総数: {len(tag_to_idx)}")
    
    # データを訓練セットと検証セットに分割
    indices = np.arange(len(all_image_paths))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_image_paths = [all_image_paths[i] for i in train_indices]
    train_tags_list = [all_tags_list[i] for i in train_indices]
    
    val_image_paths = [all_image_paths[i] for i in val_indices]
    val_tags_list = [all_tags_list[i] for i in val_indices]
    
    print(f"訓練データ数: {len(train_image_paths)}")
    print(f"検証データ数: {len(val_image_paths)}")
    
    return train_image_paths, train_tags_list, val_image_paths, val_tags_list, tag_to_idx, idx_to_tag


# 非対称損失関数（ASL）の実装
class AsymmetricLoss(nn.Module):
    def __init__(
        self, 
        gamma_neg=4, 
        gamma_pos=1, 
        clip=0.05, 
        eps=1e-8, 
        disable_torch_grad_focal_loss=False
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """"
        非対称損失関数
        Args:
            x: 予測値 (logits)
            y: 正解ラベル (0 or 1)
        """
        # Sigmoid関数を適用
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # 正例と負例の分離
        targets = y
        anti_targets = 1.0 - targets

        # クリッピング
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # 損失計算の準備
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                pt_pos = xs_pos * targets
                pt_neg = xs_neg * anti_targets
                pt_pos = pt_pos.clamp(min=self.eps)
                pt_neg = pt_neg.clamp(min=self.eps)
                
                # 重みの計算
                focal_weight_pos = pt_pos.pow(self.gamma_pos)
                focal_weight_neg = pt_neg.pow(self.gamma_neg)
            
            # 損失計算
            loss_pos = focal_weight_pos * torch.log(pt_pos)
            loss_neg = focal_weight_neg * torch.log(pt_neg)
        else:
            pt_pos = xs_pos * targets
            pt_neg = xs_neg * anti_targets
            pt_pos = pt_pos.clamp(min=self.eps)
            pt_neg = pt_neg.clamp(min=self.eps)
            
            # 重みの計算
            focal_weight_pos = pt_pos.pow(self.gamma_pos)
            focal_weight_neg = pt_neg.pow(self.gamma_neg)
            
            # 損失計算
            loss_pos = focal_weight_pos * torch.log(pt_pos)
            loss_neg = focal_weight_neg * torch.log(pt_neg)
        
        # 最終的な損失
        loss = -loss_pos.sum() - loss_neg.sum()
        return loss


# 最適化された非対称損失関数
class AsymmetricLossOptimized(nn.Module):
    def __init__(
        self, 
        gamma_neg=4, 
        gamma_pos=1, 
        clip=0.05, 
        eps=1e-8, 
        disable_torch_grad_focal_loss=False
    ):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.targets = self.anti_targets = None
        self.xs_pos = self.xs_neg = None

    def forward(self, x, y):
        """"
        最適化された非対称損失関数
        Args:
            x: 予測値 (logits)
            y: 正解ラベル (0 or 1)
        """
        self.targets = y
        self.anti_targets = 1.0 - y

        # Sigmoid関数を適用
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # クリッピング
        if self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1.0)

        # 損失計算
        loss = self._asymmetric_loss()
        return loss

    def _asymmetric_loss(self):
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                pt_pos = self.xs_pos * self.targets
                pt_neg = self.xs_neg * self.anti_targets
                pt_pos = pt_pos.clamp(min=self.eps)
                pt_neg = pt_neg.clamp(min=self.eps)
                
                # 重みの計算
                focal_weight_pos = pt_pos.pow(self.gamma_pos)
                focal_weight_neg = pt_neg.pow(self.gamma_neg)
            
            # 損失計算
            loss_pos = focal_weight_pos * torch.log(pt_pos)
            loss_neg = focal_weight_neg * torch.log(pt_neg)
        else:
            pt_pos = self.xs_pos * self.targets
            pt_neg = self.xs_neg * self.anti_targets
            pt_pos = pt_pos.clamp(min=self.eps)
            pt_neg = pt_neg.clamp(min=self.eps)
            
            # 重みの計算
            focal_weight_pos = pt_pos.pow(self.gamma_pos)
            focal_weight_neg = pt_neg.pow(self.gamma_neg)
            
            # 損失計算
            loss_pos = focal_weight_pos * torch.log(pt_pos)
            loss_neg = focal_weight_neg * torch.log(pt_neg)
        
        # 最終的な損失
        loss = -loss_pos.sum() - loss_neg.sum()
        return loss


# 評価指標の計算関数
def compute_metrics(outputs, targets, thresholds=None):
    """
    評価指標を計算する関数
    
    Args:
        outputs: モデルの出力（シグモイド適用済み）
        targets: 正解ラベル
        thresholds: 閾値のリスト（Noneの場合は[0.1, 0.2, ..., 0.9]）
    
    Returns:
        metrics: 評価指標の辞書
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    # CPU上のnumpy配列に変換
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # 各クラスのPR-AUCを計算
    from sklearn.metrics import precision_recall_curve, auc
    
    pr_aucs = []
    best_f1s = []
    best_thresholds = []
    
    # 各クラスごとに計算
    for i in range(targets_np.shape[1]):
        # クラスiのデータが存在する場合のみ計算
        if targets_np[:, i].sum() > 0:
            precision, recall, pr_thresholds = precision_recall_curve(targets_np[:, i], outputs_np[:, i])
            pr_auc = auc(recall, precision)
            pr_aucs.append(pr_auc)
            
            # 各閾値でのF1スコアを計算
            f1_scores = []
            for threshold in thresholds:
                predictions = (outputs_np[:, i] >= threshold).astype(int)
                tp = np.sum((predictions == 1) & (targets_np[:, i] == 1))
                fp = np.sum((predictions == 1) & (targets_np[:, i] == 0))
                fn = np.sum((predictions == 0) & (targets_np[:, i] == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
            
            # 最良のF1スコアとその閾値
            best_idx = np.argmax(f1_scores)
            best_f1 = f1_scores[best_idx]
            best_threshold = thresholds[best_idx]
            
            best_f1s.append(best_f1)
            best_thresholds.append(best_threshold)
    
    # マクロ平均
    macro_pr_auc = np.mean(pr_aucs) if pr_aucs else 0
    macro_f1 = np.mean(best_f1s) if best_f1s else 0
    mean_threshold = np.mean(best_thresholds) if best_thresholds else 0.5
    
    metrics = {
        'pr_auc': macro_pr_auc,
        'f1': macro_f1,
        'threshold': mean_threshold,
        'class_pr_aucs': pr_aucs,
        'class_f1s': best_f1s,
        'class_thresholds': best_thresholds
    }
    
    return metrics


# トレーニング関数
def train_model(
    model,
    train_loader,
    val_loader,
    tag_to_idx,
    idx_to_tag,
    optimizer,
    scheduler,
    criterion,
    num_epochs,
    device,
    output_dir,
    save_best='f1',
    checkpoint_interval=5,
    mixed_precision=False,
    tensorboard=True,
    existing_tags_count=0
):
    """
    モデルをトレーニングする関数
    
    Args:
        model: トレーニングするモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        tag_to_idx: タグから索引へのマッピング
        idx_to_tag: 索引からタグへのマッピング
        optimizer: オプティマイザ
        scheduler: 学習率スケジューラ
        criterion: 損失関数
        num_epochs: エポック数
        device: デバイス
        output_dir: 出力ディレクトリ
        save_best: 最良モデルの保存基準 ('f1', 'loss', 'both')
        checkpoint_interval: チェックポイント保存間隔
        mixed_precision: 混合精度を使用するかどうか
        tensorboard: TensorBoardを使用するかどうか
        existing_tags_count: 既存タグの数
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # TensorBoardの設定
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # 混合精度の設定
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.type != 'cpu' else None
    
    # 最良モデルの初期化
    best_f1 = 0.0
    best_loss = float('inf')
    
    # 初期評価
    print("初期評価を実行しています...")
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # シグモイド関数を適用
            probs = torch.sigmoid(outputs)
            all_outputs.append(probs)
            all_targets.append(targets)
    
    val_loss /= len(val_loader)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 評価指標の計算
    metrics = compute_metrics(all_outputs, all_targets)
    
    print(f"初期評価: Loss={val_loss:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}, Threshold={metrics['threshold']:.4f}")
    
    if tensorboard:
        writer.add_scalar('Loss/val', val_loss, 0)
        writer.add_scalar('Metrics/pr_auc', metrics['pr_auc'], 0)
        writer.add_scalar('Metrics/f1', metrics['f1'], 0)
        writer.add_scalar('Metrics/threshold', metrics['threshold'], 0)
    
    # トレーニングループ
    for epoch in range(1, num_epochs + 1):
        print(f"エポック {epoch}/{num_epochs}")
        
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 勾配のリセット
            optimizer.zero_grad()
            
            if mixed_precision and device.type != 'cpu':
                # 混合精度での順伝播
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # スケーラーを使用して逆伝播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 通常の順伝播と逆伝播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # シグモイド関数を適用
                probs = torch.sigmoid(outputs)
                all_outputs.append(probs)
                all_targets.append(targets)
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 評価指標の計算
        metrics = compute_metrics(all_outputs, all_targets)
        
        # 既存タグと新規タグの評価指標を分離
        if existing_tags_count > 0:
            existing_outputs = all_outputs[:, :existing_tags_count]
            existing_targets = all_targets[:, :existing_tags_count]
            new_outputs = all_outputs[:, existing_tags_count:]
            new_targets = all_targets[:, existing_tags_count:]
            
            existing_metrics = compute_metrics(existing_outputs, existing_targets)
            new_metrics = compute_metrics(new_outputs, new_targets)
            
            print(f"既存タグ: PR-AUC={existing_metrics['pr_auc']:.4f}, F1={existing_metrics['f1']:.4f}")
            print(f"新規タグ: PR-AUC={new_metrics['pr_auc']:.4f}, F1={new_metrics['f1']:.4f}")
            
            if tensorboard:
                writer.add_scalar('Metrics/existing_pr_auc', existing_metrics['pr_auc'], epoch)
                writer.add_scalar('Metrics/existing_f1', existing_metrics['f1'], epoch)
                writer.add_scalar('Metrics/new_pr_auc', new_metrics['pr_auc'], epoch)
                writer.add_scalar('Metrics/new_f1', new_metrics['f1'], epoch)
        
        print(f"エポック {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}, Threshold={metrics['threshold']:.4f}")
        
        # 学習率の更新
        if scheduler is not None:
            scheduler.step()
            if tensorboard:
                writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # TensorBoardへの記録
        if tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/pr_auc', metrics['pr_auc'], epoch)
            writer.add_scalar('Metrics/f1', metrics['f1'], epoch)
            writer.add_scalar('Metrics/threshold', metrics['threshold'], epoch)
        
        # 最良モデルの保存
        save_model = False
        if save_best == 'f1' and metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_model = True
            print(f"新しい最良F1: {best_f1:.4f}")
        elif save_best == 'loss' and val_loss < best_loss:
            best_loss = val_loss
            save_model = True
            print(f"新しい最良損失: {best_loss:.4f}")
        elif save_best == 'both':
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                save_model = True
                print(f"新しい最良F1: {best_f1:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                save_model = True
                print(f"新しい最良損失: {best_loss:.4f}")
        
        if save_model:
            # モデルの保存
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': val_loss,
                'f1': metrics['f1'],
                'pr_auc': metrics['pr_auc'],
                'threshold': metrics['threshold'],
                'tag_to_idx': tag_to_idx,
                'idx_to_tag': idx_to_tag,
                'existing_tags_count': existing_tags_count
            }
            
            # LoRAパラメータの保存（EVA02WithModuleLoRAの場合）
            if hasattr(model, 'lora_rank'):
                checkpoint['lora_rank'] = model.lora_rank
                checkpoint['lora_alpha'] = model.lora_alpha
                checkpoint['lora_dropout'] = model.lora_dropout
                checkpoint['target_modules'] = model.target_modules
            
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
            print(f"最良モデルを保存しました: {os.path.join(output_dir, 'best_model.pth')}")
        
        # 定期的なチェックポイントの保存
        if epoch % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': val_loss,
                'f1': metrics['f1'],
                'pr_auc': metrics['pr_auc'],
                'threshold': metrics['threshold'],
                'tag_to_idx': tag_to_idx,
                'idx_to_tag': idx_to_tag,
                'existing_tags_count': existing_tags_count
            }
            
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f"チェックポイントを保存しました: {os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')}")


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
    
    # トレーニングコマンド
    train_parser = subparsers.add_parser('train', help='LoRAモデルをトレーニングします')
    
    # データセット関連の引数
    train_parser.add_argument('--image_dirs', type=str, nargs='+', required=True, help='トレーニング画像のディレクトリ（複数指定可）')
    train_parser.add_argument('--val_split', type=float, default=0.1, help='検証データの割合')
    train_parser.add_argument('--min_tag_freq', type=int, default=5, help='タグの最小出現頻度')
    train_parser.add_argument('--remove_special_prefix', action='store_true', help='特殊プレフィックス（例：a@、g@など）を除去する')
    # train_parser.add_argument('--image_size', type=int, default=224, help='画像サイズ')
    train_parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    train_parser.add_argument('--num_workers', type=int, default=4, help='データローダーのワーカー数')
    
    # モデル関連の引数
    train_parser.add_argument('--lora_rank', type=int, default=4, help='LoRAのランク')
    train_parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRAのアルファ値')
    train_parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRAのドロップアウト率')
    train_parser.add_argument('--target_modules_file', type=str, default=None, help='LoRAを適用するモジュールのリストを含むファイル')
    
    # トレーニング関連の引数
    train_parser.add_argument('--num_epochs', type=int, default=10, help='エポック数')
    train_parser.add_argument('--learning_rate', type=float, default=1e-3, help='学習率')
    train_parser.add_argument('--weight_decay', type=float, default=0.01, help='重み減衰')
    train_parser.add_argument('--checkpoint_interval', type=int, default=5, help='チェックポイント保存間隔（エポック）')
    train_parser.add_argument('--save_best', type=str, default='f1', choices=['f1', 'loss', 'both'], help='最良モデルの保存基準')
    train_parser.add_argument('--output_dir', type=str, default='lora_model', help='出力ディレクトリ')
    
    # 損失関数関連の引数
    train_parser.add_argument('--loss_fn', type=str, default='bce', choices=['bce', 'asl', 'asl_optimized'], help='損失関数')
    train_parser.add_argument('--gamma_neg', type=float, default=4, help='ASL: 負例のガンマ値')
    train_parser.add_argument('--gamma_pos', type=float, default=1, help='ASL: 正例のガンマ値')
    train_parser.add_argument('--clip', type=float, default=0.05, help='ASL: クリップ値')
    
    # その他のオプション
    train_parser.add_argument('--mixed_precision', action='store_true', help='混合精度トレーニングを使用する')
    train_parser.add_argument('--use_8bit_optimizer', action='store_true', help='8-bitオプティマイザを使用する')
    train_parser.add_argument('--tensorboard', action='store_true', help='TensorBoardを使用する')
    train_parser.add_argument('--tensorboard_port', type=int, default=6006, help='TensorBoardのポート番号')
    train_parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    
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
    
    elif args.command == 'train':
        # 乱数シードの設定
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # デバイスの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {device}")
        
        # 既存のタグリストを読み込む
        print("既存のタグリストを読み込んでいます...")
        labels = load_labels_hf(repo_id=MODEL_REPO)
        existing_tags = labels.names
        print(f"既存タグ数: {len(existing_tags)}")
        
        # ターゲットモジュールの読み込み
        target_modules = None
        if args.target_modules_file:
            with open(args.target_modules_file, 'r') as f:
                target_modules = [line.strip() for line in f.readlines() if line.strip()]
            print(f"LoRAを適用するモジュール数: {len(target_modules)}")
        
        # 1. まずモデルを読み込む（この時点では既存タグのみ）
        print("モデルを読み込んでいます...")
        model = EVA02WithModuleLoRA(
            num_classes=None,  # 初期状態では既存タグのみ
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            pretrained=True
        )
        
        # モデルの期待する画像サイズを取得
        img_size = model.img_size
        print(f"モデルの期待する画像サイズを使用します: {img_size}")
        
        # 2. データセットの準備（新規タグが検出される）
        print("データセットを準備しています...")
        train_image_paths, train_tags_list, val_image_paths, val_tags_list, tag_to_idx, idx_to_tag = prepare_dataset(
            image_dirs=args.image_dirs,
            existing_tags=existing_tags,
            val_split=args.val_split,
            min_tag_freq=args.min_tag_freq,
            remove_special_prefix=args.remove_special_prefix,
            seed=args.seed
        )
        
        # 既存タグの数を記録
        existing_tags_count = len(set(existing_tags) & set(idx_to_tag.values()))
        print(f"使用される既存タグ数: {existing_tags_count}")
        print(f"新規タグ数: {len(idx_to_tag) - existing_tags_count}")
        
        # 3. モデルのヘッドを拡張（新規タグに対応）
        print("モデルのヘッドを拡張しています...")
        model.extend_head(num_classes=len(tag_to_idx))
        model = model.to(device)
        
        # データ変換の設定
        from timm.data import create_transform
        
        train_transform = create_transform(
            input_size=img_size,
            is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1'
        )
        
        val_transform = create_transform(
            input_size=img_size,
            is_training=False
        )
        
        # データセットの作成
        train_dataset = TagImageDataset(
            image_paths=train_image_paths,
            tags_list=train_tags_list,
            tag_to_idx=tag_to_idx,
            transform=train_transform,
            min_tag_freq=args.min_tag_freq,
            remove_special_prefix=args.remove_special_prefix
        )
        
        val_dataset = TagImageDataset(
            image_paths=val_image_paths,
            tags_list=val_tags_list,
            tag_to_idx=tag_to_idx,
            transform=val_transform,
            min_tag_freq=args.min_tag_freq,
            remove_special_prefix=args.remove_special_prefix
        )
        
        # データローダーの作成
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 損失関数の設定
        if args.loss_fn == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss_fn == 'asl':
            criterion = AsymmetricLoss(
                gamma_neg=args.gamma_neg,
                gamma_pos=args.gamma_pos,
                clip=args.clip
            )
        elif args.loss_fn == 'asl_optimized':
            criterion = AsymmetricLossOptimized(
                gamma_neg=args.gamma_neg,
                gamma_pos=args.gamma_pos,
                clip=args.clip
            )
        
        # オプティマイザの設定
        if args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay
                )
                print("8-bitオプティマイザを使用します")
            except ImportError:
                print("bitsandbytesがインストールされていないため、通常のAdamWを使用します")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay
                )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        
        # 学習率スケジューラの設定
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs
        )
        
        # TensorBoardの設定
        if args.tensorboard:
            try:
                import subprocess
                tensorboard_process = subprocess.Popen(
                    ['tensorboard', '--logdir', os.path.join(args.output_dir, 'tensorboard'), '--port', str(args.tensorboard_port)]
                )
                print(f"TensorBoardを起動しました: http://localhost:{args.tensorboard_port}")
            except Exception as e:
                print(f"TensorBoardの起動に失敗しました: {e}")
        
        # トレーニングの実行
        print("トレーニングを開始します...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tag_to_idx=tag_to_idx,
            idx_to_tag=idx_to_tag,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=args.num_epochs,
            device=device,
            output_dir=args.output_dir,
            save_best=args.save_best,
            checkpoint_interval=args.checkpoint_interval,
            mixed_precision=args.mixed_precision,
            tensorboard=True,
            existing_tags_count=existing_tags_count
        )
        
        # 最終モデルの保存
        final_checkpoint = {
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag,
            'existing_tags_count': existing_tags_count
        }
        
        # LoRAパラメータの保存
        if hasattr(model, 'lora_rank'):
            final_checkpoint['lora_rank'] = model.lora_rank
            final_checkpoint['lora_alpha'] = model.lora_alpha
            final_checkpoint['lora_dropout'] = model.lora_dropout
            final_checkpoint['target_modules'] = model.target_modules
        
        torch.save(final_checkpoint, os.path.join(args.output_dir, 'final_model.pth'))
        print(f"最終モデルを保存しました: {os.path.join(args.output_dir, 'final_model.pth')}")
        
        # タグマッピングの保存
        with open(os.path.join(args.output_dir, 'tag_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'tag_to_idx': tag_to_idx,
                'idx_to_tag': idx_to_tag,
                'existing_tags_count': existing_tags_count
            }, f, indent=2, ensure_ascii=False)
        
        print("トレーニングが完了しました！")
        
        # TensorBoardプロセスの終了
        if args.tensorboard and 'tensorboard_process' in locals():
            tensorboard_process.terminate()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    import math  # LoRALayer初期化に必要
    main()
