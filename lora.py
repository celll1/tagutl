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
from torchvision import transforms
import re
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from io import BytesIO
from collections import defaultdict

# デバイスの設定
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        base_model: str = 'SmilingWolf/wd-eva02-large-tagger-v3',
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
            'hf-hub:' + base_model, 
            pretrained=pretrained
        )
        
        # モデルの期待する画像サイズを取得
        self.img_size = self.backbone.patch_embed.img_size
        self.pretrained_cfg = self.backbone.pretrained_cfg
        self.transform = create_transform(**resolve_data_config(self.pretrained_cfg, model=self.backbone))

        print(f"モデルの期待する画像サイズ: {self.img_size}")
        
        # 特徴量の次元を取得
        self.feature_dim = self.backbone.head.in_features
        self.original_num_classes = self.backbone.head.out_features
        
        # 新しいクラス数が指定されている場合は、ヘッドを拡張
        if num_classes is not None and num_classes > self.original_num_classes:
            self._extend_head(num_classes)
            self.original_num_classes = num_classes
        
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
    
    def _extend_head(self, num_classes):
        """
        モデルのヘッドを拡張して新しいタグに対応する（内部メソッド）
        
        Args:
            num_classes: 新しい総クラス数（既存タグ + 新規タグ）
        """
        # 新規タグの数を計算
        self.num_new_classes = num_classes - self.original_num_classes
        
        if self.num_new_classes <= 0:
            print(f"新規タグがないか、既存タグ数より少ないため、ヘッドの拡張は行いません。(現在: {self.original_num_classes}, 要求: {num_classes})")
            return
        
        # 元のヘッドの重みとバイアスを取得
        original_weight = self.backbone.head.weight.data
        original_bias = self.backbone.head.bias.data if self.backbone.head.bias is not None else None
        
        # 新しいヘッドを作成
        new_head = nn.Linear(self.feature_dim, num_classes)
                
        # 新規タグの重みを初期化（Xavierの初期化）
        nn.init.xavier_uniform_(new_head.weight.data[self.original_num_classes:])
        if new_head.bias is not None:
            nn.init.zeros_(new_head.bias.data[self.original_num_classes:])

        # 既存タグの重みとバイアスを新しいヘッドに上書きする
        new_head.weight.data[:self.original_num_classes] = original_weight
        if original_bias is not None:
            new_head.bias.data[:self.original_num_classes] = original_bias
        
        # バックボーンのヘッドを新しいヘッドに置き換え
        self.backbone.head = new_head
        
        print(f"ヘッドを拡張しました: {self.original_num_classes} → {num_classes} クラス")
    
    def extend_head(self, num_classes):
        """
        モデルのヘッドを拡張して新しいタグに対応する（公開メソッド）
        
        Args:
            num_classes: 新しい総クラス数（既存タグ + 新規タグ）
        """
        self._extend_head(num_classes)
        # ヘッドのパラメータを訓練可能に設定
        for param in self.backbone.head.parameters():
            param.requires_grad = True
    
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


def load_model(model_path=None, metadata_path=None, base_model='SmilingWolf/wd-eva02-large-tagger-v3', device=torch_device):
    """モデルを読み込む関数"""
    print(f"モデルを読み込んでいます...")

    # huggingface のsmilingwolf のモデルの場合
    if model_path is None:
        # ラベルデータを読み込む
        labels = load_labels_hf(repo_id=base_model)
        num_classes = len(labels.names)
        
        idx_to_tag = {i: name for i, name in enumerate(labels.names)}
        tag_to_idx = {name: i for i, name in enumerate(labels.names)}

        # モデルの指定がない場合も EVA02WithModuleLoRA を使用
        print("初期化されたLoRAモデルを使用します（元のモデルと同等）")
        model = EVA02WithModuleLoRA(
            num_classes=num_classes,  # 元のモデルのクラス数
            lora_rank=4,              # デフォルト値
            lora_alpha=1.0,           # デフォルト値
            lora_dropout=0.0,         # デフォルト値
            pretrained=True
        ) 
    elif model_path.endswith('.pth') or model_path.endswith('.pt'):
        # ほかのローカルモデルの場合
        # トレーニング済みのLoRAが付与されたモデルの可能性がある
        print(f"LoRAチェックポイントを読み込んでいます: {model_path}")
        
        # PyTorch形式のチェックポイントを読み込む
        checkpoint = torch.load(model_path, map_location=device)
        
        # チェックポイントからLoRA設定を取得
        lora_rank = checkpoint.get('lora_rank', 4)
        lora_alpha = checkpoint.get('lora_alpha', 1.0) 
        lora_dropout = checkpoint.get('lora_dropout', 0.0)
        target_modules = checkpoint.get('target_modules', None)
        tag_to_idx = checkpoint.get('tag_to_idx', None)
        idx_to_tag = checkpoint.get('idx_to_tag', None)
        existing_tags_count = checkpoint.get('existing_tags_count', 0)
        
        # タグ数を決定
        if tag_to_idx is not None:
            num_classes = len(tag_to_idx)
            print(f"チェックポイントから読み込んだタグ数: {num_classes}")
            print(f"既存タグ数: {existing_tags_count}, 新規タグ数: {num_classes - existing_tags_count}")
            
            # タグマッピングがある場合は、ラベルを更新
            if idx_to_tag is not None:
                labels.names = [idx_to_tag[i] for i in range(len(idx_to_tag))]
        
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
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_path is not None and model_path.endswith('.safetensors'):
        # safetensors形式のチェックポイントを読み込む
        from safetensors.torch import load_file
        
        # メタデータを読み込む
        if metadata_path is None:
            metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
    
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # メタデータからLoRA設定を取得
            lora_rank = int(metadata.get('lora_rank', 4))
            lora_alpha = float(metadata.get('lora_alpha', 1.0))
            lora_dropout = float(metadata.get('lora_dropout', 0.0))
            
            # evalを使用する場合の注意点: 辞書のキーが整数になる
            try:
                target_modules = eval(metadata.get('target_modules', 'None'))
                tag_to_idx = eval(metadata.get('tag_to_idx', 'None'))
                idx_to_tag = eval(metadata.get('idx_to_tag', 'None'))
                tag_to_category = eval(metadata.get('tag_to_category', 'None'))
            except Exception as e:
                print(f"Warning: Error evaluating metadata: {e}")
                target_modules = None
                tag_to_idx = None
                idx_to_tag = None
                
            existing_tags_count = int(metadata.get('existing_tags_count', 0))
            
            # タグ数を決定
            if tag_to_idx is not None:
                num_classes = len(tag_to_idx)
                print(f"チェックポイントから読み込んだタグ数: {num_classes}")
                print(f"既存タグ数: {existing_tags_count}, 新規タグ数: {num_classes - existing_tags_count}")
                
                # デバッグ情報
                print(f"idx_to_tag type: {type(idx_to_tag)}")
                if idx_to_tag:
                    sample_key = next(iter(idx_to_tag.keys()))
                    print(f"Sample key type: {type(sample_key)}")
                
                # タグマッピングがある場合は、ラベルを更新
                if idx_to_tag is not None:
                    # 新しいLabelDataオブジェクトを作成
                    labels = LabelData(
                        names=[],
                        rating=[],
                        general=[],
                        character=[]
                    )
                    
                    try:
                        # キーの型を確認
                        is_int_key = False
                        if idx_to_tag:
                            sample_key = next(iter(idx_to_tag.keys()))
                            is_int_key = isinstance(sample_key, int)
                            print(f"Using {'integer' if is_int_key else 'string'} keys for idx_to_tag")
                        
                        # タグ名のリストを作成
                        tag_names = []
                        rating_indices = []
                        general_indices = []
                        character_indices = []
                        
                        # すべてのインデックスに対応するタグを取得
                        for i in range(num_classes):
                            key = i if is_int_key else str(i)
                            tag = idx_to_tag.get(key, f"unknown_{i}")
                            tag_names.append(tag)
                            
                            # カテゴリに基づいてインデックスを分類
                            if i < 4:  # 最初の4つはrating
                                rating_indices.append(np.int64(i))
                            elif tag_to_category and tag in tag_to_category:
                                category = tag_to_category[tag]
                                if category == 'Character':
                                    character_indices.append(np.int64(i))
                                else:  # General, Meta, Artist などはすべてGeneralとして扱う
                                    general_indices.append(np.int64(i))
                            else:
                                # カテゴリ情報がない場合はGeneralとして扱う
                                general_indices.append(np.int64(i))
                        
                        # LabelDataオブジェクトを更新
                        labels.names = tag_names
                        labels.rating = rating_indices
                        labels.general = general_indices
                        labels.character = character_indices
                        
                        print(f"Created labels with {len(tag_names)} tags")
                        print(f"Rating tags: {len(rating_indices)}")
                        print(f"General tags: {len(general_indices)}")
                        print(f"Character tags: {len(character_indices)}")
                        
                    except Exception as e:
                        print(f"Error creating labels: {e}")
                        # 最低限の情報でラベルを作成
                        labels.names = [f"tag_{i}" for i in range(num_classes)]
                        labels.rating = [np.int64(i) for i in range(min(4, num_classes))]
                        labels.general = [np.int64(i) for i in range(4, num_classes)]
                        labels.character = []

            # モデルの読み込み
            from safetensors.torch import load_file
            # deviceをstr型に変換して渡す
            device_str = str(device).split(':')[0]  # 'cuda:0' -> 'cuda', 'cpu' -> 'cpu'
            print(f"Loading safetensors with device: {device_str}")
            state_dict = load_file(model_path, device=device_str)

            # 出力層のサイズを取得
            num_classes = None
            for key in state_dict.keys():
                if 'head.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
            
            # モデルを作成
            model = EVA02WithModuleLoRA(
                base_model=base_model,
                num_classes=num_classes,  # safetensorsから取得したクラス数を使用
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                pretrained=True
            )
            
            # 状態辞書をモデルに読み込む
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"メタデータファイルが見つかりません: {metadata_path}")
    else:
        raise ValueError("モデルパスが指定されていません")  
    
    model = model.to(device)
    model.eval()

    return model, labels

# タグの正規化関数を拡張
def normalize_tag(tag, remove_special_prefixes=True):
    """タグを正規化する関数（スペースをアンダースコアに変換、エスケープ文字を処理など）"""
    # スペースをアンダースコアに変換
    tag = tag.replace(' ', '_')
    
    # エスケープされた文字を処理（例: \( → (）
    tag = re.sub(r'\\(.)', r'\1', tag)
    
    # 特殊プレフィックス（a@, g@など）を削除するオプション
    if remove_special_prefixes and re.match(r'^[a-zA-Z]@', tag):
        tag = tag[2:]  # プレフィックスを削除
    
    # 連続するアンダースコアを1つに
    tag = re.sub(r'_+', '_', tag)
    
    return tag

def read_tags_from_file(image_path, remove_special_prefix=True):
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

    # タグを正規化
    tags = [normalize_tag(tag, remove_special_prefixes=remove_special_prefix) for tag in tags]
    
    # print(f"Read {len(tags)} tags from {tag_path}")
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
    
def visualize_predictions_for_tensorboard(img_tensor, probs, idx_to_tag, threshold=0.35, original_tags=None,  existing_tags_count=None, tag_to_category=None):
    """
    TensorBoard 用に可視化した結果の画像を生成し、HWC形式のNumPy配列を返す関数。
    
    Args:
        img_tensor: 入力画像 (torch.Tensor, shape: C x H x W)
        probs: 各タグの予測確率（torch.Tensor または numpy配列, shape: (num_classes,)）
        idx_to_tag: インデックスからタグ名へのマッピング辞書
        threshold: タグの表示に用いる閾値
        original_tags: グラウンドトゥルースラベル (binary numpy配列または torch.Tensor, shape: (num_classes,))
        existing_tags_count: 既存タグの数
        tag_to_category: タグからカテゴリへのマッピング辞書
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch
    from collections import defaultdict

    # 確率値をnumpy配列に変換
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = probs
    
    # EVA02モデルでは入力時にRGB→BGRの変換が行われているため、
    # 表示時にはBGR→RGBに戻す必要がある
    
    # チャンネルの順序を元に戻す (BGR → RGB)
    # [2,1,0] → [0,1,2] の変換を行う
    if isinstance(img_tensor, torch.Tensor):
        img_rgb = img_tensor[[2, 1, 0]].detach().cpu()
    else:
        img_rgb = img_tensor
    
    # 正規化の逆変換（モデルの前処理に依存）
    # mean=0.5, std=0.5 で正規化されている
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    
    # 正規化の逆変換
    img_denorm = img_rgb * std + mean
    
    # [0,1]の範囲に収める
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    # PIL画像に変換
    to_pil = transforms.ToPILImage()
    image = to_pil(img_denorm)

    # グラウンドトゥルースタグリストを生成
    if original_tags is not None:
        if isinstance(original_tags, torch.Tensor):
            original_tags_np = original_tags.detach().cpu().numpy()
        else:
            original_tags_np = original_tags
        
        truth_indices = np.where(original_tags_np > 0.5)[0]
        truth_tags = [idx_to_tag[idx] for idx in truth_indices if idx in idx_to_tag]
    else:
        truth_tags = []
    
    # 正規化：スペース -> アンダースコア、エスケープされた括弧を通常に変換
    normalized_truth = []
    for tag in truth_tags:
        norm_tag = normalize_tag(tag)
        normalized_truth.append(norm_tag)
    
    # タグをカテゴリごとに分類（可視化のためだけに使用）
    category_tags = defaultdict(list)
    
    # タグを3つのカテゴリに分類
    above_threshold_in_tags = []      # 閾値以上かつグラウンドトゥルースに含まれる（True Positive）
    above_threshold_not_in_tags = []  # 閾値以上だがグラウンドトゥルースに含まれない（False Positive）
    below_threshold_in_tags = []      # 閾値未満だがグラウンドトゥルースに含まれる（False Negative）
    
    for idx, prob in enumerate(probs_np):
        if idx in idx_to_tag:
            tag = idx_to_tag[idx]
            normalized_tag = normalize_tag(tag)
            
            # タグのカテゴリを決定（可視化のためだけに使用）
            if tag_to_category and normalized_tag in tag_to_category:
                category = tag_to_category[normalized_tag]
            else:
                # カテゴリ情報がない場合はデフォルトでGeneral
                category = 'General'
            
            # 閾値に基づいて分類
            if prob >= threshold:
                if normalized_tag in normalized_truth:
                    above_threshold_in_tags.append((tag, prob, category))
                else:
                    above_threshold_not_in_tags.append((tag, prob, category))
            else:
                if normalized_tag in normalized_truth:
                    below_threshold_in_tags.append((tag, prob, category))
    
    # 降順ソート
    above_threshold_in_tags.sort(key=lambda x: x[1], reverse=True)
    above_threshold_not_in_tags.sort(key=lambda x: x[1], reverse=True)
    below_threshold_in_tags.sort(key=lambda x: x[1], reverse=True)
    
    # プロットの作成
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    
    # 左側: 入力画像
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    
    # 右側: タグ予測のバーグラフ
    all_display = above_threshold_in_tags + above_threshold_not_in_tags + below_threshold_in_tags
    tag_names = [t[0] for t in all_display]
    probs_list = [t[1] for t in all_display]
    
    # カラー指定
    colors = []
    for tag, prob, _ in all_display:
        norm_tag = normalize_tag(tag)
        if prob >= threshold and norm_tag in normalized_truth:
            colors.append('green')  # True Positive
        elif prob >= threshold:
            colors.append('red')    # False Positive
        else:
            colors.append('blue')   # False Negative
    
    # バーチャートの作成
    y_pos = range(len(tag_names))
    axs[1].barh(y_pos, probs_list, color=colors)
    axs[1].set_yticks(y_pos)
    axs[1].set_yticklabels(tag_names)
    axs[1].set_xlabel('Probability')
    axs[1].set_title('Tag Predictions')
    
    # 閾値ラインの表示
    axs[1].axvline(x=threshold, color='black', linestyle='--', alpha=0.7)
    if len(tag_names) > 0:
        axs[1].text(threshold, len(tag_names) - 1, f'Threshold: {threshold}', 
                    horizontalalignment='right', verticalalignment='bottom')
    
    # 凡例の追加
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive (threshold+ & in tags)'),
        Patch(facecolor='red', label='False Positive (threshold+ & not in tags)'),
        Patch(facecolor='blue', label='False Negative (threshold- & in tags)')
    ]
    axs[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    # プロットをPIL画像に変換
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # PIL画像をテンソルに変換
    pil_img = Image.open(buf)
    img_tensor = transforms.ToTensor()(pil_img)
    
    return img_tensor

def analyze_model_structure(base_model='SmilingWolf/wd-eva02-large-tagger-v3'):
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
    
    logger.info(f"モデル '{base_model}' の構造を分析しています...")
    
    # モデルをロード
    try:
        logger.info(f"Hugging Face Hub からモデルをロードしています...")
        model = timm.create_model('hf-hub:' + base_model, pretrained=True)
    except Exception as e:
        logger.error(f"モデルのロード中にエラーが発生しました: {e}")
        return
    
    # 結果を格納する辞書
    result = {
        "model_name": base_model,
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
    ):
        """
        画像とタグのデータセット
        
        Args:
            image_paths: 画像ファイルパスのリスト
            tags_list: 各画像に対応するタグのリスト（リストのリスト）
            tag_to_idx: タグから索引へのマッピング辞書
            transform: 画像変換関数
        """
        self.image_paths = image_paths
        self.tags_list = tags_list
        self.tag_to_idx = tag_to_idx
        self.transform = transform
        
        print(f"データセットのタグ数: {len(self.tag_to_idx)}")

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
    existing_tags, 
    val_split=0.1, 
    min_tag_freq=5, 
    remove_special_prefix=True,
    seed=42
):
    """データセットを準備する関数"""
    print("画像とタグを収集しています...")
    
    # 画像とタグを収集
    image_paths = []
    tags_list = []
    
    for image_dir in tqdm(image_dirs, desc="画像ディレクトリを処理中"):
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(root, file)
                    
                    # タグファイルのパスを取得
                    tag_file = os.path.splitext(image_path)[0] + '.txt'
                    
                    # タグファイルが存在する場合のみ処理
                    if os.path.exists(tag_file):
                        tags = read_tags_from_file(tag_file, remove_special_prefix=remove_special_prefix)
                        if tags:
                            image_paths.append(image_path)
                            tags_list.append(tags)
    
    print(f"収集された画像数: {len(image_paths)}")

    # タグの頻度を計算
    tag_freq = {}
    for tags in tags_list:
        for tag in tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
    
    # 最小頻度以上のタグを抽出
    filtered_tags = {tag for tag, freq in tag_freq.items() if freq >= min_tag_freq}

    # a@, p@など英字1文字＋@から始まるタグを除外
    if remove_special_prefix:
        filtered_tags = {tag for tag in filtered_tags if not re.match(r'^[a-zA-Z]@', tag)}
    
    new_filtered_tags = filtered_tags - set(existing_tags)
    
    print(f"既存タグ数: {len(existing_tags)}")
    print(f"最小頻度以上の新規タグ数: {len(new_filtered_tags)}")
    
    # タグをインデックスにマッピング
    # 既存タグは元のインデックスを維持し、新規タグは既存タグの後に追加
    tag_to_idx = {tag: i for i, tag in enumerate(existing_tags)}
    
    # 新規タグのインデックスを追加
    next_idx = len(existing_tags)
    for tag in sorted(new_filtered_tags):  # ソートして順序を一定に
        tag_to_idx[tag] = next_idx
        next_idx += 1
    
    # インデックスからタグへのマッピングを作成
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    
    print(f"使用されるタグの総数: {len(tag_to_idx)}")

    # idx-to-tag の最初の10件を表示
    print(f"idx-to-tag の最初の10件: {list(idx_to_tag.items())[:10]}")
    # new tagsのidx-to-tagを表示
    print(f"new tagsのidx-to-tag: {list(idx_to_tag.items())[len(existing_tags):len(existing_tags)+10]}")

    # データセットを分割
    indices = list(range(len(image_paths)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    val_size = int(len(indices) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_image_paths = [image_paths[i] for i in train_indices]
    train_tags_list = [tags_list[i] for i in train_indices]
    
    val_image_paths = [image_paths[i] for i in val_indices]
    val_tags_list = [tags_list[i] for i in val_indices]
    
    print(f"訓練データ数: {len(train_image_paths)}")
    print(f"検証データ数: {len(val_image_paths)}")
    
    return train_image_paths, train_tags_list, val_image_paths, val_tags_list, tag_to_idx, idx_to_tag


# 非対称損失関数（ASL）の実装
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """        
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # バッチ単位で平均を取る（サンプル数で割る）
        final_loss = -loss.mean()
        return final_loss


# 最適化された非対称損失関数
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    self.xs_pos_focus = self.xs_pos * self.targets
                    self.xs_neg_focus = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos_focus - self.xs_neg_focus,
                                              self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            else:
                self.xs_pos_focus = self.xs_pos * self.targets
                self.xs_neg_focus = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos_focus - self.xs_neg_focus,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            
            self.loss *= self.asymmetric_w

        # バッチ単位で平均を取る（サンプル数で割る）
        return -self.loss.mean()


# 評価指標の計算関数
def compute_metrics(outputs, targets):
    """
    評価指標を計算する関数
    
    Args:
        outputs: モデルの出力（シグモイド適用済み）
        targets: 正解ラベル
        thresholds: 閾値のリスト（Noneの場合は0.1から0.9まで0.05刻み）
    
    Returns:
        metrics: 評価指標の辞書
    """
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    import io
    from torchvision import transforms
    from sklearn.metrics import precision_recall_curve, auc, f1_score
       
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    # CPU上のnumpy配列に変換
    if isinstance(outputs, torch.Tensor):
        outputs_np = outputs.cpu().numpy()
    else:
        outputs_np = outputs

    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()
    else:
        targets_np = targets
    
    num_classes = targets_np.shape[1]
    f1_scores = np.zeros((len(thresholds), num_classes))
    pr_auc_scores = []
    
    # 各クラスごとのPR曲線とF1スコアを計算
    for i in range(num_classes):
        # クラスiのデータが十分にある場合のみ計算
        if np.sum(targets_np[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(targets_np[:, i], outputs_np[:, i])
            pr_auc = auc(recall, precision)
            pr_auc_scores.append(pr_auc)
            
            # 各閾値でのF1スコアを計算
            for t_idx, threshold in enumerate(thresholds):
                preds = (outputs_np[:, i] >= threshold).astype(int)
                if np.sum(preds) > 0:  # 予測に陽性がある場合のみ
                    f1 = f1_score(targets_np[:, i], preds)
                    f1_scores[t_idx, i] = f1
    
    # 平均PR-AUC
    macro_pr_auc = np.mean(pr_auc_scores) if pr_auc_scores else 0
    
    # 各閾値でのマクロF1スコア（クラス平均）
    macro_f1_scores = np.mean(f1_scores, axis=1)
    
    # 最適な閾値を選択（マクロF1スコアが最大となる閾値）
    best_threshold_idx = np.argmax(macro_f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_macro_f1 = macro_f1_scores[best_threshold_idx]
    
    # 最適な閾値での予測を作成
    predictions = (outputs_np >= best_threshold).astype(int)
    
    # クラスごとのF1スコア
    class_f1_scores = []
    for i in range(num_classes):
        if np.sum(predictions[:, i]) > 0 and np.sum(targets_np[:, i]) > 0:
            class_f1 = f1_score(targets_np[:, i], predictions[:, i])
            class_f1_scores.append(class_f1)
    
    # マクロF1スコア（クラス平均）
    macro_f1 = np.mean(class_f1_scores) if class_f1_scores else 0
    
    # F1スコア vs 閾値のプロット生成
    fig, ax = plt.subplots(figsize=(10, 6))
    for t_idx, threshold in enumerate(thresholds):
        ax.plot(t_idx, macro_f1_scores[t_idx], 'bo')
    ax.plot(best_threshold_idx, best_macro_f1, 'ro', markersize=10)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title(f'Macro F1 Score vs Threshold (Optimal Value: {best_threshold:.2f})')
    ax.grid(True)
    
    # プロットをバイトストリームに変換
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # 画像をテンソルに変換
    plot_image = transforms.ToTensor()(PILImage.open(buf))
    
    metrics = {
        'pr_auc': macro_pr_auc,
        'f1': macro_f1,
        'threshold': best_threshold,
        'class_f1s': class_f1_scores,
        'f1_vs_threshold_plot': plot_image
    }
    
    return metrics


# トレーニング関数
def train_model(
    model, 
    train_loader, 
    val_loader, 
    tag_to_idx,
    idx_to_tag,
    tag_to_category,
    optimizer, 
    scheduler, 
    criterion, 
    num_epochs, 
    device, 
    output_dir='lora_model',
    save_format='safetensors',
    save_best='f1',  # 'f1', 'loss', 'both'
    checkpoint_interval=1,
    mixed_precision=False,
    tensorboard=False,
    existing_tags_count=0,  # 既存タグの数
    initial_threshold=0.35,
    dynamic_threshold=True,
):
    """モデルをトレーニングする関数"""
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    def save_model(output_dir, filename, save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_f1, tag_to_idx, idx_to_tag, tag_to_category, existing_tags_count):
        os.makedirs(output_dir, exist_ok=True)
        
        # モデルの状態辞書
        model_state_dict = model.state_dict()
        
        # メタデータ情報
        metadata = {
            'epoch': epoch,
            'threshold': threshold,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'lora_rank': model.lora_rank,
            'lora_alpha': model.lora_alpha,
            'lora_dropout': model.lora_dropout,
            'target_modules': model.target_modules,
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag,
            'tag_to_category': tag_to_category,
            'existing_tags_count': existing_tags_count
        }
        
        # 完全なチェックポイント
        checkpoint = {
            'model_state_dict': model_state_dict,
            # 'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'threshold': threshold,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'lora_rank': model.lora_rank,
            'lora_alpha': model.lora_alpha,
            'lora_dropout': model.lora_dropout,
            'target_modules': model.target_modules,
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag,
            'tag_to_category': tag_to_category,
            'existing_tags_count': existing_tags_count
        }
        save_path = os.path.join(output_dir, filename)

        if save_format == 'safetensors':
            # safetensors形式ではテンソルのみ保存可能
            from safetensors.torch import save_file
            
            # モデルの状態辞書のみsafetensorsで保存
            save_file(model_state_dict, os.path.join(output_dir, f'{filename}.safetensors'), metadata={str(k): str(v) for k, v in metadata.items()})
            
            # その他の情報はJSON形式で保存
            other_info = {k: str(v) if isinstance(v, (dict, list, tuple)) else v for k, v in checkpoint.items() if k != 'model_state_dict'}
            with open(os.path.join(output_dir, f'{filename}_metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(other_info, f, indent=2, ensure_ascii=False)
            
            print(f"Model saved as {os.path.join(output_dir, f'{filename}.safetensors')} and metadata as {os.path.join(output_dir, f'{filename}_metadata.json')}")
        else:
            # PyTorch形式で保存
            torch.save(checkpoint, os.path.join(output_dir, f'{filename}.pt'))
            print(f"Model saved as {os.path.join(output_dir, f'{filename}.pt')}")
    
    # TensorBoardの設定を修正
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        
        # タイムスタンプを作成
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 実行設定の概要を含むrun名を作成
        run_name = f"run_{timestamp}_lr{optimizer.param_groups[0]['lr']}_bs{train_loader.batch_size}"
        
        # ディレクトリを作成
        tb_log_dir = os.path.join(output_dir, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        
        # run_nameをログディレクトリに含める
        writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, run_name))
        print(f"TensorBoard logs will be saved to: {os.path.join(tb_log_dir, run_name)}")
    
    # 混合精度トレーニングのスケーラー
    scaler = torch.amp.GradScaler(device=device) if mixed_precision else None
    
    # 最良のモデルを保存するための変数
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    # epochにより動的に閾値を設定する
    threshold = initial_threshold if initial_threshold is not None else 0.35
    
    # タグの総数
    total_tags = len(tag_to_idx)
    
    # 既存タグと新規タグのインデックスを分離
    existing_tag_indices = list(range(existing_tags_count))
    new_tag_indices = list(range(existing_tags_count, total_tags))
    
    print(f"既存タグ数: {len(existing_tag_indices)}, 新規タグ数: {len(new_tag_indices)}")

    # 初期検証（もともとの重みにより推論はある程度できるはず）
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Initial Validation")
        for i, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            
            if mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            
            # シグモイド関数で確率に変換
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(targets.cpu().numpy())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if i % 5 == 0 and i / 5 < 10:
                img_grid = visualize_predictions_for_tensorboard(images[0], probs[0], idx_to_tag, threshold=0.35, original_tags=targets[0], existing_tags_count=existing_tags_count)

                if tensorboard:
                    writer.add_image(f'predictions/val_{i}', img_grid, 0)
    
    # 検証メトリクスの計算
    val_loss /= len(val_loader)
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    val_metrics = compute_metrics(val_preds, val_targets)
    
    # 既存タグと新規タグに分けてメトリクスを計算
    if existing_tags_count > 0 and len(new_tag_indices) > 0:
        existing_val_metrics = compute_metrics(
            val_preds[:, existing_tag_indices], 
            val_targets[:, existing_tag_indices]
        )
        new_val_metrics = compute_metrics(
            val_preds[:, new_tag_indices], 
            val_targets[:, new_tag_indices]
        )
        
        print(f"初期検証結果 - 既存タグ F1: {existing_val_metrics['f1']:.4f}, 新規タグ F1: {new_val_metrics['f1']:.4f}")

        if tensorboard:
            writer.add_scalar('Metrics/val/F1', val_metrics['f1'], 0)
            writer.add_scalar('Metrics/val/PR-AUC', val_metrics['pr_auc'], 0)
            writer.add_scalar('Metrics/val/Threshold', threshold, 0)
            writer.add_image('Metrics/val/F1_vs_Threshold', val_metrics['f1_vs_threshold_plot'], 0)

            if existing_tags_count > 0 and len(new_tag_indices) > 0:
                writer.add_scalar('Metrics/val/F1_existing', existing_val_metrics['f1'], 0)
                writer.add_scalar('Metrics/val/F1_new', new_val_metrics['f1'], 0)
    else:
        print(f"初期検証結果 - F1: {val_metrics['f1']:.4f}")
        if tensorboard:
            writer.add_scalar('Metrics/val/F1', val_metrics['f1'], 0)
            writer.add_scalar('Metrics/val/PR-AUC', val_metrics['pr_auc'], 0)
            writer.add_scalar('Metrics/val/Threshold', threshold, 0)
            writer.add_image('Metrics/val/F1_vs_Threshold', val_metrics['f1_vs_threshold_plot'], 0)

    # トレーニングループ
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # トレーニングフェーズ
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        for i, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            
            # 混合精度トレーニング
            if mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            
            # シグモイド関数で確率に変換
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.append(probs)
            train_targets.append(targets.detach().cpu().numpy())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 定期的にTensorBoardに予測結果を記録
            if tensorboard:
                # stepごとのlossを記録
                writer.add_scalar('Train/Step_Loss', loss.item(), epoch * len(train_loader) + i)
                
                if i % 100 == 0:
                    img_grid = visualize_predictions_for_tensorboard(images[0], probs[0], idx_to_tag, threshold=0.35, original_tags=targets[0], existing_tags_count=existing_tags_count)
                    writer.add_image(f'predictions/train_epoch_{epoch}', img_grid, i)
        
        # 学習率のスケジューリング
        scheduler.step()
        if tensorboard:
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # トレーニングメトリクスの計算
        train_loss /= len(train_loader)
        train_preds = np.concatenate(train_preds)
        train_targets = np.concatenate(train_targets)
        
        train_metrics = compute_metrics(train_preds, train_targets)

        if dynamic_threshold:
            threshold = train_metrics['threshold']
        
        # 既存タグと新規タグに分けてメトリクスを計算
        if existing_tags_count > 0 and len(new_tag_indices) > 0:
            existing_train_metrics = compute_metrics(
                train_preds[:, existing_tag_indices], 
                train_targets[:, existing_tag_indices]
            )
            new_train_metrics = compute_metrics(
                train_preds[:, new_tag_indices], 
                train_targets[:, new_tag_indices]
            )

            if tensorboard:
                writer.add_scalar('Metrics/Train/Loss', train_loss, epoch+1)
                writer.add_scalar('Metrics/Train/F1_all', train_metrics['f1'], epoch+1)
                writer.add_scalar('Metrics/Train/Threshold', threshold, epoch+1)
                writer.add_image('Metrics/Train/F1_vs_Threshold', train_metrics['f1_vs_threshold_plot'], epoch+1)
                writer.add_scalar('Metrics/Train/F1_existing', existing_train_metrics['f1'], epoch+1)
                writer.add_scalar('Metrics/Train/F1_new', new_train_metrics['f1'], epoch+1)
        else:
            if tensorboard:
                writer.add_scalar('Metrics/Train/Loss', train_loss, epoch+1)
                writer.add_scalar('Metrics/Train/F1_all', train_metrics['f1'], epoch+1)
                writer.add_scalar('Metrics/Train/Threshold', threshold, epoch+1)
                writer.add_image('Metrics/Train/F1_vs_Threshold', train_metrics['f1_vs_threshold_plot'], epoch+1)

        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validation")
            for i, (images, targets) in enumerate(progress_bar):
                images = images.to(device)
                targets = targets.to(device)
                
                if mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # シグモイド関数で確率に変換
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(targets.cpu().numpy())
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if i % 5 == 0 and i / 5 < 10:
                    img_grid = visualize_predictions_for_tensorboard(images[0], probs[0], idx_to_tag, threshold=0.35, original_tags=targets[0], existing_tags_count=existing_tags_count)

                    if tensorboard:
                        writer.add_image(f'predictions/val_{i}', img_grid, epoch+1)
            
        # 検証メトリクスの計算
        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        val_metrics = compute_metrics(val_preds, val_targets)
        
        # 既存タグと新規タグに分けてメトリクスを計算
        if existing_tags_count > 0 and len(new_tag_indices) > 0:
            existing_val_metrics = compute_metrics(
                val_preds[:, existing_tag_indices], 
                val_targets[:, existing_tag_indices]
            )
            new_val_metrics = compute_metrics(
                val_preds[:, new_tag_indices], 
                val_targets[:, new_tag_indices]
            )
        
        # 結果の表示
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if existing_tags_count > 0 and len(new_tag_indices) > 0:
            print(f"既存タグ - Train F1: {existing_train_metrics['f1']:.4f}, Val F1: {existing_val_metrics['f1']:.4f}")
            print(f"新規タグ - Train F1: {new_train_metrics['f1']:.4f}, Val F1: {new_val_metrics['f1']:.4f}")
        
        # TensorBoardにメトリクスを記録
        if tensorboard:
            writer.add_scalar('Metrics/val/Loss', val_loss, epoch+1)
            writer.add_scalar('Metrics/val/F1', val_metrics['f1'], epoch+1)
            writer.add_scalar('Metrics/val/PR-AUC', val_metrics['pr_auc'], epoch+1)
            writer.add_scalar('Metrics/val/Threshold', threshold, epoch+1)
            writer.add_image('Metrics/val/F1_vs_Threshold', val_metrics['f1_vs_threshold_plot'], epoch+1)
            
            if existing_tags_count > 0 and len(new_tag_indices) > 0:
                writer.add_scalar('Metrics/val/F1_existing', existing_val_metrics['f1'], epoch+1)
                writer.add_scalar('Metrics/val/F1_new', new_val_metrics['f1'], epoch+1)
        
        # 最良のモデルを保存
        save_model_flag = False
        
        if save_best == 'f1' and val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_model_flag = True
        elif save_best == 'loss' and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_flag = True
        elif save_best == 'both':
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                save_model_flag = True
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model_flag = True

        if save_model_flag:
            # モデルの保存
            save_model(output_dir, f'best_model', save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_metrics['f1'], tag_to_idx, idx_to_tag, tag_to_category, existing_tags_count)
            print(f"Best model saved! (Val F1: {val_metrics['f1']:.4f}, Val Loss: {val_loss:.4f})")
    
        elif (epoch + 1) % checkpoint_interval == 0:
            save_model(output_dir, f'checkpoint_epoch_{epoch+1}', save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_metrics['f1'], tag_to_idx, idx_to_tag, tag_to_category, existing_tags_count)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # 最終モデルの保存
    save_model(output_dir, f'final_model', save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_metrics['f1'], tag_to_idx, idx_to_tag, tag_to_category, existing_tags_count)
    print(f"Final model saved! (Val F1: {val_metrics['f1']:.4f}, Val Loss: {val_loss:.4f})")

    save_tag_mapping(output_dir, idx_to_tag, tag_to_category)

    # 'optimizer_state_dict': optimizer.state_dict(),
    # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None, どうするか後で検討
    
    # Close the tensorboard writer
    if tensorboard:
        writer.close()
    
    return val_metrics

# def debug_model():
#     """
#     extend_head 前後の重みコピーおよび出力差分を比較するデバッグ関数
#     """
#     import copy
    
#     # デバイスの設定
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print("【検証1】extend_head 前のモデル（baseline）の読み込み")
#     model_baseline = EVA02WithModuleLoRA(
#         num_classes=None,  # 既存ヘッドのみ（pre-trained 状態）
#         lora_rank=4,
#         lora_alpha=1.0,
#         lora_dropout=0.0,
#         target_modules=None,
#         pretrained=True
#     )
#     model_baseline = model_baseline.to(device)
#     model_baseline.eval()
    
#     # baselineヘッドの重みを保持
#     baseline_head_weights = model_baseline.backbone.head.weight.data.clone()
#     baseline_head_bias = None
#     if model_baseline.backbone.head.bias is not None:
#         baseline_head_bias = model_baseline.backbone.head.bias.data.clone()
    
#     print("Baseline existing head weight stats: mean = {:.6f}, std = {:.6f}".format(
#          baseline_head_weights.mean().item(), baseline_head_weights.std().item()))
    
#     # 既存タグ数はモデル内部に保持されている (pre-trained ヘッドの out_features)
#     original_num = model_baseline.original_num_classes
#     # ここではシミュレーションとして新規タグ461件を追加する
#     total_classes = original_num + 461
#     print(f"【検証2】ヘッド拡張: {original_num} -> {total_classes} (新規タグ数: 461)")
    
#     # baselineモデルをコピーして、ヘッド拡張を実施
#     model_extended = copy.deepcopy(model_baseline)
#     model_extended.extend_head(num_classes=total_classes)
#     model_extended = model_extended.to(device)
#     model_extended.eval()
    
#     ext_head_weights = model_extended.backbone.head.weight.data.clone()
#     ext_head_bias = None
#     if model_extended.backbone.head.bias is not None:
#         ext_head_bias = model_extended.backbone.head.bias.data.clone()
    
#     # baselineの既存ヘッド部分と、extend後の既存ヘッド部分の差分を確認
#     diff_weights = ext_head_weights[:original_num] - baseline_head_weights
#     print("After extending head, difference stats for existing head portion:")
#     print("  - Difference weight: mean = {:.6f}, std = {:.6f}".format(
#          diff_weights.mean().item(), diff_weights.std().item()))
    
#     # ダミー入力を用意 (事前にモデルの期待する画像サイズを取得)
#     img_size = model_extended.img_size  # (height, width) 例：(448, 448)
#     dummy_input = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    
#     with torch.no_grad():
#         baseline_output = model_baseline(dummy_input)
#         extended_output = model_extended(dummy_input)
    
#     # 既存タグに該当する部分の出力を比較
#     baseline_existing = baseline_output[:, :original_num]
#     extended_existing = extended_output[:, :original_num]
    
#     diff_output = extended_existing - baseline_existing
#     print("Existing head output difference (after extend):")
#     print("  - Mean = {:.6f}, Std = {:.6f}".format(diff_output.mean().item(), diff_output.std().item()))
    
#     # 全体出力の統計も表示
#     print("Extended model overall output stats: mean = {:.6f}, std = {:.6f}".format(
#          extended_output.mean().item(), extended_output.std().item()))

#     print("\n【デバッグ検証終了】")

# def debug_id_mapping():
#     """
#     トレーニング時に用いられる tag_to_idx / idx_to_tag のマッピングが正しく構築されているかを確認するためのデバッグ関数
#     ＊ 特に、既存タグ（load_labels_hfで得られる）と、新規タグが正しく追加されているかを確認する
#     """
#     # 事前にload_labels_hfで得た既存のラベルリスト（pre-trainedで使われている順序）の取得
#     labels = load_labels_hf(repo_id=MODEL_REPO)
#     existing_tags = labels.names
#     num_existing = len(existing_tags)
#     print(f"pre-trainedから取得した既存タグ数: {num_existing}")
    
#     # ダミーの画像ディレクトリ（またはテスト用のタグリスト）を用意する
#     # ここでは既存タグに含まれていないタグを新規タグと仮定
#     # 例として、既存タグの一部だけ＆新規タグを含む簡易テストケースを作成します
#     image_paths = ["b.jpg", "f.jpg", "a.jpg"]
#     tags_list = [
#         # この画像は既存タグのみを含む
#         existing_tags[:10],
#         # この画像は既存タグと、新規タグ "new_tag_A", "new_tag_B" を含む
#         existing_tags[5:15] + ["new_tag_A", "new_tag_B"],
#         # この画像は新規タグのみ
#         ["new_tag_A", "new_tag_B", "new_tag_C"]
#     ]
    
#     # prepare_dataset 内部ではファイル入出力がありますので、ここではその一部処理のみ、つまり
#     # タグの頻度計算、フィルタリング、既存／新規タグの分離、そしてマッピングの構築部のみをシミュレーションします
#     tag_freq = {}
#     for tags in tags_list:
#         for tag in tags:
#             tag_freq[tag] = tag_freq.get(tag, 0) + 1
#     # min_tag_freq を1として全てのタグを採用（テスト目的）
#     filtered_tags = {tag for tag, freq in tag_freq.items() if freq >= 1}
    
#     # 既存タグと新規タグの分離
#     existing_filtered_tags = set(existing_tags) & filtered_tags
#     new_filtered_tags = filtered_tags - set(existing_tags)
    
#     print(f"フィルタリング後の既存タグ数: {len(existing_filtered_tags)}")
#     print(f"フィルタリング後の新規タグ数: {len(new_filtered_tags)}")
    
#     # マッピング作成
#     # 既存タグはそのままの順序で
#     tag_to_idx = {tag: i for i, tag in enumerate(existing_tags)}
    
#     next_idx = num_existing
#     for tag in sorted(new_filtered_tags):
#         tag_to_idx[tag] = next_idx
#         next_idx += 1
#     idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    
#     total_tags = len(tag_to_idx)
#     print(f"最終的に使用されるタグの総数: {total_tags}")
    
#     # マッピングの先頭部分（既存タグ側）と後半部分（新規タグ側）を確認
#     print("\n【既存タグマッピング（一部表示）】")
#     for i in range(min(10, num_existing)):
#         print(f"Index: {i} → {idx_to_tag[i]}")
        
#     print("\n【新規タグマッピング】")
#     for i in range(num_existing, total_tags):
#         print(f"Index: {i} → {idx_to_tag[i]}")
    
#     # 次に、シンプルな TagImageDataset を用いて、実際のターゲット生成結果を確認する
#     # 前処理に remove_special_prefix が有効の場合と無効の場合で確認可能
#     from torchvision import transforms
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
    
#     dataset = TagImageDataset(
#         image_paths=image_paths,
#         tags_list=tags_list,
#         tag_to_idx=tag_to_idx,
#         transform=transform,
#     )
    
#     print("\n【各サンプルのターゲットベクトル（nonzero indices）】")
#     for idx in range(len(dataset)):
#         image_tensor, target = dataset[idx]
#         nz = target.nonzero(as_tuple=False).squeeze().tolist()
#         print(f"Sample {idx}: ターゲットindices = {nz}")
    

def load_tag_categories():
    """
    taggroupディレクトリからタグカテゴリ情報を読み込む関数
    
    Returns:
        dict: タグからカテゴリへのマッピング辞書
    """
    tag_to_category = {}
    categories = ['Artist', 'Character', 'Copyright', 'General', 'Meta']
    
    for category in categories:
        json_path = os.path.join('taggroup', f'{category}.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    category_tags = json.load(f)
                    for tag in category_tags.keys():
                        # タグ名を正規化（スペースをアンダースコアに、エスケープされた括弧を通常に）
                        normalized_tag = tag.replace(' ', '_').replace('\\(', '(').replace('\\)', ')')
                        tag_to_category[normalized_tag] = category
                print(f"Loaded {len(category_tags)} tags for category {category}")
            except Exception as e:
                print(f"Error loading {category} tags: {e}")
    
    return tag_to_category

def save_tag_mapping(output_dir, idx_to_tag, tag_to_category):
    """
    idx_to_tag マッピングとカテゴリ情報を保存する関数
    
    Args:
        output_dir: 出力ディレクトリ
        idx_to_tag: インデックスからタグへのマッピング
        tag_to_category: タグからカテゴリへのマッピング
    """
    # idx - tag - category のマッピングを作成
    tag_mapping = {}
    for idx, tag in idx_to_tag.items():
        normalized_tag = tag.replace(' ', '_').replace('\\(', '(').replace('\\)', ')')
        category = tag_to_category.get(normalized_tag, 'General')  # デフォルトはGeneral
        tag_mapping[idx] = {
            'tag': tag,
            'category': category
        }
    
    # JSONとして保存
    os.makedirs(output_dir, exist_ok=True)
    mapping_path = os.path.join(output_dir, 'tag_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(tag_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Tag mapping saved to {mapping_path}")
    return mapping_path

def main():
    parser = argparse.ArgumentParser(description="SmilingWolf/wd-eva02-large-tagger-v3モデルを使用して画像のタグを推論し、LoRAトレーニングのための準備を行います。")
    subparsers = parser.add_subparsers(dest='command', help='コマンド')

    # ここに必要な引数を追加します（省略）
    # 分析コマンド
    analyze_parser = subparsers.add_parser('analyze', help='モデル構造を分析します')
    analyze_parser.add_argument('--base_model', type=str, default='SmilingWolf/wd-eva02-large-tagger-v3', help='使用するベースモデルのリポジトリ')
    
    # 推論コマンド
    predict_parser = subparsers.add_parser('predict', help='画像からタグを予測します')
    predict_parser.add_argument('--image', type=str, required=True, help='予測する画像ファイルのパス')
    predict_parser.add_argument('--base_model', type=str, default='SmilingWolf/wd-eva02-large-tagger-v3', help='使用するベースモデルのリポジトリ')
    predict_parser.add_argument('--model_path', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    predict_parser.add_argument('--metadata_path', type=str, default=None, help='モデルのメタデータファイルのパス（指定しない場合は元のモデルを使用）')
    predict_parser.add_argument('--output_dir', type=str, default='predictions', help='予測結果を保存するディレクトリ')
    predict_parser.add_argument('--gen_threshold', type=float, default=0.35, help='一般タグの閾値')
    predict_parser.add_argument('--char_threshold', type=float, default=0.75, help='キャラクタータグの閾値')
    
    # バッチ推論コマンド
    batch_parser = subparsers.add_parser('batch', help='複数の画像からタグを予測します')
    batch_parser.add_argument('--image_dir', type=str, required=True, help='予測する画像ファイルのディレクトリ')
    batch_parser.add_argument('--base_model', type=str, default='SmilingWolf/wd-eva02-large-tagger-v3', help='使用するベースモデルのリポジトリ')
    batch_parser.add_argument('--model_path', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    batch_parser.add_argument('--metadata_path', type=str, default=None, help='モデルのメタデータファイルのパス（指定しない場合は元のモデルを使用）')
    batch_parser.add_argument('--output_dir', type=str, default='predictions', help='予測結果を保存するディレクトリ')
    batch_parser.add_argument('--gen_threshold', type=float, default=None, help='一般タグの閾値')
    batch_parser.add_argument('--char_threshold', type=float, default=None, help='キャラクタータグの閾値')
    
    # トレーニングコマンド
    train_parser = subparsers.add_parser('train', help='LoRAモデルをトレーニングします')

    train_parser.add_argument('--base_model', type=str, default='SmilingWolf/wd-eva02-large-tagger-v3', help='使用するベースモデルのリポジトリ')
    train_parser.add_argument('--model_path', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    train_parser.add_argument('--metadata_path', type=str, default=None, help='モデルのメタデータファイルのパス（指定しない場合は_metadata.jsonを使用）')

    # データセット関連の引数
    train_parser.add_argument('--image_dirs', type=str, nargs='+', required=True, help='トレーニング画像のディレクトリ（複数指定可）')
    train_parser.add_argument('--val_split', type=float, default=0.1, help='検証データの割合')
    train_parser.add_argument('--min_tag_freq', type=int, default=5, help='タグの最小出現頻度')
    train_parser.add_argument('--remove_special_prefix', default=True, action='store_true', help='特殊プレフィックス（例：a@、g@など）を除去する')
    # train_parser.add_argument('--image_size', type=int, default=224, help='画像サイズ')
    train_parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ')
    train_parser.add_argument('--num_workers', type=int, default=4, help='データローダーのワーカー数')
    
    # モデル関連の引数
    train_parser.add_argument('--lora_rank', type=int, default=32, help='LoRAのランク')
    train_parser.add_argument('--lora_alpha', type=float, default=16, help='LoRAのアルファ値')
    train_parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRAのドロップアウト率')
    train_parser.add_argument('--target_modules_file', type=str, default=None, help='LoRAを適用するモジュールのリストを含むファイル')
    
    # トレーニング関連の引数
    train_parser.add_argument('--num_epochs', type=int, default=10, help='エポック数')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='学習率')
    train_parser.add_argument('--weight_decay', type=float, default=0.01, help='重み減衰')

    train_parser.add_argument('--initial_threshold', type=float, default=0.35, help='初期閾値')
    train_parser.add_argument('--dynamic_threshold', type=float, default=True, help='動的閾値')

    # モデルの保存関連の引数
    train_parser.add_argument('--save_format', type=str, default='safetensors', choices=['safetensors', 'pytorch'], help='モデルの保存形式')
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
    
    debug_parser = subparsers.add_parser('debug', help='モデルのデバッグを行います')

    args = parser.parse_args()
    
    if args.command == 'analyze':
        # モデル構造の分析
        analyze_model_structure(base_model=args.base_model)
    
    elif args.command == 'predict':
        # 単一画像の予測
        model, labels = load_model(args.model_path, args.metadata_path, base_model=args.base_model)
        
        # 画像に紐づくタグを読み込む
        actual_tags = read_tags_from_file(args.image)
        print(f"読み込まれたタグ: {len(actual_tags)}個")
        
        # 予測を実行
        img, caption, taglist, ratings, character, general, all_character, all_general = predict_image(
            args.image, 
            model, 
            labels, 
            gen_threshold=args.gen_threshold if args.gen_threshold is not None else model.threshold,
            char_threshold=args.char_threshold if args.char_threshold is not None else model.threshold
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
        print(f"Character tags (threshold={args.char_threshold if args.char_threshold is not None else model.threshold}):")
        for k, v in character.items():
            print(f"  {k}: {v:.3f}")
        
        print("--------")
        print(f"General tags (threshold={args.gen_threshold if args.gen_threshold is not None else model.threshold}):")
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
        model, labels = load_model(args.model_path, args.metadata_path, base_model=args.base_model)
        
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

        print(f"モデルを読み込んでいます...")
        model, labels = load_model(args.model_path, args.metadata_path, base_model=args.base_model, device=device)
        existing_tags = labels.names
        print(f"既存タグ数: {len(existing_tags)}")
        
        # ターゲットモジュールの読み込み
        target_modules = None
        if args.target_modules_file:
            with open(args.target_modules_file, 'r') as f:
                target_modules = [line.strip() for line in f.readlines() if line.strip()]
            print(f"LoRAを適用するモジュール数: {len(target_modules)}")
        
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
        existing_tags_count = len(existing_tags)
        print(f"既存タグ数: {len(set(existing_tags) & set(idx_to_tag.values()))}")
        print(f"新規タグ数: {len(idx_to_tag) - len(set(existing_tags) & set(idx_to_tag.values()))}")
        
        # 3. モデルのヘッドを拡張（新規タグに対応）
        print("モデルのヘッドを拡張しています...")
        total_classes = len(tag_to_idx)

        # 既存のモデルのクラス数と新しいタグ数を比較
        if total_classes > model.original_num_classes:
            model.extend_head(num_classes=total_classes)
        elif total_classes < model.original_num_classes:
            print(f"警告: 新しいタグ数({total_classes})が既存モデルのクラス数({model.original_num_classes})より少ないです。")
            print("既存モデルのヘッドをそのまま使用します。")
        else:
            print(f"タグ数が一致しています。ヘッドの拡張は不要です。({total_classes}クラス)")

        model = model.to(device)
        
        # データ変換の設定
        from timm.data import create_transform
        
        # データセットの作成
        train_dataset = TagImageDataset(
            image_paths=train_image_paths,
            tags_list=train_tags_list,
            tag_to_idx=tag_to_idx,
            transform=model.transform,
        )
        
        val_dataset = TagImageDataset(
            image_paths=val_image_paths,
            tags_list=val_tags_list,
            tag_to_idx=tag_to_idx,
            transform=model.transform,
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
        else:
            print("損失関数が指定されていません。BCEWithLogitsLossを使用します。")
            criterion = nn.BCEWithLogitsLoss()
        
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
                    ['tensorboard', '--logdir', os.path.join(args.output_dir, 'tensorboard_logs'), '--port', str(args.tensorboard_port)]
                )
                print(f"TensorBoardを起動しました: http://localhost:{args.tensorboard_port}")
            except Exception as e:
                print(f"TensorBoardの起動に失敗しました: {e}")

        tag_to_category = load_tag_categories()
        tag_to_category = {tag: category for tag, category in tag_to_category.items() 
                          if tag in tag_to_idx}
        
        # トレーニングの実行
        print("トレーニングを開始します...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tag_to_idx=tag_to_idx,
            idx_to_tag=idx_to_tag,
            tag_to_category=tag_to_category,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=args.num_epochs,
            device=device,
            output_dir=args.output_dir,
            save_format=args.save_format,
            save_best=args.save_best,
            checkpoint_interval=args.checkpoint_interval,
            mixed_precision=args.mixed_precision,
            tensorboard=True,
            existing_tags_count=existing_tags_count,
            initial_threshold=args.initial_threshold,
            dynamic_threshold=args.dynamic_threshold
        )
        
        print("トレーニングが完了しました！")
        
        # TensorBoardプロセスの終了
        if args.tensorboard and 'tensorboard_process' in locals():
            tensorboard_process.terminate()

    elif args.command == 'debug':
        debug_model()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    import math  # LoRALayer初期化に必要
    main()
