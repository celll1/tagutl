import os
import re
import json
import ast
import copy
import argparse
import requests

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from PIL import Image
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import timm
from timm.data import create_transform, resolve_data_config
import gc
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import transforms
import bitsandbytes as bnb
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from io import BytesIO
from tqdm.auto import tqdm

# ZClipのインポートを試みる
# MIT License
# https://github.com/bluorion-com/ZClip
try:
    from zclip import ZClip
    zclip_available = True
except ImportError:
    ZClip = None
    zclip_available = False

# デバイスの設定
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None

# xformersのインポート（オプション引数が有効な場合のみ）
def import_xformers():
    try:
        import xformers
        import xformers.ops
        print("xformersが正常にインポートされました")
        return True
    except ImportError:
        print("xformersをインポートできませんでした。pip install xformersでインストールしてください")
        return False
    
def replace_timm_attn_with_xformers(model, use_xformers=False):
    """
    EVA02モデルのattentionをxformersのメモリ効率の良いバージョンに置き換えます。

    ※注意:
       - attn_mask は元々利用されていないため、今回の乖離はスケーリングのタイミングに起因している可能性が高いです。
       - そのため、事前に q に対して self.scale を掛けるのではなく、xformers に内部スケーリングを任せます。

    Args:
        model: EVA02モデル
        use_xformers: xformersを使用するかどうか
    """
    if not use_xformers:
        return model

    if not import_xformers():
        print("xformersが利用できないため、標準のattentionを使用します")
        return model

    print("xformersが正常にインポートされました")

    import xformers.ops
    from timm.models.eva import EvaAttention, apply_rot_embed_cat

    class XFormersEvaAttention(nn.Module):
        def __init__(self, original_module):
            super().__init__()
            # 属性のコピー
            self.num_heads = original_module.num_heads
            self.scale = original_module.scale
            self.num_prefix_tokens = original_module.num_prefix_tokens
            self.fused_attn = False  # xformers利用時はFalse
            self.qkv_bias_separate = getattr(original_module, 'qkv_bias_separate', False)

            # 層のコピー
            self.qkv = original_module.qkv
            self.q_proj = original_module.q_proj
            self.k_proj = original_module.k_proj
            self.v_proj = original_module.v_proj
            self.q_bias = original_module.q_bias
            self.k_bias = original_module.k_bias
            self.v_bias = original_module.v_bias

            self.attn_drop = original_module.attn_drop
            self.norm = original_module.norm
            self.proj = original_module.proj
            self.proj_drop = original_module.proj_drop

        def forward(self, x, rope=None, attn_mask=None):
            B, N, C = x.shape

            # q, k, v計算（連続性のためcontiguous()を呼ぶ）
            if self.qkv is not None:
                if self.q_bias is None:
                    qkv = self.qkv(x)
                else:
                    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                    if self.qkv_bias_separate:
                        qkv = self.qkv(x)
                        qkv += qkv_bias
                    else:
                        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
                qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
                q, k, v = qkv.unbind(0)  # 各テンソル shape: (B, num_heads, N, head_dim)
            else:
                q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2).contiguous()
                k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2).contiguous()
                v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2).contiguous()

            # ropeの適用（rotary embedding）
            if rope is not None:
                npt = self.num_prefix_tokens
                q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
                k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

            # ※ここでは q の事前スケーリングを削除します
            # q = q * self.scale

            # xformers の attention 呼び出し時に内部スケーリングを指定
            try:
                x_out = xformers.ops.memory_efficient_attention(
                    q, k, v,
                    attn_bias=None,  # attn_mask はもともと利用されていない前提
                    p=self.attn_drop.p if self.training else 0.0,
                    scale=self.scale,  # 内部でのスケーリングを行う
                )
            except Exception as e:
                print(f"xformersのattentionでエラーが発生しました: {e}")
                print("標準のattentionにフォールバックします")
                attn = (q @ k.transpose(-2, -1))
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x_out = attn @ v

            # 元の形状に戻し、正規化・投影処理
            x_out = x_out.transpose(1, 2).reshape(B, N, -1)
            x_out = self.norm(x_out)
            x_out = self.proj(x_out)
            x_out = self.proj_drop(x_out)
            return x_out

    # モデル内の EvaAttention モジュールを置換
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, EvaAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                new_attn = XFormersEvaAttention(module)
                setattr(parent, child_name, new_attn)
                replaced_count += 1
                print(f"xformers対応に置き換え: {name}")

    print(f"モデルのattentionをxformersのメモリ効率の良いバージョンに置き換えました（{replaced_count}層）")
    return model

# SageAttentionのインポートを試みる
def import_sageattention():
    try:
        import sageattention
        print("sageattentionが正常にインポートされました")
        # sageattn 関数が存在するか確認
        if hasattr(sageattention, 'sageattn'):
             return True
        else:
            print("sageattentionのバージョンが古いか、正しくインストールされていません。sageattn関数が見つかりません。")
            return False
    except ImportError:
        print("sageattentionをインポートできませんでした。pip install sageattention==1.0.6 またはソースからビルドしてください")
        return False
    
def replace_timm_attn_with_sageattention(model, use_sageattention=False):
    """
    EVA02モデルのattentionをSageAttentionに置き換えます。

    Args:
        model: EVA02モデル
        use_sageattention: SageAttentionを使用するかどうか
    """
    if not use_sageattention:
        return model

    # ★★★ ここで import_sageattention() を呼び出す ★★★
    if not import_sageattention():
        print("sageattentionが利用できないため、標準のattentionを使用します")
        return model

    import sageattention # import_sageattention() で成功を確認しているので、ここでは単純にimport
    from timm.models.eva import EvaAttention, apply_rot_embed_cat

    class SageEvaAttention(nn.Module):
        def __init__(self, original_module):
            super().__init__()
            # 属性のコピー
            self.num_heads = original_module.num_heads
            self.scale = original_module.scale
            self.num_prefix_tokens = original_module.num_prefix_tokens
            # SageAttentionにはfused_attnやattn_dropに相当する直接的な引数はない
            # self.fused_attn = False # 不要
            # self.attn_drop = original_module.attn_drop # SageAttention内部では扱わない
            self.qkv_bias_separate = getattr(original_module, 'qkv_bias_separate', False)

            # 層のコピー
            self.qkv = original_module.qkv
            self.q_proj = original_module.q_proj
            self.k_proj = original_module.k_proj
            self.v_proj = original_module.v_proj
            self.q_bias = original_module.q_bias
            self.k_bias = original_module.k_bias
            self.v_bias = original_module.v_bias

            self.norm = original_module.norm
            self.proj = original_module.proj
            self.proj_drop = original_module.proj_drop # Attention後のDropoutは維持

        def forward(self, x, rope=None, attn_mask=None):
            B, N, C = x.shape

            # q, k, v計算（連続性のためcontiguous()を呼ぶ）
            if self.qkv is not None:
                if self.q_bias is None:
                    qkv = self.qkv(x)
                else:
                    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                    if self.qkv_bias_separate:
                        qkv = self.qkv(x)
                        qkv += qkv_bias
                    else:
                        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
                qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
                q, k, v = qkv.unbind(0)  # 各テンソル shape: (B, num_heads, N, head_dim)
            else:
                q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2).contiguous()
                k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2).contiguous()
                v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2).contiguous()

            # ropeの適用（rotary embedding）
            if rope is not None:
                npt = self.num_prefix_tokens
                q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
                k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

            # SageAttention呼び出し
            # attn_mask は is_causal で代替。現状のEVAではattn_mask未使用前提。
            # scale は SageAttention に渡す引数がないため、適用しない。
            # ドロップアウトもSageAttentionには引数がない。
            # データ型は FP16/BF16 が期待される。mixed_precision=True での実行を想定。
            try:
                # is_causal は元の EvaAttention にはないため、False固定とする
                # 必要であれば、モデルの使い方に応じて変更する
                is_causal = False
                # ★★★ sageattention モジュール名を直接使用 ★★★
                x_out = sageattention.sageattn(
                    q, k, v,
                    tensor_layout="HND",
                    is_causal=is_causal,
                    # scale=self.scale # sageattnにscale引数はない
                    # attn_bias=None # attn_maskはis_causalで代替
                    # p=self.attn_drop.p # sageattnにdropout引数はない
                )
            except Exception as e:
                print(f"SageAttentionの呼び出しでエラーが発生しました: {e}")
                print("標準のattentionにフォールバックします（性能に影響が出る可能性があります）")
                # フォールバックとして元の計算を模倣（ただし非効率）
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))
                attn = attn.softmax(dim=-1)
                # attn = self.attn_drop(attn) # ドロップアウトは適用できない
                x_out = attn @ v


            # 元の形状に戻し、正規化・投影処理
            x_out = x_out.transpose(1, 2).reshape(B, N, -1)
            x_out = self.norm(x_out)
            x_out = self.proj(x_out)
            x_out = self.proj_drop(x_out) # Attention後のdropoutは適用
            return x_out

    # モデル内の EvaAttention モジュールを置換
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, EvaAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                new_attn = SageEvaAttention(module)
                setattr(parent, child_name, new_attn)
                replaced_count += 1
                # print(f"SageAttention対応に置き換え: {name}") # tqdmとかぶるのでコメントアウト推奨

    print(f"モデルのattentionをSageAttentionに置き換えました（{replaced_count}層）")
    return model

# flash-attentionのインポートを試みる
def import_flash_attention():
    try:
        import flash_attn
        print("flash-attentionが正常にインポートされました")
        # flash_attn_func が存在するか確認 (v2.0以降の主要関数)
        if hasattr(flash_attn, 'flash_attn_func'):
            return True
        else:
            print("flash-attention v2.0以降が必要です。flash_attn_funcが見つかりません。")
            return False
    except ImportError:
        print("flash-attentionをインポートできませんでした。pip install flash-attn でインストールしてください")
        return False

def replace_timm_attn_with_flashattention(model, use_flashattention=False):
    """
    EVA02モデルのattentionをFlashAttention (v2)に置き換えます。

    Args:
        model: EVA02モデル
        use_flashattention: FlashAttentionを使用するかどうか
    """
    if not use_flashattention:
        return model

    if not import_flash_attention():
        print("flash-attentionが利用できないため、標準のattentionを使用します")
        return model

    print("flash-attentionを正常にインポートしました")

    from flash_attn import flash_attn_func
    from timm.models.eva import EvaAttention, apply_rot_embed_cat

    class FlashEvaAttention(nn.Module):
        def __init__(self, original_module):
            super().__init__()
            # 属性のコピー
            self.num_heads = original_module.num_heads
            self.scale = original_module.scale
            self.num_prefix_tokens = original_module.num_prefix_tokens
            self.attn_drop = original_module.attn_drop # flash_attn_funcに渡すため保持
            self.qkv_bias_separate = getattr(original_module, 'qkv_bias_separate', False)

            # 層のコピー
            self.qkv = original_module.qkv
            self.q_proj = original_module.q_proj
            self.k_proj = original_module.k_proj
            self.v_proj = original_module.v_proj
            self.q_bias = original_module.q_bias
            self.k_bias = original_module.k_bias
            self.v_bias = original_module.v_bias

            self.norm = original_module.norm
            self.proj = original_module.proj
            self.proj_drop = original_module.proj_drop # Attention後のDropoutは維持

        def forward(self, x, rope=None, attn_mask=None):
            B, N, C = x.shape

            # q, k, v計算（連続性のためcontiguous()を呼ぶ）
            if self.qkv is not None:
                if self.q_bias is None:
                    qkv = self.qkv(x)
                else:
                    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                    if self.qkv_bias_separate:
                        qkv = self.qkv(x)
                        qkv += qkv_bias
                    else:
                        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
                # FlashAttentionは (B, N, 3, num_heads, head_dim) を期待するため、permuteを調整
                qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
                q, k, v = qkv.unbind(2) # dim=2 で分割 -> (B, N, num_heads, head_dim)
                # FlashAttentionは (B, N, num_heads, head_dim) の入力を受け付ける
            else:
                # FlashAttention用に reshape を調整
                q = self.q_proj(x).reshape(B, N, self.num_heads, -1)
                k = self.k_proj(x).reshape(B, N, self.num_heads, -1)
                v = self.v_proj(x).reshape(B, N, self.num_heads, -1)


            # ropeの適用（rotary embedding）
            if rope is not None:
                npt = self.num_prefix_tokens
                # FlashAttentionの入力形状 (B, N, num_heads, head_dim) に合わせてropeを適用
                q_rope = q[:, npt:, :, :]
                k_rope = k[:, npt:, :, :]
                # apply_rot_embed_cat は (B, num_heads, N, C) を期待するため、transposeが必要
                q_rope_perm = q_rope.transpose(1, 2) # (B, num_heads, N-npt, head_dim)
                k_rope_perm = k_rope.transpose(1, 2) # (B, num_heads, N-npt, head_dim)
                q_rope_rotated = apply_rot_embed_cat(q_rope_perm, rope).transpose(1, 2) # (B, N-npt, num_heads, head_dim)
                k_rope_rotated = apply_rot_embed_cat(k_rope_perm, rope).transpose(1, 2) # (B, N-npt, num_heads, head_dim)

                q = torch.cat([q[:, :npt, :, :], q_rope_rotated], dim=1).type_as(v)
                k = torch.cat([k[:, :npt, :, :], k_rope_rotated], dim=1).type_as(v)


            # FlashAttention呼び出し (flash_attn_funcを使用)
            # 入力形状: (batch_size, seqlen, nheads, headdim)
            # causal=False (EvaAttentionでは未使用), softmax_scaleを設定
            try:
                # flash_attn_func は dropout_p を受け付ける
                x_out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    softmax_scale=self.scale, # 事前スケーリングの代わりに引数で渡す
                    causal=False, # EvaAttentionではcausalマスクは使われていない想定
                    # attn_bias は None (EvaAttentionでは未使用)
                    deterministic=True
                )
            except Exception as e:
                print(f"FlashAttentionの呼び出しでエラーが発生しました: {e}")
                print("標準のattentionにフォールバックします（性能に影響が出る可能性があります）")
                # フォールバック (xformers版から流用、ただし非効率)
                # (B, N, H, D) -> (B, H, N, D) に変換して計算
                q_perm = q.transpose(1, 2)
                k_perm = k.transpose(1, 2)
                v_perm = v.transpose(1, 2)
                q_perm = q_perm * self.scale # フォールバック時は事前スケール
                attn = (q_perm @ k_perm.transpose(-2, -1))
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn) # フォールバック時はDropout適用
                x_out_perm = attn @ v_perm
                # (B, H, N, D) -> (B, N, H, D) に戻す
                x_out = x_out_perm.transpose(1, 2)

            # 元の形状に戻し、正規化・投影処理
            # FlashAttentionの出力は (B, N, nheads*headdim) ではないので reshape の前に形状を確認
            # x_out は (B, N, num_heads, head_dim) のはず
            x_out = x_out.reshape(B, N, -1) # ここで (B, N, C) になるはず
            x_out = self.norm(x_out)
            x_out = self.proj(x_out)
            x_out = self.proj_drop(x_out) # Attention後のdropoutは適用
            return x_out

    # モデル内の EvaAttention モジュールを置換
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, EvaAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                new_attn = FlashEvaAttention(module)
                setattr(parent, child_name, new_attn)
                replaced_count += 1
                # print(f"FlashAttention対応に置き換え: {name}") # tqdmとかぶるのでコメントアウト推奨

    print(f"モデルのattentionをFlashAttentionに置き換えました（{replaced_count}層）")
    return model

# Loraレイヤーの定義
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=32, alpha=16.0):
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
    def __init__(self, base_layer, rank=32, alpha=16.0, dropout=0.0):
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
    artist: list[np.int64]
    character: list[np.int64]
    copyright: list[np.int64]
    meta: list[np.int64]
    quality: list[np.int64]


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
        artist=list(np.where(df["category"] == 1)[0]),
        character=list(np.where(df["category"] == 4)[0]),
        copyright=list(np.where(df["category"] == 3)[0]),
        meta=list(np.where(df["category"] == 5)[0]),
        quality=[],
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

    # Artist labels with all probabilities
    all_artist_labels = dict([probs_list[i] for i in labels.artist])

    # Character labels with all probabilities
    all_char_labels = dict([probs_list[i] for i in labels.character])

    # Copyright labels with all probabilities
    all_copyright_labels = dict([probs_list[i] for i in labels.copyright])

    # Meta labels with all probabilities
    all_meta_labels = dict([probs_list[i] for i in labels.meta])

    # Quality labels with all probabilities
    quality_labels = dict([probs_list[i] for i in labels.quality])

    # Filtered general labels (above threshold)
    gen_labels = {k: v for k, v in all_gen_labels.items() if v > gen_threshold}
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Filtered artist labels (above threshold)
    artist_labels = {k: v for k, v in all_artist_labels.items() if v > gen_threshold}
    artist_labels = dict(sorted(artist_labels.items(), key=lambda item: item[1], reverse=True))

    # Filtered character labels (above threshold)
    char_labels = {k: v for k, v in all_char_labels.items() if v > char_threshold}
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Filtered copyright labels (above threshold)
    copyright_labels = {k: v for k, v in all_copyright_labels.items() if v > gen_threshold}
    copyright_labels = dict(sorted(copyright_labels.items(), key=lambda item: item[1], reverse=True))

    # Filtered meta labels (above threshold)
    meta_labels = {k: v for k, v in all_meta_labels.items() if v > gen_threshold}
    meta_labels = dict(sorted(meta_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels for caption
    combined_names = list(gen_labels.keys())
    combined_names.extend(list(char_labels.keys()))

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, gen_labels, artist_labels, char_labels, quality_labels, copyright_labels, meta_labels, all_gen_labels, all_artist_labels, all_char_labels, all_copyright_labels, all_meta_labels


# EVA02モデルにモジュールごとのLoRAを適用するクラス
class EVA02WithModuleLoRA(nn.Module):
    def __init__(
        self, 
        base_model: str = 'SmilingWolf/wd-eva02-large-tagger-v3',
        model_path: str = None,
        num_classes=None,  # 初期化時にはNoneでも可能に
        threshold=0.35,
        idx_to_tag=None,
        tag_to_idx=None,
        tag_to_category=None,
        lora_rank=32, 
        lora_alpha=16, 
        lora_dropout=0.0,
        target_modules=None,
        pretrained=True
    ):
        super().__init__()

        self.base_model = base_model
        
        # LoRAのハイパーパラメータを保存
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.threshold = threshold
        self.idx_to_tag = idx_to_tag
        self.tag_to_idx = tag_to_idx
        self.tag_to_category = tag_to_category
        
        # バックボーンを作成
        self.backbone = timm.create_model(
            'hf-hub:' + self.base_model, 
            pretrained=pretrained
        )
        
        # モデルの期待する画像サイズを取得
        self.img_size = self.backbone.patch_embed.img_size
        self.pretrained_cfg = self.backbone.pretrained_cfg
        self.transform = create_transform(**resolve_data_config(self.pretrained_cfg, model=self.backbone))

        print(f"モデルの期待する画像サイズ: {self.img_size}")
        print(f"モデルの前処理設定：{self.pretrained_cfg}")
        
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

        if self.lora_rank is not None and self.lora_rank > 0:
            self.apply_lora_to_modules()
    
    def _extend_head(self, num_classes):
        """
        モデルのヘッドを拡張して新しいタグに対応する（内部メソッド）
        
        Args:
            num_classes: 新しい総クラス数（既存タグ + 新規タグ）
        """
        current_num_classes = self.backbone.head.out_features # ★★★ 現在の実際のクラス数を取得 ★★★
        self.num_new_classes = num_classes - current_num_classes # ★★★ 現在のクラス数と比較 ★★★

        if self.num_new_classes <= 0:
            # ★★★ メッセージのクラス数を修正 ★★★
            print(f"新規タグがないか、要求クラス数({num_classes})が現在のクラス数({current_num_classes})以下であるため、ヘッドの拡張は行いません。")
            # 拡張しない場合でも、original_num_classesは現在の状態に合わせておく
            self.original_num_classes = current_num_classes
            return

        print(f"ヘッドを拡張します: {current_num_classes} → {num_classes} クラス ({self.num_new_classes} 個の新規タグ)")

        # 元のヘッドの重みとバイアスを取得
        original_weight = self.backbone.head.weight.data
        original_bias = self.backbone.head.bias.data if self.backbone.head.bias is not None else None
        original_in_features = self.backbone.head.in_features # 入力特徴量を取得

        # 新しいヘッドを作成
        new_head = nn.Linear(original_in_features, num_classes) # 入力特徴量を指定

        # 新規タグの重みを初期化
        nn.init.zeros_(new_head.weight.data[current_num_classes:]) # ★★★ current_num_classes を使用 ★★★
        if new_head.bias is not None:
            nn.init.zeros_(new_head.bias.data[current_num_classes:]) # ★★★ current_num_classes を使用 ★★★

        # 既存タグの重みとバイアスを新しいヘッドに上書きする
        new_head.weight.data[:current_num_classes] = original_weight # ★★★ current_num_classes を使用 ★★★
        if original_bias is not None and new_head.bias is not None: # ★★★ new_head.bias もチェック ★★★
            new_head.bias.data[:current_num_classes] = original_bias # ★★★ current_num_classes を使用 ★★★

        # バックボーンのヘッドを新しいヘッドに置き換え
        self.backbone.head = new_head
        self.original_num_classes = num_classes # ★★★ 拡張後のクラス数に更新 ★★★

        print(f"ヘッドの拡張が完了しました。新しいクラス数: {num_classes}")
    
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
    
    def apply_lora_to_modules(self):
        """モデルの各モジュールにLoRAを適用する"""
        # 正規表現パターンをコンパイル
        patterns = [re.compile(pattern) for pattern in self.target_modules]
        
        # 新しいLoRA層を追跡
        self.lora_layers = {}
        
        # 各モジュールを調査
        for name, module in tqdm(self.backbone.named_modules(), desc="LoRA適用中", leave=True):
            # パターンに一致するか確認
            if isinstance(module, nn.Linear) and any(pattern.search(name) for pattern in patterns):
                # debug(コメントを消さない)
                # tqdm.write(f"Applying LoRA to module: {name}")
                
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

        self.freeze_non_lora_parameters()
        
        print(f"Applied LoRA to {len(self.lora_layers)} modules, rank: {self.lora_rank}, alpha: {self.lora_alpha}, dropout: {self.lora_dropout}")

    def _remove_lora_from_modules(self):
        """モデルからすべてのLoRA層を削除するメソッド"""
        # 削除したLoRA層の数をカウント
        removed_count = 0
        
        # 記録されているLoRA層を元に戻す
        for name, lora_module in list(self.lora_layers.items()):
            # debug(コメントを消さない)
            # print(f"Removing LoRA from module: {name}")
            
            # nameが"backbone."で始まる場合、それを取り除く
            if name.startswith("backbone."):
                name = name[9:]  # "backbone."の長さ(9文字)を取り除く
            
            # モジュールの親を取得
            if "." in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.backbone
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                # 直接backboneの属性の場合
                parent = self.backbone
                child_name = name
            
            # 元の線形層を取得して置き換え
            original_module = lora_module.base_layer
            setattr(parent, child_name, original_module)
            removed_count += 1
        
        # LoRA関連の属性をリセット
        self.lora_layers = {}
        
        print(f"Removed LoRA from {removed_count} modules")
        return self
    
    def freeze_non_lora_parameters(self):
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

    # def freeze_non_head_parameters(self):
    #     """ヘッドのパラメータのみ訓練可能に設定する"""
    #     for param in self.backbone.parameters():
    #         param.requires_grad = False

    #     for param in self.backbone.head.parameters():
    #         param.requires_grad = True

    def freeze_non_new_head_parameters(self):
        """新しいヘッドのパラメータのみ訓練可能に設定する"""
        # すべてのパラメータを凍結
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 新規タグのヘッド部分のみを訓練可能に設定
        if hasattr(self, 'original_num_classes') and hasattr(self.backbone.head, 'weight'):
            # ヘッドのパラメータを訓練可能に
            self.backbone.head.weight.requires_grad = True
            if hasattr(self.backbone.head, 'bias') and self.backbone.head.bias is not None:
                self.backbone.head.bias.requires_grad = True

            # 古いタグ部分の勾配を0にするためのフックを設定
            def weight_hook(grad):
                # 古いタグ部分の勾配を0に設定
                grad_clone = grad.clone()
                grad_clone[:self.original_num_classes] = 0
                return grad_clone

            def bias_hook(grad):
                # 古いタグ部分の勾配を0に設定
                grad_clone = grad.clone()
                grad_clone[:self.original_num_classes] = 0
                return grad_clone

            # フックを登録
            self.backbone.head.weight.register_hook(weight_hook)
            if hasattr(self.backbone.head, 'bias') and self.backbone.head.bias is not None:
                self.backbone.head.bias.register_hook(bias_hook)

    def remove_tags(self, tags_to_remove: list[str]):
        """
        モデルから指定されたタグを削除し、マッピングを更新する
        
        Args:
            tags_to_remove (list[str]): 削除するタグのリスト
        """
        print(f"\n削除対象のタグ数: {len(tags_to_remove)}")
        
        indices_to_remove = [
            self.tag_to_idx[tag]
            for tag in tags_to_remove
            if tag in self.tag_to_idx
        ]
        
        print(f"実際に削除可能なタグ数: {len(indices_to_remove)}")
        if len(indices_to_remove) < len(tags_to_remove):
            not_found = [tag for tag in tags_to_remove if tag not in self.tag_to_idx]
            print("見つからなかったタグ:")
            for tag in not_found[:10]:  # 最初の10個だけ表示
                print(f"  - {tag}")
            if len(not_found) > 10:
                print(f"  ... 他 {len(not_found) - 10} 個")
        
        if not indices_to_remove:
            print("削除可能なタグが見つかりませんでした。")
            return

        original_head = self.backbone.head
        original_size = original_head.out_features
        in_features = original_head.in_features # 入力次元を取得
        original_device = original_head.weight.device # 元のデバイスを取得

        keep_mask = torch.ones(original_size, dtype=torch.bool, device=original_device) # デバイスを合わせる
        keep_mask[indices_to_remove] = False

        # 保持する重みとバイアスを抽出
        new_weight = original_head.weight.data[keep_mask]
        new_bias_data = None
        has_bias = original_head.bias is not None
        if has_bias:
            new_bias_data = original_head.bias.data[keep_mask]

        new_size = new_weight.size(0)
        print(f"\nモデルのヘッドを再構築中...")
        print(f"ヘッドの次元を更新: {original_size} → {new_size}")

        # 新しい nn.Linear モジュールを作成
        new_head = nn.Linear(in_features, new_size, bias=has_bias, device=original_device) # デバイスを指定
        new_head.weight = nn.Parameter(new_weight)
        if has_bias:
            new_head.bias = nn.Parameter(new_bias_data)

        # バックボーンのヘッドを新しいヘッドに置き換え
        self.backbone.head = new_head
        print("モデルのヘッド再構築完了。")

        # マッピングの更新
        print("\nタグマッピングを更新中...")
        new_idx_to_tag = {}
        new_tag_to_idx = {}
        new_tag_to_category = {} # カテゴリ情報も更新

        new_idx = 0
        # 元のマッピング情報を保持 (remove_tagsが複数回呼ばれる可能性も考慮)
        original_idx_to_tag_copy = copy.deepcopy(self.idx_to_tag)
        original_tag_to_category_copy = copy.deepcopy(self.tag_to_category) if hasattr(self, 'tag_to_category') and self.tag_to_category else {}

        # 削除前のインデックスでループ
        num_tags_before_removal = original_size
        for old_idx in range(num_tags_before_removal):
            if old_idx not in indices_to_remove:
                # コピーした元のマッピングからタグを取得
                if old_idx in original_idx_to_tag_copy:
                    tag = original_idx_to_tag_copy[old_idx]
                    new_idx_to_tag[new_idx] = tag
                    new_tag_to_idx[tag] = new_idx
                    # カテゴリ情報も引き継ぐ
                    if tag in original_tag_to_category_copy:
                        new_tag_to_category[tag] = original_tag_to_category_copy[tag]
                    new_idx += 1
                else:
                    # print(f"Warning: old_idx {old_idx} not found in original_idx_to_tag during mapping update.")
                    pass # 元のマッピングにないインデックスはスキップ (発生しないはずだが念のため)

        # クラス変数を更新
        self.idx_to_tag = new_idx_to_tag
        self.tag_to_idx = new_tag_to_idx
        self.tag_to_category = new_tag_to_category
        self.original_num_classes = new_size # 削除後のクラス数を反映

        print(f"\n処理完了:")
        print(f"- 削除されたタグ数: {len(indices_to_remove)}")
        print(f"- 残りのタグ数: {len(self.idx_to_tag)}")

    def merge_lora_to_base_model(self, scale=1.0, new_lora_rank=None, new_lora_alpha=None, new_lora_dropout=None):
        """
        LoRAの重みをベースモデルにマージする
        
        Args:
            scale: マージ時のスケーリング係数（デフォルト: 1.0）
        
        Returns:
            マージされたレイヤーの数
        """
        # lora_layerが存在しない場合は、モデル内のLoRALinearモジュールを検索
        if not hasattr(self, 'lora_layers') or not self.lora_layers:
            print("lora_layersが見つかりません。モデル内のLoRALinearモジュールを検索します...")
            self.lora_layers = {}
            
            # モデル内のすべてのLoRALinearモジュールを検索
            for name, module in self.named_modules():
                if isinstance(module, LoRALinear):
                    self.lora_layers[name] = module
                    # print(f"Loraレイヤーを検出: {name}")
        
        if not self.lora_layers:
            print("マージするLoraレイヤーがありません。")
            return 0
        
        merged_count = 0
        for name, module in tqdm(self.lora_layers.items(), desc="LoRAレイヤーのマージ", leave=True):
            if isinstance(module, LoRALinear):
                # LoRAの計算: x @ (A @ B) * scale
                # ベースの重みに A @ B * scale を加算

                # debug(コメントを消さない)
                # tqdm.write(f"マージ中のモジュール: {name}")

                with torch.no_grad():
                    # 行列のサイズを確認
                    base_weight_shape = module.base_layer.weight.shape
                    lora_a_shape = module.lora.lora_A.shape
                    lora_b_shape = module.lora.lora_B.shape
                    
                    # debug(コメントを消さない)
                    # tqdm.write(f"base_weight: {base_weight_shape}, lora_A: {lora_a_shape}, lora_B: {lora_b_shape}")
                    
                    # 行列の掛け算と転置を適切に行う
                    if base_weight_shape[0] == lora_b_shape[1] and base_weight_shape[1] == lora_a_shape[0]:
                        # ケース1: base_weight が out_features x in_features で
                        # lora_A が in_features x rank, lora_B が rank x out_features の場合
                        # (B.T @ A.T).T = A @ B を計算
                        lora_weight = (module.lora.lora_B.t() @ module.lora.lora_A.t()).t()
                    elif base_weight_shape[0] == lora_a_shape[0] and base_weight_shape[1] == lora_b_shape[1]:
                        # ケース2: base_weight が out_features x in_features で
                        # lora_A が out_features x rank, lora_B が rank x in_features の場合
                        # A @ B を計算
                        lora_weight = module.lora.lora_A @ module.lora.lora_B
                    else:
                        # tqdm.write(f"  警告: 行列のサイズが一致しません。このモジュールはスキップします。")
                        continue
                    
                    # スケーリングを適用
                    lora_weight = lora_weight * (module.lora.scale * scale)
                    
                    # 行列のサイズを確認
                    # debug(コメントを消さない)
                    # tqdm.write(f"  計算されたlora_weight: {lora_weight.shape}")
                    
                    # ベースの重みに加算する前に次元が一致するか確認
                    if lora_weight.shape == base_weight_shape:
                        module.base_layer.weight.data += lora_weight.to(module.base_layer.weight.data.device)
                        merged_count += 1
                    else:
                        # tqdm.write(f"  警告: 計算されたLoRA重みのサイズ({lora_weight.shape})がベース重みのサイズ({base_weight_shape})と一致しません。")
                        # 転置して試してみる
                        if lora_weight.t().shape == base_weight_shape:
                            # tqdm.write(f"  転置後のサイズが一致するため、転置して加算します。")
                            module.base_layer.weight.data += lora_weight.t().to(module.base_layer.weight.data.device)
                            merged_count += 1
                        else:
                            # tqdm.write(f"このモジュールはスキップします。")
                            pass
                del module
        
        print(f"{merged_count}個のLoraレイヤーをベースモデルにマージしました（スケール: {scale}）")
        
        # マージ後はLoraレイヤーをリセット
        self._remove_lora_from_modules()
        
        gc.collect()
        torch.cuda.empty_cache()

        if new_lora_rank is not None:
            self.lora_rank = new_lora_rank
            if new_lora_alpha is not None:
                self.lora_alpha = new_lora_alpha
            if new_lora_dropout is not None:
                self.lora_dropout = new_lora_dropout

        if self.lora_rank is not None and self.lora_rank > 0:   
            # LoRAを再適用（空のLoRAを適用）
            self.apply_lora_to_modules()
            print(f"LoRAを再適用しました。")
        
        return merged_count
    
    def forward(self, x):
        # モデル全体を通して推論
        return self.backbone(x)
    
    def to(self, device):
        self.backbone.to(device)
        return self

def load_model(model_path=None,
               metadata_path=None,
               base_model='SmilingWolf/wd-eva02-large-tagger-v3',
               lora_rank=None,
               lora_alpha=None,
               lora_dropout=None,
               pretrained=True,
               device=torch_device,
               use_xformers=False, # xformersを使用するかどうかのフラグ
               use_sageattention=False, # SageAttentionを使用するかどうかのフラグを追加
               use_flashattention=False # FlashAttentionを使用するかどうかのフラグを追加
    ) -> tuple[EVA02WithModuleLoRA, LabelData]:
    """モデルを読み込む関数"""
    print(f" === Load Model === ")

    # xformersとsageattentionの同時使用は不可
    if use_xformers and use_sageattention:
        raise ValueError("xformersとSageAttentionを同時に有効にすることはできません。どちらか一方を選択してください。")
    if use_xformers:
        print("xformersを使用します。")
    if use_sageattention:
        print("SageAttentionを使用します。")
    if use_flashattention:
        print("FlashAttentionを使用します。")
    else:
        print("xformersとSageAttentionとFlashAttentionを使用しません。")    

    if model_path is None:
        print(f"ベースモデルを読み込んでいます...")

        # ラベルデータを読み込む
        print(f"ラベルデータを読み込んでいます...")
        labels = load_labels_hf(repo_id=base_model)
        num_classes = len(labels.names)
        
        idx_to_tag = {i: name for i, name in enumerate(labels.names)}
        tag_to_idx = {name: i for i, name in enumerate(labels.names)}
        tag_to_category = {name: 'Rating' if i in labels.rating else 'Character' if i in labels.character else 'General' for i, name in enumerate(labels.names)}
        print(f"カテゴリごとのタグ数: Rating: {len(labels.rating)}, Character: {len(labels.character)}, General: {len(labels.general)}, Artist: {len(labels.artist)}, Copyright: {len(labels.copyright)}, Meta: {len(labels.meta)}")

        print(f"初期化されたLoRAモデルを使用します（ベースモデル: {base_model}, lora_rank: {lora_rank}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}")

        model = EVA02WithModuleLoRA(
            base_model=base_model,
            model_path = None,
            num_classes=num_classes,  # 元のモデルのクラス数
            threshold=0.35,
            idx_to_tag=idx_to_tag,
            tag_to_idx=tag_to_idx,
            tag_to_category=tag_to_category,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            pretrained=pretrained
        )
    elif model_path.endswith('.pth') or model_path.endswith('.pt'):
        # ほかのローカルモデルの場合
        # トレーニング済みのLoRAが付与されたモデルの可能性がある
        print(f"LoRA適用モデル(PyTorch形式)を読み込んでいます: {model_path}")
        
        # PyTorch形式のチェックポイントを読み込む
        print(f"チェックポイントを読み込んでいます...")
        checkpoint = torch.load(model_path, map_location=device)

        # ベースモデル設定を上書き
        base_model = checkpoint.get('base_model', None)

        # チェックポイントからLoRA設定を取得
        print(f"チェックポイントからLoRA設定を取得しています...")
        lora_rank = checkpoint.get('lora_rank', lora_rank)
        lora_alpha = checkpoint.get('lora_alpha', lora_alpha) 
        lora_dropout = checkpoint.get('lora_dropout', lora_dropout)
        print(f"LoRA設定: lora_rank: {lora_rank}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}")
        target_modules = checkpoint.get('target_modules', None)
        threshold = checkpoint.get('threshold', 0.35)
        tag_to_idx = checkpoint.get('tag_to_idx', None)
        idx_to_tag = checkpoint.get('idx_to_tag', None)
        tag_to_category = checkpoint.get('tag_to_category', None)

        labels = LabelData(names=[], rating=[], general=[], character=[], artist=[], copyright=[], meta=[], quality=[])

        if idx_to_tag is not None:
            num_classes = len(idx_to_tag)
            labels.names = [idx_to_tag[i] for i in range(len(idx_to_tag))]
            labels.rating = [i for i, name in enumerate(labels.names) if tag_to_category[name] == 'Rating']
            labels.general = [i for i, name in enumerate(labels.names) if tag_to_category[name] == 'General']
            labels.artist = [i for i, name in enumerate(labels.names) if tag_to_category[name] == 'Artist']
            labels.character = [i for i, name in enumerate(labels.names) if tag_to_category[name] == 'Character']
            labels.meta = [i for i, name in enumerate(labels.names) if tag_to_category[name] == 'Meta']
            labels.quality = [i for i, name in enumerate(labels.names) if tag_to_category[name] == 'Quality']
            
        else:
            # データが欠落しているときは、ベースモデルから読み込む
            print(f"チェックポイントからラベルデータが見つかりません。ベースモデルから読み込んでいます...")
            labels = load_labels_hf(repo_id=base_model)
            num_classes = len(labels.names)
            idx_to_tag = {i: name for i, name in enumerate(labels.names)}
            tag_to_idx = {name: i for i, name in enumerate(labels.names)}
            tag_to_category = {name: 'Rating' if i in labels.rating else 'Character' if i in labels.character else 'General' for i, name in enumerate(labels.names)}
            # 'general': 0, 'sensitive': 1, 'questionable': 2, 'explicit': 3はratingに入るように
            # labels.rating.extend([i for i, name in enumerate(labels.names) if name not in tag_to_category and name in ['general', 'sensitive', 'questionable', 'explicit']])
            # labels.general.extend([i for i, name in enumerate(labels.names) if name not in tag_to_category and name not in ['general', 'sensitive', 'questionable', 'explicit']])  

        print(f"カテゴリごとのタグ数: Rating: {len(labels.rating)}, Character: {len(labels.character)}, General: {len(labels.general)}, Artist: {len(labels.artist)}, Copyright: {len(labels.copyright)}, Meta: {len(labels.meta)}")
        
        # モデルを作成
        model = EVA02WithModuleLoRA(
            base_model=base_model,
            model_path = model_path,
            num_classes=num_classes,
            threshold=threshold,
            idx_to_tag=idx_to_tag,
            tag_to_idx=tag_to_idx,
            tag_to_category=tag_to_category,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            pretrained=False
        )
        # 状態辞書をモデルに読み込む
        model.load_state_dict(checkpoint['model_state_dict'])

        # Loraレイヤーを明示的に検出
        model.lora_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                model.lora_layers[name] = module

        print(f"検出されたLoraレイヤー数: {len(model.lora_layers)}")

    elif model_path is not None and model_path.endswith('.safetensors'):
        # safetensors形式のチェックポイントを読み込む
        from safetensors.torch import load_file
        
        # メタデータを読み込む
        if metadata_path is None:
            metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
            print(f"Read metadata from {metadata_path}")
    
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # メタデータからLoRA設定を取得
            try:
                base_model = metadata.get('base_model', 'SmilingWolf/wd-eva02-large-tagger-v3')
                lora_rank = metadata.get('lora_rank', lora_rank)
                lora_alpha = metadata.get('lora_alpha', lora_alpha)
                lora_dropout = metadata.get('lora_dropout', lora_dropout)
                print(f"LoRA設定: lora_rank: {lora_rank}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}")
            except Exception as e:
                print(f"チェックポイントからLoRA設定が見つかりません...")
            
            # evalを使用する場合の注意点: 辞書のキーが整数になる
            try:
                target_modules = eval(metadata.get('target_modules', 'None'))
            except Exception as e:
                print(f"チェックポイントからtarget_modulesが見つかりません...")
                target_modules = None

            labels = LabelData(names=[], rating=[], general=[], character=[], artist=[], copyright=[], meta=[], quality=[])

            try:
                idx_to_tag = ast.literal_eval(metadata['idx_to_tag'])
                tag_to_idx = ast.literal_eval(metadata['tag_to_idx'])
                tag_to_category = ast.literal_eval(metadata['tag_to_category'])
                threshold = float(metadata.get('threshold', 0.35))
                if idx_to_tag is not None:
                    num_classes = len(idx_to_tag)
                    labels.names = [idx_to_tag[i] for i in range(len(idx_to_tag))]
                    labels.rating = [i for i, name in enumerate(labels.names) if tag_to_category.get(name) == 'Rating']
                    labels.general = [i for i, name in enumerate(labels.names) if tag_to_category.get(name) == 'General']
                    labels.artist = [i for i, name in enumerate(labels.names) if tag_to_category.get(name) == 'Artist']
                    labels.character = [i for i, name in enumerate(labels.names) if tag_to_category.get(name) == 'Character'] 
                    labels.meta = [i for i, name in enumerate(labels.names) if tag_to_category.get(name) == 'Meta']
                    labels.quality = [i for i, name in enumerate(labels.names) if tag_to_category.get(name) == 'Quality']

                    # tag_to_category のキーに存在しないtagは、暫定的にGeneralに分類
                    # 'general': 0, 'sensitive': 1, 'questionable': 2, 'explicit': 3はratingに入るように
                    labels.rating.extend([i for i, name in enumerate(labels.names) if name not in tag_to_category and name in ['general', 'sensitive', 'questionable', 'explicit']])
                    labels.general.extend([i for i, name in enumerate(labels.names) if name not in tag_to_category and name not in ['general', 'sensitive', 'questionable', 'explicit']])

                    # 'best_quality', 'high_quality', 'normal_quality', 'medium_quality', 'low_quality', 'bad_quality', 'worst_quality'はqualityに入るように
                    labels.quality.extend([i for i, name in enumerate(labels.names) if name not in tag_to_category and name in ['best_quality', 'high_quality', 'normal_quality', 'medium_quality', 'low_quality', 'bad_quality', 'worst_quality']])
                    
            except Exception as e:
                print(f"チェックポイントからラベルデータが見つかりません。{e}, ベースモデルから読み込んでいます...")
                target_modules = None
                labels = load_labels_hf(repo_id=base_model)
                num_classes = len(labels.names)
                threshold = 0.35
                idx_to_tag = {i: name for i, name in enumerate(labels.names)}
                tag_to_idx = {name: i for i, name in enumerate(labels.names)}
                tag_to_category = {name: 'Rating' if i in labels.rating else 'Character' if i in labels.character else 'General' for i, name in enumerate(labels.names)}

            print(f"カテゴリごとのタグ数: Rating: {len(labels.rating)}, Character: {len(labels.character)}, General: {len(labels.general)}, Artist: {len(labels.artist)}, Copyright: {len(labels.copyright)}, Meta: {len(labels.meta)}, Quality: {len(labels.quality)}")

            # モデルの読み込み
            from safetensors.torch import load_file
            # deviceをstr型に変換して渡す
            device_str = str(device).split(':')[0]  # 'cuda:0' -> 'cuda', 'cpu' -> 'cpu'
            print(f"Loading safetensors with device: {device_str}")
            state_dict = load_file(model_path, device=device_str)

            # 出力層のサイズを取得
            head_size = None
            for key in state_dict.keys():
                if 'head.weight' in key:
                    head_size = state_dict[key].shape[0]
                    break
            if head_size != num_classes:
                print(f"警告: ベースモデルのラベル数とチェックポイントのラベル数が一致しません。")
                print(f"ベースモデルのラベル数: {num_classes}, チェックポイントのラベル数: {head_size}")
            
            # モデルを作成
            model = EVA02WithModuleLoRA(
                base_model=base_model,
                model_path = model_path,
                num_classes=num_classes,  # safetensorsから取得したクラス数を使用
                threshold=threshold,
                idx_to_tag=idx_to_tag,
                tag_to_idx=tag_to_idx,
                tag_to_category=tag_to_category,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                pretrained=False
            )

            model.load_state_dict(state_dict)
            
            # Loraレイヤーを明示的に検出
            model.lora_layers = {}
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    model.lora_layers[name] = module
            
            print(f"検出されたLoraレイヤー数: {len(model.lora_layers)}")
        else:
            raise ValueError(f"メタデータファイルが見つかりません: {metadata_path}")
    else:
        # ベースモデルもチェックポイントも指定されない場合 (以前はValueErrorだったが、初期LoRAを許容するならここに来る)
        print(f"モデルパスが指定されていません。ベースモデル '{base_model}' から初期化します。")
        # ラベルデータを読み込む
        print(f"ラベルデータを読み込んでいます...")
        labels = load_labels_hf(repo_id=base_model)
        num_classes = len(labels.names)

        idx_to_tag = {i: name for i, name in enumerate(labels.names)}
        tag_to_idx = {name: i for i, name in enumerate(labels.names)}
        # カテゴリ情報を付与 (load_labels_hf の情報だけでは不十分なため、taggroup から読み込む方が良いが、ここでは暫定)
        tag_to_category = {name: 'Rating' if i in labels.rating else
                                  'Character' if i in labels.character else
                                  'General' for i, name in enumerate(labels.names)}
        print(f"カテゴリごとのタグ数: Rating: {len(labels.rating)}, Character: {len(labels.character)}, General: {len(labels.general)}, Artist: {len(labels.artist)}, Copyright: {len(labels.copyright)}, Meta: {len(labels.meta)}, Quality: {len(labels.quality)}")

        print(f"初期化されたLoRAモデルを使用します（ベースモデル: {base_model}, lora_rank: {lora_rank}, lora_alpha: {lora_alpha}, lora_dropout: {lora_dropout}")

        model = EVA02WithModuleLoRA(
            base_model=base_model,
            model_path=None,
            num_classes=num_classes,
            threshold=0.35,
            idx_to_tag=idx_to_tag,
            tag_to_idx=tag_to_idx,
            tag_to_category=tag_to_category,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            pretrained=pretrained # ベースからなのでTrue
        )

    # xformers または SageAttention を使用する場合はattentionを置き換え
    if use_xformers:
        model = replace_timm_attn_with_xformers(model, use_xformers=True)
    elif use_sageattention:
        model = replace_timm_attn_with_sageattention(model, use_sageattention=True)
    elif use_flashattention:
        model = replace_timm_attn_with_flashattention(model, use_flashattention=True)

    model = model.to(device)
    model.eval()

    print(f" === Load Model Completed === ")

    return model, labels

# タグの正規化関数を拡張
def normalize_tag(tag):
    """タグを正規化する関数（スペースをアンダースコアに変換、エスケープ文字を処理など）"""
    # スペースをアンダースコアに変換
    tag = tag.replace(' ', '_')
    
    # エスケープされた文字を処理（例: \( → (）
    tag = re.sub(r'\\(.)', r'\1', tag)
    
    # 連続するアンダースコアを1つに
    tag = re.sub(r'_+', '_', tag)
    
    return tag

def read_tags_from_file(image_path, remove_special_prefix="remove"):
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
    tags = [normalize_tag(tag) for tag in tags]

    # タグを処理
    processed_tags = []
    for tag in tags:
        # 特殊プレフィックスを持つタグの処理
        if re.match(r'^[a-zA-Z]@', tag):
            if remove_special_prefix == "remove":
                # プレフィックスを削除して追加
                processed_tags.append(tag[2:])
            elif remove_special_prefix == "omit":
                # 特殊プレフィックスを持つタグは完全にスキップ
                continue
        else:
            # 特殊プレフィックスを持たないタグはそのまま正規化して追加
            processed_tags.append(tag)
    
    return processed_tags

def predict_image(image_path, model, labels, gen_threshold = None, char_threshold = None, device=torch_device):
    """画像からタグを予測する関数"""
    # 画像の読み込みと前処理
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        img_input = Image.open(BytesIO(response.content))
    else:
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
    caption, taglist, ratings, general, artist, character, copyright, meta, quality, all_general, all_artist, all_character, all_copyright, all_meta = get_tags(
        probs=outputs.squeeze(0),
        labels=labels,
        gen_threshold=gen_threshold if gen_threshold is not None else model.threshold,
        char_threshold=char_threshold if char_threshold is not None else model.threshold,
    )
    
    return img_input, caption, taglist, ratings, general, artist, character, copyright, meta, quality, all_general, all_artist, all_character, all_copyright, all_meta


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
    caption, taglist, ratings, character, general, meta, quality, all_character, all_general = predictions
    
    # タグの正規化（スペースを_に変換、エスケープされた括弧を通常の括弧に変換）
    normalized_tags = [normalize_tag(tag) for tag in tags]
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
    # total_tags = len(above_threshold_in_tags) + len(above_threshold_not_in_tags) + len(below_threshold_in_tags)
    # if total_tags > max_tags:
    #     # 各カテゴリから均等に選択
    #     tags_per_category = max(1, max_tags // 3)
    #     above_threshold_in_tags = above_threshold_in_tags[:tags_per_category]
    #     above_threshold_not_in_tags = above_threshold_not_in_tags[:tags_per_category]
    #     below_threshold_in_tags = below_threshold_in_tags[:tags_per_category]
    
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
    
def visualize_predictions_for_tensorboard(img_tensor, probs, idx_to_tag, threshold=0.35, original_tags=None, tag_to_category=None, neg_means=None, neg_variance=None, neg_counts=None):
    """
    TensorBoard 用に可視化した結果の画像を生成し、HWC形式のNumPy配列を返す関数。
    
    Args:
        img_tensor: 入力画像 (torch.Tensor, shape: C x H x W)
        probs: 各タグの予測確率（torch.Tensor または numpy配列, shape: (num_classes,)）
        idx_to_tag: インデックスからタグ名へのマッピング辞書
        threshold: タグの表示に用いる閾値
        original_tags: グラウンドトゥルースラベル (binary numpy配列または torch.Tensor, shape: (num_classes,))
        tag_to_category: タグからカテゴリへのマッピング辞書
        neg_means: 陰性群の平均値
        neg_variance: 陰性群の分散
        neg_counts: 陰性群のサンプル数
    """

    # 確率値をnumpy配列に変換
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = probs
    
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
    
    # 偽陽性を100件までに制限
    above_threshold_not_in_tags = above_threshold_not_in_tags[:100]
    
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
    
    # 陰性群の平均値を追加
    if all(x is not None for x in [neg_means, neg_variance, neg_counts]):
        # タグごとのインデックスを取得
        for i, (tag, prob, category) in enumerate(all_display):
            # 偽陽性（赤いバー）のタグのみを処理
            if colors[i] == 'red':
                # 元のインデックスを取得
                tag_idx = [idx for idx, t in idx_to_tag.items() if t == tag][0]
                
                mean_val = neg_means[tag_idx]
                std_val = np.sqrt(neg_variance[tag_idx])
                n = neg_counts[tag_idx]
                
                # シグモイド関数の出力値を対数オッズに戻す
                mean_logit = np.log(mean_val + 1e-7) - np.log(1 - mean_val + 1e-7)
                
                # ロジットスケールでの標準誤差を計算
                # デルタ法による分散の伝播
                sigmoid_derivative = mean_val * (1 - mean_val)
                std_logit = std_val / sigmoid_derivative
                
                # 95%信頼区間の計算（ロジットスケール）
                from scipy import stats

                eps = 0.0005 / len(neg_means) # Bonferroni correction

                if n < 30:
                    ci_logit = stats.t.ppf(1-eps, df=n-1) * (std_logit / np.sqrt(n))
                else:
                    ci_logit = stats.norm.ppf(1-eps) * (std_logit / np.sqrt(n))
                
                # ロジットスケールの信頼区間を確率スケールに変換
                ci_lower = 1 / (1 + np.exp(-(mean_logit - ci_logit)))
                ci_upper = 1 / (1 + np.exp(-(mean_logit + ci_logit)))
                # 有意な統計情報がある場合のみ表示
                if mean_val > 0 and n >= 5:  # サンプル数が5以上の場合のみ表示
                    # 平均値をシアンのドットで表示
                    axs[1].scatter(
                        mean_val,
                        i,
                        color='blue',
                        alpha=0.7,
                        s=30,
                        zorder=5
                    )
                    
                    # 95%信頼区間をエラーバーとして表示
                    axs[1].hlines(
                        y=i,
                        xmin=ci_lower,
                        xmax=ci_upper,
                        color='blue',
                        alpha=0.5,
                        linewidth=2,
                        zorder=4
                    )
                    
                    # 統計情報をテキストで表示
                    stats_text = f'{ci_lower:.2f} ~ {ci_upper:.2f}'
                    
                    # テキストの位置を調整（信頼区間の右端から少し離す）
                    text_x = min(ci_upper + 0.02, 0.98)
                    axs[1].text(
                        x=text_x,          # x座標
                        y=i,               # y座標
                        s=stats_text,      # 表示テキスト
                        color='blue',
                        alpha=1,
                        fontsize=10,
                        verticalalignment='center'
                    )

    # 凡例の追加
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='green', label='True Positive'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative'),
        Line2D([0], [0], color='cyan', marker='o', linestyle='-', 
               label='Neg. Mean & 95% CI', markersize=5, alpha=0.7)
    ]
    axs[1].legend(handles=legend_elements, loc='lower right', fontsize=8)
    
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
        cache_dir=None,  # キャッシュディレクトリを指定するパラメータ
        force_recache=False,  # キャッシュを強制的に再作成するフラグ
    ):
        """
        画像とタグのデータセット
        
        Args:
            image_paths: 画像ファイルパスのリスト
            tags_list: 各画像に対応するタグのリスト（リストのリスト）
            tag_to_idx: タグから索引へのマッピング辞書
            transform: 画像変換関数
            cache_dir: キャッシュディレクトリ（Noneの場合はキャッシュを使用しない）
            force_recache: 既存のキャッシュを無視して再作成するかどうか
        """
        self.image_paths = image_paths
        self.tags_list = tags_list
        self.tag_to_idx = tag_to_idx
        self.transform = transform
        self.cache_dir = cache_dir
        self.force_recache = force_recache
        
        print(f"データセットのタグ数: {len(self.tag_to_idx)}")
        
        # キャッシュ関連の初期化
        self.using_cache = cache_dir is not None
        self.cache_metadata_file = None
        self.cache_files_exist = False
        
        # キャッシュディレクトリが指定されている場合
        if self.using_cache:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_metadata_file = os.path.join(cache_dir, 'metadata.json')
            
            # メタデータファイルが存在するか確認
            if os.path.exists(self.cache_metadata_file) and not force_recache:
                with open(self.cache_metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # キャッシュが有効か確認（データサイズや構成が一致するか）
                if (metadata.get('dataset_size') == len(self.image_paths) and 
                    metadata.get('num_classes') == len(self.tag_to_idx)):
                    self.cache_files_exist = True
                    print(f"有効なキャッシュメタデータを見つけました: {len(self.image_paths)}個のサンプル")
                else:
                    print(f"キャッシュメタデータが一致しません。新しいキャッシュを作成します。")
                    self.cache_files_exist = False
            
            # キャッシュが存在しない場合は作成
            if not self.cache_files_exist or force_recache:
                self._create_cache()
    
    def _get_cache_path(self, idx):
        """インデックスからキャッシュファイルのパスを生成"""
        return os.path.join(self.cache_dir, f"sample_{idx}.pt")
    
    def _create_cache(self):
        """個別のファイルにデータをキャッシュする"""
        print("データセットのキャッシュを作成中...")
        
        # キャッシュ作成のために情報を収集
        dataset_info = {
            'dataset_size': len(self.image_paths),
            'num_classes': len(self.tag_to_idx),
            'missing_samples': []  # 読み込みに失敗したサンプルのリスト
        }
        
        for idx in tqdm(range(len(self.image_paths)), desc="Caching dataset"):
            # 画像を読み込み前処理を適用
            try:
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
                
                # 個別のファイルとしてキャッシュに保存
                cache_path = self._get_cache_path(idx)
                torch.save((image, label), cache_path)
                
            except (Image.UnidentifiedImageError, OSError, IOError) as e:
                # 画像が読み込めない場合のエラーをスキップし、メタデータに記録
                print(f"Warning: キャッシュ作成中に画像の読み込みに失敗しました: {img_path}, エラー: {str(e)}")
                dataset_info['missing_samples'].append(idx)
        
        # メタデータをディスクに保存
        with open(self.cache_metadata_file, 'w') as f:
            json.dump(dataset_info, f)
            
        print(f"キャッシュの作成が完了しました。{len(self.image_paths)}個のサンプルが処理されました。")
        print(f"読み込みに失敗したサンプル: {len(dataset_info['missing_samples'])}")
        
        # キャッシュが作成されたことを記録
        self.cache_files_exist = True

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # キャッシュを使用している場合はキャッシュファイルからデータを取得
        if self.using_cache and self.cache_files_exist:
            cache_path = self._get_cache_path(idx)
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    print(f"Warning: キャッシュファイルの読み込みに失敗しました: {cache_path}, エラー: {str(e)}")
                    # キャッシュ読み込み失敗時は通常の処理にフォールバック
        
        # キャッシュがない場合や読み込み失敗時は通常通り処理
        # 最大再試行回数
        max_retries = 5
        current_idx = idx
        
        for _ in range(max_retries):
            # 画像の読み込みと前処理
            img_path = self.image_paths[current_idx]
            
            try:
                image = Image.open(img_path)
                image = pil_ensure_rgb(image)
                image = pil_pad_square(image)
                
                if self.transform:
                    image = self.transform(image)
                    # RGB to BGR for EVA02 model
                    image = image[[2, 1, 0]]
                
                # タグをone-hotエンコーディング
                tags = self.tags_list[current_idx]
                num_classes = len(self.tag_to_idx)
                label = torch.zeros(num_classes)
                
                for tag in tags:
                    if tag in self.tag_to_idx:
                        label[self.tag_to_idx[tag]] = 1.0
                
                # キャッシュが有効でファイルが存在しない場合は、遅延キャッシュを作成
                if self.using_cache and self.cache_files_exist:
                    cache_path = self._get_cache_path(current_idx)
                    if not os.path.exists(cache_path):
                        torch.save((image, label), cache_path)
                
                return image, label
                
            except (Image.UnidentifiedImageError, OSError, IOError) as e:
                # 画像が読み込めない場合のエラーログ
                print(f"Warning: 画像の読み込みに失敗しました: {img_path}, エラー: {str(e)}")
                
                # 次のインデックスを試す
                current_idx = (current_idx + 1) % len(self)
        
        # すべての再試行が失敗した場合、ダミーデータを返す
        print(f"Error: {max_retries}回の再試行後も画像を読み込めませんでした。ダミーデータを返します。")
        dummy_image = torch.zeros(3, 224, 224)  # モデルの入力サイズに合わせて調整
        dummy_target = torch.zeros(len(self.tag_to_idx))
        
        return dummy_image, dummy_target


# データセットの準備関数
def prepare_dataset(
    model: EVA02WithModuleLoRA,
    image_dirs,  # メインの学習データディレクトリ
    reg_image_dirs=None,  # 正則化用データディレクトリ（オプション）
    val_split=0.1,
    min_tag_freq=5,
    remove_special_prefix="remove",
    seed=42,
    cache_dir=None,  # キャッシュディレクトリのパラメータを追加
    tags_to_remove=None # 削除対象のタグリストを追加
):
    """
    データセットを準備し、必要に応じてモデルのヘッドを拡張します。
    メインデータセットと正則化データセットを別々に処理し、
    タグの統合と拡張は一度だけ行います。

    Args:
        tags_to_remove: データセットとモデルから削除するタグのリスト

    Returns:
        train_image_paths: 訓練用画像パスのリスト
        train_tags_list: 訓練用タグのリスト
        val_image_paths: 検証用画像パスのリスト
        val_tags_list: 検証用タグのリスト
        reg_image_paths: 正則化用画像パスのリスト
        reg_tags_list: 正則化用タグのリスト
        num_existing_tags: 既存タグの数（タグ削除後）
        num_new_tags: 新規追加タグの数
    """
    import random
    import re # re をインポート

    # シャッフルのためのシードを設定
    random.seed(seed)
    np.random.seed(seed)

    print("画像とタグを収集しています...")
    tags_to_remove_set = set()
    if tags_to_remove:
        print(f"データセットから {len(tags_to_remove)} 個のタグを削除します。")
        tags_to_remove_set = set(tags_to_remove) # 高速な検索のためセットに変換

    # メインデータセットから画像とタグを収集
    main_image_paths = []
    main_tags_list = []

    for image_dir in tqdm(image_dirs, desc="メインディレクトリを処理中"):
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(root, file)
                    try:
                        tags = read_tags_from_file(image_path, remove_special_prefix)
                        if tags_to_remove_set:
                            # 削除対象タグを除外
                            tags = [tag for tag in tags if tag not in tags_to_remove_set]

                        if tags:  # タグが存在する場合のみ追加
                            main_image_paths.append(image_path)
                            main_tags_list.append(tags)
                    except Exception as e:
                        print(f"画像 {image_path} の処理中にエラーが発生: {e}")

    print(f"メインデータセットの画像数: {len(main_image_paths)}")

    # 正則化データセットから画像とタグを収集（指定がある場合）
    reg_image_paths = []
    reg_tags_list = []

    if reg_image_dirs:
        for image_dir in tqdm(reg_image_dirs, desc="正則化ディレクトリを処理中"):
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_path = os.path.join(root, file)
                        try:
                            tags = read_tags_from_file(image_path, remove_special_prefix)
                            if tags_to_remove_set:
                                # 削除対象タグを除外
                                tags = [tag for tag in tags if tag not in tags_to_remove_set]

                            if tags:  # タグが存在する場合のみ追加
                                reg_image_paths.append(image_path)
                                reg_tags_list.append(tags)
                        except Exception as e:
                            print(f"画像 {image_path} の処理中にエラーが発生: {e}")

        print(f"正則化データセットの画像数: {len(reg_image_paths)}")

    # 全てのタグを収集してモデルを拡張 (タグ削除後のデータから収集)
    all_tags = []
    all_tags.extend([tag for tags in main_tags_list for tag in tags])
    if reg_tags_list:
        all_tags.extend([tag for tags in reg_tags_list for tag in tags])

    # タグの頻度を計算
    tag_freq = {}
    for tag in all_tags:
        tag_freq[tag] = tag_freq.get(tag, 0) + 1

    # 最小頻度以上のタグを抽出
    filtered_tags = {tag for tag, freq in tag_freq.items() if freq >= min_tag_freq}

    # 特殊プレフィックスを持つタグを除外 (read_tags_from_fileで処理される想定だが念のため)
    if remove_special_prefix == "omit":
         filtered_tags = {tag for tag in filtered_tags if not re.match(r'^[a-zA-Z]@', tag)}


    # モデルに現在存在するタグを取得 (model.remove_tags 実行後の状態)
    existing_tags = list(model.idx_to_tag.values())
    # データセットに存在するタグのうち、モデルにまだない新規タグを特定
    new_filtered_tags = filtered_tags - set(existing_tags)

    print(f"既存タグ数 (モデル側・削除後): {len(existing_tags)}")
    print(f"最小頻度以上の新規タグ数 (データセット側・削除後): {len(new_filtered_tags)}")

    # idx_to_tagとtag_to_idxのlenが念のため同じかどうかを確認
    if not len(model.idx_to_tag) == len(model.tag_to_idx):
        raise ValueError("idx_to_tagとtag_to_idxのlenが異なります。")

    num_existing_tags_before_extend = len(model.idx_to_tag) # 拡張前の既存タグ数

    if not len(new_filtered_tags) == 0:
        print(f"モデルに {len(new_filtered_tags)} 個の新規タグを追加します。")
        # 新規タグのインデックスを追加
        next_idx = len(model.idx_to_tag)
        for tag in sorted(new_filtered_tags):  # ソートして順序を一定に
            model.tag_to_idx[tag] = next_idx
            model.idx_to_tag[next_idx] = tag # idx_to_tagも更新
            # カテゴリ情報も追加 (デフォルトはGeneral)
            if not hasattr(model, 'tag_to_category') or model.tag_to_category is None:
                 model.tag_to_category = {} # tag_to_category がなければ初期化
            model.tag_to_category[tag] = 'General' # 新規タグはGeneralに分類
            next_idx += 1

        print(f"使用されるタグの総数: {len(model.tag_to_idx)}")

        print(f"モデルのヘッドを拡張します。")
        num_classes = len(model.idx_to_tag)
        model.extend_head(num_classes) # extend_head は内部で original_num_classes を使うので、拡張前に呼ぶ
    else:
        print("データセットに新規タグはありませんでした。ヘッドの拡張は行いません。")


    # idx-to-tag の最初の10件を表示
    print(f"idx-to-tag の最初の10件: {list(model.idx_to_tag.items())[:10]}")
    # new tagsのidx-to-tagを表示
    if len(new_filtered_tags) > 0:
        print(f"追加された新規タグのidx-to-tag: {list(model.idx_to_tag.items())[num_existing_tags_before_extend:num_existing_tags_before_extend+min(10, len(new_filtered_tags))]}")

    # メインデータセットを訓練用と検証用に分割
    main_indices = list(range(len(main_image_paths)))
    random.shuffle(main_indices)

    val_size = int(len(main_indices) * val_split)
    train_indices = main_indices[val_size:]
    val_indices = main_indices[:val_size]

    train_image_paths = [main_image_paths[i] for i in train_indices]
    train_tags_list = [main_tags_list[i] for i in train_indices]

    val_image_paths = [main_image_paths[i] for i in val_indices]
    val_tags_list = [main_tags_list[i] for i in val_indices]

    print(f"訓練用画像: {len(train_image_paths)}")
    print(f"検証用画像: {len(val_image_paths)}")
    if reg_image_paths:
        print(f"正則化用画像: {len(reg_image_paths)}")

    # 元の関数の戻り値の形式に合わせる
    num_existing_tags_after_removal = len(existing_tags) # remove_tags 後の既存タグ数
    num_new_tags_added = len(new_filtered_tags)         # 今回追加された新規タグ数

    if reg_image_dirs:
        return train_image_paths, train_tags_list, val_image_paths, val_tags_list, reg_image_paths, reg_tags_list, num_existing_tags_after_removal, num_new_tags_added
    else:
        return train_image_paths, train_tags_list, val_image_paths, val_tags_list, None, None, num_existing_tags_after_removal, num_new_tags_added

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

def compute_metrics(outputs, targets):
    """
    評価指標を計算する関数（リファクタリング済み・高速化版・閾値選択ロジック変更）

    Args:
        outputs: モデルの出力（シグモイド適用済み）
        targets: 正解ラベル

    Returns:
        metrics: 評価指標の辞書。さらに各クラスごとの最適な閾値も追加される。
    """
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    import io
    from torchvision import transforms
    from sklearn.metrics import precision_recall_curve, auc
    from tqdm import tqdm
    import numpy as np
    import torch

    # 閾値（0.1～0.85, 0.05刻み）
    thresholds = np.arange(0.1, 0.9, 0.05)

    # torch.Tensorの場合はNumPy配列へ変換
    if isinstance(outputs, torch.Tensor):
        outputs_np = outputs.cpu().numpy()
    else:
        outputs_np = outputs

    if isinstance(targets, torch.Tensor):
        targets_np = targets.cpu().numpy()
    else:
        targets_np = targets

    num_classes = targets_np.shape[1]

    # --- PR-AUC の計算 --- (変更なし)
    pr_auc_scores = []
    for i in tqdm(range(num_classes), desc="Calculating PR-AUC", leave=False):
        if targets_np[:, i].sum() > 0:
            precision, recall, _ = precision_recall_curve(targets_np[:, i], outputs_np[:, i])
            # AUC計算時にNaNが発生しないようにチェック
            if not (np.isnan(recall).any() or np.isnan(precision).any()):
                pr_auc_scores.append(auc(recall, precision))
    macro_pr_auc = np.mean(pr_auc_scores) if pr_auc_scores else 0

    # --- 閾値毎のマクロF1スコアの計算（ベクトル演算） --- (変更なし)
    # 全閾値に対し、全サンプル・全クラスでの予測を一括計算
    preds_all = (outputs_np[None, :, :] >= thresholds[:, None, None]).astype(int)  # shape: (n_thresholds, n_samples, num_classes)
    TP_all = (preds_all * targets_np[None, :, :]).sum(axis=1)  # 各閾値・各クラスのTP
    pred_sum_all = preds_all.sum(axis=1)                        # 各閾値・各クラスの予測陽性数
    true_sum = targets_np.sum(axis=0)                           # 各クラスの正解陽性数

    # 各閾値・各クラスで、予測と正解両方に陽性がある場合のF1を計算（それ以外は nan とする）
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_all = np.where((pred_sum_all > 0) & (true_sum[None, :] > 0),
                          2 * TP_all / (pred_sum_all + true_sum[None, :]),
                          np.nan)
    # 各閾値で、有効なクラスのみのF1の平均を算出
    macro_f1_scores = np.array([np.nanmean(row) if np.any(~np.isnan(row)) else 0 for row in f1_all])

    # --- ★★★ 新しい閾値選択ロジック ★★★ ---
    best_threshold_idx = 0 # デフォルト初期化
    best_threshold = thresholds[0]
    best_macro_f1 = 0.0

    if len(macro_f1_scores) > 0 and macro_f1_scores.max() > 0: # F1スコアが計算できているか確認
        max_f1_idx_orig = np.argmax(macro_f1_scores)
        max_f1_score = macro_f1_scores[max_f1_idx_orig]

        # 目標F1スコアを設定 (最大F1の95%以上、かつ最低でも0.75以上とする)
        # これらの値 (0.95, 0.75) は調整可能です
        target_f1 = max(max_f1_score * 0.95, 0.70)

        # 目標F1スコア以上となる最初の閾値のインデックスを探す
        candidate_indices = np.where(macro_f1_scores >= target_f1)[0]

        if len(candidate_indices) > 0:
            # 候補の中で最もインデックスが小さい（=閾値が低い）ものを選ぶ
            best_threshold_idx = candidate_indices[0]
            best_threshold = thresholds[best_threshold_idx]
            print(f"閾値選択: 目標F1({target_f1:.4f})以上となる最小閾値({best_threshold:.2f})を選択 (MaxF1: {max_f1_score:.4f} @ {thresholds[max_f1_idx_orig]:.2f})")
        else:
            # 適切な候補が見つからなかった場合は、元の最大値を選ぶ
            best_threshold_idx = max_f1_idx_orig
            best_threshold = thresholds[best_threshold_idx]
            print(f"閾値選択: 目標F1({target_f1:.4f})以上となる閾値が見つからず、元の最大F1閾値({best_threshold:.2f})を選択 (MaxF1: {max_f1_score:.4f})")

        best_macro_f1 = macro_f1_scores[best_threshold_idx] # 選択された閾値でのF1

    else:
        # F1スコアが計算できなかった場合
        print("警告: マクロF1スコアが計算できませんでした。デフォルト閾値を使用します。")
        # macro_f1_scoresが空の場合や最大値が0の場合のargmaxは0を返すため、それでよい
        best_threshold_idx = np.argmax(macro_f1_scores) if len(macro_f1_scores) > 0 else 0
        best_threshold = thresholds[best_threshold_idx]
        best_macro_f1 = macro_f1_scores[best_threshold_idx] if len(macro_f1_scores) > 0 else 0.0


    # --- 全体の予測（選択された閾値） --- (変更なし)
    predictions = (outputs_np >= best_threshold).astype(int)

    # --- 最適閾値での各クラスF1スコア計算（有効なクラスのみ対象） --- (変更なし)
    pred_sum_best = predictions.sum(axis=0)
    TP_best = (predictions * targets_np).sum(axis=0)
    true_sum_best = targets_np.sum(axis=0)
    valid = (pred_sum_best > 0) & (true_sum_best > 0)
    f1_scores = np.zeros(num_classes, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'): # ゼロ除算警告を抑制
        f1_scores[valid] = 2 * TP_best[valid] / (pred_sum_best[valid] + true_sum_best[valid])
    # validなクラスのみ抽出（元のコードと同様、条件を満たさないクラスは除外）
    class_f1_scores = [f1_scores[i] for i in range(num_classes) if valid[i]]
    # macro_f1 = np.mean(class_f1_scores) if class_f1_scores else 0 # best_macro_f1 を使うので不要

    # --- 各クラスごとの最適閾値の計算（ベクトル演算） --- (変更なし)
    # 無効な閾値では f1 を -1 に設定しておく
    valid_mask = (pred_sum_all > 0) & (true_sum[None, :] > 0)
    f1_all_thresh = np.full_like(pred_sum_all, -1, dtype=float)
    denom = pred_sum_all + true_sum[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_all_thresh[valid_mask] = 2 * TP_all[valid_mask] / denom[valid_mask]

    class_best_thresholds = []
    for i in range(num_classes):
        if true_sum[i] > 0:
            max_f1_class = np.max(f1_all_thresh[:, i])
            # 元のコードと同様、最大F1が0以下なら None を設定
            if max_f1_class > 0:
                best_idx_class = np.argmax(f1_all_thresh[:, i])
                class_best_thresholds.append(thresholds[best_idx_class])
            else:
                class_best_thresholds.append(None) # 該当クラスで良い閾値が見つからない
        else:
            # 正例が存在しないクラスはグローバル最適閾値を適用 (Noneでも良いかもしれない)
            class_best_thresholds.append(best_threshold)

    # --- F1スコアと閾値の関係プロット --- (変更なし)
    fig, ax = plt.subplots(figsize=(10, 6))
    # x軸をインデックスではなく閾値の値にする
    ax.plot(thresholds, macro_f1_scores, 'bo-', label="Macro F1 Score")
    # 選択された閾値にマーク
    ax.plot(best_threshold, best_macro_f1, 'ro', markersize=10, label="Selected Threshold")
    # ax.set_xticks(range(len(thresholds))) # 不要
    # ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45) # 不要
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title(f'Macro F1 Score vs Threshold (Selected Value: {best_threshold:.2f})') # タイトルを修正
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_image = transforms.ToTensor()(PILImage.open(buf))

    metrics = {
        'pr_auc': macro_pr_auc,
        'f1': best_macro_f1, # 選択された閾値でのF1を返す
        'threshold': best_threshold,
        'class_f1s': class_f1_scores,
        'f1_vs_threshold_plot': plot_image,
        'class_best_thresholds': class_best_thresholds  # 各クラスごとの最適閾値リスト
    }
    return metrics

# トレーニング関数
def train_model(
    model, 
    base_model,
    train_loader, 
    val_loader, 
    reg_loader,
    tag_to_idx,
    idx_to_tag,
    tag_to_category,
    old_tags_count,
    learning_rate,
    weight_decay,
    optimizer_type, # ★追加★
    criterion, 
    num_epochs, 
    device, 
    output_dir='lora_model',
    save_format='safetensors',
    save_best='f1',  # 'f1', 'loss', 'both'
    checkpoint_interval=1,
    merge_every_n_epochs=None,
    mixed_precision=False,
    tensorboard=False,
    initial_threshold=0.35,
    dynamic_threshold=True,
    head_only_train_steps=0,
    cache_dir=None,  # キャッシュディレクトリのパラメータを追加
    use_zclip=False # zclipを使用するかどうかのフラグを追加
):
    """モデルをトレーニングする関数"""
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    def save_model(output_dir, filename, base_model, save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_f1, tag_to_idx, idx_to_tag, tag_to_category):
        os.makedirs(output_dir, exist_ok=True)
        
        # モデルの状態辞書
        model_state_dict = model.state_dict()
        
        # メタデータ情報
        metadata = {
            'base_model': base_model,
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
        }
        
        # 完全なチェックポイント
        checkpoint = {
            'model_state_dict': model_state_dict,
            # 'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'base_model': base_model,
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
        }

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
        run_name = f"run_{timestamp}_bs{train_loader.batch_size}"
        
        # ディレクトリを作成
        tb_log_dir = os.path.join(output_dir, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        
        # run_nameをログディレクトリに含める
        writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, run_name))
        print(f"TensorBoard logs will be saved to: {os.path.join(tb_log_dir, run_name)}")

    def setup_optimizer(model, learning_rate, weight_decay, optimizer_type): # ★引数変更★
        """オプティマイザを設定する関数

        Args:
            model: 対象のモデル
            learning_rate (float): 学習率
            weight_decay (float): Weight decay
            optimizer_type (str): オプティマイザの種類 ('adamw', 'adamw8bit', 'lion', 'lion8bit') ★変更★

        Returns:
            optimizer: 設定されたオプティマイザ
        """
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"訓練可能なパラメータ数: {sum(p.numel() for p in trainable_params)}")

        optimizer_type = optimizer_type.lower() # 小文字に統一

        if optimizer_type == 'adamw8bit':
            try:
                optimizer = bnb.optim.Adam8bit(
                    trainable_params,
                    lr=learning_rate,
                    betas=(0.9, 0.99), # AdamWのデフォルトに近いbeta値
                    weight_decay=weight_decay
                )
                print("AdamW8bit オプティマイザを使用します")
            except Exception as e:
                print(f"AdamW8bitの初期化に失敗しました ({e})。通常のAdamWを使用します。")
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
        elif optimizer_type == 'lion8bit':
            try:
                optimizer = bnb.optim.Lion8bit(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                print("Lion8bit オプティマイザを使用します")
            except Exception as e:
                print(f"Lion8bitの初期化に失敗しました ({e})。通常のLionを使用します。")
                try:
                    optimizer = bnb.optim.Lion(
                        trainable_params,
                        lr=learning_rate,
                        weight_decay=weight_decay
                    )
                    print("Lion オプティマイザを使用します")
                except Exception as e_lion:
                     print(f"Lionの初期化にも失敗しました ({e_lion})。AdamWを使用します。")
                     optimizer = torch.optim.AdamW(
                         trainable_params,
                         lr=learning_rate,
                         weight_decay=weight_decay
                     )
        elif optimizer_type == 'lion':
            try:
                optimizer = bnb.optim.Lion(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                print("Lion オプティマイザを使用します")
            except Exception as e:
                print(f"Lionの初期化に失敗しました ({e})。AdamWを使用します。")
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
            print("AdamW オプティマイザを使用します")
        else:
            print(f"未対応のオプティマイザタイプ: {optimizer_type}。デフォルトのAdamW8bitを使用します。")
            try:
                optimizer = bnb.optim.Adam8bit(
                    trainable_params,
                    lr=learning_rate,
                    betas=(0.9, 0.99),
                    weight_decay=weight_decay
                )
            except Exception as e:
                print(f"AdamW8bitの初期化に失敗しました ({e})。通常のAdamWを使用します。")
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
        return optimizer

    # オプティマイザの設定
    optimizer = setup_optimizer(model, learning_rate, weight_decay, optimizer_type) # ★引数変更★
    
    # 学習率スケジューラの設定
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # 混合精度トレーニングのスケーラー
    scaler = torch.amp.GradScaler(device=device) if mixed_precision else None
    
    # 最良のモデルを保存するための変数
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    # epochにより動的に閾値を設定する
    threshold = initial_threshold if initial_threshold is not None else 0.35

    # 初期検証（もともとの重みにより推論はある程度できるはず）
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []

    val_neg_preds = []
    val_pos_preds = []
    
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
            # 最新100件のみを保持
            val_preds = (val_preds + [probs])[-100:]
            val_targets = (val_targets + [targets.detach().cpu().numpy()])[-100:]
            neg_probs = probs*(1-targets.cpu().numpy())
            pos_probs = probs*targets.cpu().numpy()
            val_neg_preds = (val_neg_preds + [neg_probs])[-100:]
            val_pos_preds = (val_pos_preds + [pos_probs])[-100:]
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # タグ拡張が多いとき著しく重いので、いったんコメントアウト
            # if i % 5 == 0 and i / 5 < 10:
                # img_grid = visualize_predictions_for_tensorboard(images[0], probs[0], idx_to_tag, threshold=threshold, original_tags=targets[0])

                # if tensorboard:
                #     writer.add_image(f'predictions/val_{i}', img_grid, 0)
    
    def calculate_group_stats(preds_list):
        """
        予測値の平均と分散を計算する関数
        
        Args:
            preds_list: バッチごとの予測値のリスト
            num_classes: クラス数
            
        Returns:
            stats: 平均、分散、サンプル数を含む辞書
        """
        # 各タグごとの統計量を格納する配列
        sum_vals = np.zeros(preds_list[0].shape[1])
        count = np.zeros(preds_list[0].shape[1])
        sq_sum = np.zeros(preds_list[0].shape[1])  # 二乗和（分散計算用）

        # バッチごとの統計量を集計
        for batch in preds_list:
            for pred in batch:
                # シグモイド関数の出力値を対数オッズに戻す
                # logit(p) = log(p/(1-p))
                logits = np.log(pred + 1e-7) - np.log(1 - pred + 1e-7)
                
                # マスク（サンプルの特定）
                mask = pred > 0
                
                # 統計量の更新
                sum_vals[mask] += logits[mask]
                count[mask] += 1
                sq_sum[mask] += logits[mask] ** 2
        
        # ゼロ除算を防ぐ
        count = np.maximum(count, 1)
        
        # 平均と分散を計算（logitスケール）
        means_logit = sum_vals / count
        variance_logit = (sq_sum / count) - (means_logit ** 2)
        
        # logitスケールから確率スケールに変換
        means = 1 / (1 + np.exp(-means_logit))
        
        # 分散の伝播（デルタ法）
        # Var(sigmoid(x)) ≈ (sigmoid'(μ))^2 * Var(x)
        sigmoid_derivative = means * (1 - means)
        variance = (sigmoid_derivative ** 2) * variance_logit

        # 結果を返す
        return {
            'means': means,
            'variance': variance,
            'counts': count
        }

    # 陽性群と陰性群の統計を計算
    # pos_stats = calculate_group_stats(val_pos_preds)
    neg_stats = calculate_group_stats(val_neg_preds)
    
    val_preds = [pred for batch in val_preds for pred in batch]
    val_targets = [target for batch in val_targets for target in batch]
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # 検証メトリクスの計算
    val_loss /= len(val_loader)
    val_metrics = compute_metrics(val_preds, val_targets)
    
    # 既存タグと新規タグに分けてメトリクスを計算
    if old_tags_count > 0 and len(model.tag_to_idx) > old_tags_count:
        existing_val_metrics = compute_metrics(
            val_preds[:, :old_tags_count],
            val_targets[:, :old_tags_count]
        )
        new_val_metrics = compute_metrics(
            val_preds[:, old_tags_count:],
            val_targets[:, old_tags_count:]
        )
        
        print(f"初期検証結果 - 既存タグ F1: {existing_val_metrics['f1']:.4f}, 新規タグ F1: {new_val_metrics['f1']:.4f}")

        if tensorboard:
            writer.add_scalar('Metrics/val/F1', val_metrics['f1'], 0)
            writer.add_scalar('Metrics/val/PR-AUC', val_metrics['pr_auc'], 0)
            writer.add_scalar('Metrics/val/Threshold', threshold, 0)
            writer.add_image('Metrics/val/F1_vs_Threshold', val_metrics['f1_vs_threshold_plot'], 0)

            if old_tags_count > 0 and len(model.tag_to_idx) > old_tags_count:
                writer.add_scalar('Metrics/val/F1_existing', existing_val_metrics['f1'], 0)
                writer.add_scalar('Metrics/val/F1_new', new_val_metrics['f1'], 0)
        del existing_val_metrics, new_val_metrics
    else:
        print(f"初期検証結果 - F1: {val_metrics['f1']:.4f}")
        if tensorboard:
            writer.add_scalar('Metrics/val/F1', val_metrics['f1'], 0)
            writer.add_scalar('Metrics/val/PR-AUC', val_metrics['pr_auc'], 0)
            writer.add_scalar('Metrics/val/Threshold', threshold, 0)
            writer.add_image('Metrics/val/F1_vs_Threshold', val_metrics['f1_vs_threshold_plot'], 0)

    del val_preds, val_targets, val_neg_preds, val_pos_preds

    global_step = 0
    # トレーニングループの前に以下を追加
    if head_only_train_steps > 0:
        print(f"最初の{head_only_train_steps}ステップではヘッド部分のみを学習します")
        model.freeze_non_new_head_parameters()

    # ZClipの初期化 (use_zclipがTrueの場合)
    zclip = None
    if use_zclip:
        if zclip_available:
            zclip = ZClip(mode="zscore", alpha=0.97, z_thresh=2.5, clip_option="adaptive_scaling", max_grad_norm=1.0, clip_factor=1.0)
            print("ZClipによる勾配クリッピングを有効化しました。")
        else:
            print("警告: --use_zclipが指定されましたが、zclip.pyが見つからないためZClipは使用されません。")

    try:
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
                global_step += 1
                images = images.to(device)
                targets = targets.to(device)
                
                # 通常のデータでの学習
                if mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, targets)

                    scaler.scale(loss).backward()
                    if zclip:
                        zclip.step(model) # 勾配クリッピングを適用
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                    loss.backward()
                    if zclip:
                        zclip.step(model) # 勾配クリッピングを適用
                    optimizer.step()
                    optimizer.zero_grad()

                # 正則化データセットでの学習
                if reg_loader is not None:
                    try:
                        reg_images, reg_targets = next(reg_iter)
                    except (StopIteration, NameError):
                        reg_iter = iter(reg_loader)
                        reg_images, reg_targets = next(reg_iter)

                    reg_images = reg_images.to(device)
                    reg_targets = reg_targets.to(device)

                    if mixed_precision:
                        with torch.amp.autocast('cuda'):
                            reg_outputs = model(reg_images)
                            reg_loss = criterion(reg_outputs, reg_targets)

                        scaler.scale(reg_loss).backward()
                        if zclip:
                            zclip.step(model) # 勾配クリッピングを適用
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        reg_outputs = model(reg_images)
                        reg_loss = criterion(reg_outputs, reg_targets)

                        reg_loss.backward()
                        if zclip:
                            zclip.step(model) # 勾配クリッピングを適用
                        optimizer.step()
                        optimizer.zero_grad()

                    loss = (loss + reg_loss) / 2

                train_loss += loss.item()

                # シグモイド関数で確率に変換
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                # 最新100件のみを保持
                train_preds = (train_preds + [probs])[-100:]
                train_targets = (train_targets + [targets.detach().cpu().numpy()])[-100:]

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                # 定期的にTensorBoardに予測結果を記録
                if tensorboard:
                    # stepごとのlossを記録
                    writer.add_scalar('Train/Step_Loss', loss.item(), epoch * len(train_loader) + i)

                    if i % 100 == 0:
                        img_grid = visualize_predictions_for_tensorboard(
                            images[0],
                            probs[0],
                            idx_to_tag,
                            threshold=threshold,
                            original_tags=targets[0],
                            tag_to_category=tag_to_category,
                            neg_means=neg_stats['means'],  # 陰性群の平均値を追加
                            neg_variance=neg_stats['variance'],  # 陰性群の分散を追加
                            neg_counts=neg_stats['counts']  # 陰性群のサンプル数を追加
                        )
                        writer.add_image(f'predictions/train_epoch_{epoch}', img_grid, i)

                if global_step == head_only_train_steps:
                    tqdm.write("ヘッド部分の訓練を終了します")
                    model.freeze_non_lora_parameters()
            
            # 学習率のスケジューリング
            scheduler.step()
            if tensorboard:
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # トレーニングメトリクスの計算
            train_loss /= len(train_loader)

            train_preds = [pred for batch in train_preds for pred in batch]
            train_targets = [target for batch in train_targets for target in batch]
            train_preds = np.array(train_preds)
            train_targets = np.array(train_targets)

            train_metrics = compute_metrics(train_preds,train_targets)

            if dynamic_threshold:
                threshold = round(train_metrics['threshold'], 2) - 0.1
            
            # 既存タグと新規タグに分けてメトリクスを計算
            if old_tags_count > 0 and len(model.tag_to_idx) > old_tags_count:
                existing_train_metrics = compute_metrics(train_preds[:,:old_tags_count],train_targets[:,:old_tags_count])
                new_train_metrics = compute_metrics(train_preds[:,old_tags_count:],train_targets[:,old_tags_count:])

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
            del train_preds, train_targets

            # 検証フェーズ
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            val_neg_preds = []
            val_pos_preds = []
            
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
                    # 最新100件のみを保持
                    val_preds = (val_preds + [probs])[-100:]
                    val_targets = (val_targets + [targets.detach().cpu().numpy()])[-100:]
                    neg_probs = probs*(1-targets.cpu().numpy())
                    pos_probs = probs*targets.cpu().numpy()
                    val_neg_preds = (val_neg_preds + [neg_probs])[-100:]
                    val_pos_preds = (val_pos_preds + [pos_probs])[-100:]
                    
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                    if i % 5 == 0 and i / 5 < 10:                        
                        # 予測の可視化
                        img_grid = visualize_predictions_for_tensorboard(
                            images[0], 
                            probs[0], 
                            idx_to_tag, 
                            threshold=threshold, 
                            original_tags=targets[0],
                            tag_to_category=tag_to_category,
                            neg_means=neg_stats['means'],  # 陰性群の平均値を追加
                            neg_variance=neg_stats['variance'],  # 陰性群の分散を追加
                            neg_counts=neg_stats['counts']  # 陰性群のサンプル数を追加
                        )

                        if tensorboard:
                            writer.add_image(f'predictions/val_{i}', img_grid, epoch+1)


            # 陽性群と陰性群の統計を計算
            # pos_stats = calculate_group_stats(val_pos_preds)
            neg_stats = calculate_group_stats(val_neg_preds)

            # 検証メトリクスの計算
            val_loss /= len(val_loader)

            val_preds = [pred for batch in val_preds for pred in batch]
            val_targets = [target for batch in val_targets for target in batch]
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)

            val_metrics = compute_metrics(val_preds, val_targets)
            
            # 結果の表示
            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # 既存タグと新規タグに分けてメトリクスを計算
            if old_tags_count > 0 and len(model.tag_to_idx) > old_tags_count:
                existing_val_metrics = compute_metrics(
                    val_preds[:, :old_tags_count],
                    val_targets[:, :old_tags_count]
                )
                new_val_metrics = compute_metrics(
                    val_preds[:, old_tags_count:],
                    val_targets[:, old_tags_count:]
                )
                
                print(f"epoch {epoch+1} 検証結果 - 既存タグ F1: {existing_val_metrics['f1']:.4f}, 新規タグ F1: {new_val_metrics['f1']:.4f}")

                if tensorboard:
                    writer.add_scalar('Metrics/val/F1', val_metrics['f1'], epoch+1)
                    writer.add_scalar('Metrics/val/PR-AUC', val_metrics['pr_auc'], epoch+1)
                    writer.add_scalar('Metrics/val/Threshold', threshold, epoch+1)
                    writer.add_image('Metrics/val/F1_vs_Threshold', val_metrics['f1_vs_threshold_plot'], epoch+1)

                    if old_tags_count > 0 and len(model.tag_to_idx) > old_tags_count:
                        writer.add_scalar('Metrics/val/F1_existing', existing_val_metrics['f1'], epoch+1)
                        writer.add_scalar('Metrics/val/F1_new', new_val_metrics['f1'], epoch+1)
                del existing_val_metrics, new_val_metrics
            else:
                print(f"epoch {epoch+1} 検証結果 - F1: {val_metrics['f1']:.4f}")
                if tensorboard:
                    writer.add_scalar('Metrics/val/F1', val_metrics['f1'], epoch+1)
                    writer.add_scalar('Metrics/val/PR-AUC', val_metrics['pr_auc'], epoch+1)
                    writer.add_scalar('Metrics/val/Threshold', threshold, epoch+1)
                    writer.add_image('Metrics/val/F1_vs_Threshold', val_metrics['f1_vs_threshold_plot'], epoch+1)

            del val_preds, val_targets, val_neg_preds, val_pos_preds
            
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
                save_model(output_dir, f'best_model', base_model, save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_metrics['f1'], tag_to_idx, idx_to_tag, tag_to_category)
                print(f"Best model saved! (Val F1: {val_metrics['f1']:.4f}, Val Loss: {val_loss:.4f})")
        
            if (epoch + 1) % checkpoint_interval == 0:
                save_model(output_dir, f'checkpoint_epoch_{epoch+1}', base_model, save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_metrics['f1'], tag_to_idx, idx_to_tag, tag_to_category)
                print(f"Checkpoint saved at epoch {epoch+1}")

            if merge_every_n_epochs is not None and (epoch + 1) % merge_every_n_epochs == 0:
                model.merge_lora_to_base_model(
                    scale=1.0,
                    new_lora_rank=None,
                    new_lora_alpha=None,
                    new_lora_dropout=None
                )
                model.to(device)
                # optimizer, schedulerをリセット
                optimizer = setup_optimizer(model, learning_rate, weight_decay, optimizer_type)
                # schedulerは通常、warmupをやり直すが、ここでは未実装
                
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        # ★★★ 割り込み時やエラー時にもモデルを保存 ★★★
        try:
            print("エラーまたは中断により、現在の状態を保存します...")
            save_model(
                output_dir,
                f'interrupted_checkpoint_epoch_{epoch+1 if "epoch" in locals() else 0}', # epochが存在すれば使う
                base_model,
                save_format,
                model,
                optimizer,
                scheduler,
                epoch if "epoch" in locals() else 0,
                threshold if "threshold" in locals() else initial_threshold,
                val_loss if "val_loss" in locals() else float('inf'),
                val_metrics['f1'] if "val_metrics" in locals() else 0.0,
                tag_to_idx,
                idx_to_tag,
                tag_to_category
            )
        except Exception as save_e:
            print(f"状態の保存中にさらにエラーが発生しました: {save_e}")
        # ★★★ ここまで追加 ★★★
        raise e # エラーを再送出


    finally:
        # 最終モデルの保存
        save_model(output_dir, f'final_model', base_model, save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_metrics['f1'], tag_to_idx, idx_to_tag, tag_to_category)
        print(f"Final model saved! (Val F1: {val_metrics['f1']:.4f}, Val Loss: {val_loss:.4f})")

        save_tag_mapping(output_dir, idx_to_tag, tag_to_category)
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None, どうするか後で検討
        
        # Close the tensorboard writer
        if tensorboard:
            writer.close()
        
        return val_metrics    

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
    predict_parser.add_argument('--gen_threshold', type=float, default=None, help='一般タグの閾値')
    predict_parser.add_argument('--char_threshold', type=float, default=None, help='キャラクタータグの閾値')
    predict_parser.add_argument('--xformers', action='store_true', help='xformersを使用してメモリ効率の良いattentionを有効化')
    predict_parser.add_argument('--sageattention', action='store_true', help='SageAttentionを使用してメモリ効率の良いattentionを有効化')
    predict_parser.add_argument('--flashattention', action='store_true', help='FlashAttentionを使用してメモリ効率の良いattentionを有効化')

    # バッチ推論コマンド
    batch_parser = subparsers.add_parser('batch', help='複数の画像からタグを予測します')
    batch_parser.add_argument('--image_dir', type=str, required=True, help='予測する画像ファイルのディレクトリ')
    batch_parser.add_argument('--base_model', type=str, default='SmilingWolf/wd-eva02-large-tagger-v3', help='使用するベースモデルのリポジトリ')
    batch_parser.add_argument('--model_path', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    batch_parser.add_argument('--metadata_path', type=str, default=None, help='モデルのメタデータファイルのパス（指定しない場合は元のモデルを使用）')
    batch_parser.add_argument('--output_dir', type=str, default='predictions', help='予測結果を保存するディレクトリ')
    batch_parser.add_argument('--gen_threshold', type=float, default=None, help='一般タグの閾値')
    batch_parser.add_argument('--char_threshold', type=float, default=None, help='キャラクタータグの閾値')
    batch_parser.add_argument('--xformers', action='store_true', help='xformersを使用してメモリ効率の良いattentionを有効化')
    batch_parser.add_argument('--sageattention', action='store_true', help='SageAttentionを使用してメモリ効率の良いattentionを有効化')
    batch_parser.add_argument('--flashattention', action='store_true', help='FlashAttentionを使用してメモリ効率の良いattentionを有効化')
    
    # トレーニングコマンド
    train_parser = subparsers.add_parser('train', help='LoRAモデルをトレーニングします')

    train_parser.add_argument('--base_model', type=str, default='SmilingWolf/wd-eva02-large-tagger-v3', help='使用するベースモデルのリポジトリ')
    train_parser.add_argument('--model_path', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    train_parser.add_argument('--metadata_path', type=str, default=None, help='モデルのメタデータファイルのパス（指定しない場合は_metadata.jsonを使用）')

    train_parser.add_argument('--merge_before_train', nargs='?', const=1, type=float, default=None, 
                         help='トレーニング前にLoRAの重みをベースモデルにマージする。数値を指定するとスケーリング係数として使用（デフォルト: 1.0）')
    train_parser.add_argument('--merge_every_n_epochs', type=int, default=None, help='トレーニング中にマージするエポック数')

    # データセット関連の引数
    train_parser.add_argument('--image_dirs', type=str, nargs='+', required=True, help='トレーニング画像のディレクトリ（複数指定可）')
    train_parser.add_argument('--val_split', type=float, default=0.1, help='検証データの割合')
    train_parser.add_argument('--reg_image_dirs', type=str, nargs='+', default=None, help='正則化画像のディレクトリ（複数指定可）')
    train_parser.add_argument('--min_tag_freq', type=int, default=10, help='タグの最小出現頻度')
    train_parser.add_argument('--remove_special_prefix', default="omit", choices=["remove", "omit"], help='特殊プレフィックス（例：a@、g@など）を除去する')
    # train_parser.add_argument('--image_size', type=int, default=224, help='画像サイズ')
    train_parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ')
    train_parser.add_argument('--num_workers', type=int, default=4, help='データローダーのワーカー数')
    train_parser.add_argument('--cache_dir', type=str, default=None, help='キャッシュディレクトリのパス')
    train_parser.add_argument('--force_recache', action='store_true', help='キャッシュを強制的に再作成する')
    
    # モデル関連の引数
    train_parser.add_argument('--lora_rank', type=int, default=32, help='LoRAのランク')
    train_parser.add_argument('--lora_alpha', type=float, default=16, help='LoRAのアルファ値')
    train_parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRAのドロップアウト率')
    train_parser.add_argument('--target_modules_file', type=str, default=None, help='LoRAを適用するモジュールのリストを含むファイル')
    
    # トレーニング関連の引数
    train_parser.add_argument('--num_epochs', type=int, default=10, help='エポック数')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='学習率')
    train_parser.add_argument('--weight_decay', type=float, default=0.01, help='重み減衰')
    train_parser.add_argument('--optimizer', type=str, default='adamw8bit',
                              choices=['adamw', 'adamw8bit', 'lion', 'lion8bit'],
                              help='使用するオプティマイザの種類') # ★追加★
    train_parser.add_argument('--use_zclip', action='store_true', help='ZClipによる勾配クリッピングを使用する') # ZClip引数を追加

    train_parser.add_argument('--tags_to_remove', type=str, default=None, help='削除するタグのファイルパス')

    train_parser.add_argument('--head_only_train_steps', type=int, default=0,
                      help='最初のN回のステップでヘッド部分のみを学習し、他のパラメータをフリーズします')

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
    train_parser.add_argument('--tensorboard', default=True, action='store_true', help='TensorBoardを使用する')
    train_parser.add_argument('--tensorboard_port', type=int, default=6006, help='TensorBoardのポート番号')
    train_parser.add_argument('--bind_all', action='store_true', help='TensorBoardをすべてのネットワークインターフェースにバインドする')
    train_parser.add_argument('--seed', type=int, default=42, help='乱数シード')

    train_parser.add_argument('--xformers', action='store_true', help='xformersを使用してメモリ効率の良いattentionを有効化')
    train_parser.add_argument('--sageattention', action='store_true', help='SageAttentionを使用してメモリ効率の良いattentionを有効化')
    train_parser.add_argument('--flashattention', action='store_true', help='FlashAttentionを使用してメモリ効率の良いattentionを有効化')
    # モデルマージコマンド
    merge_parser = subparsers.add_parser('merge', help='LoRAモデルをマージします')

    merge_parser.add_argument('--model_path', type=str, default=None, help='使用するLoRAモデルのパス（指定しない場合は元のモデルを使用）')
    merge_parser.add_argument('--metadata_path', type=str, default=None, help='モデルのメタデータファイルのパス（指定しない場合は_metadata.jsonを使用）')
    merge_parser.add_argument('--base_model', type=str, default='SmilingWolf/wd-eva02-large-tagger-v3', help='使用するベースモデルのリポジトリ')
    merge_parser.add_argument('--output_dir', type=str, default='merged_model', help='出力ディレクトリ')
    merge_parser.add_argument('--remove_tags', type=str, default=None, help='削除するタグのファイルパス')
    merge_parser.add_argument('--scale', type=float, default=1.0, help='マージ時のスケーリング係数（デフォルト: 1.0）')
    merge_parser.add_argument('--save_format', type=str, default='safetensors', choices=['safetensors', 'pytorch', 'onnx'], help='モデルの保存形式')
    merge_parser.add_argument('--fp16', action='store_true', help='FP16モデルを保存する')

    merge_parser.add_argument('--merge_type', type=str, default='lora', choices=['lora'], help='マージするモデルの種類')
    """
    lora: LoRAモデルを元のモデルにマージし、LoRA層を削除する
    """
    
    # debug_parser = subparsers.add_parser('debug', help='モデルのデバッグを行います')

    args = parser.parse_args()
    
    if args.command == 'analyze':
        # モデル構造の分析
        analyze_model_structure(base_model=args.base_model)
    
    elif args.command == 'predict':

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用デバイス: {device}")

        # 単一画像の予測
        model, labels = load_model(args.model_path, args.metadata_path, base_model=args.base_model, device=device, use_xformers=args.xformers, use_sageattention=args.sageattention, use_flashattention=args.flashattention)
        
        # 画像に紐づくタグを読み込む
        actual_tags = read_tags_from_file(args.image)
        print(f"読み込まれたタグ: {len(actual_tags)}個")
        
        # 予測を実行
        img, caption, taglist, ratings, general, artist, character, copyright, meta, quality, all_general, all_artist, all_character, all_copyright, all_meta = predict_image(
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
        print(f"Copyright tags (threshold={args.gen_threshold if args.gen_threshold is not None else model.threshold}):")
        for k, v in copyright.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"Artist tags (threshold={args.gen_threshold if args.gen_threshold is not None else model.threshold}):")
        for k, v in artist.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"General tags (threshold={args.gen_threshold if args.gen_threshold is not None else model.threshold}):")
        for k, v in general.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"Meta tags (threshold={args.gen_threshold if args.gen_threshold is not None else model.threshold}):")
        for k, v in meta.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"Quality tags (threshold={args.gen_threshold if args.gen_threshold is not None else model.threshold}):")
        for k, v in quality.items():
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
            predictions=(caption, taglist, ratings, character, general, meta, quality, all_character, all_general),
            threshold=args.gen_threshold if args.gen_threshold is not None else model.threshold,
            output_path=output_path
        )
    
    elif args.command == 'batch':

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用デバイス: {device}")

        # モデルの読み込み
        model, labels = load_model(args.model_path, args.metadata_path, base_model=args.base_model, device=device, use_xformers=args.xformers, use_sageattention=args.sageattention, use_flashattention=args.flashattention)
        
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
                img, caption, taglist, ratings, character, general, meta, quality, all_character, all_general = predict_image(
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
                    predictions=(caption, taglist, ratings, character, general, meta, quality, all_character, all_general),
                    threshold=args.gen_threshold,
                    output_path=output_path
                )
                
                # 結果を保存
                results[os.path.basename(image_file)] = {
                    'caption': caption,
                    'ratings': ratings,
                    'character': {k: float(v) for k, v in character.items()},  # JSON化のためfloatに変換
                    'general': {k: float(v) for k, v in general.items()},  # JSON化のためfloatに変換
                    'meta': {k: float(v) for k, v in meta.items()},  # JSON化のためfloatに変換
                    'quality': quality,
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

        if args.sageattention and sageattention_available:
            print("SageAttentionを使用します。")

        print(f"モデルを読み込んでいます...")
        model, labels = load_model(args.model_path, args.metadata_path, base_model=args.base_model, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, device=device, use_xformers=args.xformers, use_sageattention=args.sageattention, use_flashattention=args.flashattention)
        
        # --- ここから tags_to_remove の処理を追加 ---
        tags_to_remove_list = None
        if args.tags_to_remove: # 引数が指定されているかチェック
            print(f"--tags_to_remove が指定されました。タグファイル {args.tags_to_remove} を読み込みます。")
            if os.path.exists(args.tags_to_remove):
                # read_tags_from_file を使用してタグリストを取得
                # 特殊プレフィックスは削除するように指定 (remove_special_prefix="remove")
                tags_to_remove_list = read_tags_from_file(args.tags_to_remove, remove_special_prefix="remove")
                if tags_to_remove_list:
                    print(f"{len(tags_to_remove_list)} 個のタグをモデルから削除します...")
                    model.remove_tags(tags_to_remove_list) # モデルからタグを削除
                    print("モデルからのタグ削除が完了しました。")
                else:
                    print(f"警告: タグファイル {args.tags_to_remove} は空または読み込めませんでした。タグ削除は行われません。")
                    tags_to_remove_list = None # 念のためNoneに戻す
            else:
                print(f"警告: タグファイル {args.tags_to_remove} が見つかりません。タグ削除は行われません。")
                tags_to_remove_list = None # 念のためNoneに戻す
        # --- tags_to_remove の処理ここまで ---

        # ターゲットモジュールの読み込み
        target_modules = None
        if args.target_modules_file:
            with open(args.target_modules_file, 'r') as f:
                target_modules = [line.strip() for line in f.readlines() if line.strip()]
            print(f"LoRAを適用するモジュール数: {len(target_modules)}")

        # モデルの期待する画像サイズを取得
        img_size = model.img_size
        print(f"モデルの期待する画像サイズを使用します: {img_size}")

        # 2. データセットの準備（新規タグが検出される）,ヘッド拡張もこの中で行われる
        print("データセットを準備しています...")
        train_image_paths, train_tags_list, val_image_paths, val_tags_list, reg_image_paths, reg_tags_list, old_tags_count, _ = prepare_dataset(
            model=model,
            image_dirs=args.image_dirs,
            reg_image_dirs=args.reg_image_dirs,
            val_split=args.val_split,
            min_tag_freq=args.min_tag_freq,
            remove_special_prefix=args.remove_special_prefix,
            seed=args.seed,
            tags_to_remove=tags_to_remove_list # 読み込んだリストを渡す
        )

        # トレーニング前にLoRAをマージ（指定された場合）
        if args.merge_before_train is not None:
            # モデルがLoRAを持っているか確認（ベースモデルから始める場合は何もしない）
            if hasattr(model, 'lora_layers') and model.lora_layers: # lora_layersが存在し、空でないことを確認
                print(f"トレーニング前にLoRAをベースモデルにマージします（スケール: {args.merge_before_train}）...")
                model.merge_lora_to_base_model(scale=1.0, new_lora_rank=args.lora_rank, new_lora_alpha=args.lora_alpha, new_lora_dropout=args.lora_dropout)
            else:
                print("マージするLoraレイヤーがありません。ベースモデルまたは既存の状態でトレーニングを開始します。")

        # modelにまだLoRAが適用されていない場合は適用する
        if not hasattr(model, 'lora_layers') or not model.lora_layers: # lora_layersが存在しないか空の場合
            if args.lora_rank is not None and args.lora_rank > 0:
                print(f"LoRAを適用します（lora_rank: {args.lora_rank}, lora_alpha: {args.lora_alpha}, lora_dropout: {args.lora_dropout})")
                model.lora_rank = args.lora_rank
                model.lora_alpha = args.lora_alpha
                model.lora_dropout = args.lora_dropout
                # ターゲットモジュールが指定されていればそれを使用
                if target_modules:
                    model.target_modules = target_modules
                model.apply_lora_to_modules()
            else:
                 print("LoRAランクが指定されていないか0以下であるため、LoRAは適用されません。")
        else:
             print("LoRAがすでに適用されています。")


        model = model.to(device)

        # データセットの作成
        train_dataset = TagImageDataset(
            image_paths=train_image_paths,
            tags_list=train_tags_list,
            tag_to_idx=model.tag_to_idx,
            transform=model.transform,
            cache_dir=None if args.cache_dir is None else os.path.join(args.cache_dir, 'train'),  # トレーニング用キャッシュディレクトリ
            force_recache=False,  # キャッシュを強制的に再作成するフラグ
        )

        val_dataset = TagImageDataset(
            image_paths=val_image_paths,
            tags_list=val_tags_list,
            tag_to_idx=model.tag_to_idx,
            transform=model.transform,
            cache_dir=None if args.cache_dir is None else os.path.join(args.cache_dir, 'val'),  # 検証用キャッシュディレクトリ
            force_recache=False,  # キャッシュを強制的に再作成するフラグ
        )

        if args.reg_image_dirs:
            reg_dataset = TagImageDataset(
                image_paths=reg_image_paths,
                tags_list=reg_tags_list,
                tag_to_idx=model.tag_to_idx,
                transform=model.transform,
                cache_dir=None if args.cache_dir is None else os.path.join(args.cache_dir, 'reg'),  # 正則化用キャッシュディレクトリ
                force_recache=False,  # キャッシュを強制的に再作成するフラグ
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

        if args.reg_image_dirs:
            reg_loader = torch.utils.data.DataLoader(
                reg_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            reg_loader = None
        
        # 損失関数の設定
        if args.loss_fn == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss_fn == 'asl':
            criterion = AsymmetricLoss(
                gamma_neg=args.gamma_neg,
                gamma_pos=args.gamma_pos,
                clip=args.clip,
            )
        elif args.loss_fn == 'asl_optimized':
            criterion = AsymmetricLossOptimized(
                gamma_neg=args.gamma_neg,
                gamma_pos=args.gamma_pos,
                clip=args.clip,
            )
        else:
            print("損失関数が指定されていません。BCEWithLogitsLossを使用します。")
            criterion = nn.BCEWithLogitsLoss()
        
        # TensorBoardの設定
        if args.tensorboard:
            try:
                import subprocess
                tensorboard_process = subprocess.Popen(
                    ['tensorboard', '--logdir', os.path.join(args.output_dir, 'tensorboard_logs'), '--port', str(args.tensorboard_port), '--bind_all'] if args.bind_all else ['tensorboard', '--logdir', os.path.join(args.output_dir, 'tensorboard_logs'), '--port', str(args.tensorboard_port)]
                )
                print(f"TensorBoardを起動しました: http://localhost:{args.tensorboard_port}")
            except Exception as e:
                print(f"TensorBoardの起動に失敗しました: {e}")

        tag_to_category = load_tag_categories()
        # model.tag_to_idx に存在するタグのみを tag_to_category に含める
        valid_tag_to_category = {tag: category for tag, category in tag_to_category.items()
                          if tag in model.tag_to_idx}
        # 不足しているタグがあればGeneralとして追加
        for tag_idx, tag in model.idx_to_tag.items():
            if tag not in valid_tag_to_category:
                 valid_tag_to_category[tag] = 'General'
        # Ratingタグを強制的に上書き
        for rating_tag in ['general', 'sensitive', 'questionable', 'explicit']:
             if rating_tag in model.tag_to_idx: # モデルに存在する場合のみ上書き
                 valid_tag_to_category[rating_tag] = 'Rating'
        # Qualityタグを強制的に上書き
        for quality_tag in ['best_quality', 'high_quality', 'normal_quality', 'medium_quality', 'low_quality', 'bad_quality', 'worst_quality']:
            if quality_tag in model.tag_to_idx: # モデルに存在する場合のみ上書き
                valid_tag_to_category[quality_tag] = 'Quality'
        # model の tag_to_category を更新
        model.tag_to_category = valid_tag_to_category


        # トレーニングの実行
        print("トレーニングを開始します...")

        train_model(
            model=model,
            base_model=args.base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            reg_loader=reg_loader,
            tag_to_idx=model.tag_to_idx,
            idx_to_tag=model.idx_to_tag,
            tag_to_category=model.tag_to_category,
            old_tags_count=old_tags_count, # prepare_datasetから返された削除後の既存タグ数を渡す
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer, # ★引数変更★
            criterion=criterion,
            num_epochs=args.num_epochs,
            device=device,
            output_dir=args.output_dir,
            save_format=args.save_format,
            save_best=args.save_best,
            checkpoint_interval=args.checkpoint_interval,
            merge_every_n_epochs=args.merge_every_n_epochs,
            mixed_precision=args.mixed_precision,
            tensorboard=args.tensorboard, # args.tensorboard を渡す
            initial_threshold=args.initial_threshold,
            dynamic_threshold=args.dynamic_threshold,
            head_only_train_steps=args.head_only_train_steps,
            cache_dir=False, # args.cache_dir を渡す
            use_zclip=args.use_zclip
        )
        print("トレーニングが完了しました！")

        # TensorBoardプロセスの終了
        if args.tensorboard and 'tensorboard_process' in locals():
            tensorboard_process.terminate()
            print("TensorBoardプロセスを終了しました。")

    elif args.command == 'merge':

        def save_model(output_dir, filename, base_model, save_format, model, optimizer, scheduler, epoch, threshold, val_loss, val_f1, tag_to_idx, idx_to_tag, tag_to_category):
            os.makedirs(output_dir, exist_ok=True)
            
            # モデルの状態辞書
            model_state_dict = model.state_dict()
            
            # メタデータ情報
            metadata = {
                'base_model': base_model,
                'epoch': epoch,
                'threshold': threshold,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'lora_rank': model.lora_rank if hasattr(model, 'lora_rank') else None, # マージ後はNoneになる可能性
                'lora_alpha': model.lora_alpha if hasattr(model, 'lora_alpha') else None,
                'lora_dropout': model.lora_dropout if hasattr(model, 'lora_dropout') else None,
                'target_modules': model.target_modules if hasattr(model, 'target_modules') else None,
                'tag_to_idx': str(tag_to_idx), # JSON互換のため文字列化
                'idx_to_tag': str(idx_to_tag),
                'tag_to_category': str(tag_to_category),
            }
            
            # 完全なチェックポイント
            checkpoint = {
                'model_state_dict': model_state_dict,
                # 'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'base_model': base_model,
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
            }

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

        def export_onnx(model, output_dir, filename, fp16=False ,simplify=True, optimize_for_gpu=True):
            # ONNX形式で保存
            from timm.utils.model import reparameterize_model
            from timm.utils.onnx import onnx_export
            import os
            import torch

            onnx_path = os.path.join(output_dir, f'{filename}.onnx')
            optimized_path = os.path.join(output_dir, f'{filename}_optimized.onnx')

            # モデルの再パラメータ化
            model = reparameterize_model(model)

            # モデルをCPUに移動
            model.to("cpu")

            print(f"ONNX形式で保存します: {onnx_path}")
            
            # timmのonnx_exportを使用
            onnx_export(
                model,
                onnx_path,
                opset=14,
                dynamic_size=True,
                aten_fallback=False,
                keep_initializers=False,
                check_forward=True,
                training=False,
                verbose=True,
                use_dynamo=False,
                input_size=(3, 448, 448),
                batch_size=1,
            )

            # ONNXモデルの変換と簡略化
            try:
                import onnx
                
                # ONNXモデルの読み込み
                model_onnx = onnx.load(onnx_path)
                
                # FP16変換
                if fp16:
                    from onnxconverter_common import float16
                    model_onnx =  float16.convert_float_to_float16(
                        model_onnx,
                        keep_io_types=True,
                        op_block_list=['Sigmoid']  # Sigmoidなど特定の演算子はfloat32のまま
                    )
                
                # モデルの簡略化
                if simplify:
                    from onnxsim import simplify
                    print("ONNXモデルを簡略化しています...")
                    model_onnx, check = simplify(model_onnx)
                    
                    if not check:
                        print("警告: 簡略化されたモデルの検証に失敗しました")
                    else:
                        print("モデルの簡略化に成功しました")
                
                # 変換後のモデルを保存
                onnx.save(model_onnx, onnx_path)
                print(f"変換されたONNXモデルを保存しました: {onnx_path}")
                
            except Exception as e:
                print(f"モデルの変換/簡略化中にエラーが発生しました: {e}")
            
            # GPU向けの最適化
            if optimize_for_gpu:
                try:
                    import onnxruntime as ort
                    
                    print("ONNX Runtimeを使用してモデルをGPU向けに最適化しています...")
                    
                    # セッションオプションを設定
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    sess_options.optimized_model_filepath = optimized_path
                    
                    # 利用可能なプロバイダーを確認
                    providers = ort.get_available_providers()
                    print(f"利用可能なプロバイダー: {providers}")
                    
                    if 'CUDAExecutionProvider' in providers:
                        provider_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        print("CUDAプロバイダーを使用して最適化します")
                    else:
                        provider_list = ['CPUExecutionProvider']
                        print("CPUプロバイダーを使用して最適化します")
                    
                    # セッションを作成して最適化
                    _ = ort.InferenceSession(onnx_path, sess_options, providers=provider_list)
                    
                    print(f"最適化されたモデルを保存しました: {optimized_path}")
                    
                    # 最適化されたモデルが正常に生成されたか確認
                    if os.path.exists(optimized_path) and os.path.getsize(optimized_path) > 0:
                        print(f"最適化されたモデルのサイズ: {os.path.getsize(optimized_path) / (1024*1024):.2f} MB")
                        return optimized_path
                    else:
                        print("警告: 最適化されたモデルが正常に生成されませんでした。元のモデルを使用します。")
                        return onnx_path
                    
                except Exception as e:
                    print(f"GPU最適化中にエラーが発生しました: {e}")
                    print("元のONNXモデルを使用します")
                    return onnx_path
            
            return onnx_path

        # モデルの読み込み
        model, labels = load_model(args.model_path, args.metadata_path, base_model=args.base_model, device=torch_device)

        # 出力ディレクトリがない場合は作成
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # マージタイプに応じた処理
        if args.merge_type == 'lora':
            # LoRAをベースモデルにマージ
            model.lora_rank = None
            model.lora_alpha = None
            model.lora_dropout = None

            model.merge_lora_to_base_model(scale=args.scale)

            if args.remove_tags:
                # タグファイルを読み込み
                tags_to_remove = read_tags_from_file(args.remove_tags)
                
                # タグの削除
                model.remove_tags(tags_to_remove)
            
            # LoRA層を削除
            model._remove_lora_from_modules()
            # 入力ファイル名から拡張子を除去し、'_merged'を追加
            filename = os.path.splitext(os.path.basename(args.model_path))[0] + '_merged'

            if args.save_format == 'safetensors':
                save_model(
                    output_dir=args.output_dir,
                    filename=filename,
                    base_model=args.base_model,
                    save_format=args.save_format,
                    model=model,
                optimizer=None,
                scheduler=None,
                epoch=None,
                threshold=model.threshold,
                val_loss=None,
                val_f1=None,
                tag_to_idx=model.tag_to_idx,
                idx_to_tag=model.idx_to_tag,
                tag_to_category=model.tag_to_category
                )

            elif args.save_format == 'onnx':
                export_onnx(model, args.output_dir, filename, fp16=args.fp16)

            # タグマッピングの保存
            save_tag_mapping(args.output_dir, model.idx_to_tag, model.tag_to_category)
        else:
            print(f"未対応のマージタイプです: {args.merge_type}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


