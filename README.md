# tagutl

SmilingWolf/wd-eva02-large-tagger-v3のLoRA Finetuner

## 必要条件

- Python 3.8+
- PyTorch
- torchvision
- timm
- Pillow
- matplotlib
- tqdm
- tensorboard

## インストール

```bash
git clone https://github.com/celll1/tagutl
cd tagutl
pip install -r requirements.txt
```
  
必要に応じてvenvを作ってください。  

## 使用方法

### 画像タグの予測

```bash
python test.py predict --image_path path/to/image.jpg --model_path path/to/model.safetensors --threshold 0.35
```

### LoRAによるファインチューニング

```bash
python test.py train --image_dirs path/to/images --output_dir lora_model --num_epochs 10 --batch_size 8 --learning_rate 1e-4 --lora_rank 32 --lora_alpha 16 --tensorboard
```

### 主なオプション

#### 予測

- `--image_path`: 予測する画像のパス
- `--model_path`: モデルファイルのパス
- `--threshold`: タグ表示の閾値（デフォルト: 0.35）
- `--output_path`: 結果を保存するパス（指定しない場合は表示のみ）

#### トレーニング

- `--image_dirs`: トレーニング画像のディレクトリ（複数指定可）
- `--val_split`: 検証データの割合（デフォルト: 0.1）
- `--min_tag_freq`: タグの最小出現頻度（デフォルト: 5）
- `--lora_rank`: LoRAのランク（デフォルト: 32)
- `--lora_alpha`: LoRAのアルファ値（デフォルト: 16）
- `--num_epochs`: トレーニングのエポック数（デフォルト: 10）
- `--learning_rate`: 学習率（デフォルト: 1e-3）
- `--batch_size`: バッチサイズ（デフォルト: 32）
- `--output_dir`: モデルの保存先ディレクトリ（デフォルト: lora_model）
- `--loss_fn`: 損失関数（bce, asl, asl_optimized）
- `--mixed_precision`: 混合精度トレーニングの有効化
- `--tensorboard`: TensorBoardによるモニタリングの有効化

そのほかはargs参照。

## ライセンス

[Apache License 2.0](LICENSE)