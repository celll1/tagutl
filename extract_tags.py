import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

def load_tag_mapping(mapping_path: str) -> Dict:
    """tag_mapping.jsonを読み込む"""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_tags_by_category(
    mapping: Dict,
    categories: List[str],
    exclude_categories: Optional[List[str]] = None
) -> List[str]:
    """指定されたカテゴリのタグを抽出する"""
    tags = []
    
    # カテゴリ名を小文字に変換（大文字小文字を区別しない）
    categories = [cat.lower() for cat in categories] if categories else []
    exclude_categories = [cat.lower() for cat in exclude_categories] if exclude_categories else []
    
    # タグの抽出
    for tag_data in mapping.values():
        if isinstance(tag_data, dict):
            tag = tag_data.get("tag")
            category = tag_data.get("category", "").lower()
            
            if not tag or not category:
                continue
                
            if categories and category not in categories:
                continue
            if exclude_categories and category in exclude_categories:
                continue
                
            tags.append(tag)

    return sorted(tags)  # アルファベット順にソート

def main():
    parser = argparse.ArgumentParser(description='タグマッピングから特定カテゴリのタグを抽出')
    parser.add_argument('--mapping', type=str, required=True,
                      help='tag_mapping.jsonのパス')
    parser.add_argument('--output', type=str, required=True,
                      help='出力ファイルパス')
    parser.add_argument('--categories', type=str, nargs='+',
                      help='抽出するカテゴリ（スペース区切りで複数指定可能）')
    parser.add_argument('--exclude', type=str, nargs='+',
                      help='除外するカテゴリ（スペース区切りで複数指定可能）')
    parser.add_argument('--newline', action='store_true',
                      help='カンマ区切りの代わりに改行区切りで出力')
    
    args = parser.parse_args()
    
    # マッピングの読み込み
    mapping = load_tag_mapping(args.mapping)
    
    # タグの抽出
    tags = extract_tags_by_category(
        mapping,
        args.categories,
        args.exclude
    )
    
    # 出力
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if args.newline:
            f.write('\n'.join(tags))
        else:
            f.write(', '.join(tags))

    print(f"抽出されたタグ数: {len(tags)}")
    print(f"出力先: {output_path}")
    
    # デバッグ情報の表示
    if len(tags) == 0:
        print("\nデバッグ情報:")
        print(f"指定されたカテゴリ: {args.categories}")
        print(f"最初の5つのエントリ:")
        for i, (key, value) in enumerate(list(mapping.items())[:5]):
            print(f"{key}: {value}")

if __name__ == '__main__':
    main() 