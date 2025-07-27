import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set

def load_tag_mapping(mapping_path: str) -> Dict:
    """tag_mapping.jsonを読み込む"""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_tags(tags_path: str) -> List[str]:
    """既存のタグリストファイル（カンマ区切り）を読み込む"""
    try:
        with open(tags_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            # カンマ区切りで分割し、前後の空白を削除
            tags = [tag.strip() for tag in content.split(',') if tag.strip()]
            return tags
    except FileNotFoundError:
        print(f"既存のタグファイルが見つかりません: {tags_path}")
        return []

def merge_tags(existing_tags: List[str], new_tags: List[str]) -> List[str]:
    """既存のタグと新しいタグを統合し、重複を除去してソートする"""
    # セットを使って重複を除去
    all_tags: Set[str] = set(existing_tags + new_tags)
    return sorted(list(all_tags))

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
    parser.add_argument('--mapping', '-m', type=str, required=True,
                      help='tag_mapping.jsonのパス')
    parser.add_argument('--output', '-o', type=str, required=True,
                      help='出力ファイルパス')
    parser.add_argument('--categories', type=str, nargs='+',
                      help='抽出するカテゴリ（スペース区切りで複数指定可能）')
    parser.add_argument('--exclude', type=str, nargs='+',
                      help='除外するカテゴリ（スペース区切りで複数指定可能）')
    parser.add_argument('--newline', action='store_true',
                      help='カンマ区切りの代わりに改行区切りで出力')
    parser.add_argument('--existing-tags', '-e', type=str,
                      help='既存のタグリストファイル（カンマ区切り）のパス。指定すると新しく抽出されたタグを追加して保存')
    
    args = parser.parse_args()
    
    # マッピングの読み込み
    mapping = load_tag_mapping(args.mapping)
    
    # タグの抽出
    new_tags = extract_tags_by_category(
        mapping,
        args.categories,
        args.exclude
    )
    
    # 既存のタグリストがある場合は統合
    if args.existing_tags:
        existing_tags = load_existing_tags(args.existing_tags)
        final_tags = merge_tags(existing_tags, new_tags)
        print(f"既存のタグ数: {len(existing_tags)}")
        print(f"新しく抽出されたタグ数: {len(new_tags)}")
        print(f"統合後のタグ数: {len(final_tags)}")
        print(f"追加されたタグ数: {len(final_tags) - len(existing_tags)}")
    else:
        final_tags = new_tags
        print(f"抽出されたタグ数: {len(final_tags)}")
    
    # 出力
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if args.newline:
            f.write('\n'.join(final_tags))
        else:
            f.write(', '.join(final_tags))

    print(f"出力先: {output_path}")
    
    # デバッグ情報の表示
    if len(new_tags) == 0:
        print("\nデバッグ情報:")
        print(f"指定されたカテゴリ: {args.categories}")
        print(f"最初の5つのエントリ:")
        for i, (key, value) in enumerate(list(mapping.items())[:5]):
            print(f"{key}: {value}")

if __name__ == '__main__':
    main() 