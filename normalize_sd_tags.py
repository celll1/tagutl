import os
import argparse
import re
import json
import random

CATEGORY_ORDER = [
    'rating', 'quality', 'character', 'copyright', 'artist', 'general_special', 'general_person_count', 'general', 'unknown', 'meta', 'model'
]

# Rating と Quality タグのハードコードリスト (仮)
HARDCODED_RATING_TAGS = ['general', 'sensitive', 'questionable', 'explicit']
HARDCODED_QUALITY_TAGS = ['best quality', 'high quality', 'normal quality', 'low quality', 'worst quality', 'bad quality'] # bad quality を追加

# 人数関連タグを識別するための正規表現パターン
# 例: 1girl, 2girls, 1boy, multiple boys, 1other, multiple_others などにマッチ
PERSON_COUNT_TAG_PATTERNS = [
    re.compile(r"^\d+girls?$"),
    re.compile(r"^\d+boys?$"),
    re.compile(r"^\d+others?$"),
    re.compile(r"^no_humans$"),
    re.compile(r"^multiple_girls$"),
    re.compile(r"^multiple_boys$"),
    re.compile(r"^multiple_others$"),
    re.compile(r"^group$"), # group タグも人数関連として扱う
    re.compile(r"^solo$"),   # solo も人数関連
    re.compile(r"^.*_focus$"),  # 任意の*_focusタグ（solo_focus, male_focus等）
    re.compile(r"^still_life$")  # still_life も人数関連として扱う
]

# 特殊ケースの定義
DOUBLE_BACKSLASH_SLASH = "\\//"
PLACEHOLDER = "__DOUBLE_BACKSLASH_SLASH_PLACEHOLDER__"

def is_learning_exclusion_tag(tag: str) -> bool:
    """指定されたタグが学習除外タグ（a@xxx形式）かどうかを判定する。"""
    return re.match(r'^[a-zA-Z]@', tag) is not None

def is_person_count_tag(normalized_tag_for_sd: str) -> bool:
    """指定されたタグが人数関連タグかどうかを判定する。
    判定は normalize_tag_for_sd で正規化された後のタグに対して行う。
    normalize_tag_for_sd は小文字化し、括弧をエスケープ、アンダースコアをスペースにする。
    ただし、人数関連タグのパターンはアンダースコア区切りを想定しているため、
    判定前にスペースをアンダースコアに戻すか、パターン側をスペース区切りにする必要がある。
    ここでは、渡されるタグがスペース区切りになっていることを前提とする。
    """
    # normalize_tag_for_sd はスペースを維持するので、パターン側もスペースを許容するか、
    # もしくは判定前にスペースをアンダースコアに置換して比較する。
    # ここでは、パターンはアンダースコアを想定し、入力タグのスペースをアンダースコアに置換して比較する。
    tag_for_pattern_check = normalized_tag_for_sd.replace(' ', '_') 
    # 括弧エスケープはパターンに含まれないので、パターンマッチング前には影響しない想定
    for pattern in PERSON_COUNT_TAG_PATTERNS:
        if pattern.match(tag_for_pattern_check):
            return True
    return False

def normalize_tag_for_sd(tag: str, debug_print: bool = False) -> str:
    """Stable Diffusion用のタグ形式に正規化する関数。
    括弧をエスケープし、すべて小文字にする。スペースは維持。
    エスケープは必ず1回だけ行われるようにする。'\\//' は保護する。
    """
    original_tag_for_debug = tag # デバッグ用に元のタグを保持
    tag_changed_in_step2 = False
    tag_changed_in_step3 = False

    tag = tag.lower()

    tag = tag.replace(DOUBLE_BACKSLASH_SLASH, PLACEHOLDER)

    # ステップ2: 過剰なエスケープの修正
    before_step2 = tag
    tag = re.sub(r'\\{2,}\(', r'\\(', tag)
    tag = re.sub(r'\\{2,}\)', r'\\)', tag)
    if before_step2 != tag:
        tag_changed_in_step2 = True

    # ステップ3: 未エスケープの括弧のエスケープ
    before_step3 = tag
    tag = re.sub(r'(?<!\\)\(', r'\\(', tag)
    tag = re.sub(r'(?<!\\)\)', r'\\)', tag)
    if before_step3 != tag:
        tag_changed_in_step3 = True
    
    tag = tag.replace(PLACEHOLDER, DOUBLE_BACKSLASH_SLASH)

    # デバッグ表示 (ステップ2または3で変更があった場合のみ)
    if debug_print and (tag_changed_in_step2 or tag_changed_in_step3):
        print(f"  [DEBUG] Tag normalized: '{original_tag_for_debug}' -> '{tag.replace('_', ' ')}' (Step2Change: {tag_changed_in_step2}, Step3Change: {tag_changed_in_step3})")

    tag = tag.replace('_', ' ')
    return tag.strip()

def load_tag_aliases(alias_file_path: str) -> dict:
    """タグエイリアスを読み込む。古いタグ名をキー、新しいタグ名を値とする辞書を返す。"""
    if not os.path.isfile(alias_file_path):
        return {}
    
    try:
        with open(alias_file_path, 'r', encoding='utf-8') as f:
            aliases = json.load(f)
        if not isinstance(aliases, dict):
            print(f"Warning: Expected a dictionary in {alias_file_path}, but got {type(aliases)}.")
            return {}
        return aliases
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {alias_file_path}.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading aliases from {alias_file_path}: {e}")
        return {}

def load_categories_from_tag_groups(tag_group_dir: str) -> dict:
    """taggroupディレクトリからカテゴリ情報を読み込む。"""
    tag_to_category = {}

    # 1. ハードコードされたRatingタグを登録
    for tag_name in HARDCODED_RATING_TAGS:
        normalized_key = tag_name.lower().replace('_', ' ') # 検索キーの正規化
        tag_to_category[normalized_key] = 'rating'

    # 2. ハードコードされたQualityタグを登録
    for tag_name in HARDCODED_QUALITY_TAGS:
        normalized_key = tag_name.lower().replace('_', ' ') # 検索キーの正規化
        tag_to_category[normalized_key] = 'quality'

    # 3. taggroupディレクトリ内のJSONファイルからカテゴリ情報を読み込む
    if not os.path.isdir(tag_group_dir):
        print(f"Warning: Tag group directory not found at {tag_group_dir}. Only hardcoded tags will be categorized.")
        return tag_to_category

    expected_files = {
        'character': 'Character.json',
        'copyright': 'Copyright.json',
        'artist': 'Artist.json',
        'general': 'General.json',
        'meta': 'Meta.json',
        'model': 'Model.json',
    }

    for category_name_code, filename in expected_files.items():
        file_path = os.path.join(tag_group_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for tag_name_from_file in data.keys():
                    # ファイルから読み込んだタグ名も検索キーとして正規化
                    # タグファイル処理時と同じ正規化を適用（括弧エスケープ除去を含む）
                    search_key = tag_name_from_file.lower().replace('_', ' ')
                    normalized_key = search_key.replace('\\(', '(').replace('\\)', ')')
                    # 既にRating/Qualityで登録されていなければ、ファイル由来のカテゴリを登録
                    if normalized_key not in tag_to_category: 
                        tag_to_category[normalized_key] = category_name_code
                    # else: # 既にRating/Qualityで登録されている場合は上書きしない
                        # print(f"Tag '{normalized_key}' already categorized as '{tag_to_category[normalized_key]}', not overwriting with '{category_name_code}' from {filename}")
            else:
                print(f"Warning: Expected a dictionary in {file_path}, but got {type(data)}.")
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {tag_group_dir}.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}.")
        except Exception as e:
            print(f"Warning: Error loading tags from {file_path}: {e}")
            
    return tag_to_category

def apply_tag_aliases(tag: str, tag_aliases: dict) -> str:
    """タグエイリアスを適用する。エイリアスが見つかればそれを返し、なければ元のタグを返す。"""
    # タグを小文字に変換してエイリアスを検索
    lower_tag = tag.lower()
    if lower_tag in tag_aliases:
        return tag_aliases[lower_tag]
    
    # スペースとアンダースコアを考慮した検索
    # アンダースコアをスペースに変換して検索
    tag_with_spaces = lower_tag.replace('_', ' ')
    if tag_with_spaces in tag_aliases:
        return tag_aliases[tag_with_spaces]
    
    # スペースをアンダースコアに変換して検索
    tag_with_underscores = lower_tag.replace(' ', '_')
    if tag_with_underscores in tag_aliases:
        return tag_aliases[tag_with_underscores]
    
    return tag

def process_tag_file(file_path: str, tag_to_category: dict, tag_aliases: dict = None, apply_aliases: bool = False, debug_normalization: bool = False, randomize_categories: list = None):
    """単一のタグファイル（TXTまたはJSON）を処理し、カテゴリ順にソート後、正規化する。"""
    import json
    
    try:
        # ファイル形式の判定
        is_json = file_path.lower().endswith('.json')
        
        if is_json:
            # JSONファイルの読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # tagsフィールドから文字列を取得
            if 'tags' not in json_data:
                if debug_normalization:
                    print(f"No 'tags' field found in JSON file: {file_path}")
                return
            
            tags_string = json_data['tags']
            if isinstance(tags_string, list):
                # リスト形式の場合
                tags_in_file = [tag.strip() for tag in tags_string if tag.strip()]
            else:
                # 文字列形式（カンマ区切り）の場合
                tags_in_file = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
        else:
            # TXTファイルの読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tags_in_file = [tag.strip() for tag in content.split(',') if tag.strip()]
        
        if not tags_in_file:
            return

        if debug_normalization:
            print(f"Processing file for debug: {file_path}")

        categorized_tags = {cat: set() for cat in CATEGORY_ORDER}  # リストではなくセットを使用
        duplicate_count = 0  # 重複カウント用

        for tag_str_from_file in tags_in_file:
            # エイリアスを適用（オプションが有効な場合）
            if apply_aliases and tag_aliases:
                tag_str_after_alias = apply_tag_aliases(tag_str_from_file, tag_aliases)
                if tag_str_after_alias != tag_str_from_file and debug_normalization:
                    print(f"  [DEBUG] Alias applied: '{tag_str_from_file}' -> '{tag_str_after_alias}'")
                tag_str_from_file = tag_str_after_alias
            
            search_key = tag_str_from_file.lower().replace('_', ' ')
            search_key_no_escape_no_sd_norm = search_key.replace('\\\\(', '(').replace('\\\\)', ')')
            category = tag_to_category.get(search_key_no_escape_no_sd_norm)
            # normalize_tag_for_sd にデバッグフラグを渡す
            final_normalized_tag = normalize_tag_for_sd(tag_str_from_file, debug_print=debug_normalization)

            # 学習除外タグかどうかをチェック
            if is_learning_exclusion_tag(tag_str_from_file):
                category_to_add = 'general_special'
            # 人数関連タグの処理（General カテゴリの場合のみ）
            elif category == 'general' and is_person_count_tag(final_normalized_tag):
                category_to_add = 'general_person_count'
            elif category and category in categorized_tags:
                category_to_add = category
            else:
                # カテゴリが不明または存在しない場合はunknownに分類
                category_to_add = 'unknown'
            
            # 重複チェック（すでに同じ正規化タグが存在するかチェック）
            if final_normalized_tag in categorized_tags[category_to_add]:
                duplicate_count += 1
                if debug_normalization:
                    print(f"  [DEBUG] Duplicate tag skipped: '{final_normalized_tag}' (category: {category_to_add})")
            else:
                categorized_tags[category_to_add].add(final_normalized_tag)
        
        # セットをリストに変換してソート（またはランダム化）
        for cat in categorized_tags:
            if cat == 'general_special':
                # 学習除外タグは常にアルファベット順（ランダム化対象外）
                categorized_tags[cat] = sorted(list(categorized_tags[cat]))
            elif cat == 'general_person_count':
                # 人数関連タグの処理
                if randomize_categories and 'general_person_count' in randomize_categories:
                    tag_list = list(categorized_tags[cat])
                    random.shuffle(tag_list)
                    categorized_tags[cat] = tag_list
                else:
                    categorized_tags[cat] = sorted(list(categorized_tags[cat]))
            elif cat == 'general':
                # generalカテゴリ（人数タグは既に分離済み）
                if randomize_categories and 'general' in randomize_categories:
                    tag_list = list(categorized_tags[cat])
                    random.shuffle(tag_list)
                    categorized_tags[cat] = tag_list
                else:
                    categorized_tags[cat] = sorted(list(categorized_tags[cat]))
            elif cat == 'unknown':
                # unknownカテゴリの処理
                if randomize_categories and 'unknown' in randomize_categories:
                    tag_list = list(categorized_tags[cat])
                    random.shuffle(tag_list)
                    categorized_tags[cat] = tag_list
                else:
                    categorized_tags[cat] = sorted(list(categorized_tags[cat]))
            elif randomize_categories and cat in randomize_categories:
                # その他でランダム化が指定されたカテゴリの場合
                tag_list = list(categorized_tags[cat])
                random.shuffle(tag_list)
                categorized_tags[cat] = tag_list
            else:
                categorized_tags[cat] = sorted(list(categorized_tags[cat]))  # セットをソート済みリストに変換

        sorted_normalized_tags = []
        for cat_name in CATEGORY_ORDER:
            sorted_normalized_tags.extend(categorized_tags[cat_name])
        # sorted_normalized_tags.extend(unknown_tags) # 不要
        
        new_tags_content = ", ".join(sorted_normalized_tags)

        if duplicate_count > 0:
            print(f"  Removed {duplicate_count} duplicate tag(s) in {file_path}")

        # 変更があった場合のみ保存
        if is_json:
            # JSON形式での保存
            original_tags_string = json_data['tags']
            if isinstance(original_tags_string, list):
                # 元がリスト形式の場合はリスト形式で保存
                json_data['tags'] = sorted_normalized_tags
            else:
                # 元が文字列形式の場合は文字列形式で保存
                json_data['tags'] = new_tags_content
            
            # 元のJSON文字列と比較して変更があるかチェック
            original_content = json.dumps(json_data, ensure_ascii=False, separators=(',', ':'))
            if isinstance(original_tags_string, list):
                temp_data = json_data.copy()
                temp_data['tags'] = ", ".join(original_tags_string) if original_tags_string else ""
                original_comparison = json.dumps(temp_data, ensure_ascii=False, separators=(',', ':'))
            else:
                original_comparison = json.dumps({**json_data, 'tags': original_tags_string}, ensure_ascii=False, separators=(',', ':'))
            
            if original_comparison != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"Updated: {file_path}")
        else:
            # TXT形式での保存
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read().strip()
            
            if original_content != new_tags_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_tags_content)
                print(f"Updated: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Normalize and sort tags in text files for Stable Diffusion training. Example: 'tag with spaces (and parens)' -> 'tag with spaces \\\\(and parens\\\\)'")
    parser.add_argument('directories', nargs='+', help='One or more directories to process recursively.')
    # parser.add_argument('--tag_mapping', type=str, required=True, help='Path to the tag_mapping.json file for category information.')
    parser.add_argument('--tag_group_dir', type=str, required=True, help='Path to the directory containing category JSON files (e.g., Character.json, General.json).')
    parser.add_argument('--tag_aliases', type=str, help='Path to the tag_aliases.json file for converting old tags to current tags.')
    parser.add_argument('--apply_aliases', action='store_true', help='Apply tag aliases to convert old tags to current tags.')
    parser.add_argument('--debug_normalization', action='store_true', help='Print debug information for tags whose normalization changed due to escaping.')
    parser.add_argument('--randomize', type=str, nargs='*', help='Randomize tags within specified categories. Use "all" for all categories, or specify category names (e.g., general character)')

    args = parser.parse_args()

    # tag_categories = load_tag_categories(args.tag_mapping)
    tag_categories = load_categories_from_tag_groups(args.tag_group_dir)
    if not tag_categories:
        print("Warning: No category information loaded (or only hardcoded ones). Tags will be normalized but might not be sorted by category as expected.")
    # else:
        # print(f"Loaded {len(tag_categories)} tag-category mappings.") # デバッグ用

    # タグエイリアスの読み込み
    tag_aliases = {}
    if args.apply_aliases:
        if not args.tag_aliases:
            print("Error: --apply_aliases flag is set but --tag_aliases path is not provided.")
            return
        tag_aliases = load_tag_aliases(args.tag_aliases)
        if tag_aliases:
            print(f"Loaded {len(tag_aliases)} tag aliases.")
        else:
            print("Warning: No tag aliases loaded or file not found.")
    
    # ランダム化するカテゴリの設定
    randomize_categories = None
    if args.randomize is not None:
        if len(args.randomize) == 0 or 'all' in args.randomize:
            # --randomize または --randomize all の場合、全カテゴリをランダム化
            randomize_categories = CATEGORY_ORDER
            print("Randomizing tags in all categories.")
        else:
            # 指定されたカテゴリのみランダム化
            randomize_categories = [cat for cat in args.randomize if cat in CATEGORY_ORDER]
            invalid_cats = [cat for cat in args.randomize if cat not in CATEGORY_ORDER]
            if invalid_cats:
                print(f"Warning: Invalid category names ignored: {invalid_cats}")
            if randomize_categories:
                print(f"Randomizing tags in categories: {randomize_categories}")

    for directory in args.directories:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory. Skipping.")
            continue
        
        print(f"Processing directory: {directory}...")
        for root, _, files in os.walk(directory):
            for file in files:
                # TXTファイルとJSONファイルの両方を処理
                if file.lower().endswith('.txt') or file.lower().endswith('.json'):
                    file_path = os.path.join(root, file)
                    process_tag_file(file_path, tag_categories, tag_aliases, args.apply_aliases, args.debug_normalization, randomize_categories)
        print(f"Finished processing directory: {directory}")

if __name__ == '__main__':
    main()