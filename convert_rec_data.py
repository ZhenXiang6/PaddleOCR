#!/usr/bin/env python3
"""
Recognition 資料轉換腳本
將各種格式的手寫字資料轉換成 PaddleOCR Recognition 訓練格式
"""
import argparse
import json
import random
from pathlib import Path
from shutil import copy2

def parse_args():
    parser = argparse.ArgumentParser(description='轉換 Recognition 訓練資料')
    parser.add_argument('--input', type=str, required=True,
                        help='輸入資料路徑（可以是目錄或標註文件）')
    parser.add_argument('--format', type=str, default='auto',
                        choices=['auto', 'filename', 'json', 'csv', 'txt'],
                        help='輸入格式（默認: auto 自動檢測）')
    parser.add_argument('--output_dir', type=str, default='train_data/rec',
                        help='輸出目錄（默認: train_data/rec）')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='訓練集比例（默認: 0.9）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='隨機種子（默認: 42）')
    parser.add_argument('--encoding', type=str, default='utf-8',
                        help='文件編碼（默認: utf-8）')
    return parser.parse_args()

def detect_format(input_path):
    """自動檢測輸入格式"""
    input_path = Path(input_path)

    if input_path.is_dir():
        # 檢查是否為圖片目錄（檔名包含標籤）
        image_files = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.png'))
        if image_files:
            return 'filename'
        return None

    # 檢查文件副檔名
    suffix = input_path.suffix.lower()
    if suffix == '.json':
        return 'json'
    elif suffix == '.csv':
        return 'csv'
    elif suffix == '.txt':
        return 'txt'

    return None

def parse_filename_format(image_dir):
    """
    從檔名解析標籤
    支持格式:
    - 水_001.jpg → 水
    - water_水.jpg → 水
    - 001_水.jpg → 水
    """
    image_dir = Path(image_dir)
    data = []

    image_files = list(image_dir.glob('*.jpg')) + \
                  list(image_dir.glob('*.png')) + \
                  list(image_dir.glob('*.jpeg'))

    for img_path in image_files:
        # 嘗試從檔名提取標籤
        filename = img_path.stem  # 不含副檔名

        # 方法1: 檔名就是標籤（純文字）
        # 例如: 水.jpg
        if len(filename) == 1:
            label = filename
        # 方法2: 標籤_序號 格式
        # 例如: 水_001.jpg
        elif '_' in filename:
            parts = filename.split('_')
            # 取第一部分作為標籤
            label = parts[0]
            # 如果第一部分是數字，取第二部分
            if parts[0].isdigit() and len(parts) > 1:
                label = parts[1]
        # 方法3: 整個檔名作為標籤
        else:
            label = filename

        if label:
            data.append({
                'image_path': str(img_path),
                'label': label
            })

    return data

def parse_json_format(json_file):
    """
    從 JSON 文件解析
    格式: [{"image": "path/to/img.jpg", "text": "標籤"}, ...]
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    data = []
    for item in json_data:
        if 'image' in item and 'text' in item:
            data.append({
                'image_path': item['image'],
                'label': item['text']
            })

    return data

def parse_csv_format(csv_file, encoding='utf-8'):
    """
    從 CSV 文件解析
    格式: image_path,label
    """
    import csv

    data = []
    with open(csv_file, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳過表頭（如果有）

        for row in reader:
            if len(row) >= 2:
                data.append({
                    'image_path': row[0],
                    'label': row[1]
                })

    return data

def parse_txt_format(txt_file, encoding='utf-8'):
    """
    從 TXT 文件解析
    支持格式:
    - image_path[TAB]label
    - image_path label
    - image_path,label
    """
    data = []
    with open(txt_file, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 嘗試不同的分隔符
            for sep in ['\t', ' ', ',']:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    data.append({
                        'image_path': parts[0],
                        'label': parts[1]
                    })
                    break

    return data

def load_data(input_path, format_type, encoding='utf-8'):
    """載入資料"""
    input_path = Path(input_path)

    # 自動檢測格式
    if format_type == 'auto':
        format_type = detect_format(input_path)
        if format_type is None:
            raise ValueError(f"無法自動檢測格式: {input_path}")
        print(f"✅ 自動檢測格式: {format_type}")

    # 根據格式載入資料
    if format_type == 'filename':
        return parse_filename_format(input_path)
    elif format_type == 'json':
        return parse_json_format(input_path)
    elif format_type == 'csv':
        return parse_csv_format(input_path, encoding)
    elif format_type == 'txt':
        return parse_txt_format(input_path, encoding)
    else:
        raise ValueError(f"不支援的格式: {format_type}")

def validate_data(data):
    """驗證資料"""
    valid_data = []

    for item in data:
        image_path = Path(item['image_path'])
        label = item['label']

        # 檢查圖片是否存在
        if not image_path.exists():
            print(f"警告：圖片不存在: {image_path}")
            continue

        # 檢查標籤是否為空
        if not label or not label.strip():
            print(f"警告：標籤為空: {image_path}")
            continue

        valid_data.append(item)

    return valid_data

def split_train_val(data, train_ratio, random_seed):
    """分割訓練集和驗證集"""
    random.seed(random_seed)
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return train_data, val_data

def save_recognition_labels(data, output_file, output_dir):
    """
    保存 Recognition 標註文件
    格式: image_path[TAB]label
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 複製圖片到輸出目錄
            src_path = Path(item['image_path'])
            dest_name = src_path.name

            # 如果檔名重複，加上前綴
            dest_path = image_dir / dest_name
            counter = 1
            while dest_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                dest_name = f"{stem}_{counter}{suffix}"
                dest_path = image_dir / dest_name
                counter += 1

            copy2(src_path, dest_path)

            # 寫入標註（相對路徑）
            rel_path = f"images/{dest_name}"
            label = item['label']
            f.write(f"{rel_path}\t{label}\n")

def main():
    args = parse_args()

    print("=" * 70)
    print("Recognition 資料轉換")
    print("=" * 70)

    # 檢查輸入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n❌ 錯誤：輸入路徑不存在: {input_path}")
        return 1

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    print(f"\n載入資料: {input_path}")
    data = load_data(input_path, args.format, args.encoding)
    print(f"✅ 載入 {len(data)} 筆資料")

    # 驗證資料
    print("\n驗證資料...")
    valid_data = validate_data(data)
    print(f"✅ 有效資料: {len(valid_data)} 筆")
    print(f"⚠️  無效資料: {len(data) - len(valid_data)} 筆")

    if len(valid_data) == 0:
        print("\n❌ 錯誤：沒有有效的資料")
        return 1

    # 分割訓練集和驗證集
    print(f"\n分割訓練集和驗證集（比例: {args.train_ratio:.1%}）...")
    train_data, val_data = split_train_val(
        valid_data, args.train_ratio, args.random_seed
    )
    print(f"✅ 訓練集: {len(train_data)} 筆")
    print(f"✅ 驗證集: {len(val_data)} 筆")

    # 保存標註文件
    print("\n保存標註文件...")
    train_file = output_dir / 'train.txt'
    val_file = output_dir / 'val.txt'

    save_recognition_labels(train_data, train_file, output_dir)
    save_recognition_labels(val_data, val_file, output_dir)

    print(f"✅ 訓練集標註: {train_file}")
    print(f"✅ 驗證集標註: {val_file}")

    # 統計信息
    print("\n" + "=" * 70)
    print("轉換完成！")
    print("=" * 70)

    print(f"\n統計信息:")
    print(f"  總資料數: {len(valid_data)}")
    print(f"  訓練集: {len(train_data)} 筆")
    print(f"  驗證集: {len(val_data)} 筆")

    # 統計標籤長度
    label_lengths = [len(item['label']) for item in valid_data]
    avg_length = sum(label_lengths) / len(label_lengths)
    max_length = max(label_lengths)
    min_length = min(label_lengths)

    print(f"  平均標籤長度: {avg_length:.1f} 字")
    print(f"  最長標籤: {max_length} 字")
    print(f"  最短標籤: {min_length} 字")

    print(f"\n輸出目錄: {output_dir}")
    print(f"  - images/    # 所有圖片")
    print(f"  - train.txt  # 訓練集標註")
    print(f"  - val.txt    # 驗證集標註")

    print("\n下一步:")
    print("1. 檢查輸出的標註文件格式")
    print("2. 如有作文資料，執行 crop_essay_lines.py 裁剪文字行")
    print("3. 合併資料: cat train.txt essay_lines/train.txt > all_train.txt")
    print("4. 修改配置文件 configs/rec/custom_rec.yml")
    print("5. 開始訓練: python tools/train.py -c configs/rec/custom_rec.yml")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
