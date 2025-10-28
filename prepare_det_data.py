#!/usr/bin/env python3
"""
Detection 資料準備腳本
將 PPOCRLabel 輸出轉換成 PaddleOCR 訓練格式
"""
import argparse
import json
import random
from pathlib import Path
from shutil import copy2

def parse_args():
    parser = argparse.ArgumentParser(description='準備 Detection 訓練資料')
    parser.add_argument('--label_file', type=str, required=True,
                        help='PPOCRLabel 輸出的 Label.txt 路徑')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='圖片目錄路徑')
    parser.add_argument('--output_dir', type=str, default='train_data/det',
                        help='輸出目錄（默認: train_data/det）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='訓練集比例（默認: 0.8）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='隨機種子（默認: 42）')
    return parser.parse_args()

def read_label_file(label_file):
    """讀取 PPOCRLabel 輸出的標註文件"""
    labels = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 分割圖片路徑和標註
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"警告：跳過格式錯誤的行: {line[:50]}...")
                continue

            image_name = parts[0]
            try:
                annotations = json.loads(parts[1])
                labels.append({
                    'image_name': image_name,
                    'annotations': annotations
                })
            except json.JSONDecodeError:
                print(f"警告：跳過 JSON 解析失敗的行: {image_name}")
                continue

    return labels

def validate_annotations(labels):
    """驗證標註數據"""
    valid_labels = []

    for item in labels:
        image_name = item['image_name']
        annotations = item['annotations']

        # 檢查是否有標註
        if not annotations:
            print(f"警告：{image_name} 沒有標註，跳過")
            continue

        # 驗證每個標註
        valid_annotations = []
        for ann in annotations:
            # 檢查必要字段
            if 'points' not in ann or 'transcription' not in ann:
                print(f"警告：{image_name} 的標註缺少必要字段，跳過此標註")
                continue

            points = ann['points']

            # 檢查點數量
            if len(points) != 4:
                print(f"警告：{image_name} 的標註點數不是4個，跳過此標註")
                continue

            # 檢查每個點是否有 x, y 坐標
            if not all(len(p) == 2 for p in points):
                print(f"警告：{image_name} 的標註點格式錯誤，跳過此標註")
                continue

            valid_annotations.append(ann)

        if valid_annotations:
            valid_labels.append({
                'image_name': image_name,
                'annotations': valid_annotations
            })

    return valid_labels

def split_train_val(labels, train_ratio, random_seed):
    """分割訓練集和驗證集"""
    random.seed(random_seed)
    random.shuffle(labels)

    split_idx = int(len(labels) * train_ratio)
    train_labels = labels[:split_idx]
    val_labels = labels[split_idx:]

    return train_labels, val_labels

def save_labels(labels, output_file):
    """保存標註文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in labels:
            image_name = item['image_name']
            annotations = item['annotations']

            # 轉換成 JSON 字符串
            json_str = json.dumps(annotations, ensure_ascii=False)

            # 寫入文件
            f.write(f"{image_name}\t{json_str}\n")

def copy_images(labels, image_dir, output_dir):
    """複製圖片到輸出目錄"""
    image_dir = Path(image_dir)
    output_image_dir = Path(output_dir) / 'images'
    output_image_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_count = 0

    for item in labels:
        image_name = item['image_name']
        source_path = image_dir / image_name

        if not source_path.exists():
            print(f"警告：圖片不存在: {source_path}")
            missing_count += 1
            continue

        dest_path = output_image_dir / image_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        copy2(source_path, dest_path)
        copied_count += 1

    return copied_count, missing_count

def main():
    args = parse_args()

    print("=" * 70)
    print("Detection 資料準備")
    print("=" * 70)

    # 檢查輸入文件
    label_file = Path(args.label_file)
    if not label_file.exists():
        print(f"\n❌ 錯誤：標註文件不存在: {label_file}")
        return 1

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"\n❌ 錯誤：圖片目錄不存在: {image_dir}")
        return 1

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 讀取標註
    print(f"\n讀取標註文件: {label_file}")
    labels = read_label_file(label_file)
    print(f"✅ 讀取 {len(labels)} 個標註")

    # 驗證標註
    print("\n驗證標註數據...")
    valid_labels = validate_annotations(labels)
    print(f"✅ 有效標註: {len(valid_labels)} 個")
    print(f"⚠️  無效標註: {len(labels) - len(valid_labels)} 個")

    if len(valid_labels) == 0:
        print("\n❌ 錯誤：沒有有效的標註數據")
        return 1

    # 分割訓練集和驗證集
    print(f"\n分割訓練集和驗證集（比例: {args.train_ratio:.1%}）...")
    train_labels, val_labels = split_train_val(
        valid_labels, args.train_ratio, args.random_seed
    )
    print(f"✅ 訓練集: {len(train_labels)} 個")
    print(f"✅ 驗證集: {len(val_labels)} 個")

    # 保存標註文件
    print("\n保存標註文件...")
    train_file = output_dir / 'train.txt'
    val_file = output_dir / 'val.txt'

    save_labels(train_labels, train_file)
    save_labels(val_labels, val_file)

    print(f"✅ 訓練集標註: {train_file}")
    print(f"✅ 驗證集標註: {val_file}")

    # 複製圖片
    print("\n複製圖片到輸出目錄...")
    all_labels = train_labels + val_labels
    copied, missing = copy_images(all_labels, image_dir, output_dir)

    print(f"✅ 已複製: {copied} 個圖片")
    if missing > 0:
        print(f"⚠️  缺失: {missing} 個圖片")

    # 統計信息
    print("\n" + "=" * 70)
    print("準備完成！")
    print("=" * 70)

    # 計算平均標註數
    total_annotations = sum(len(item['annotations']) for item in train_labels)
    avg_annotations_train = total_annotations / len(train_labels) if train_labels else 0

    total_annotations_val = sum(len(item['annotations']) for item in val_labels)
    avg_annotations_val = total_annotations_val / len(val_labels) if val_labels else 0

    print(f"\n統計信息:")
    print(f"  總圖片數: {len(valid_labels)}")
    print(f"  訓練集: {len(train_labels)} 個圖片")
    print(f"  驗證集: {len(val_labels)} 個圖片")
    print(f"  訓練集平均標註數/圖: {avg_annotations_train:.1f}")
    print(f"  驗證集平均標註數/圖: {avg_annotations_val:.1f}")

    print(f"\n輸出目錄: {output_dir}")
    print(f"  - images/    # 所有圖片")
    print(f"  - train.txt  # 訓練集標註")
    print(f"  - val.txt    # 驗證集標註")

    print("\n下一步:")
    print("1. 檢查輸出的標註文件格式是否正確")
    print("2. 修改配置文件 configs/det/custom_det.yml")
    print("3. 開始訓練: python tools/train.py -c configs/det/custom_det.yml")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
