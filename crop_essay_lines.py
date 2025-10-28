#!/usr/bin/env python3
"""
從 Detection 標註裁剪文字行
用於 Recognition 訓練的補充資料
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='從 Detection 標註裁剪文字行')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Detection 標註文件（train.txt 或 val.txt）')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='圖片目錄路徑')
    parser.add_argument('--output_dir', type=str, default='train_data/rec/essay_lines',
                        help='輸出目錄（默認: train_data/rec/essay_lines）')
    parser.add_argument('--min_text_length', type=int, default=2,
                        help='最小文字長度（默認: 2，過濾太短的文字）')
    parser.add_argument('--padding', type=int, default=5,
                        help='裁剪邊距（默認: 5 像素）')
    return parser.parse_args()

def read_det_labels(label_file):
    """讀取 Detection 標註文件"""
    labels = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
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

def get_crop_bbox(points, padding=5):
    """
    從 4 點座標獲取裁剪邊界框
    points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    points = np.array(points, dtype=np.float32)

    # 獲取最小外接矩形
    x_min = max(0, int(np.min(points[:, 0])) - padding)
    y_min = max(0, int(np.min(points[:, 1])) - padding)
    x_max = int(np.max(points[:, 0])) + padding
    y_max = int(np.max(points[:, 1])) + padding

    return x_min, y_min, x_max, y_max

def crop_text_region(image, points, padding=5):
    """裁剪文字區域"""
    x_min, y_min, x_max, y_max = get_crop_bbox(points, padding)

    # 裁剪
    crop = image[y_min:y_max, x_min:x_max]

    return crop

def main():
    args = parse_args()

    print("=" * 70)
    print("從 Detection 標註裁剪文字行")
    print("=" * 70)

    # 檢查輸入
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
    crop_dir = output_dir / 'crop_img'
    crop_dir.mkdir(parents=True, exist_ok=True)

    # 讀取標註
    print(f"\n讀取標註文件: {label_file}")
    labels = read_det_labels(label_file)
    print(f"✅ 讀取 {len(labels)} 個標註")

    # 裁剪文字區域
    print(f"\n開始裁剪...")
    label_lines = []
    crop_count = 0
    skip_count = 0

    for item in labels:
        image_name = item['image_name']
        annotations = item['annotations']

        # 讀取圖片
        image_path = image_dir / image_name
        if not image_path.exists():
            print(f"警告：圖片不存在: {image_path}")
            continue

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"警告：無法讀取圖片: {image_path}")
                continue
        except Exception as e:
            print(f"警告：讀取圖片失敗: {image_path}, 錯誤: {e}")
            continue

        # 處理每個文字區域
        for idx, ann in enumerate(annotations):
            points = ann['points']
            transcription = ann.get('transcription', '')

            # 過濾太短的文字
            if len(transcription) < args.min_text_length:
                skip_count += 1
                continue

            # 跳過佔位符
            if transcription in ['###', '***']:
                skip_count += 1
                continue

            # 裁剪文字區域
            try:
                crop = crop_text_region(image, points, args.padding)

                # 檢查裁剪結果
                if crop.size == 0:
                    print(f"警告：裁剪結果為空: {image_name}, 索引 {idx}")
                    skip_count += 1
                    continue

                # 保存裁剪圖片
                stem = Path(image_name).stem
                crop_name = f"{stem}_{idx}.jpg"
                crop_path = crop_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                # 記錄標註
                rel_path = f"crop_img/{crop_name}"
                label_lines.append(f"{rel_path}\t{transcription}\n")

                crop_count += 1

            except Exception as e:
                print(f"警告：裁剪失敗: {image_name}, 索引 {idx}, 錯誤: {e}")
                skip_count += 1
                continue

    # 保存標註文件
    label_output = output_dir / 'labels.txt'
    with open(label_output, 'w', encoding='utf-8') as f:
        f.writelines(label_lines)

    # 完成
    print("\n" + "=" * 70)
    print("裁剪完成！")
    print("=" * 70)

    print(f"\n統計信息:")
    print(f"  總圖片數: {len(labels)}")
    print(f"  成功裁剪: {crop_count} 個文字區域")
    print(f"  跳過: {skip_count} 個文字區域")

    print(f"\n輸出目錄: {output_dir}")
    print(f"  - crop_img/  # {crop_count} 個裁剪圖片")
    print(f"  - labels.txt # 標註文件")

    print("\n標註文件格式:")
    if label_lines:
        print("  " + label_lines[0].strip())
        if len(label_lines) > 1:
            print("  " + label_lines[1].strip())
        print("  ...")

    print("\n下一步:")
    print("1. 檢查裁剪的圖片質量")
    print("2. 合併單字資料和作文行資料:")
    print(f"   cat char_data/train.txt {label_output} > all_train.txt")
    print("3. 分割訓練集和驗證集")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
