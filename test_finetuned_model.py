#!/usr/bin/env python3
"""
測試 Fine-tuned 模型
對比預訓練模型和 Fine-tuned 模型的效果
"""
import argparse
import time
import json
from pathlib import Path
from paddleocr import PaddleOCR
import cv2
import numpy as np

# 使用現有的可視化功能
import sys
sys.path.insert(0, str(Path(__file__).parent))

def parse_args():
    parser = argparse.ArgumentParser(description='測試 Fine-tuned 模型')
    parser.add_argument('--test_images', type=str, required=True,
                        help='測試圖片目錄或單個圖片路徑')
    parser.add_argument('--det_model', type=str, default=None,
                        help='自定義 Detection 模型路徑（不指定則使用預訓練）')
    parser.add_argument('--rec_model', type=str, default=None,
                        help='自定義 Recognition 模型路徑（不指定則使用預訓練）')
    parser.add_argument('--output_dir', type=str, default='output/model_comparison',
                        help='輸出目錄（默認: output/model_comparison）')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='設備類型（默認: cpu）')
    parser.add_argument('--compare_pretrained', action='store_true',
                        help='是否同時運行預訓練模型進行對比')
    return parser.parse_args()

def draw_boxes_on_image(image, lines, color=(0, 255, 0)):
    """在圖片上繪製檢測框"""
    vis_image = image.copy()

    for i, line in enumerate(lines):
        bbox = line['bbox']
        text = line['text']
        score = line['score']

        # 轉換坐標格式
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))

        # 繪製檢測框
        cv2.polylines(vis_image, [pts], True, color, 2)

        # 添加文本標籤
        x = int(bbox[0][0])
        y = int(bbox[0][1]) - 5
        # 截斷過長的文本
        label = text[:15] if len(text) <= 15 else text[:12] + "..."
        cv2.putText(vis_image, label, (x, max(y, 15)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis_image

def test_single_image(ocr, img_path, model_name):
    """測試單個圖片"""
    try:
        # 讀取圖片
        img_array = cv2.imread(str(img_path))
        if img_array is None:
            return None
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # OCR 處理
        start_time = time.time()
        result = ocr.predict(input=str(img_path))
        ocr_time = time.time() - start_time

        # 提取結果
        first_result = result[0]
        rec_texts = first_result.get('rec_texts', [])
        rec_scores = first_result.get('rec_scores', [])
        dt_polys = first_result.get('dt_polys', [])

        # 構建 lines 列表
        lines = [
            {
                'text': rec_texts[i],
                'score': float(rec_scores[i]) if i < len(rec_scores) else 0,
                'bbox': dt_polys[i].tolist() if i < len(dt_polys) else None
            }
            for i in range(len(rec_texts))
        ]

        # 統計
        num_lines = len(lines)
        total_chars = sum(len(text) for text in rec_texts)
        avg_conf = sum(rec_scores) / len(rec_scores) if rec_scores else 0
        chars_per_line = total_chars / num_lines if num_lines > 0 else 0

        return {
            'model_name': model_name,
            'image_name': Path(img_path).name,
            'ocr_time': ocr_time,
            'num_lines': num_lines,
            'total_chars': total_chars,
            'avg_confidence': avg_conf,
            'chars_per_line': chars_per_line,
            'lines': lines,
            'image_array': img_array
        }

    except Exception as e:
        print(f"錯誤：處理圖片失敗 {img_path}: {e}")
        return None

def save_visualization(result, output_dir, model_name):
    """保存可視化結果"""
    vis_dir = Path(output_dir) / f"vis_{model_name}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_name = result['image_name']
    vis_image = draw_boxes_on_image(result['image_array'], result['lines'])

    vis_path = vis_dir / f"{Path(image_name).stem}.jpg"
    cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return vis_path

def save_text_results(result, output_dir, model_name):
    """保存文字結果"""
    txt_dir = Path(output_dir) / f"txt_{model_name}"
    txt_dir.mkdir(parents=True, exist_ok=True)

    image_name = result['image_name']
    txt_path = txt_dir / f"{Path(image_name).stem}.txt"

    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in result['lines']:
            f.write(line['text'] + '\n')

    return txt_path

def save_json_results(result, output_dir, model_name):
    """保存 JSON 結果"""
    json_dir = Path(output_dir) / f"json_{model_name}"
    json_dir.mkdir(parents=True, exist_ok=True)

    image_name = result['image_name']
    json_path = json_dir / f"{Path(image_name).stem}.json"

    # 移除 image_array（不能序列化）
    result_copy = result.copy()
    result_copy.pop('image_array', None)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_copy, f, ensure_ascii=False, indent=2)

    return json_path

def compare_results(pretrained_result, finetuned_result):
    """對比兩個模型的結果"""
    comparison = {
        'image_name': pretrained_result['image_name'],
        'pretrained': {
            'num_lines': pretrained_result['num_lines'],
            'total_chars': pretrained_result['total_chars'],
            'avg_confidence': pretrained_result['avg_confidence'],
            'chars_per_line': pretrained_result['chars_per_line'],
            'ocr_time': pretrained_result['ocr_time']
        },
        'finetuned': {
            'num_lines': finetuned_result['num_lines'],
            'total_chars': finetuned_result['total_chars'],
            'avg_confidence': finetuned_result['avg_confidence'],
            'chars_per_line': finetuned_result['chars_per_line'],
            'ocr_time': finetuned_result['ocr_time']
        },
        'improvements': {}
    }

    # 計算改善
    if pretrained_result['chars_per_line'] > 0:
        comparison['improvements']['chars_per_line_ratio'] = \
            finetuned_result['chars_per_line'] / pretrained_result['chars_per_line']

    comparison['improvements']['confidence_diff'] = \
        finetuned_result['avg_confidence'] - pretrained_result['avg_confidence']

    comparison['improvements']['lines_diff'] = \
        finetuned_result['num_lines'] - pretrained_result['num_lines']

    return comparison

def main():
    args = parse_args()

    print("=" * 70)
    print("測試 Fine-tuned 模型")
    print("=" * 70)

    # 檢查輸入
    test_path = Path(args.test_images)
    if not test_path.exists():
        print(f"\n❌ 錯誤：測試路徑不存在: {test_path}")
        return 1

    # 獲取測試圖片
    if test_path.is_file():
        test_images = [test_path]
    else:
        test_images = list(test_path.glob('*.jpg')) + \
                      list(test_path.glob('*.png')) + \
                      list(test_path.glob('*.jpeg'))

    if not test_images:
        print(f"\n❌ 錯誤：沒有找到測試圖片")
        return 1

    print(f"\n找到 {len(test_images)} 個測試圖片")

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Fine-tuned 模型
    print("\n初始化 Fine-tuned 模型...")
    finetuned_config = {
        'use_doc_orientation_classify': True,
        'use_doc_unwarping': False,
        'use_textline_orientation': True,
        'device': args.device
    }

    if args.det_model:
        finetuned_config['det_model_dir'] = args.det_model
        print(f"  ✅ Detection 模型: {args.det_model}")
    else:
        print(f"  ℹ️  Detection 模型: 預訓練")

    if args.rec_model:
        finetuned_config['rec_model_dir'] = args.rec_model
        print(f"  ✅ Recognition 模型: {args.rec_model}")
    else:
        print(f"  ℹ️  Recognition 模型: 預訓練")

    ocr_finetuned = PaddleOCR(**finetuned_config)

    # 初始化預訓練模型（如果需要對比）
    ocr_pretrained = None
    if args.compare_pretrained:
        print("\n初始化預訓練模型（用於對比）...")
        pretrained_config = {
            'use_doc_orientation_classify': True,
            'use_doc_unwarping': False,
            'use_textline_orientation': True,
            'device': args.device
        }
        ocr_pretrained = PaddleOCR(**pretrained_config)

    # 測試圖片
    print(f"\n開始測試 {len(test_images)} 個圖片...")
    print("-" * 70)

    finetuned_results = []
    pretrained_results = []
    comparisons = []

    for img_path in test_images:
        print(f"\n處理: {img_path.name}")

        # 測試 Fine-tuned 模型
        result_finetuned = test_single_image(ocr_finetuned, img_path, 'finetuned')
        if result_finetuned:
            finetuned_results.append(result_finetuned)

            # 保存結果
            save_visualization(result_finetuned, output_dir, 'finetuned')
            save_text_results(result_finetuned, output_dir, 'finetuned')
            save_json_results(result_finetuned, output_dir, 'finetuned')

            print(f"  Fine-tuned:")
            print(f"    區塊數: {result_finetuned['num_lines']}")
            print(f"    字符數: {result_finetuned['total_chars']}")
            print(f"    字/區塊: {result_finetuned['chars_per_line']:.1f}")
            print(f"    信心度: {result_finetuned['avg_confidence']:.3f}")
            print(f"    耗時: {result_finetuned['ocr_time']:.2f}秒")

        # 測試預訓練模型（如果需要）
        if ocr_pretrained:
            result_pretrained = test_single_image(ocr_pretrained, img_path, 'pretrained')
            if result_pretrained:
                pretrained_results.append(result_pretrained)

                # 保存結果
                save_visualization(result_pretrained, output_dir, 'pretrained')
                save_text_results(result_pretrained, output_dir, 'pretrained')
                save_json_results(result_pretrained, output_dir, 'pretrained')

                print(f"  預訓練:")
                print(f"    區塊數: {result_pretrained['num_lines']}")
                print(f"    字符數: {result_pretrained['total_chars']}")
                print(f"    字/區塊: {result_pretrained['chars_per_line']:.1f}")
                print(f"    信心度: {result_pretrained['avg_confidence']:.3f}")
                print(f"    耗時: {result_pretrained['ocr_time']:.2f}秒")

                # 對比
                if result_finetuned:
                    comparison = compare_results(result_pretrained, result_finetuned)
                    comparisons.append(comparison)

                    print(f"  改善:")
                    print(f"    字/區塊比例: {comparison['improvements']['chars_per_line_ratio']:.2f}x")
                    print(f"    信心度提升: {comparison['improvements']['confidence_diff']:+.3f}")

    # 匯總統計
    print("\n" + "=" * 70)
    print("測試完成！")
    print("=" * 70)

    if finetuned_results:
        avg_lines = sum(r['num_lines'] for r in finetuned_results) / len(finetuned_results)
        avg_chars = sum(r['total_chars'] for r in finetuned_results) / len(finetuned_results)
        avg_conf = sum(r['avg_confidence'] for r in finetuned_results) / len(finetuned_results)
        avg_chars_per_line = sum(r['chars_per_line'] for r in finetuned_results) / len(finetuned_results)

        print(f"\nFine-tuned 模型統計:")
        print(f"  平均區塊數: {avg_lines:.1f}")
        print(f"  平均字符數: {avg_chars:.1f}")
        print(f"  平均字/區塊: {avg_chars_per_line:.1f}")
        print(f"  平均信心度: {avg_conf:.3f}")

    if pretrained_results:
        avg_lines = sum(r['num_lines'] for r in pretrained_results) / len(pretrained_results)
        avg_chars = sum(r['total_chars'] for r in pretrained_results) / len(pretrained_results)
        avg_conf = sum(r['avg_confidence'] for r in pretrained_results) / len(pretrained_results)
        avg_chars_per_line = sum(r['chars_per_line'] for r in pretrained_results) / len(pretrained_results)

        print(f"\n預訓練模型統計:")
        print(f"  平均區塊數: {avg_lines:.1f}")
        print(f"  平均字符數: {avg_chars:.1f}")
        print(f"  平均字/區塊: {avg_chars_per_line:.1f}")
        print(f"  平均信心度: {avg_conf:.3f}")

    if comparisons:
        # 保存對比報告
        comparison_report = output_dir / 'comparison_report.json'
        with open(comparison_report, 'w', encoding='utf-8') as f:
            json.dump(comparisons, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 對比報告已保存: {comparison_report}")

    print(f"\n✅ 所有結果已保存至: {output_dir}")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
