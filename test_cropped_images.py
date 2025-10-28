#!/usr/bin/env python3
"""
裁剪图片 OCR 测试脚本
专门处理已裁剪的作文区域图片（PNG 格式）
"""
import sys
import time
import json
from pathlib import Path
from paddleocr import PaddleOCR
from tqdm import tqdm
import cv2
import numpy as np

# 最佳配置：Mobile 模型
OPTIMAL_CONFIG = {
    "ocr_version": "PP-OCRv5",
    "text_detection_model_name": "PP-OCRv5_mobile_det",
    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    "use_doc_orientation_classify": True,
    "use_doc_unwarping": False,
    "use_textline_orientation": True,
    "device": "cpu",
    "text_det_limit_side_len": 960,
    "text_det_thresh": 0.3,
    "text_det_box_thresh": 0.6,
    "text_det_unclip_ratio": 1.6,
    "text_rec_score_thresh": 0.5,
    "text_recognition_batch_size": 16
}

def draw_boxes_on_image(image, lines):
    """
    在图片上绘制检测框（全部绿色，因为都是手写文字）

    Args:
        image: numpy array 图片
        lines: OCR 结果的 lines 列表

    Returns:
        vis_image: 带检测框的图片
    """
    vis_image = image.copy()

    for i, line in enumerate(lines):
        bbox = line['bbox']
        text = line['text']
        score = line['score']

        # 转换坐标格式
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))

        # 绿色框表示识别的手写文字
        color = (0, 255, 0)
        thickness = 2

        # 绘制检测框
        cv2.polylines(vis_image, [pts], True, color, thickness)

        # 添加文本标签
        x = int(bbox[0][0])
        y = int(bbox[0][1]) - 5
        # 截断过长的文本
        label = text[:10] if len(text) <= 10 else text[:8] + "..."
        cv2.putText(vis_image, label, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis_image

def process_single_image(ocr, img_path, output_dir, progress_bar=None):
    """处理单个图片文件"""
    try:
        # 更新进度条
        if progress_bar:
            progress_bar.set_description(f"处理 {img_path.name[:30]}")

        # 读取图片
        img_array = cv2.imread(str(img_path))
        if img_array is None:
            return {
                'success': False,
                'image_name': img_path.name,
                'error': '无法读取图片'
            }

        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # OCR 处理
        start_time = time.time()
        result = ocr.predict(input=str(img_path))
        ocr_time = time.time() - start_time

        # 提取结果
        first_result = result[0]
        rec_texts = first_result.get('rec_texts', [])
        rec_scores = first_result.get('rec_scores', [])
        dt_polys = first_result.get('dt_polys', [])

        # 构建 lines 列表
        lines = [
            {
                'text': rec_texts[i],
                'score': float(rec_scores[i]) if i < len(rec_scores) else 0,
                'bbox': dt_polys[i].tolist() if i < len(dt_polys) else None
            }
            for i in range(len(rec_texts))
        ]

        num_lines = len(lines)
        total_chars = sum(len(text) for text in rec_texts)
        avg_conf = sum(rec_scores) / len(rec_scores) if rec_scores else 0

        # 保存可视化图片
        vis_dir = output_dir / "vis_cropped"
        vis_dir.mkdir(parents=True, exist_ok=True)

        vis_image = draw_boxes_on_image(img_array, lines)
        vis_path = vis_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        # 保存 TXT（所有识别文字）
        txt_dir = output_dir / "txt_cropped"
        txt_dir.mkdir(parents=True, exist_ok=True)
        txt_path = txt_dir / f"{img_path.stem}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line['text'] + '\n')

        # 保存详细 JSON
        json_dir = output_dir / "json_cropped"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"{img_path.stem}.json"

        json_data = {
            'image_name': img_path.name,
            'image_size': {
                'width': img_array.shape[1],
                'height': img_array.shape[0]
            },
            'ocr_time': ocr_time,
            'statistics': {
                'num_lines': num_lines,
                'total_chars': total_chars,
                'avg_confidence': avg_conf,
                'min_confidence': min(rec_scores) if rec_scores else 0,
                'max_confidence': max(rec_scores) if rec_scores else 0
            },
            'lines': lines
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"\n   ✅ {img_path.name}")
        print(f"      识别区块: {num_lines} 个")
        print(f"      总字符数: {total_chars}")
        print(f"      平均信心度: {avg_conf:.3f}")
        print(f"      OCR 耗时: {ocr_time:.2f} 秒")

        return {
            'success': True,
            'image_name': img_path.name,
            'num_lines': num_lines,
            'total_chars': total_chars,
            'avg_confidence': avg_conf,
            'ocr_time': ocr_time
        }

    except Exception as e:
        return {
            'success': False,
            'image_name': img_path.name,
            'error': str(e)
        }

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("裁剪作文图片 OCR 测试")
    print("使用 PP-OCRv5 Mobile 模型")
    print("=" * 70)

    # 检查数据目录
    data_dir = Path("/app/data/crop")
    if not data_dir.exists():
        print(f"\n❌ 数据目录不存在: {data_dir}")
        return 1

    # 获取所有 PNG 图片
    image_files = sorted(data_dir.glob("*.png"))
    if not image_files:
        print(f"\n❌ 没有找到 PNG 图片")
        return 1

    total_images = len(image_files)
    print(f"\n找到 {total_images} 个 PNG 图片")

    # 输出目录
    output_dir = Path("/app/output/cropped_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 OCR
    print("\n初始化 PP-OCRv5 Mobile 模型...")
    start_init = time.time()
    ocr = PaddleOCR(**OPTIMAL_CONFIG)
    init_time = time.time() - start_init
    print(f"✅ 初始化完成 (耗时: {init_time:.2f} 秒)")

    # 批量处理
    print(f"\n开始处理 {total_images} 个图片...")
    print("-" * 70)

    results = []
    start_batch = time.time()

    with tqdm(total=total_images, desc="总进度", unit="图片") as pbar:
        for img_path in image_files:
            result = process_single_image(ocr, img_path, output_dir, pbar)
            results.append(result)
            pbar.update(1)

    batch_time = time.time() - start_batch

    # 统计结果
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n总处理数: {total_images}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(failed)}")
    print(f"总耗时: {batch_time:.2f} 秒")
    print(f"平均每张图片: {batch_time/total_images:.2f} 秒")

    if successful:
        total_lines = sum(r['num_lines'] for r in successful)
        total_chars = sum(r['total_chars'] for r in successful)
        avg_conf = sum(r['avg_confidence'] for r in successful) / len(successful)

        print(f"\n统计数据:")
        print(f"  总文字区块: {total_lines}")
        print(f"  总字符数: {total_chars}")
        print(f"  平均信心度: {avg_conf:.3f}")
        print(f"  平均每张图片区块数: {total_lines/len(successful):.1f}")

    if failed:
        print(f"\n失败的文件:")
        for r in failed:
            print(f"  - {r['image_name']}: {r['error']}")

    # 保存处理报告
    report_path = output_dir / "processing_report.json"
    report_data = {
        'total_images': total_images,
        'successful': len(successful),
        'failed': len(failed),
        'total_time': batch_time,
        'avg_time_per_image': batch_time / total_images,
        'config': OPTIMAL_CONFIG,
        'results': results
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 处理报告已保存: {report_path}")
    print(f"✅ TXT 文件保存在: {output_dir / 'txt_cropped'}")
    print(f"✅ JSON 文件保存在: {output_dir / 'json_cropped'}")
    print(f"✅ 可视化图片保存在: {output_dir / 'vis_cropped'}")

    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
