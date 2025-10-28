#!/usr/bin/env python3
"""
智能过滤 + 可视化 OCR 处理脚本
功能：
1. 过滤打印文字（表头、标题、注意事项）
2. 过滤格子编号（纯数字）
3. 生成可视化图片（带检测框）
4. 输出过滤统计报告
"""
import sys
import time
import json
from pathlib import Path
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
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

# 过滤规则配置
FILTER_CONFIG = {
    # 关键词黑名单（打印文字）
    "keywords_blacklist": [
        "答題卷", "作答區", "注意", "國語文", "法人", "考試中心",
        "基金會", "學科能力", "測驗", "限在", "作答", "應標明"
    ],

    # X坐标阈值（右侧表头区域）
    "x_max_threshold": 3000,

    # 纯数字过滤（格子编号）
    "filter_pure_numbers": True,
    "max_number_length": 3,

    # 高信心度短文本过滤
    "high_conf_threshold": 0.95,
    "short_text_length": 3
}

def should_filter(text, score, bbox, filter_reasons):
    """
    判断文本是否应该被过滤掉

    Args:
        text: 识别的文本
        score: 信心度
        bbox: 边界框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        filter_reasons: 用于记录过滤原因的字典

    Returns:
        (should_filter: bool, reason: str)
    """
    # 提取 X 坐标
    x_min = min(bbox[0][0], bbox[3][0])
    x_max = max(bbox[1][0], bbox[2][0])

    # 规则 1: 关键词黑名单
    for keyword in FILTER_CONFIG["keywords_blacklist"]:
        if keyword in text:
            filter_reasons["keyword"] += 1
            return True, f"关键词: {keyword}"

    # 规则 2: X 坐标过滤（右侧表头）
    if x_min > FILTER_CONFIG["x_max_threshold"]:
        filter_reasons["x_coordinate"] += 1
        return True, f"X坐标: {x_min} > {FILTER_CONFIG['x_max_threshold']}"

    # 规则 3: 纯数字过滤（格子编号）
    if FILTER_CONFIG["filter_pure_numbers"]:
        stripped_text = text.strip()
        if stripped_text.isdigit() and len(stripped_text) <= FILTER_CONFIG["max_number_length"]:
            filter_reasons["pure_number"] += 1
            return True, f"纯数字: {stripped_text}"

    # 规则 4: 高信心度 + 短文本（可能是打印的标记）
    if (score > FILTER_CONFIG["high_conf_threshold"] and
        len(text.strip()) <= FILTER_CONFIG["short_text_length"]):
        filter_reasons["high_confidence_short"] += 1
        return True, f"高信心度短文本: {text} (信心度: {score:.3f})"

    return False, ""

def draw_boxes_on_image(image, lines, filter_status):
    """
    在图片上绘制检测框

    Args:
        image: numpy array 图片
        lines: OCR 结果的 lines 列表
        filter_status: 每个文本块的过滤状态列表

    Returns:
        vis_image: 带检测框的图片
    """
    vis_image = image.copy()

    for i, (line, status) in enumerate(zip(lines, filter_status)):
        bbox = line['bbox']
        text = line['text']
        score = line['score']

        # 转换坐标格式
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))

        # 根据过滤状态选择颜色
        if status['filtered']:
            color = (0, 0, 255)  # 红色 = 已过滤
            thickness = 1
        else:
            color = (0, 255, 0)  # 绿色 = 保留
            thickness = 2

        # 绘制检测框
        cv2.polylines(vis_image, [pts], True, color, thickness)

        # 添加文本标签（可选：只显示保留的文本）
        if not status['filtered']:
            x = int(bbox[0][0])
            y = int(bbox[0][1]) - 5
            # 截断过长的文本
            label = text[:10] if len(text) <= 10 else text[:8] + "..."
            cv2.putText(vis_image, label, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis_image

def process_single_pdf(ocr, pdf_path, output_dir, progress_bar=None):
    """处理单个 PDF 文件（带过滤和可视化）"""
    try:
        # 更新进度条
        if progress_bar:
            progress_bar.set_description(f"处理 {pdf_path.name[:30]}")

        # 转换 PDF 为图片
        images = convert_from_path(str(pdf_path), dpi=300, first_page=1, last_page=1)

        # 保存临时图片
        temp_img = Path("/tmp") / f"{pdf_path.stem}.jpg"
        images[0].save(temp_img, "JPEG")

        # 读取图片为 numpy array（用于可视化）
        img_array = cv2.imread(str(temp_img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # OCR 处理
        start_time = time.time()
        result = ocr.predict(input=str(temp_img))
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

        # 应用过滤规则
        filter_reasons = {
            "keyword": 0,
            "x_coordinate": 0,
            "pure_number": 0,
            "high_confidence_short": 0
        }

        filter_status = []
        kept_lines = []

        for line in lines:
            filtered, reason = should_filter(
                line['text'],
                line['score'],
                line['bbox'],
                filter_reasons
            )

            status = {
                'filtered': filtered,
                'reason': reason
            }
            filter_status.append(status)

            if not filtered:
                kept_lines.append(line)

        # 统计数据
        total_boxes = len(lines)
        filtered_count = sum(1 for s in filter_status if s['filtered'])
        kept_count = total_boxes - filtered_count

        # 保存可视化图片
        vis_dir_all = output_dir / "vis_all"
        vis_dir_filtered = output_dir / "vis_filtered"
        vis_dir_final = output_dir / "vis_final"

        for d in [vis_dir_all, vis_dir_filtered, vis_dir_final]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. 所有检测框（红色）
        vis_all = img_array.copy()
        for line in lines:
            pts = np.array(line['bbox'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_all, [pts], True, (255, 0, 0), 2)
        cv2.imwrite(str(vis_dir_all / f"{pdf_path.stem}.jpg"), cv2.cvtColor(vis_all, cv2.COLOR_RGB2BGR))

        # 2. 过滤对比图（绿色=保留，红色=移除）
        vis_filtered = draw_boxes_on_image(img_array, lines, filter_status)
        cv2.imwrite(str(vis_dir_filtered / f"{pdf_path.stem}.jpg"), cv2.cvtColor(vis_filtered, cv2.COLOR_RGB2BGR))

        # 3. 只显示保留的文本（绿色）
        vis_final = img_array.copy()
        for line, status in zip(lines, filter_status):
            if not status['filtered']:
                pts = np.array(line['bbox'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_final, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(str(vis_dir_final / f"{pdf_path.stem}.jpg"), cv2.cvtColor(vis_final, cv2.COLOR_RGB2BGR))

        # 保存过滤后的 TXT（只包含手写文字）
        txt_path = output_dir / "txt" / f"{pdf_path.stem}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in kept_lines:
                f.write(line['text'] + '\n')

        # 保存详细 JSON（包含过滤信息）
        json_path = output_dir / "json" / f"{pdf_path.stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        json_data = {
            'pdf_name': pdf_path.name,
            'ocr_time': ocr_time,
            'filter_config': FILTER_CONFIG,
            'statistics': {
                'total_boxes': total_boxes,
                'filtered_count': filtered_count,
                'kept_count': kept_count,
                'filter_reasons': filter_reasons,
                'avg_confidence': sum(l['score'] for l in kept_lines) / len(kept_lines) if kept_lines else 0,
                'total_chars': sum(len(l['text']) for l in kept_lines)
            },
            'all_lines': [
                {
                    **line,
                    'filtered': filter_status[i]['filtered'],
                    'filter_reason': filter_status[i]['reason']
                }
                for i, line in enumerate(lines)
            ],
            'kept_lines': kept_lines
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 删除临时文件
        temp_img.unlink()

        return {
            'success': True,
            'pdf_name': pdf_path.name,
            'total_boxes': total_boxes,
            'filtered_count': filtered_count,
            'kept_count': kept_count,
            'filter_reasons': filter_reasons,
            'ocr_time': ocr_time
        }

    except Exception as e:
        return {
            'success': False,
            'pdf_name': pdf_path.name,
            'error': str(e)
        }

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("学测作文 PDF 智能过滤 + 可视化处理")
    print("使用 PP-OCRv5 Mobile 模型 + 多层过滤系统")
    print("=" * 70)

    # 检查数据目录
    data_dir = Path("/app/data")
    if not data_dir.exists():
        print(f"\n❌ 数据目录不存在: {data_dir}")
        return 1

    # 获取所有 PDF 文件
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ 没有找到 PDF 文件")
        return 1

    total_pdfs = len(pdf_files)
    print(f"\n找到 {total_pdfs} 个 PDF 文件")

    # 输出目录
    output_dir = Path("/app/output/batch_ocr_filtered")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印过滤配置
    print("\n过滤规则配置:")
    print(f"  - 关键词黑名单: {len(FILTER_CONFIG['keywords_blacklist'])} 个")
    print(f"  - X坐标阈值: > {FILTER_CONFIG['x_max_threshold']}")
    print(f"  - 纯数字过滤: {FILTER_CONFIG['max_number_length']} 位以内")
    print(f"  - 高信心度短文本: 信心度 > {FILTER_CONFIG['high_conf_threshold']}, 长度 ≤ {FILTER_CONFIG['short_text_length']}")

    # 初始化 OCR
    print("\n初始化 PP-OCRv5 Mobile 模型...")
    start_init = time.time()
    ocr = PaddleOCR(**OPTIMAL_CONFIG)
    init_time = time.time() - start_init
    print(f"✅ 初始化完成 (耗时: {init_time:.2f} 秒)")

    # 批量处理
    print(f"\n开始批量处理 {total_pdfs} 个 PDF...")
    print("-" * 70)

    results = []
    start_batch = time.time()

    with tqdm(total=total_pdfs, desc="总进度", unit="PDF") as pbar:
        for pdf_path in pdf_files:
            result = process_single_pdf(ocr, pdf_path, output_dir, pbar)
            results.append(result)
            pbar.update(1)

    batch_time = time.time() - start_batch

    # 统计结果
    print("\n" + "=" * 70)
    print("批量处理完成！")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n总处理数: {total_pdfs}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(failed)}")
    print(f"总耗时: {batch_time:.2f} 秒")
    print(f"平均每个 PDF: {batch_time/total_pdfs:.2f} 秒")

    if successful:
        total_boxes = sum(r['total_boxes'] for r in successful)
        total_filtered = sum(r['filtered_count'] for r in successful)
        total_kept = sum(r['kept_count'] for r in successful)

        print(f"\n过滤统计:")
        print(f"  总文字区块: {total_boxes}")
        print(f"  已过滤: {total_filtered} ({total_filtered/total_boxes*100:.1f}%)")
        print(f"  保留（手写文字）: {total_kept} ({total_kept/total_boxes*100:.1f}%)")

        # 汇总过滤原因
        reason_totals = {
            "keyword": 0,
            "x_coordinate": 0,
            "pure_number": 0,
            "high_confidence_short": 0
        }
        for r in successful:
            for key in reason_totals:
                reason_totals[key] += r['filter_reasons'][key]

        print(f"\n过滤原因分布:")
        print(f"  关键词黑名单: {reason_totals['keyword']}")
        print(f"  X坐标过滤: {reason_totals['x_coordinate']}")
        print(f"  纯数字: {reason_totals['pure_number']}")
        print(f"  高信心度短文本: {reason_totals['high_confidence_short']}")

    if failed:
        print(f"\n失败的文件:")
        for r in failed:
            print(f"  - {r['pdf_name']}: {r['error']}")

    # 保存处理报告
    report_path = output_dir / "processing_report.json"
    report_data = {
        'total_pdfs': total_pdfs,
        'successful': len(successful),
        'failed': len(failed),
        'total_time': batch_time,
        'avg_time_per_pdf': batch_time / total_pdfs,
        'filter_config': FILTER_CONFIG,
        'ocr_config': OPTIMAL_CONFIG,
        'results': results
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 处理报告已保存: {report_path}")
    print(f"✅ TXT 文件保存在: {output_dir / 'txt'}")
    print(f"✅ JSON 文件保存在: {output_dir / 'json'}")
    print(f"✅ 可视化图片保存在:")
    print(f"    - {output_dir / 'vis_all'} (所有检测框)")
    print(f"    - {output_dir / 'vis_filtered'} (过滤对比)")
    print(f"    - {output_dir / 'vis_final'} (仅手写文字)")

    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
