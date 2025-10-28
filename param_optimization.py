#!/usr/bin/env python3
"""
PaddleOCR 參數優化測試腳本
測試三種配置以找出最適合學測作文稿紙的參數組合
"""
import sys
import time
import json
from pathlib import Path
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# 三種測試配置
CONFIGS = {
    "baseline_1200": {
        "name": "基準配置 (1200px)",
        "description": "基準解析度，針對稿紙格式優化",
        "params": {
            "text_det_limit_side_len": 1200,
            "text_det_thresh": 0.2,
            "text_det_box_thresh": 0.5,
            "text_det_unclip_ratio": 1.6,
            "text_rec_score_thresh": 0.35,
            "text_recognition_batch_size": 8
        }
    },
    "high_res_1440": {
        "name": "高解析度配置 (1440px)",
        "description": "更高解析度，更嚴格閾值，減少碎片化檢測",
        "params": {
            "text_det_limit_side_len": 1440,
            "text_det_thresh": 0.25,  # 稍高，減少誤檢
            "text_det_box_thresh": 0.6,  # 更高，只保留高信心度檢測
            "text_det_unclip_ratio": 1.8,  # 更大，擴展檢測框避免切字
            "text_rec_score_thresh": 0.4,  # 更高，只保留高信心度識別
            "text_recognition_batch_size": 6  # 稍低，因解析度更高
        }
    },
    "fast_mobile": {
        "name": "快速配置 (Mobile 模型)",
        "description": "使用 Mobile 模型加速處理",
        "params": {
            "text_det_limit_side_len": 960,
            "text_det_thresh": 0.3,
            "text_det_box_thresh": 0.6,
            "text_det_unclip_ratio": 1.6,
            "text_rec_score_thresh": 0.5,
            "text_recognition_batch_size": 16
        },
        "model_names": {
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_mobile_rec"
        }
    }
}

def test_config(config_key, config_data, pdf_path, output_dir):
    """測試單一配置"""
    print("\n" + "=" * 70)
    print(f"測試配置: {config_data['name']}")
    print(f"說明: {config_data['description']}")
    print("=" * 70)

    try:
        # 初始化 OCR
        print("\n[1/3] 初始化 OCR...")
        start_time = time.time()

        # 基本參數
        ocr_kwargs = {
            "ocr_version": "PP-OCRv5",
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "device": "cpu",
            **config_data['params']
        }

        # 如果有自定義模型名稱（如 mobile 模型）
        if "model_names" in config_data:
            ocr_kwargs.update(config_data['model_names'])
        else:
            # 使用 server 模型
            ocr_kwargs.update({
                "text_detection_model_name": "PP-OCRv5_server_det",
                "text_recognition_model_name": "PP-OCRv5_server_rec"
            })

        ocr = PaddleOCR(**ocr_kwargs)
        init_time = time.time() - start_time
        print(f"   ✅ 初始化完成 (耗時: {init_time:.2f} 秒)")

        # 轉換 PDF
        print("\n[2/3] 轉換 PDF...")
        images = convert_from_path(str(pdf_path), dpi=300, first_page=1, last_page=1)
        print(f"   ✅ 轉換完成")

        # 保存臨時圖片
        temp_img = output_dir / "temp_image.jpg"
        images[0].save(temp_img, "JPEG")

        # OCR 處理
        print("\n[3/3] 執行 OCR...")
        start_time = time.time()
        result = ocr.predict(input=str(temp_img))
        ocr_time = time.time() - start_time

        # 分析結果
        first_result = result[0]
        rec_texts = first_result.get('rec_texts', [])
        rec_scores = first_result.get('rec_scores', [])
        dt_polys = first_result.get('dt_polys', [])

        num_lines = len(rec_texts)
        avg_conf = sum(rec_scores) / len(rec_scores) if rec_scores else 0

        print(f"\n   ✅ OCR 完成！")
        print(f"      耗時: {ocr_time:.2f} 秒")
        print(f"      檢測區塊: {num_lines} 個")
        print(f"      平均信心度: {avg_conf:.3f}")
        if rec_scores:
            print(f"      最低信心度: {min(rec_scores):.3f}")
            print(f"      最高信心度: {max(rec_scores):.3f}")

        # 計算統計數據
        # 字元數量（估計每行平均字數）
        total_chars = sum(len(text) for text in rec_texts)
        avg_chars_per_line = total_chars / num_lines if num_lines > 0 else 0

        # 信心度分佈
        high_conf_count = sum(1 for s in rec_scores if s >= 0.9)
        mid_conf_count = sum(1 for s in rec_scores if 0.7 <= s < 0.9)
        low_conf_count = sum(1 for s in rec_scores if s < 0.7)

        print(f"\n   統計數據:")
        print(f"      總字元數: {total_chars}")
        print(f"      平均每行字數: {avg_chars_per_line:.1f}")
        print(f"      高信心度區塊 (≥0.9): {high_conf_count} ({high_conf_count/num_lines*100:.1f}%)")
        print(f"      中信心度區塊 (0.7-0.9): {mid_conf_count} ({mid_conf_count/num_lines*100:.1f}%)")
        print(f"      低信心度區塊 (<0.7): {low_conf_count} ({low_conf_count/num_lines*100:.1f}%)")

        # 保存結果
        config_output_dir = output_dir / config_key
        config_output_dir.mkdir(exist_ok=True)

        # 保存 TXT
        txt_path = config_output_dir / f"{pdf_path.stem}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for text in rec_texts:
                f.write(text + '\n')

        # 保存 JSON
        json_path = config_output_dir / f"{pdf_path.stem}.json"
        json_data = {
            'config': config_data,
            'performance': {
                'init_time': init_time,
                'ocr_time': ocr_time,
                'total_time': init_time + ocr_time
            },
            'statistics': {
                'num_lines': num_lines,
                'total_chars': total_chars,
                'avg_chars_per_line': avg_chars_per_line,
                'avg_confidence': avg_conf,
                'min_confidence': min(rec_scores) if rec_scores else 0,
                'max_confidence': max(rec_scores) if rec_scores else 0,
                'high_conf_count': high_conf_count,
                'mid_conf_count': mid_conf_count,
                'low_conf_count': low_conf_count
            },
            'lines': [
                {
                    'text': rec_texts[i],
                    'score': float(rec_scores[i]) if i < len(rec_scores) else 0,
                    'bbox': dt_polys[i].tolist() if i < len(dt_polys) else None
                }
                for i in range(len(rec_texts))
            ]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"\n   ✅ 結果已保存:")
        print(f"      TXT: {txt_path}")
        print(f"      JSON: {json_path}")

        return {
            'success': True,
            'config_key': config_key,
            'config_name': config_data['name'],
            'num_lines': num_lines,
            'avg_confidence': avg_conf,
            'ocr_time': ocr_time,
            'avg_chars_per_line': avg_chars_per_line
        }

    except Exception as e:
        print(f"\n   ❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'config_key': config_key,
            'error': str(e)
        }

def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("PaddleOCR 參數優化測試")
    print("測試三種配置以找出最適合學測作文稿紙的參數")
    print("=" * 70)

    # 檢查數據目錄
    data_dir = Path("/app/data")
    if not data_dir.exists():
        print(f"\n❌ 數據目錄不存在: {data_dir}")
        return 1

    # 獲取測試 PDF
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ 沒有找到 PDF 檔案")
        return 1

    # 選擇測試檔案（使用第一個）
    test_pdf = pdf_files[0]
    print(f"\n測試檔案: {test_pdf.name}")

    # 輸出目錄
    output_dir = Path("/app/output/param_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 測試所有配置
    results = []
    for config_key, config_data in CONFIGS.items():
        result = test_config(config_key, config_data, test_pdf, output_dir)
        results.append(result)

    # 生成比較報告
    print("\n" + "=" * 70)
    print("測試結果比較")
    print("=" * 70)

    print(f"\n{'配置':<20} {'區塊數':<10} {'平均字/行':<12} {'信心度':<10} {'耗時(秒)':<10}")
    print("-" * 70)

    for r in results:
        if r['success']:
            print(f"{r['config_name']:<20} {r['num_lines']:<10} "
                  f"{r['avg_chars_per_line']:<12.1f} {r['avg_confidence']:<10.3f} "
                  f"{r['ocr_time']:<10.2f}")
        else:
            print(f"{r['config_key']:<20} {'失敗':<10}")

    # 保存比較報告
    report_path = output_dir / "comparison_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 比較報告已保存: {report_path}")
    print("\n建議:")
    print("  - 如果需要最高準確度，選擇高解析度配置")
    print("  - 如果需要平衡速度和準確度，選擇基準配置")
    print("  - 如果需要最快速度，選擇快速配置")
    print("  - 根據「平均字/行」判斷檢測碎片化程度（越高越好）")

    return 0

if __name__ == "__main__":
    sys.exit(main())
