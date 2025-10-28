#!/usr/bin/env python3
"""
批量處理學測作文 PDF 的 OCR 腳本
使用優化後的 Mobile 模型配置（平衡速度與準確度）
"""
import sys
import time
import json
from pathlib import Path
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm

# 最佳配置：Mobile 模型（速度快、碎片化少）
OPTIMAL_CONFIG = {
    "ocr_version": "PP-OCRv5",
    "text_detection_model_name": "PP-OCRv5_mobile_det",
    "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    "use_doc_orientation_classify": True,
    "use_doc_unwarping": False,
    "use_textline_orientation": True,
    "device": "cpu",

    # Detection 參數（快速配置）
    "text_det_limit_side_len": 960,
    "text_det_thresh": 0.3,
    "text_det_box_thresh": 0.6,
    "text_det_unclip_ratio": 1.6,

    # Recognition 參數
    "text_rec_score_thresh": 0.5,
    "text_recognition_batch_size": 16
}

def process_single_pdf(ocr, pdf_path, output_dir, progress_bar=None):
    """處理單個 PDF 檔案"""
    try:
        # 更新進度條描述
        if progress_bar:
            progress_bar.set_description(f"處理 {pdf_path.name[:30]}")

        # 轉換 PDF 為圖片（每個 PDF 只處理第一頁）
        images = convert_from_path(str(pdf_path), dpi=300, first_page=1, last_page=1)

        # 保存臨時圖片
        temp_img = Path("/tmp") / f"{pdf_path.stem}.jpg"
        images[0].save(temp_img, "JPEG")

        # OCR 處理
        start_time = time.time()
        result = ocr.predict(input=str(temp_img))
        ocr_time = time.time() - start_time

        # 提取結果
        first_result = result[0]
        rec_texts = first_result.get('rec_texts', [])
        rec_scores = first_result.get('rec_scores', [])
        dt_polys = first_result.get('dt_polys', [])

        # 計算統計數據
        num_lines = len(rec_texts)
        total_chars = sum(len(text) for text in rec_texts)
        avg_conf = sum(rec_scores) / len(rec_scores) if rec_scores else 0

        # 保存 TXT（純文字）
        txt_path = output_dir / "txt" / f"{pdf_path.stem}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            for text in rec_texts:
                f.write(text + '\n')

        # 保存 JSON（包含詳細信息）
        json_path = output_dir / "json" / f"{pdf_path.stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = {
            'pdf_name': pdf_path.name,
            'ocr_time': ocr_time,
            'statistics': {
                'num_lines': num_lines,
                'total_chars': total_chars,
                'avg_confidence': avg_conf,
                'min_confidence': min(rec_scores) if rec_scores else 0,
                'max_confidence': max(rec_scores) if rec_scores else 0
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

        # 刪除臨時檔案
        temp_img.unlink()

        return {
            'success': True,
            'pdf_name': pdf_path.name,
            'num_lines': num_lines,
            'total_chars': total_chars,
            'avg_confidence': avg_conf,
            'ocr_time': ocr_time
        }

    except Exception as e:
        return {
            'success': False,
            'pdf_name': pdf_path.name,
            'error': str(e)
        }

def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("學測作文 PDF 批量 OCR 處理")
    print("使用 PP-OCRv5 Mobile 模型（優化配置）")
    print("=" * 70)

    # 檢查數據目錄
    data_dir = Path("/app/data")
    if not data_dir.exists():
        print(f"\n❌ 數據目錄不存在: {data_dir}")
        return 1

    # 獲取所有 PDF 檔案
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ 沒有找到 PDF 檔案")
        return 1

    total_pdfs = len(pdf_files)
    print(f"\n找到 {total_pdfs} 個 PDF 檔案")

    # 輸出目錄
    output_dir = Path("/app/output/batch_ocr_mobile")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 OCR
    print("\n初始化 PP-OCRv5 Mobile 模型...")
    start_init = time.time()
    ocr = PaddleOCR(**OPTIMAL_CONFIG)
    init_time = time.time() - start_init
    print(f"✅ 初始化完成 (耗時: {init_time:.2f} 秒)")

    # 批量處理
    print(f"\n開始批量處理 {total_pdfs} 個 PDF...")
    print("-" * 70)

    results = []
    start_batch = time.time()

    # 使用 tqdm 顯示進度條
    with tqdm(total=total_pdfs, desc="總進度", unit="PDF") as pbar:
        for pdf_path in pdf_files:
            result = process_single_pdf(ocr, pdf_path, output_dir, pbar)
            results.append(result)
            pbar.update(1)

    batch_time = time.time() - start_batch

    # 統計結果
    print("\n" + "=" * 70)
    print("批量處理完成！")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n總處理數: {total_pdfs}")
    print(f"成功: {len(successful)}")
    print(f"失敗: {len(failed)}")
    print(f"總耗時: {batch_time:.2f} 秒")
    print(f"平均每個 PDF: {batch_time/total_pdfs:.2f} 秒")

    if successful:
        total_lines = sum(r['num_lines'] for r in successful)
        total_chars = sum(r['total_chars'] for r in successful)
        avg_conf = sum(r['avg_confidence'] for r in successful) / len(successful)

        print(f"\n統計數據:")
        print(f"  總文字區塊: {total_lines}")
        print(f"  總字元數: {total_chars}")
        print(f"  平均信心度: {avg_conf:.3f}")
        print(f"  平均每個 PDF 區塊數: {total_lines/len(successful):.1f}")

    if failed:
        print(f"\n失敗的檔案:")
        for r in failed:
            print(f"  - {r['pdf_name']}: {r['error']}")

    # 保存處理報告
    report_path = output_dir / "processing_report.json"
    report_data = {
        'total_pdfs': total_pdfs,
        'successful': len(successful),
        'failed': len(failed),
        'total_time': batch_time,
        'avg_time_per_pdf': batch_time / total_pdfs,
        'config': OPTIMAL_CONFIG,
        'results': results
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 處理報告已保存: {report_path}")
    print(f"✅ TXT 檔案保存在: {output_dir / 'txt'}")
    print(f"✅ JSON 檔案保存在: {output_dir / 'json'}")

    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
