#!/usr/bin/env python3
"""
Docker 容器内的 PaddleOCR 3.3.0 测试脚本
此脚本将在 Linux x86_64 环境中运行（完整相容性）
"""
import sys
import time
from pathlib import Path

def test_docker_ocr():
    """在 Docker 容器内测试 PP-OCRv5"""
    print("=" * 70)
    print("PaddleOCR 3.3.0 (PP-OCRv5) Docker 環境測試")
    print("Linux x86_64 環境 - 完整相容性")
    print("=" * 70)

    try:
        # 导入库
        print("\n[1/4] 導入庫...")
        from paddleocr import PaddleOCR
        from pdf2image import convert_from_path
        print("✅ 導入成功")

        # 初始化 OCR（基準配置）
        print("\n[2/4] 初始化 PP-OCRv5（基準配置）...")
        start_time = time.time()
        ocr = PaddleOCR(
            ocr_version="PP-OCRv5",
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="cpu",

            # Detection 參數（針對稿紙優化）
            text_det_limit_side_len=1200,
            text_det_thresh=0.2,
            text_det_box_thresh=0.5,
            text_det_unclip_ratio=1.6,

            # Recognition 參數
            text_rec_score_thresh=0.35,
            text_recognition_batch_size=8
        )
        init_time = time.time() - start_time
        print(f"✅ 初始化成功 (耗時: {init_time:.2f} 秒)")

        # 检查数据目录
        print("\n[3/4] 檢查資料目錄...")
        data_dir = Path("/app/data")
        if not data_dir.exists():
            print("❌ /app/data 目錄不存在")
            print("   請確保使用 -v 參數掛載了 essay_data 目錄")
            return False

        pdf_files = list(data_dir.glob("*.pdf"))
        print(f"✅ 找到 {len(pdf_files)} 個 PDF 檔案")

        if not pdf_files:
            print("⚠️  沒有 PDF 檔案可供測試")
            return True

        # 测试第一个 PDF
        print("\n[4/4] 測試 OCR 處理...")
        first_pdf = pdf_files[0]
        print(f"   PDF: {first_pdf.name}")

        # 轉換 PDF 為圖片
        print("   轉換 PDF...")
        images = convert_from_path(str(first_pdf), dpi=300, first_page=1, last_page=1)
        print(f"   ✅ 轉換完成，共 {len(images)} 頁")

        # 保存臨時圖片（PaddleOCR 不支持直接使用 PIL Image）
        temp_img_path = "/tmp/test_essay.jpg"
        images[0].save(temp_img_path, "JPEG")

        # OCR 處理
        print("   執行 OCR...")
        start_time = time.time()
        result = ocr.predict(input=temp_img_path)
        ocr_time = time.time() - start_time

        # 顯示結果
        if result and len(result) > 0:
            print(f"   ✅ OCR 完成 (耗時: {ocr_time:.2f} 秒)")

            # 調試：打印結果類型和屬性
            print(f"\n   調試信息：")
            print(f"   - result 類型: {type(result)}")
            print(f"   - result[0] 類型: {type(result[0])}")
            print(f"   - result[0] 屬性: {dir(result[0])}")

            # OCRResult 對象是字典式訪問
            first_result = result[0]

            # 獲取識別結果（字典鍵訪問 - 注意是複數形式）
            rec_texts = first_result.get('rec_texts', [])
            rec_scores = first_result.get('rec_scores', [])
            dt_polys = first_result.get('dt_polys', [])

            if rec_texts and len(rec_texts) > 0:
                num_lines = len(rec_texts)
                print(f"\n   ✅ 檢測到 {num_lines} 個文字區塊")

                print("\n   前 10 行識別結果：")
                print("   " + "-" * 66)
                for i in range(min(10, len(rec_texts))):
                    text = rec_texts[i]
                    score = rec_scores[i] if i < len(rec_scores) else 0
                    # 截斷過長的文字
                    display_text = text if len(text) <= 40 else text[:37] + "..."
                    print(f"   {i+1:2d}. {display_text:<40} (信心度: {score:.3f})")
                print("   " + "-" * 66)

                # 計算平均信心度
                if rec_scores:
                    avg_conf = sum(rec_scores) / len(rec_scores)
                    print(f"\n   平均信心度: {avg_conf:.3f}")
                    print(f"   最低信心度: {min(rec_scores):.3f}")
                    print(f"   最高信心度: {max(rec_scores):.3f}")

                # 保存結果
                output_dir = Path("/app/output")
                output_dir.mkdir(exist_ok=True)

                # 保存 TXT（純文字）
                txt_path = output_dir / f"{first_pdf.stem}_test.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for text in rec_texts:
                        f.write(text + '\n')
                print(f"\n   ✅ 純文字已保存到: {txt_path}")

                # 保存 JSON（包含信心度和座標）
                import json
                json_path = output_dir / f"{first_pdf.stem}_test.json"
                json_data = {
                    'total_lines': len(rec_texts),
                    'avg_confidence': avg_conf if rec_scores else 0,
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
                print(f"   ✅ JSON 已保存到: {json_path}")

                return True
            else:
                print("   ⚠️  未檢測到文字")
                print(f"   rec_texts 長度: {len(rec_texts)}")
                return False
        else:
            print("   ❌ OCR 執行失敗")
            return False

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print()
    success = test_docker_ocr()
    print("\n" + "=" * 70)
    if success:
        print("✅ Docker 環境測試成功！")
        print("   PP-OCRv5 可在 Docker 容器中正常運行")
        print("\n下一步：")
        print("   1. 調整參數配置以達最佳效果")
        print("   2. 批次處理全部 41 個 PDF")
    else:
        print("❌ Docker 環境測試失敗")
        print("   請檢查錯誤訊息並重試")
    print("=" * 70)

    sys.exit(0 if success else 1)
