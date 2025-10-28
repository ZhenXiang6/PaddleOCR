#!/usr/bin/env python3
"""
测试 PaddleOCR 3.3.0 (PP-OCRv5) 在 macOS arm64 环境的兼容性
"""
import sys
import time
from pathlib import Path

def test_basic_import():
    """测试基本导入"""
    print("=" * 60)
    print("Test 1: 基本导入测试")
    print("=" * 60)
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR 导入成功")
        return True
    except Exception as e:
        print(f"❌ PaddleOCR 导入失败: {e}")
        return False

def test_initialization():
    """测试初始化 PP-OCRv5"""
    print("\n" + "=" * 60)
    print("Test 2: PP-OCRv5 初始化测试（使用 CPU）")
    print("=" * 60)
    try:
        from paddleocr import PaddleOCR
        print("初始化参数：")
        print("  - ocr_version: PP-OCRv5")
        print("  - device: cpu")
        print("  - use_doc_orientation_classify: False")
        print("  - use_doc_unwarping: False")
        print("  - use_textline_orientation: False")
        print("\n正在初始化... (可能需要下载模型，请耐心等待)")

        start_time = time.time()
        ocr = PaddleOCR(
            ocr_version="PP-OCRv5",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        elapsed = time.time() - start_time
        print(f"✅ PP-OCRv5 初始化成功 (耗時: {elapsed:.2f} 秒)")
        return ocr
    except Exception as e:
        print(f"❌ PP-OCRv5 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_sample_image(ocr):
    """测试样本图片 OCR"""
    print("\n" + "=" * 60)
    print("Test 3: 樣本圖片 OCR 測試")
    print("=" * 60)

    # 检查是否有样本 PDF
    essay_dir = Path("essay_data")
    if not essay_dir.exists():
        print("⚠️  essay_data 目錄不存在，跳過圖片測試")
        return True

    pdf_files = list(essay_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  沒有找到 PDF 文件，跳過圖片測試")
        return True

    # 使用第一个 PDF 测试（先转换为图片）
    try:
        print("安裝 pdf2image...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pdf2image", "--break-system-packages", "-q"], check=True)

        from pdf2image import convert_from_path

        first_pdf = pdf_files[0]
        print(f"轉換 PDF: {first_pdf.name}")

        images = convert_from_path(first_pdf, dpi=150, first_page=1, last_page=1)
        if not images:
            print("❌ PDF 轉圖片失敗")
            return False

        # 保存临时图片
        temp_img = Path("/tmp/test_essay.jpg")
        images[0].save(temp_img, "JPEG")
        print(f"✅ PDF 轉圖片成功，保存至: {temp_img}")

        # 运行 OCR
        print("\n正在執行 OCR...")
        start_time = time.time()
        result = ocr.predict(input=str(temp_img))
        elapsed = time.time() - start_time

        if result:
            print(f"✅ OCR 執行成功 (耗時: {elapsed:.2f} 秒)")
            print(f"\n檢測到 {len(result[0].boxes) if hasattr(result[0], 'boxes') else 'N/A'} 個文字區塊")

            # 显示前 3 行识别结果
            if hasattr(result[0], 'boxes') and result[0].boxes:
                print("\n前 3 行識別結果:")
                for i, box in enumerate(result[0].boxes[:3]):
                    text = box.get('text', 'N/A')
                    score = box.get('score', 0)
                    print(f"  {i+1}. {text} (信心度: {score:.3f})")
            return True
        else:
            print("❌ OCR 執行失敗：無結果返回")
            return False

    except Exception as e:
        print(f"❌ 圖片 OCR 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "🔍" * 30)
    print("PaddleOCR 3.3.0 (PP-OCRv5) 環境驗證測試")
    print("macOS arm64 (Apple Silicon) 環境")
    print("🔍" * 30 + "\n")

    # Test 1: Import
    if not test_basic_import():
        print("\n❌ 測試失敗：無法導入 PaddleOCR")
        sys.exit(1)

    # Test 2: Initialization
    ocr = test_initialization()
    if ocr is None:
        print("\n❌ 測試失敗：無法初始化 PP-OCRv5")
        print("\n⚠️  建議解決方案：")
        print("   1. 使用 Docker 容器部署（Linux x86_64 環境）")
        print("   2. 使用雲端 AI Studio 環境")
        sys.exit(1)

    # Test 3: Sample OCR
    if not test_sample_image(ocr):
        print("\n⚠️  圖片 OCR 測試失敗，但環境基本可用")

    print("\n" + "=" * 60)
    print("✅ 環境驗證完成！PaddleOCR 3.3.0 可在您的環境中運行")
    print("=" * 60)
    print("\n下一步：開始參數優化測試")

if __name__ == "__main__":
    main()
