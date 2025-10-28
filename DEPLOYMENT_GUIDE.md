# PaddleOCR 3.3.0 部署指南（macOS arm64 環境）

## ⚠️ 問題診斷

**測試結果**：PaddleOCR 3.3.0 在 macOS arm64 (Apple Silicon) 上遇到段錯誤（Segmentation Fault, exit code 139）

**原因**：PaddlePaddle Inference 模組在 Apple Silicon 上的相容性問題（與您之前遇到的 v2.x 問題相同）

---

## 🎯 推薦解決方案（按優先級排序）

### 方案 A：Docker 容器部署 ⭐⭐⭐⭐⭐ （強烈推薦）

**優點**：
- ✅ Linux x86_64 環境，完整相容性
- ✅ 一次配置，永久使用
- ✅ 隔離環境，不影響 macOS 系統
- ✅ 可在任何機器上重現

**安裝 Docker**：
```bash
# 方法 1: 使用 Homebrew (推薦)
brew install --cask docker

# 方法 2: 下載 Docker Desktop for Mac
# https://www.docker.com/products/docker-desktop
```

**使用步驟**：

1. 構建 Docker 鏡像：
```bash
cd /Users/morrisliao/Desktop/git-repo/paddleOCR/PaddleOCR
docker build -f Dockerfile.essay_ocr -t paddleocr-essay:latest .
```

2. 運行 OCR（單個 PDF）：
```bash
docker run --rm \
  -v "$PWD/essay_data":/app/data:ro \
  -v "$PWD/output":/app/output \
  paddleocr-essay:latest \
  python3 -c "
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import json

# 初始化 OCR
ocr = PaddleOCR(
    ocr_version='PP-OCRv5',
    use_doc_orientation_classify=True,
    use_textline_orientation=True,
    device='cpu'
)

# 處理 PDF
images = convert_from_path('/app/data/107 學測原卷1-1.pdf', dpi=300)
result = ocr.predict(input=images[0])
print(result[0].boxes[:3])  # 顯示前 3 行
"
```

3. 批次處理（稍後會創建專用腳本）

**預估時間**：
- Docker 安裝：5-10 分鐘
- 鏡像構建：5-10 分鐘
- 總計：15-20 分鐘

---

### 方案 B：AI Studio 雲端環境 ⭐⭐⭐⭐

**優點**：
- ✅ 官方環境，100% 相容
- ✅ 免費 GPU 資源（處理更快）
- ✅ 無需本地安裝
- ✅ 已預裝 PaddleOCR

**使用步驟**：

1. 註冊/登入 AI Studio：
   - 網址：https://aistudio.baidu.com/
   - 使用百度帳號登入

2. 創建 Notebook 專案：
   - 選擇「Notebook」
   - 環境選擇：Python 3.10 + PaddlePaddle 3.2.0
   - GPU 配置：CPU 或 V100（免費額度）

3. 上傳作文 PDF 檔案：
   - 上傳 `essay_data/` 資料夾
   - 或使用 AI Studio 的「數據集」功能

4. 運行 OCR 代碼（與本地相同）

5. 下載結果：
   - JSON、TXT 檔案
   - 打包下載到本地

**預估時間**：
- 帳號註冊：5 分鐘
- 環境配置：5 分鐘
- 上傳資料：視檔案大小（約 5-10 分鐘）
- 總計：15-20 分鐘

---

### 方案 C：使用 ONNX Runtime 後端 ⭐⭐⭐

**實驗性方案**，可能避開 PaddlePaddle Inference 的相容性問題

**步驟**：

1. 安裝 ONNX Runtime：
```bash
pip install onnxruntime --break-system-packages
```

2. 導出模型為 ONNX 格式（需要在 Linux 環境或方案 A/B 中執行）
3. 使用 ONNX 模型進行推理

**注意**：此方案需要額外配置，且不保證完全支援 PP-OCRv5 的所有功能

---

### 方案 D：短期過渡方案（不推薦）⭐

**僅用於快速驗證想法**，不適合生產環境

使用已安裝的 PaddleOCR 2.10.0（PP-OCRv4）：

**限制**：
- ❌ 繁體中文識別準確率較低（比 PP-OCRv5 低約 30-40%）
- ❌ 不支援多種文字類型混合
- ❌ 無法滿足您的「必須使用最新版本」的要求

**僅建議用於**：
- 快速測試工作流程
- 評估參數設置
- 確認資料格式

---

## 📊 方案比較

| 方案 | 相容性 | 準確率 | 配置難度 | 處理速度 | 推薦度 |
|------|--------|--------|----------|----------|--------|
| A. Docker | ✅✅✅✅✅ | PP-OCRv5 | 中 | 中-快 | ⭐⭐⭐⭐⭐ |
| B. AI Studio | ✅✅✅✅✅ | PP-OCRv5 | 低 | 快 (GPU) | ⭐⭐⭐⭐ |
| C. ONNX | ✅✅✅ | PP-OCRv5 | 高 | 中 | ⭐⭐⭐ |
| D. v2.10.0 | ✅✅✅✅✅ | PP-OCRv4 | 無 | 快 | ⭐ |

---

## 💡 建議決策流程

**如果您有 30 分鐘時間**：
→ 選擇**方案 A（Docker）**
  - 一次配置，長期使用
  - 完整控制權
  - 適合開發迭代

**如果您想立即測試**：
→ 選擇**方案 B（AI Studio）**
  - 最快上手
  - 免費 GPU
  - 適合單次處理或驗證效果

**如果您是 Docker/ONNX 專家**：
→ 可嘗試**方案 C（ONNX）**
  - 技術挑戰性高
  - 可能獲得最佳性能

---

## 🚀 下一步行動

**請告訴我您選擇哪個方案，我將協助您完成部署**

```bash
# 方案 A: 安裝 Docker
brew install --cask docker

# 方案 B: 開啟 AI Studio
open https://aistudio.baidu.com/

# 方案 D: 使用現有環境（臨時測試）
python3 -c "from paddleocr import PaddleOCR; print(PaddleOCR.__version__)"  # 2.10.0
```

---

## 📝 備註

- **Docker 是最推薦的方案**，因為：
  1. 與您的開發機器隔離
  2. 可以在任何環境重現
  3. 適合後續的模型訓練階段

- **AI Studio 適合快速驗證**，但：
  1. 需要上傳/下載資料
  2. 依賴網路連接
  3. 免費額度有限

- 根據您的需求（「無論如何都要用新版本」），**不推薦方案 D**
