# PaddleOCR Fine-tune 快速開始

## 🚀 30 分鐘完整流程

本指南幫助你在 30-60 分鐘內完成第一次 Fine-tune 測試。

---

## 前置準備

### 1. 環境檢查

```bash
# 檢查 Docker 是否安裝
docker --version

# 檢查 Docker 鏡像
docker images | grep paddleocr-essay

# 如果鏡像不存在，構建它
docker build --platform linux/amd64 -f Dockerfile.essay_ocr -t paddleocr-essay:latest .
```

### 2. 安裝 PPOCRLabel

```bash
# 在 macOS 本地安裝（不在 Docker 內）
pip install PPOCRLabel
```

---

## Detection Fine-tune（推薦先做）

### 階段 1：資料標註（10-20 分鐘）

#### 1.1 準備圖片

```bash
# 創建圖片目錄
mkdir -p essay_images

# 將作文圖片放入此目錄
# 例如：essay_001.jpg, essay_002.jpg, ...
```

#### 1.2 啟動 PPOCRLabel

```bash
PPOCRLabel --lang ch
```

#### 1.3 快速標註流程

1. **載入圖片**：
   - 點擊「打開目錄」→ 選擇 `essay_images/`

2. **自動標註**：
   - 點擊「自動標註」（或按 `Ctrl+A`）
   - 等待 1-2 分鐘

3. **快速修正** （僅修正明顯錯誤）：
   - 按 `D` 鍵逐張查看
   - 發現碎片化：刪除小框 (`Delete`) → 重繪大框 (`W`)
   - 文字標註可簡化或使用 "###"

4. **完成**：
   - 標註會自動保存到 `essay_images/Label.txt`
   - 關閉 PPOCRLabel

**時間估計**：
- 50 張圖片：約 15 分鐘
- 100 張圖片：約 30 分鐘

### 階段 2：資料準備（1-2 分鐘）

```bash
# 準備 Detection 訓練資料
python3 prepare_det_data.py \
  --label_file essay_images/Label.txt \
  --image_dir essay_images \
  --output_dir train_data/det \
  --train_ratio 0.8

# 檢查輸出
ls train_data/det/
# 應該看到: images/, train.txt, val.txt
```

### 階段 3：配置訓練（2-3 分鐘）

#### 3.1 複製配置文件

```bash
cp configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml configs/det/custom_det.yml
```

#### 3.2 修改關鍵配置

編輯 `configs/det/custom_det.yml`：

```yaml
Global:
  epoch_num: 100  # 測試用減少 epoch（原 500）
  save_model_dir: ./output/custom_det/

Train:
  dataset:
    data_dir: ./train_data/det/images/
    label_file_list:
      - ./train_data/det/train.txt
  loader:
    batch_size_per_card: 4  # CPU 使用小 batch

Eval:
  dataset:
    data_dir: ./train_data/det/images/
    label_file_list:
      - ./train_data/det/val.txt
```

### 階段 4：開始訓練（5-10 分鐘）

```bash
# 賦予執行權限
chmod +x run_train_docker.sh train_detection.sh

# 開始訓練
./run_train_docker.sh detection
```

**預期輸出**：
```
======================================================================
Detection 模型訓練
======================================================================

配置信息:
  配置文件: configs/det/custom_det.yml
  預訓練模型: ./PP-OCRv5_mobile_det_train/best_accuracy
  設備: cpu

正在下載預訓練模型...
✅ 預訓練模型下載完成

開始訓練...
[Train] epoch: 1, iter: 10, loss: 2.341, ...
[Eval] hmean: 0.7812, precision: 0.8234, recall: 0.7432
...
```

**訓練時間**：
- 50 張圖片，100 epochs，CPU：約 3-5 小時
- 建議：先用 10 epochs 快速驗證（約 30 分鐘）

### 階段 5：導出模型（1 分鐘）

```bash
# 在 Docker 容器內執行
docker run --rm --platform linux/amd64 \
  -v $(pwd)/output:/app/paddleocr/output \
  -v $(pwd)/configs:/app/paddleocr/configs \
  paddleocr-essay:latest \
  bash -c "cd /app/paddleocr && python3 tools/export_model.py \
    -c configs/det/custom_det.yml \
    -o Global.pretrained_model=./output/custom_det/best_accuracy \
       Global.save_inference_dir=./output/inference/custom_det_model"
```

### 階段 6：測試效果（2-3 分鐘）

```bash
# 測試 fine-tuned 模型
python3 test_finetuned_model.py \
  --test_images essay_data/crop/ \
  --det_model ./output/inference/custom_det_model \
  --output_dir output/det_test \
  --compare_pretrained

# 查看結果
ls output/det_test/
# vis_finetuned/  - Fine-tuned 模型可視化
# vis_pretrained/ - 預訓練模型可視化（對比）
# comparison_report.json - 對比報告
```

**查看對比**：
```bash
# macOS 打開可視化圖片
open output/det_test/vis_finetuned/*.jpg
open output/det_test/vis_pretrained/*.jpg

# 查看對比報告
cat output/det_test/comparison_report.json | jq
```

---

## Recognition Fine-tune（可選）

### 階段 1：資料準備（5-10 分鐘）

#### 方法 1：轉換已有的單字資料

```bash
# 假設你的單字資料格式：char_images/水_001.jpg, 能_001.jpg, ...
python3 convert_rec_data.py \
  --input /path/to/char_images \
  --format filename \
  --output_dir train_data/rec \
  --train_ratio 0.9
```

#### 方法 2：從 Detection 標註裁剪

```bash
# 從 Detection 標註裁剪文字行
python3 crop_essay_lines.py \
  --label_file train_data/det/train.txt \
  --image_dir train_data/det/images \
  --output_dir train_data/rec/essay_lines
```

#### 合併資料

```bash
# 合併單字資料和作文行資料
cat train_data/rec/train.txt train_data/rec/essay_lines/labels.txt > train_data/rec/all_train.txt

# 分割訓練集和驗證集（90/10）
total_lines=$(wc -l < train_data/rec/all_train.txt)
train_lines=$((total_lines * 9 / 10))

head -n $train_lines train_data/rec/all_train.txt > train_data/rec/train.txt
tail -n +$((train_lines + 1)) train_data/rec/all_train.txt > train_data/rec/val.txt
```

### 階段 2：配置訓練（2-3 分鐘）

```bash
# 複製配置文件
cp configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml configs/rec/custom_rec.yml
```

編輯 `configs/rec/custom_rec.yml`：

```yaml
Global:
  epoch_num: 100  # 測試用減少 epoch
  save_model_dir: ./output/custom_rec/

Train:
  dataset:
    data_dir: ./train_data/rec/
    label_file_list:
      - ./train_data/rec/train.txt
  loader:
    batch_size_per_card: 64  # CPU 使用小 batch

Eval:
  dataset:
    data_dir: ./train_data/rec/
    label_file_list:
      - ./train_data/rec/val.txt
```

### 階段 3：開始訓練

```bash
chmod +x train_recognition.sh

./run_train_docker.sh recognition
```

### 階段 4：測試效果

```bash
# 導出模型
docker run --rm --platform linux/amd64 \
  -v $(pwd)/output:/app/paddleocr/output \
  -v $(pwd)/configs:/app/paddleocr/configs \
  paddleocr-essay:latest \
  bash -c "cd /app/paddleocr && python3 tools/export_model.py \
    -c configs/rec/custom_rec.yml \
    -o Global.pretrained_model=./output/custom_rec/best_accuracy \
       Global.save_inference_dir=./output/inference/custom_rec_model"

# 測試
python3 test_finetuned_model.py \
  --test_images essay_data/crop/ \
  --det_model ./output/inference/custom_det_model \
  --rec_model ./output/inference/custom_rec_model \
  --output_dir output/full_test \
  --compare_pretrained
```

---

## 使用 Fine-tuned 模型

### Python API

```python
from paddleocr import PaddleOCR

# 使用自定義模型
ocr = PaddleOCR(
    det_model_dir='./output/inference/custom_det_model',
    rec_model_dir='./output/inference/custom_rec_model',  # 可選
    use_doc_orientation_classify=True,
    device='cpu'
)

# 處理圖片
result = ocr.predict(input='test_essay.jpg')

# 獲取文字
for res in result:
    texts = res['rec_texts']
    for text in texts:
        print(text)
```

### 整合到現有系統

```python
# 替換現有的 filter_and_visualize.py 中的 OCR 初始化
ocr = PaddleOCR(
    **OPTIMAL_CONFIG,
    det_model_dir='./output/inference/custom_det_model',  # 添加這行
    rec_model_dir='./output/inference/custom_rec_model',  # 添加這行
)
```

---

## 常見問題

### Q: 訓練時間太長怎麼辦？

**解決方案**：
1. 減少 `epoch_num`（先用 10-20 epochs 測試）
2. 減少訓練數據（先用 50 張測試）
3. 使用更小的 `batch_size`

### Q: 訓練中斷了怎麼辦？

**恢復訓練**：
```bash
# 找到最後的 checkpoint
ls output/custom_det/

# 恢復訓練
RESUME_CHECKPOINT="./output/custom_det/iter_500" ./run_train_docker.sh detection
```

### Q: 效果不理想怎麼辦？

**檢查清單**：
1. 標註質量：檢查 `Label.txt` 是否有大量碎片化標註
2. 數據量：至少需要 100 張圖片
3. Epoch 數：可能需要更多訓練輪次
4. 查看訓練曲線：`loss` 是否下降

### Q: 如何判斷訓練效果？

**指標**：
1. **Detection**：
   - `hmean` (F1分數) 應該 > 0.80
   - `precision` 和 `recall` 應該平衡

2. **Recognition**：
   - `acc` (準確率) 應該 > 0.85
   - 在驗證集上應該穩定

3. **實際測試**：
   - 對比 `chars_per_line`：應該明顯增加
   - 對比可視化圖片：碎片化減少

---

## 時間估算

### Detection Fine-tune（50 張圖片）

| 階段 | 時間 |
|------|------|
| 資料標註 | 15 分鐘 |
| 資料準備 | 2 分鐘 |
| 配置訓練 | 3 分鐘 |
| 訓練（10 epochs） | 30 分鐘 |
| 導出測試 | 3 分鐘 |
| **總計** | **~53 分鐘** |

### 完整訓練（500 張圖片 + Recognition）

| 階段 | 時間 |
|------|------|
| Detection 標註 | 4-6 小時 |
| Detection 訓練 (100 epochs) | 8-12 小時 |
| Recognition 資料準備 | 1-2 小時 |
| Recognition 訓練 (100 epochs) | 12-24 小時 |
| **總計** | **~2-3 天** |

---

## 下一步

1. ✅ 完成 Detection Fine-tune
2. ✅ 測試效果並迭代改進
3. ✅ 收集更多資料（目標 500-1000 張）
4. ✅ 完成 Recognition Fine-tune（如需要）
5. ✅ 整合到作文批改系統

**祝 Fine-tune 順利！** 🚀

---

## 參考文檔

- [完整 Fine-tune 指南](./FINETUNE_GUIDE.md)
- [PPOCRLabel 教學](./PPOCRLABEL_TUTORIAL.md)
- [PaddleOCR 官方文檔](https://paddlepaddle.github.io/PaddleOCR/)
