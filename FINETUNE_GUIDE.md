# PaddleOCR Fine-tune 完整教學指南

## 📋 目錄

1. [概述](#概述)
2. [環境準備](#環境準備)
3. [Detection Fine-tune](#detection-fine-tune)
4. [Recognition Fine-tune](#recognition-fine-tune)
5. [模型評估與對比](#模型評估與對比)
6. [常見問題](#常見問題)

---

## 概述

### 為什麼需要 Fine-tune？

PaddleOCR 預訓練模型在通用場景表現良好，但在特定場景（如手寫作文稿紙）可能存在：
- **Detection 問題**：碎片化（文字被切成很多小塊）、漏檢、誤檢
- **Recognition 問題**：識別錯誤、繁簡體混淆、特殊字符

### 兩個獨立模型

```
完整 OCR 系統 = Detection 模型 + Recognition 模型

原始圖片 → [Detection] → 文字區域邊界框 → [Recognition] → 文字內容
```

- **Detection**：找出文字在哪裡（輸出：邊界框座標）
- **Recognition**：識別文字是什麼（輸出：文字內容）

### 數據需求對比

| 模型 | 最少數據 | 推薦數據 | 標註難度 | 標註內容 |
|------|---------|---------|---------|---------|
| Detection | 500 張 | 1000 張 | ⭐⭐ 簡單 | 邊界框位置 |
| Recognition | 5000 張 | 10000 張 | ⭐⭐⭐⭐ 複雜 | 完整文字內容 |

---

## 環境準備

### 1. Docker 環境

由於 macOS arm64 不支援訓練，必須使用 Docker（Linux x86_64）環境。

**使用現有鏡像**：
```bash
# 檢查鏡像是否存在
docker images | grep paddleocr-essay

# 如果不存在，重新構建
docker build --platform linux/amd64 -f Dockerfile.essay_ocr -t paddleocr-essay:latest .
```

**啟動容器（互動模式）**：
```bash
docker run -it --rm --platform linux/amd64 \
  -v $(pwd)/essay_data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd):/app/paddleocr \
  paddleocr-essay:latest \
  /bin/bash
```

### 2. 安裝 PPOCRLabel（標註工具）

**在 macOS 本地安裝**（不在 Docker 內）：
```bash
pip install PPOCRLabel
```

**啟動**：
```bash
PPOCRLabel --lang ch
```

---

## Detection Fine-tune

### 階段 1：數據標註

#### 1.1 啟動 PPOCRLabel

```bash
PPOCRLabel --lang ch
```

#### 1.2 載入圖片

1. 點擊「打開目錄」→ 選擇作文圖片目錄
2. 點擊「自動標註」→ 使用預訓練模型檢測（會自動標註）

#### 1.3 手動修正標註

**關鍵操作**：

1. **合併碎片化的框**：
   - 將多個單字框合併成完整句子框
   - 方法：刪除多餘的框，重新繪製大框包含整句話

2. **調整邊界框**：
   - 確保框完整包含所有文字
   - 框不要太緊（留一點邊距）

3. **文字標註**：
   - 可以簡化（只需大致正確）
   - 或直接使用 "###" 佔位（Detection 不需要準確的文字內容）

**標註示範**：

❌ **錯誤**（碎片化）：
```
框1: [100,50,120,70] → "水"
框2: [121,50,141,70] → "能"
框3: [142,50,162,70] → "載"
```

✅ **正確**（完整句子）：
```
框1: [100,50,500,70] → "水能載舟亦能覆舟"
```

#### 1.4 輸出標註

PPOCRLabel 會生成：
- `Label.txt` - 標註文件
- `crop_img/` - 裁剪的文字圖片（用於 Recognition）

**標註文件格式**：
```
essay_001.jpg	[{"transcription": "水能載舟亦能覆舟", "points": [[100, 50], [500, 50], [500, 100], [100, 100]]}, ...]
```

### 階段 2：數據準備

#### 2.1 使用準備腳本

```bash
python prepare_det_data.py \
  --label_file /path/to/PPOCRLabel/Label.txt \
  --image_dir /path/to/images \
  --output_dir train_data/det \
  --train_ratio 0.8
```

**輸出**：
```
train_data/det/
├── images/          # 所有圖片
├── train.txt        # 訓練集標註（80%）
└── val.txt          # 驗證集標註（20%）
```

### 階段 3：配置訓練

#### 3.1 下載預訓練模型

在 Docker 容器內執行：

```bash
cd /app/paddleocr

# 下載 PP-OCRv5 Mobile Detection 預訓練模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_det_train.tar
tar -xf PP-OCRv5_mobile_det_train.tar

# 檢查
ls PP-OCRv5_mobile_det_train/
# 應該看到: best_accuracy.pdparams, best_accuracy.pdopt
```

#### 3.2 修改配置文件

複製並修改配置：

```bash
cp configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml configs/det/custom_det.yml
```

**關鍵修改**（`custom_det.yml`）：

```yaml
Global:
  pretrained_model: ./PP-OCRv5_mobile_det_train/best_accuracy
  save_model_dir: ./output/custom_det/
  epoch_num: 200  # 調整 epoch 數

Train:
  dataset:
    data_dir: ./train_data/det/images/
    label_file_list:
      - ./train_data/det/train.txt
  loader:
    batch_size_per_card: 4  # 根據記憶體調整

Eval:
  dataset:
    data_dir: ./train_data/det/images/
    label_file_list:
      - ./train_data/det/val.txt
```

### 階段 4：執行訓練

#### 4.1 開始訓練

```bash
python tools/train.py -c configs/det/custom_det.yml
```

**訓練過程**：
- 每個 epoch 會輸出訓練損失
- 定期進行驗證（驗證集）
- 自動保存 best_accuracy 模型

**預計訓練時間**：
- 500 張圖片，200 epochs
- CPU：約 8-12 小時
- GPU：約 2-4 小時

#### 4.2 監控訓練

**查看日誌**：
```bash
tail -f output/custom_det/train.log
```

**關鍵指標**：
- `loss`：訓練損失（應該逐漸降低）
- `hmean`：F1 分數（應該逐漸提升）
- `precision`、`recall`：檢測準確率和召回率

#### 4.3 恢復訓練（如果中斷）

```bash
python tools/train.py -c configs/det/custom_det.yml \
  -o Global.checkpoints=./output/custom_det/iter_5000
```

### 階段 5：評估模型

#### 5.1 評估驗證集

```bash
python tools/eval.py -c configs/det/custom_det.yml \
  -o Global.checkpoints=./output/custom_det/best_accuracy
```

#### 5.2 導出推理模型

```bash
python tools/export_model.py -c configs/det/custom_det.yml \
  -o Global.pretrained_model=./output/custom_det/best_accuracy \
     Global.save_inference_dir=./inference/custom_det_model
```

**輸出**：
```
inference/custom_det_model/
├── inference.pdiparams      # 模型參數
├── inference.pdiparams.info
└── inference.pdmodel         # 模型結構
```

#### 5.3 測試推理

```bash
python test_finetuned_model.py \
  --model_type detection \
  --model_path ./inference/custom_det_model \
  --test_images ./test_images/ \
  --output_dir ./output/det_test/
```

---

## Recognition Fine-tune

### 階段 1：數據準備

#### 1.1 轉換已有的單字資料

假設你的單字資料格式：
```
char_images/
├── 水_001.jpg
├── 能_001.jpg
└── ...
```

**使用轉換腳本**：
```bash
python convert_rec_data.py \
  --image_dir /path/to/char_images \
  --output_dir train_data/rec \
  --format auto  # 自動識別格式
```

#### 1.2 裁剪作文文字行（補充資料）

從 Detection 標註中裁剪：

```bash
python crop_essay_lines.py \
  --label_file train_data/det/train.txt \
  --image_dir train_data/det/images \
  --output_dir train_data/rec/essay_lines
```

#### 1.3 合併資料

```bash
cat train_data/rec/char_data.txt train_data/rec/essay_lines.txt > train_data/rec/all_train.txt

# 分割訓練集和驗證集
python split_rec_data.py \
  --input train_data/rec/all_train.txt \
  --output_dir train_data/rec \
  --train_ratio 0.9
```

**最終格式**（`train.txt`）：
```
char_images/水_001.jpg	水
char_images/能_001.jpg	能
essay_lines/line_001.jpg	水能載舟亦能覆舟
essay_lines/line_002.jpg	活中擁有健康的人際關係
```

### 階段 2：配置訓練

#### 2.1 下載預訓練模型

```bash
cd /app/paddleocr

wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_rec_train.tar
tar -xf PP-OCRv5_mobile_rec_train.tar
```

#### 2.2 修改配置文件

```bash
cp configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml configs/rec/custom_rec.yml
```

**關鍵修改**：

```yaml
Global:
  pretrained_model: ./PP-OCRv5_mobile_rec_train/best_accuracy
  save_model_dir: ./output/custom_rec/
  epoch_num: 300
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt  # 確認包含繁體字

Train:
  dataset:
    data_dir: ./train_data/rec/
    label_file_list:
      - ./train_data/rec/train.txt
  loader:
    batch_size_per_card: 128

Eval:
  dataset:
    data_dir: ./train_data/rec/
    label_file_list:
      - ./train_data/rec/val.txt
```

### 階段 3：執行訓練

```bash
python tools/train.py -c configs/rec/custom_rec.yml
```

**預計訓練時間**：
- 5000 張圖片，300 epochs
- CPU：約 12-24 小時
- GPU：約 4-8 小時

### 階段 4：評估與導出

```bash
# 評估
python tools/eval.py -c configs/rec/custom_rec.yml \
  -o Global.checkpoints=./output/custom_rec/best_accuracy

# 導出
python tools/export_model.py -c configs/rec/custom_rec.yml \
  -o Global.pretrained_model=./output/custom_rec/best_accuracy \
     Global.save_inference_dir=./inference/custom_rec_model
```

---

## 模型評估與對比

### 使用 Fine-tuned 模型

#### 方法 1：Python API

```python
from paddleocr import PaddleOCR

# 使用自定義模型
ocr = PaddleOCR(
    det_model_dir='./inference/custom_det_model',
    rec_model_dir='./inference/custom_rec_model',
    use_doc_orientation_classify=True,
    device='cpu'
)

result = ocr.predict(input='test.jpg')
print(result[0]['rec_texts'])
```

#### 方法 2：測試腳本

```bash
python test_finetuned_model.py \
  --det_model ./inference/custom_det_model \
  --rec_model ./inference/custom_rec_model \
  --test_images ./test_images/ \
  --output_dir ./output/comparison/
```

### 對比評估

**對比指標**：

| 模型 | 平均區塊數 | 字/區塊 | 準確率 | 碎片化改善 |
|------|----------|---------|--------|-----------|
| 預訓練 | 109 | 1.9 | 84.3% | - |
| Fine-tuned | ? | ? | ? | ? |

---

## 常見問題

### Q1: 訓練時記憶體不足怎麼辦？

**解決方案**：
- 減少 `batch_size_per_card`
- 降低 `text_det_limit_side_len`（Detection）
- 使用更小的模型（mobile 而非 server）

### Q2: 訓練損失不下降？

**可能原因**：
1. Learning rate 太高或太低 → 調整 `Optimizer.lr`
2. 數據標註錯誤 → 檢查標註質量
3. 模型已經收斂 → 提前停止訓練

### Q3: 訓練完模型效果反而變差？

**可能原因**：
1. **過擬合**：訓練數據太少或太單一
   - 解決：增加數據多樣性，使用數據增強
2. **標註錯誤**：標註質量差
   - 解決：重新檢查和修正標註
3. **訓練過度**：epoch 太多
   - 解決：使用更早的 checkpoint

### Q4: PPOCRLabel 標註太慢？

**加速技巧**：
1. 使用「自動標註」功能（半自動）
2. 使用快捷鍵（W/E/D/X 等）
3. 只標註關鍵樣本（不需要全部標註）
4. 文字標註可以簡化（Detection 只需位置準確）

### Q5: 42 張圖片夠嗎？

**現實評估**：
- **Detection**：42 張可能不足，建議至少 200-500 張
- **Recognition**：如果有幾千個單字資料，加上從作文裁剪的行，**可能足夠**

**建議**：
- 先用 42 張驗證流程（證明方向正確）
- 逐步收集更多資料（目標 500-1000 張）

### Q6: 如何判斷是否需要繼續訓練？

**觀察指標**：
- `train loss` 下降但 `val loss` 上升 → 過擬合，停止訓練
- `train loss` 和 `val loss` 都不動 → 收斂，停止訓練
- `train loss` 和 `val loss` 都在下降 → 繼續訓練

### Q7: 可以只 Fine-tune Detection 嗎？

**可以！** 如果主要問題是碎片化，只 Fine-tune Detection 即可。

**使用方式**：
```python
ocr = PaddleOCR(
    det_model_dir='./inference/custom_det_model',  # 自定義 Detection
    # rec_model_dir 不指定，使用預訓練 Recognition
    device='cpu'
)
```

---

## 附錄：資料格式範例

### Detection 標註格式（Label.txt）

```txt
essay_001.jpg	[{"transcription": "水能載舟亦能覆舟", "points": [[100, 50], [500, 50], [500, 100], [100, 100]]}, {"transcription": "活中擁有健康的人際關係", "points": [[100, 120], [480, 120], [480, 170], [100, 170]]}]
essay_002.jpg	[{"transcription": "互動的同時更有和現實脫節的風險", "points": [[100, 50], [520, 50], [520, 100], [100, 100]]}]
```

### Recognition 標註格式（train.txt）

```txt
crop_img/text_001.jpg	水能載舟亦能覆舟
crop_img/text_002.jpg	活中擁有健康的人際關係
crop_img/text_003.jpg	互動的同時更有和現實脫節的風險
char_img/水_001.jpg	水
char_img/能_001.jpg	能
```

---

## 參考資源

- [PaddleOCR 官方文檔](https://paddlepaddle.github.io/PaddleOCR/)
- [PP-OCRv5 介紹](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- [PPOCRLabel 使用指南](https://github.com/PaddlePaddle/PaddleOCR/tree/main/PPOCRLabel)
- [Fine-tuning 教學](http://www.paddleocr.ai/v2.10.0/en/ppocr/model_train/finetune.html)

---

## 下一步

1. ✅ 閱讀本指南
2. ✅ 安裝 PPOCRLabel
3. ✅ 開始標註 Detection 資料（50-100 張快速驗證）
4. ✅ 準備 Recognition 資料（轉換格式）
5. ✅ 在 Docker 中執行訓練
6. ✅ 評估效果並迭代改進

**祝你 Fine-tune 順利！** 🚀
