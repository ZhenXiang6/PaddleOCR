#!/bin/bash
# Detection 模型訓練腳本
# 在 Docker 容器內執行

set -e  # 遇到錯誤立即退出

echo "======================================================================"
echo "Detection 模型訓練"
echo "======================================================================"

# 配置參數
CONFIG_FILE="${CONFIG_FILE:-configs/det/custom_det.yml}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-./PP-OCRv5_mobile_det_train/best_accuracy}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DEVICE="${DEVICE:-cpu}"

echo ""
echo "配置信息:"
echo "  配置文件: $CONFIG_FILE"
echo "  預訓練模型: $PRETRAINED_MODEL"
echo "  設備: $DEVICE"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  恢復訓練: $RESUME_CHECKPOINT"
fi

# 檢查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo ""
    echo "❌ 錯誤：配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "請先創建配置文件，參考:"
    echo "  cp configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml $CONFIG_FILE"
    echo "  然後修改數據路徑和訓練參數"
    exit 1
fi

# 檢查預訓練模型（如果不是恢復訓練）
if [ -z "$RESUME_CHECKPOINT" ]; then
    if [ ! -f "${PRETRAINED_MODEL}.pdparams" ]; then
        echo ""
        echo "⚠️  警告：預訓練模型不存在: ${PRETRAINED_MODEL}.pdparams"
        echo ""
        echo "正在下載 PP-OCRv5 Mobile Detection 預訓練模型..."

        # 下載預訓練模型
        wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_det_train.tar
        tar -xf PP-OCRv5_mobile_det_train.tar
        rm PP-OCRv5_mobile_det_train.tar

        echo "✅ 預訓練模型下載完成"
    fi
fi

# 創建輸出目錄
mkdir -p output/custom_det

# 開始訓練
echo ""
echo "======================================================================"
echo "開始訓練..."
echo "======================================================================"
echo ""

if [ -n "$RESUME_CHECKPOINT" ]; then
    # 恢復訓練
    python3 tools/train.py \
        -c "$CONFIG_FILE" \
        -o Global.checkpoints="$RESUME_CHECKPOINT" \
           Global.device="$DEVICE"
else
    # 從預訓練模型開始
    python3 tools/train.py \
        -c "$CONFIG_FILE" \
        -o Global.pretrained_model="$PRETRAINED_MODEL" \
           Global.device="$DEVICE"
fi

# 訓練完成
echo ""
echo "======================================================================"
echo "訓練完成！"
echo "======================================================================"
echo ""
echo "模型保存位置: output/custom_det/"
echo "  - best_accuracy.pdparams  # 最佳模型"
echo "  - latest.pdparams          # 最新模型"
echo "  - train.log                # 訓練日誌"
echo ""
echo "下一步:"
echo "1. 評估模型: ./eval_detection.sh"
echo "2. 導出模型: ./export_detection.sh"
echo "3. 測試模型: python3 test_finetuned_model.py --det_model ./inference/custom_det_model"
