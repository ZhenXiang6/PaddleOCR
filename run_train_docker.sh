#!/bin/bash
# 在 Docker 容器中執行訓練
# 使用方法: ./run_train_docker.sh [detection|recognition]

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 參數
TASK_TYPE="${1:-detection}"
DEVICE="${DEVICE:-cpu}"

echo -e "${GREEN}======================================================================"
echo "PaddleOCR Fine-tune 訓練 (Docker)"
echo -e "======================================================================${NC}"
echo ""

# 檢查參數
if [ "$TASK_TYPE" != "detection" ] && [ "$TASK_TYPE" != "recognition" ]; then
    echo -e "${RED}❌ 錯誤：無效的任務類型: $TASK_TYPE${NC}"
    echo ""
    echo "使用方法:"
    echo "  ./run_train_docker.sh detection      # 訓練 Detection 模型"
    echo "  ./run_train_docker.sh recognition    # 訓練 Recognition 模型"
    exit 1
fi

# 檢查 Docker 鏡像
if ! docker images | grep -q "paddleocr-essay"; then
    echo -e "${YELLOW}⚠️  警告：Docker 鏡像不存在${NC}"
    echo ""
    echo "正在構建 Docker 鏡像..."
    docker build --platform linux/amd64 -f Dockerfile.essay_ocr -t paddleocr-essay:latest .
    echo -e "${GREEN}✅ Docker 鏡像構建完成${NC}"
    echo ""
fi

# 檢查資料目錄
if [ "$TASK_TYPE" = "detection" ]; then
    if [ ! -d "train_data/det" ]; then
        echo -e "${RED}❌ 錯誤：Detection 訓練資料不存在${NC}"
        echo ""
        echo "請先準備訓練資料:"
        echo "  1. 使用 PPOCRLabel 標註圖片"
        echo "  2. 執行: python3 prepare_det_data.py \\"
        echo "       --label_file /path/to/Label.txt \\"
        echo "       --image_dir /path/to/images \\"
        echo "       --output_dir train_data/det"
        exit 1
    fi
elif [ "$TASK_TYPE" = "recognition" ]; then
    if [ ! -d "train_data/rec" ]; then
        echo -e "${RED}❌ 錯誤：Recognition 訓練資料不存在${NC}"
        echo ""
        echo "請先準備訓練資料:"
        echo "  1. 轉換單字資料: python3 convert_rec_data.py ..."
        echo "  2. 裁剪作文文字行: python3 crop_essay_lines.py ..."
        echo "  3. 合併資料並分割訓練/驗證集"
        exit 1
    fi
fi

# 啟動 Docker 容器並執行訓練
echo -e "${GREEN}開始訓練 $TASK_TYPE 模型...${NC}"
echo ""

SCRIPT_NAME="train_${TASK_TYPE}.sh"

docker run --rm --platform linux/amd64 \
  -v "$(pwd)/train_data:/app/paddleocr/train_data" \
  -v "$(pwd)/output:/app/paddleocr/output" \
  -v "$(pwd)/configs:/app/paddleocr/configs" \
  -v "$(pwd)/${SCRIPT_NAME}:/app/paddleocr/${SCRIPT_NAME}" \
  -e DEVICE="$DEVICE" \
  paddleocr-essay:latest \
  bash -c "cd /app/paddleocr && chmod +x ${SCRIPT_NAME} && ./${SCRIPT_NAME}"

echo ""
echo -e "${GREEN}======================================================================"
echo "訓練完成！"
echo -e "======================================================================${NC}"
echo ""
echo "模型保存位置: ./output/custom_${TASK_TYPE}/"
echo ""
echo "下一步:"
echo "  1. 評估模型性能"
echo "  2. 導出推理模型"
echo "  3. 測試 fine-tuned 模型效果"
