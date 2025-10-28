#!/bin/bash
# Docker OCR 快速启动脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================================${NC}"
echo -e "${GREEN}PaddleOCR Docker 環境 - 快速啟動${NC}"
echo -e "${GREEN}===========================================================${NC}"

# 检查 Docker 镜像
if ! docker image inspect paddleocr-essay:latest > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker 鏡像不存在${NC}"
    echo -e "${YELLOW}請先執行：docker build --platform linux/amd64 -f Dockerfile.essay_ocr -t paddleocr-essay:latest .${NC}"
    exit 1
fi

# 检查数据目录
if [ ! -d "essay_data" ]; then
    echo -e "${RED}❌ essay_data 目錄不存在${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p output

# 运行测试
echo -e "\n${GREEN}[1/2] 執行環境測試...${NC}"
docker run --rm --platform linux/amd64 \
  -v "$PWD/essay_data":/app/data:ro \
  -v "$PWD/output":/app/output \
  -v "$PWD/docker_ocr_test.py":/app/docker_ocr_test.py:ro \
  paddleocr-essay:latest \
  python3 /app/docker_ocr_test.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Docker 環境測試成功！${NC}"
    echo -e "${YELLOW}測試結果已保存至: output/${NC}"
else
    echo -e "\n${RED}❌ Docker 環境測試失敗${NC}"
    exit 1
fi

echo -e "\n${GREEN}===========================================================${NC}"
echo -e "${GREEN}下一步：參數優化與批次處理${NC}"
echo -e "${GREEN}===========================================================${NC}"
