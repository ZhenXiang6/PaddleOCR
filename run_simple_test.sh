#!/bin/bash
cd /Users/morrisliao/Desktop/git-repo/paddleOCR/PaddleOCR
docker run --rm --platform linux/amd64 \
  -v /Users/morrisliao/Desktop/git-repo/paddleOCR/PaddleOCR/essay_data:/app/data:ro \
  -v /Users/morrisliao/Desktop/git-repo/paddleOCR/PaddleOCR/output:/app/output \
  -v /Users/morrisliao/Desktop/git-repo/paddleOCR/PaddleOCR/docker_ocr_test.py:/app/docker_ocr_test.py:ro \
  paddleocr-essay:latest \
  python3 /app/docker_ocr_test.py
