# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaddleOCR is a production-ready OCR and document AI engine built on PaddlePaddle. Version 3.x represents a major rewrite with modern Python API (paddleocr/) separated from the legacy training framework (ppocr/, tools/).

**Key Components:**
- **PP-OCRv5**: Universal scene text recognition supporting 5 text types (Simplified Chinese, Traditional Chinese, English, Japanese, Pinyin) with 13% accuracy improvement
- **PP-StructureV3**: Complex document parsing that converts PDFs/images to Markdown/JSON with structure preservation
- **PP-ChatOCRv4**: Intelligent information extraction using ERNIE 4.5 for key-value pair extraction
- **PaddleOCR-VL**: 0.9B parameter vision-language model for multilingual document parsing (109 languages)

## Architecture

### Two Parallel Systems

1. **Modern API (paddleocr/)** - v3.x user-facing API
   - Built on top of PaddleX framework
   - High-level pipelines: `PaddleOCR`, `PPStructureV3`, `PPChatOCRv4Doc`, `PaddleOCRVL`
   - Individual models: `TextDetection`, `TextRecognition`, `LayoutDetection`, etc.
   - Entry point: `paddleocr/__main__.py` (CLI), `paddleocr/__init__.py` (Python API)

2. **Training Framework (ppocr/, tools/)** - Legacy system for model training
   - Model architectures: `ppocr/modeling/`
   - Data processing: `ppocr/data/`
   - Training losses: `ppocr/losses/`
   - Training scripts: `tools/train.py`, `tools/eval.py`, `tools/export_model.py`

### Directory Structure

```
paddleocr/          # Modern v3.x API (inference only)
├── _pipelines.py   # High-level pipelines
├── _models.py      # Individual model wrappers
└── _cli.py         # Command-line interface

ppocr/              # Legacy training framework
├── modeling/       # Model architectures (backbones, necks, heads)
├── data/           # Data loading and augmentation
├── losses/         # Loss functions for training
├── metrics/        # Evaluation metrics
└── postprocess/    # Post-processing for inference

tools/              # Training and inference scripts
├── train.py        # Model training
├── eval.py         # Model evaluation
├── export_model.py # Export to inference format
├── infer_*.py      # Inference scripts
└── infer/          # Inference utilities

configs/            # YAML configuration files
├── det/            # Text detection configs
├── rec/            # Text recognition configs
├── cls/            # Text orientation classification
├── table/          # Table recognition configs
└── kie/            # Key information extraction

deploy/             # Deployment solutions
├── cpp_infer/      # C++ inference
├── paddle2onnx/    # ONNX conversion
├── lite/           # Mobile deployment
└── docker/         # Docker deployment

data/               # User data directory (not in repo)
output/             # Training/inference outputs
```

## Common Commands

### Installation

```bash
# Basic OCR only (minimal dependencies)
pip install paddleocr

# Full features (document parsing, information extraction, translation)
pip install "paddleocr[all]"

# Specific feature groups
pip install "paddleocr[doc-parser]"  # Document parsing
pip install "paddleocr[ie]"          # Information extraction
pip install "paddleocr[trans]"       # Document translation
```

### CLI Inference

```bash
# Basic OCR
paddleocr ocr -i image.png

# Document parsing to Markdown
paddleocr pp_structurev3 -i document.pdf

# Key information extraction (requires API key)
paddleocr pp_chatocrv4_doc -i invoice.png -k "total amount" --qianfan_api_key YOUR_KEY

# PaddleOCR-VL document parsing
paddleocr doc_parser -i document.png

# Get help for any command
paddleocr ocr --help
```

### Python API Usage

```python
from paddleocr import PaddleOCR

# Initialize with options
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Run inference
result = ocr.predict(input="image.png")

# Process results
for res in result:
    res.print()                      # Print to console
    res.save_to_img("output")        # Save visualization
    res.save_to_json("output")       # Save JSON results
```

### Training Commands

**Important**: Training uses the legacy framework (ppocr/, tools/, configs/)

```bash
# Train a model
python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml

# Resume training
python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.checkpoints=./output/iter_1000

# Evaluate a model
python tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.checkpoints=./output/best_accuracy

# Export to inference format
python tools/export_model.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=./output/best_accuracy Global.save_inference_dir=./inference/det_model

# Run inference with legacy tools (useful for debugging training)
python tools/infer_det.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.infer_img="./test_imgs/" Global.pretrained_model="./inference/det_model"
```

### Testing

```bash
# Run basic tests (excluding resource-intensive tests)
pytest

# Run all tests including resource-intensive ones
pytest -m ""
```

## Configuration System (YAML)

All model training configs are in `configs/` organized by task:

```yaml
# Example structure from configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml
Global:
  pretrained_model: null
  save_model_dir: ./output/
  epoch_num: 500

Architecture:
  model_type: det
  algorithm: DB
  Backbone:
    name: MobileNetV3
  Neck:
    name: DBFPN
  Head:
    name: DBHead

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/train_list.txt
  loader:
    batch_size_per_card: 8
```

**Key Config Parameters:**
- `Global.pretrained_model`: Path to pretrained weights
- `Global.save_model_dir`: Output directory for checkpoints
- `Architecture`: Model structure (Backbone/Neck/Head)
- `Train/Eval.dataset`: Data configuration
- `Optimizer/Loss`: Training hyperparameters

**Override configs via CLI:**
```bash
python tools/train.py -c config.yml -o Global.epoch_num=100 Train.loader.batch_size_per_card=16
```

## Version 2.x to 3.x Migration

**Breaking Changes:**
1. **API Redesign**: Old `paddleocr` module completely replaced
   - v2.x: `from paddleocr import PaddleOCR; ocr = PaddleOCR(); result = ocr.ocr(img_path)`
   - v3.x: `from paddleocr import PaddleOCR; ocr = PaddleOCR(); result = ocr.predict(input=img_path)`

2. **Method Names Changed**:
   - v2.x: `ocr.ocr()` returns `[[[bbox, (text, confidence)]]]`
   - v3.x: `ocr.predict()` returns structured result objects with `.print()`, `.save_to_json()` methods

3. **Dependency Groups**: Core vs optional features separated
   - v2.x: All features installed by default
   - v3.x: Install only what you need (`paddleocr[all]`, `paddleocr[doc-parser]`, etc.)

4. **Training Framework Unchanged**: `ppocr/` and `tools/` remain compatible

**For v2.x Compatibility:**
- If running existing v2.x code, downgrade: `pip install paddleocr==2.10.0`
- Training code using `ppocr/` and `tools/` works in both versions

## macOS Apple Silicon Notes

**Known Issues (as of v3.3.0):**
- PaddleOCR 3.x + PaddlePaddle 3.x has stability issues on macOS arm64
- Symptoms: Segmentation faults, freezing, slow inference

**Workaround:**
- Use PaddleOCR 2.x: `pip install paddleocr==2.10.0 paddlepaddle==3.2.0`
- Or use Docker/Linux for production workloads
- See user CLAUDE.md (in `.claude/CLAUDE.md`) for detailed macOS troubleshooting

## Heterogeneous Hardware Support

PaddleOCR supports multiple hardware accelerators:

```bash
# CPU (default)
paddleocr ocr -i image.png

# GPU (NVIDIA)
paddleocr ocr -i image.png --device gpu

# Ascend NPU (Huawei)
# Install: pip install paddle-custom-device-npu
paddleocr ocr -i image.png --device npu

# Kunlunxin XPU
# Install: pip install paddle-custom-device-xpu
paddleocr ocr -i image.png --device xpu
```

## Development Workflow

### Typical Training Workflow

1. **Prepare Data**:
   ```
   data/
   ├── train_images/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── train_list.txt  # Format: img_path\tlabel
   ```

2. **Select/Create Config**: Choose from `configs/{det,rec,cls}/` or modify existing config

3. **Train**: `python tools/train.py -c config.yml`

4. **Evaluate**: `python tools/eval.py -c config.yml -o Global.checkpoints=./output/best_accuracy`

5. **Export**: `python tools/export_model.py -c config.yml -o Global.pretrained_model=./output/best_accuracy`

6. **Use in API**: Place exported model in appropriate directory and update API config

### Debugging Tips

1. **Test with legacy tools first**: If training a new model, test with `tools/infer_*.py` before integrating with v3 API

2. **Check data loading**: Add `print()` statements in `ppocr/data/*.py` to verify data format

3. **Visualize augmentation**: Use `ppocr/data/imaug/` modules to debug data augmentation

4. **Compare configs**: Use `diff` to compare similar configs when something works vs doesn't work

5. **Enable logging**: Set environment variable `PADDLEOCR_DEBUG=1` for verbose output

## Model Types

- **Detection**: DB, EAST, SAST, PSE, FCE, PGNet, CT
- **Recognition**: CRNN, SVTR, ABINet, NRTR, SAR, ASTER, ViTSTR, ParseQ, PARSeq
- **Table**: TableMaster, SLANet
- **Formula**: LaTeX-OCR, UniMERNet, PP-FormulaNet
- **Layout**: PP-Layout, LayoutLM series
- **KIE**: LayoutLM, LayoutXLM, VI-LayoutXLM, SDMGR

## Data Format

### Detection Labels
```
# Format: image_path\t[{points, transcription, difficult}]
train_data/img1.jpg\t[{"transcription": "hello", "points": [[0,0],[100,0],[100,50],[0,50]], "difficult": false}]
```

### Recognition Labels
```
# Format: image_path\ttranscription
train_data/img1.jpg\thello world
```

### Table Recognition
```
# Format: image_path\tHTML_table
train_data/table1.jpg\t<html><body><table>...</table></body></html>
```

## Deployment Options

1. **Python API**: Direct `paddleocr` package usage (easiest)
2. **HTTP Service**: `paddleocr serve` (coming soon)
3. **C++ Inference**: `deploy/cpp_infer/` (high performance)
4. **ONNX Export**: `deploy/paddle2onnx/` (cross-platform)
5. **Mobile**: `deploy/lite/` (iOS/Android)
6. **Docker**: `deploy/docker/` (containerized)
7. **MCP Server**: Integration with Claude Desktop and other Agent applications

## Important Notes

- **Always read the documentation matching your PaddleOCR version** (2.x vs 3.x have different APIs)
- **Training code is stable across versions** - only inference API changed in v3.x
- **Use dependency groups** to avoid installing unnecessary packages
- **Check hardware compatibility** before deploying to production
- **YAML configs are the source of truth** for model architecture and training settings
