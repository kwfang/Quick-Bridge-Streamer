# 🤖 Quick-Bridge-Streamer: A BridgeData V2 RLDS Viewer

> **"Preview BridgeData without polluting your PyTorch environment."**

A lightweight BridgeData V2 viewer for **PyTorch users** who want to inspect TFRecord datasets **without installing full TensorFlow** or risking CUDA conflicts.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- ⚡ **PyTorch-Safe** — TensorFlow-CPU only (GPU disabled). Zero CUDA conflicts with your PyTorch environment.
- 🎥 **Real-time visualization** — High-performance streaming via Rerun SDK
- 🔍 **Smart analytics** — Detects frozen robots and gripper spam automatically
- 🎯 **PyTorch code generation** — Auto-generates ready-to-use Dataset code on exit
- 🌐 **Web-based viewer** — Opens in browser automatically, no CLI tools needed

## 🚀 Quick Start

### Step 1: Clone & Install

```bash
git clone https://github.com/kwfang/Quick-Bridge-Streamer.git
cd Quick-Bridge-Streamer
pip install -r requirements.txt  # Safe: only tensorflow-cpu, no CUDA deps
```

### Step 2: Prepare Your Data

Place your BridgeData V2 TFRecord files in a folder (e.g., `data/`):

```
data/
├── bridge_dataset-train.tfrecord-00000-of-XXXXX
├── bridge_dataset-train.tfrecord-00001-of-XXXXX
└── dataset_info.json
```


### Step 3: Run

```bash
python main.py data --limit 100
```

**Parameters:**
- `data`: Path to dataset folder (containing `.tfrecord-*` files)
- `--limit`: Number of frames to stream (`-1` = unlimited, default)
- `--split`: Data split (`train` or `val`, default: `train`)
- `--no-interactive`: Auto-select first dataset without prompting

### Step 4: Exit

When you press `Ctrl+C` to exit, the terminal automatically prints:

```python
[SUCCESS] Session ended.

Need to load this data in PyTorch? Here is your snippet:
---------------------------------------------------------
class BridgeDataV2Dataset(Dataset):
    def __init__(self, data_path="...", split="train"):
        ...
    def __getitem__(self, idx):
        ...
---------------------------------------------------------
Copy the above to start training!
```

## 📁 Project Structure

```
Quick-Bridge-Streamer/
├── oxview/                 # Core package
│   ├── __init__.py         # Version info
│   ├── loader.py           # TFRecord parsing & streaming
│   ├── viewer.py           # Rerun web visualization
│   ├── analytics.py        # Quality detection algorithms
│   └── exporter.py         # PyTorch code generation
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🔧 Advanced Usage

### Scan multiple datasets

```bash
# If data/ contains multiple subfolders with datasets
python main.py data --no-interactive
```

### Stream unlimited frames

```bash
python main.py data --limit -1
```

### Select specific dataset

```bash
python main.py data --dataset 2 --limit 50
```

## 🧪 Dataset Format

**BridgeData V2** structure:

| Feature | Type | Shape | Description |
|---------|------|-------|-------------|
| `action` | float32 | [7] | [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper] |
| `language_instruction` | string | — | Natural language task description |
| `observation/image_0` | uint8 | [256, 256, 3] | Main camera RGB (JPEG) |
| `observation/state` | float32 | [7] | Robot joint angles + gripper position |

## 🐛 Troubleshooting

### ImportError: No module named 'tensorflow'

```bash
pip install -r requirements.txt
```

### Browser doesn't open

Check firewall settings or manually open: `http://localhost:9090`

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

**Version**: 0.1.0 
