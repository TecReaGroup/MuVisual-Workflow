# MuVisual-Workflow

instrument separation model download link: <https://huggingface.co/buckets/Trgroup/BS-ROFO-SW-Fixed-bucket>

## Introduction

MuVisual is a tool for visualizing music.
Now, The default sheet music is in 4/4 time.

## Develop

```bash
conda create -n muvisual python=3.12 -y
conda activate muvisual

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu130>
python -m pip install -r requirements.txt
```
