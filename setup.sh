#!/bin/bash
set -e

# Tạo env riêng nếu muốn
python3 -m venv venv
source venv/bin/activate

# Update pip
pip install --upgrade pip

# Cài requirements chung
pip install -r requirements.txt

# ---------------- DeepSolo ----------------
if [ ! -d "DeepSolo" ]; then
  git clone https://github.com/ViTAE-Transformer/DeepSolo.git
fi
cd DeepSolo
pip install -r requirements.txt
pip install -e .
cd ..

# ---------------- PARSeq ----------------
if [ ! -d "parseq" ]; then
  git clone https://github.com/baudm/parseq.git
fi
cd parseq
# Nếu dùng GPU thì dùng core.cu.txt, còn CPU thì core.cpu.txt
pip install -r requirements/core.cu.txt
pip install -e .
cd ..

echo "✅ Setup xong! Giờ bạn có thể chạy main.py"
