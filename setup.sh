#!/usr/bin/env bash
set -e

echo "========================================"
echo "SER Pipeline - Instalacja (Linux/macOS)"
echo "========================================"
echo ""

# Sprawdz czy Python jest dostepny
if ! command -v python3 &> /dev/null; then
    echo "[BLAD] Python 3 nie znaleziony. Zainstaluj Python 3.10+."
    exit 1
fi

# Sprawdz czy CUDA GPU jest dostepna
if command -v nvidia-smi &> /dev/null; then
    echo "GPU NVIDIA wykryta - instalacja PyTorch z CUDA 11.8."
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu118"
else
    echo "Brak GPU NVIDIA - instalacja PyTorch CPU."
    TORCH_INDEX=""
fi

# Tworzenie venv
if [ ! -d "venv" ]; then
    echo ""
    echo "Tworzenie srodowiska wirtualnego..."
    python3 -m venv venv
    echo "OK."
else
    echo "Srodowisko wirtualne juz istnieje."
fi

# Aktywacja
source venv/bin/activate

# Instalacja PyTorch (GPU lub CPU)
echo ""
echo "Instalacja PyTorch..."
pip install --upgrade pip --quiet
pip install $TORCH_INDEX torch torchvision torchaudio --quiet

# Instalacja reszty zaleznosci
echo ""
echo "Instalacja pozostalych zaleznosci..."
pip install -r requirements.txt --quiet

# Weryfikacja
echo ""
echo "Weryfikacja instalacji..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================"
echo "Instalacja zakonczona."
echo ""
echo "Aby uruchomic pipeline:"
echo "  1. source venv/bin/activate"
echo "  2. Skopiuj foldery TESS, RAVDESS, SAVEE, CREMA-D do data/"
echo "  3. python run.py"
echo ""
echo "Dostepne opcje CLI:"
echo "  python run.py --help"
echo "========================================"
