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
if [ ! -f "venv/bin/python" ]; then
    echo ""
    echo "Tworzenie srodowiska wirtualnego..."

    # Proba 1: standardowe
    if python3 -m venv venv 2>/dev/null && [ -f "venv/bin/python" ]; then
        echo "OK."
    else
        # Proba 2: --without-pip + reczne bootstrapping pip
        echo "Standardowe venv nie powiodlo sie, proba z --without-pip..."
        rm -rf venv 2>/dev/null
        python3 -m venv venv --without-pip
        if [ ! -f "venv/bin/python" ]; then
            echo "[BLAD] Nie udalo sie utworzyc venv."
            echo "Uzyj pelnej instalacji Pythona (nie ze snap/store)."
            echo "Debian/Ubuntu: sudo apt install python3-venv python3-pip"
            exit 1
        fi
        # Bootstrapping pip
        echo "Pobieranie pip..."
        venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
        venv/bin/python get-pip.py --quiet
        rm -f get-pip.py
        echo "OK (pip zainstalowany recznie)."
    fi
else
    echo "Srodowisko wirtualne juz istnieje."
fi

# Aktywacja
source venv/bin/activate

# Instalacja PyTorch (GPU lub CPU)
echo ""
echo "Instalacja PyTorch..."
pip install --upgrade pip --quiet
pip install $TORCH_INDEX torch torchvision torchaudio --quiet || echo "[UWAGA] Instalacja PyTorch nie powiodla sie."

# Instalacja reszty zaleznosci
echo ""
echo "Instalacja pozostalych zaleznosci..."
pip install -r requirements.txt --quiet || echo "[UWAGA] Instalacja niektorych zaleznosci nie powiodla sie."

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
" 2>/dev/null || echo "[UWAGA] Weryfikacja PyTorch nie powiodla sie."

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
