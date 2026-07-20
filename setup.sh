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

# Sprawdz czy CUDA jest dostepna (opcjonalnie)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA GPU wykryta."
    PIP_INDEX=""
else
    echo "Brak CUDA GPU - instalacja PyTorch CPU."
    PIP_INDEX="--index-url https://download.pytorch.org/whl/cpu"
fi

# Tworzenie venv
if [ ! -d "venv" ]; then
    echo "Tworzenie srodowiska wirtualnego..."
    python3 -m venv venv
    echo "OK."
else
    echo "Srodowisko wirtualne juz istnieje."
fi

# Aktywacja i instalacja
echo ""
echo "Instalacja zaleznosci..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install $PIP_INDEX -r requirements.txt

echo ""
echo "========================================"
echo "Instalacja zakonczona."
echo ""
echo "Aby uruchomic pipeline:"
echo "  1. Aktywuj venv:   source venv/bin/activate"
echo "  2. Umiesc dane:    skopiuj foldery TESS, RAVDESS, SAVEE, CREMA-D do data/"
echo "  3. Uruchom:        python run.py"
echo ""
echo "Dostepne opcje:"
echo "  python run.py                      - wszystkie 4 datasety x 5 modeli"
echo "  python run.py --combined           - tryb polaczonych danych (6 klas)"
echo "  python run.py --models dscnn,lstm  - tylko wybrane modele"
echo "  python run.py --datasets SAVEE     - tylko jeden dataset"
echo "========================================"
