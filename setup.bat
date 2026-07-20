@echo off
echo ========================================
echo SER Pipeline - Instalacja (Windows)
echo ========================================
echo.

:: Sprawdz czy Python jest dostepny
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [BLAD] Python nie znaleziony. Zainstaluj Python 3.10+ i dodaj do PATH.
    pause
    exit /b 1
)

:: Sprawdz czy CUDA GPU jest dostepna
set USE_CUDA=0
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo GPU NVIDIA wykryta - instalacja PyTorch z CUDA.
    set USE_CUDA=1
) else (
    echo Brak GPU NVIDIA lub sterownikow CUDA - instalacja PyTorch CPU.
)

:: Tworzenie venv
if not exist "venv\" (
    echo.
    echo Tworzenie srodowiska wirtualnego...
    python -m venv venv
    echo OK.
) else (
    echo Srodowisko wirtualne juz istnieje.
)

:: Aktywacja
call venv\Scripts\activate.bat

:: Instalacja PyTorch (GPU lub CPU)
echo.
echo Instalacja PyTorch...
pip install --upgrade pip --quiet

if %USE_CUDA% equ 1 (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
) else (
    pip install torch torchvision torchaudio --quiet
)

:: Instalacja reszty zaleznosci
echo.
echo Instalacja pozostalych zaleznosci...
pip install -r requirements.txt --quiet
echo.

:: Weryfikacja
echo Weryfikacja instalacji...
python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo [UWAGA] Weryfikacja PyTorch nie powiodla sie.
)

echo.
echo ========================================
echo Instalacja zakonczona.
echo.
echo Aby uruchomic pipeline:
echo   1. venv\Scripts\activate
echo   2. Skopiuj foldery TESS, RAVDESS, SAVEE, CREMA-D do data\
echo   3. python run.py
echo.
echo Dostepne opcje CLI:
echo   python run.py --help
echo ========================================
pause
