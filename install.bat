@echo off
setlocal enabledelayedexpansion

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
    echo Brak GPU NVIDIA - instalacja PyTorch CPU.
)

:: Tworzenie venv (z fallbackiem dla Python bez ensurepip)
if not exist "venv\Scripts\python.exe" (
    echo.
    echo Tworzenie srodowiska wirtualnego...

    :: Proba 1: standardowe tworzenie venv
    python -m venv venv 2>nul
    if exist "venv\Scripts\python.exe" (
        echo OK.
        goto :venv_done
    )

    :: Proba 2: venv z --without-pip + reczne bootstrapping pip
    echo Standardowe venv nie powiodlo sie, proba z --without-pip...
    rmdir /s /q venv 2>nul
    python -m venv venv --without-pip
    if not exist "venv\Scripts\python.exe" (
        echo [BLAD] Nie udalo sie utworzyc venv.
        echo Uzyj pelnej instalacji Pythona z python.org, a nie ze Microsoft Store.
        pause
        exit /b 1
    )

    :: Pobierz i zainstaluj pip recznie
    echo Pobieranie pip...
    venv\Scripts\python.exe -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')" 2>nul
    if exist "get-pip.py" (
        venv\Scripts\python.exe get-pip.py --quiet
        del get-pip.py
    ) else (
        echo [BLAD] Nie udalo sie pobrac pip. Sprawdz polaczenie internetowe.
        pause
        exit /b 1
    )
    echo OK (pip zainstalowany recznie).
) else (
    echo Srodowisko wirtualne juz istnieje.
)
:venv_done

:: Sprawdz czy aktywacja dziala
if not exist "venv\Scripts\activate.bat" (
    echo [BLAD] Brak pliku activate.bat - venv mogl nie zostac poprawnie utworzony.
    pause
    exit /b 1
)

:: Aktywacja
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [BLAD] Aktywacja venv nie powiodla sie.
    pause
    exit /b 1
)

:: Instalacja PyTorch (GPU lub CPU)
echo.
echo Instalacja PyTorch...
pip install --upgrade pip --quiet

if %USE_CUDA% equ 1 (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
) else (
    pip install torch torchvision torchaudio --quiet
)
if %errorlevel% neq 0 (
    echo [UWAGA] Instalacja PyTorch nie powiodla sie.
    echo Sprobuj recznie: pip install torch
)

:: Instalacja reszty zaleznosci
echo.
echo Instalacja pozostalych zaleznosci...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [UWAGA] Instalacja niektorych zaleznosci nie powiodla sie.
)

:: Weryfikacja
echo.
echo Weryfikacja instalacji...
python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo [UWAGA] Weryfikacja PyTorch nie powiodla sie - sprawdz czy torch zostal poprawnie zainstalowany.
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
