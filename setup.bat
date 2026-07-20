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

:: Tworzenie venv
if not exist "venv\" (
    echo Tworzenie srodowiska wirtualnego...
    python -m venv venv
    echo OK.
) else (
    echo Srodowisko wirtualne juz istnieje.
)

:: Aktywacja i instalacja zaleznosci
echo.
echo Instalacja zaleznosci...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo.

echo ========================================
echo Instalacja zakonczona.
echo.
echo Aby uruchomic pipeline:
echo   1. Aktywuj venv:   venv\Scripts\activate
echo   2. Umiesc dane:    skopiuj foldery TESS, RAVDESS, SAVEE, CREMA-D do data/
echo   3. Uruchom:        python run.py
echo.
echo Dostepne opcje:
echo   python run.py                      - wszystkie 4 datasety x 5 modeli
echo   python run.py --combined           - tryb polaczonych danych (6 klas)
echo   python run.py --models dscnn,lstm  - tylko wybrane modele
echo ========================================
pause
