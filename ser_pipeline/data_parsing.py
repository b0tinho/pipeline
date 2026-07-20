"""
Ładowanie i parsowanie plików audio dla czterech niezależnych zbiorów
danych: TESS, RAVDESS, SAVEE, CREMA-D.

Każdy zbiór danych stosuje inną konwencję nazewnictwa plików oraz inny
oryginalny zestaw kodów emocji. Funkcje w tym module tłumaczą (mapują)
specyficzne dla danego zbioru kody na wspólny, czytelny słownik nazw
tekstowych (np. "Angry", "Happy", "Sad"), NIE łącząc przy tym danych
pomiędzy zbiorami - `load_dataset_dataframe` buduje i zwraca zawsze
JEDEN, niezależny DataFrame dla jednego, wskazanego datasetu.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# TESS - Toronto Emotional Speech Set
# ---------------------------------------------------------------------------
# Konwencja nazw plików: "{actor}_{word}_{emotion}.wav",
# np. "OAF_back_angry.wav" lub "YAF_witch_ps.wav".
# Aktorki: OAF (Older Adult Female), YAF (Younger Adult Female).
# Oryginalne kody emocji to ostatni token nazwy pliku (przed ".wav"):
#   angry, disgust, fear, happy, neutral, ps (pleasant surprise), sad
TESS_LABEL_MAP: Dict[str, str] = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "ps": "Surprise",  # "pleasant surprise" w oryginalnej taksonomii TESS
    "sad": "Sad",
}


def parse_tess_filename(file_path: Path) -> Optional[str]:
    """Wyciąga etykietę emocji z nazwy pliku ze zbioru TESS.

    Args:
        file_path: Ścieżka do pliku .wav.

    Returns:
        Czytelna nazwa emocji (np. "Angry") albo None, gdy nazwa pliku
        nie odpowiada znanej konwencji TESS.
    """
    stem = file_path.stem.lower()
    code = stem.split("_")[-1]
    return TESS_LABEL_MAP.get(code)


# ---------------------------------------------------------------------------
# RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song
# ---------------------------------------------------------------------------
# Konwencja nazw plików - siedem dwucyfrowych pól oddzielonych myślnikiem:
#   Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
#   np. "03-01-06-01-02-01-12.wav"
# Trzecie pole (indeks 2, liczony od zera) koduje emocję:
#   01=neutral, 02=calm, 03=happy, 04=sad, 05=angry,
#   06=fearful, 07=disgust, 08=surprised
RAVDESS_LABEL_MAP: Dict[str, str] = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise",
}


def parse_ravdess_filename(
    file_path: Path, merge_calm_into_neutral: bool = True
) -> Optional[str]:
    """Wyciąga etykietę emocji z nazwy pliku ze zbioru RAVDESS.

    Args:
        file_path: Ścieżka do pliku .wav.
        merge_calm_into_neutral: RAVDESS jest jedynym z czterech
            zbiorów posiadającym dodatkową klasę "Calm" (spokój), która
            nie występuje w TESS, SAVEE ani CREMA-D. Wiele prac z
            zakresu SER łączy tę klasę z "Neutral" ze względu na dużą
            trudność ich rozróżnienia akustycznego oraz aby ograniczyć
            liczbę klas do 7 kanonicznych emocji podstawowych. Domyślnie
            włączone (True); ustaw False, aby zachować oryginalne 8 klas.

    Returns:
        Czytelna nazwa emocji albo None, gdy nazwa pliku jest
        niepoprawna lub nierozpoznana.
    """
    parts = file_path.stem.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    label = RAVDESS_LABEL_MAP.get(emotion_code)
    if label == "Calm" and merge_calm_into_neutral:
        return "Neutral"
    return label


# ---------------------------------------------------------------------------
# SAVEE - Surrey Audio-Visual Expressed Emotion
# ---------------------------------------------------------------------------
# Konwencja nazw plików: "{actor}_{emotion_code}{number}.wav",
# np. "DC_a01.wav" lub "JE_su05.wav".
# Aktorzy: DC, JE, JK, KL. Kody emocji (litery bezpośrednio przed numerem):
#   a=anger, d=disgust, f=fear, h=happiness,
#   n=neutral, sa=sadness, su=surprise
SAVEE_LABEL_MAP: Dict[str, str] = {
    "a": "Angry",
    "d": "Disgust",
    "f": "Fear",
    "h": "Happy",
    "n": "Neutral",
    "sa": "Sad",
    "su": "Surprise",
}

_SAVEE_CODE_PATTERN = re.compile(r"^([a-zA-Z]+)")


def parse_savee_filename(file_path: Path) -> Optional[str]:
    """Wyciąga etykietę emocji z nazwy pliku ze zbioru SAVEE.

    Args:
        file_path: Ścieżka do pliku .wav.

    Returns:
        Czytelna nazwa emocji albo None, gdy nie rozpoznano formatu.
    """
    stem = file_path.stem
    if "_" not in stem:
        return None
    _, code_and_number = stem.split("_", maxsplit=1)
    match = _SAVEE_CODE_PATTERN.match(code_and_number.lower())
    if not match:
        return None
    return SAVEE_LABEL_MAP.get(match.group(1))


# ---------------------------------------------------------------------------
# CREMA-D - Crowd-sourced Emotional Multimodal Actors Dataset
# ---------------------------------------------------------------------------
# Konwencja nazw plików: "{ActorID}_{Sentence}_{Emotion}_{Level}.wav",
# np. "1001_DFA_ANG_XX.wav".
# Trzecie pole koduje emocję: ANG, DIS, FEA, HAP, NEU, SAD.
# Uwaga: CREMA-D, w przeciwieństwie do pozostałych zbiorów, nie zawiera
# klasy "Surprise" - jest to jego oryginalna, węższa taksonomia emocji.
CREMAD_LABEL_MAP: Dict[str, str] = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}


def parse_cremad_filename(file_path: Path) -> Optional[str]:
    """Wyciąga etykietę emocji z nazwy pliku ze zbioru CREMA-D.

    Args:
        file_path: Ścieżka do pliku .wav.

    Returns:
        Czytelna nazwa emocji albo None, gdy nie rozpoznano formatu.
    """
    parts = file_path.stem.split("_")
    if len(parts) < 3:
        return None
    return CREMAD_LABEL_MAP.get(parts[2].upper())


# Rejestr parserów - klucz to nazwa datasetu używana w całym pipeline'ie.
# Dzięki temu rejestrowi `load_dataset_dataframe` może pozostać w pełni
# generyczna względem konkretnego datasetu.
DATASET_PARSERS: Dict[str, Callable[[Path], Optional[str]]] = {
    "TESS": parse_tess_filename,
    "RAVDESS": parse_ravdess_filename,
    "SAVEE": parse_savee_filename,
    "CREMA-D": parse_cremad_filename,
}


# ---------------------------------------------------------------------------
# Mapowanie emocji precyzyjnych ("fine") na kategorie zgrubne ("coarse")
# ---------------------------------------------------------------------------
# Wykorzystywane przez `dataset.build_dataloaders` do automatycznego
# wyprowadzenia pomocniczej etykiety zgrubnej - potrzebnej do treningu
# modeli hierarchicznych/wielozadaniowych (patrz `model.HierarchicalMultiTaskCRNN`
# oraz połączona funkcja straty w `engine.train_model`). Żaden z czterech
# datasetów nie definiuje własnej hierarchii emocji, więc etykieta zgrubna
# jest wyprowadzana deterministycznie z etykiety precyzyjnej - bez potrzeby
# dodatkowej ręcznej anotacji - na podstawie ugruntowanego w literaturze SER
# podziału emocji według walencji (valence):
#   - Negative: Angry, Disgust, Fear, Sad
#   - Positive: Happy, Surprise
#   - Neutral:  Neutral, Calm
COARSE_EMOTION_MAP: Dict[str, str] = {
    "Angry": "Negative",
    "Disgust": "Negative",
    "Fear": "Negative",
    "Sad": "Negative",
    "Happy": "Positive",
    "Surprise": "Positive",
    "Neutral": "Neutral",
    "Calm": "Neutral",
}


# ---------------------------------------------------------------------------
# Mapowanie aktorów/mówców na płeć (gender) - wyprowadzane automatycznie
# z nazw plików każdego datasetu, BEZ potrzeby ręcznej anotacji.
# ---------------------------------------------------------------------------
# Każdy z czterech datasetów ma odrębną konwencję identyfikacji
# mówcy/aktora w nazwie pliku, a płeć jest znana z oficjalnej
# dokumentacji zbiorów (lub wprost z prefixu nazwy dla TESS).
#
# TESS:   tylko Female (starsza / młodsza aktorka) - OAF, YAF.
# RAVDESS: aktorzy 1-12 = Female, 13-24 = Male (dokumentacja).
# SAVEE:  DC (David Cowan), KL = Male; JE (James), JK = Female.
# CREMA-D: parzyste ActorID = Female, nieparzyste = Male.

TESS_GENDER_MAP: Dict[str, str] = {"OAF": "Female", "YAF": "Female"}

SAVEE_GENDER_MAP: Dict[str, str] = {
    "DC": "Male",
    "JE": "Female",
    "JK": "Female",
    "KL": "Male",
}


def parse_tess_gender(file_path: Path) -> Optional[str]:
    """Wyciąga płeć z nazwy pliku TESS (pierwszy token prefixu aktora).

    Prefiks pierwszego tokena: OAF/YAF -> Female. Dla nietypowych
    nazw plikow (np. \"OA_bite_neutral.wav\" zamiast \"OAF_...\")
    fallback do nazwy folderu nadrzędnego.
    """
    stem = file_path.stem.upper()
    parts = stem.split("_")
    if len(parts) >= 2:
        result = TESS_GENDER_MAP.get(parts[0])
        if result:
            return result
    # Fallback: sprawdź prefiks w nazwie folderu (np. "OAF_neutral").
    parent = file_path.parent.name.upper()
    for prefix in TESS_GENDER_MAP:
        if parent.startswith(prefix):
            return TESS_GENDER_MAP[prefix]
    return None


def parse_ravdess_gender(file_path: Path) -> Optional[str]:
    """Wyciąga płeć z nazwy pliku RAVDESS (siódme pole - numer aktora).

    Aktorzy 1-12 = Female, 13-24 = Male (oficjalna dokumentacja RAVDESS).
    """
    parts = file_path.stem.split("-")
    if len(parts) < 7:
        return None
    try:
        actor_num = int(parts[6])
    except (ValueError, IndexError):
        return None
    return "Female" if 1 <= actor_num <= 12 else "Male"


def parse_savee_gender(file_path: Path) -> Optional[str]:
    """Wyciąga płeć z nazwy pliku SAVEE (pierwszy token - dwuliterowy kod aktora)."""
    stem = file_path.stem.upper()
    if "_" not in stem:
        return None
    actor_code = stem.split("_")[0]
    return SAVEE_GENDER_MAP.get(actor_code)


def parse_cremad_gender(file_path: Path) -> Optional[str]:
    """Wyciąga płeć z nazwy pliku CREMA-D (pierwszy token - ID aktora).

    Parzyste ActorID = Female, nieparzyste = Male (oficjalna dokumentacja CREMA-D).
    """
    parts = file_path.stem.split("_")
    if len(parts) < 1:
        return None
    try:
        actor_id = int(parts[0])
    except (ValueError, IndexError):
        return None
    return "Female" if actor_id % 2 == 0 else "Male"


# Rejestr parserów płci - klucz to nazwa datasetu.
GENDER_PARSERS: Dict[str, Callable[[Path], Optional[str]]] = {
    "TESS": parse_tess_gender,
    "RAVDESS": parse_ravdess_gender,
    "SAVEE": parse_savee_gender,
    "CREMA-D": parse_cremad_gender,
}


def map_to_coarse_emotion(fine_emotion: str) -> str:
    """Mapuje precyzyjną etykietę emocji na kategorię zgrubną (walencja).

    Args:
        fine_emotion: Czytelna nazwa emocji precyzyjnej (np. "Angry"),
            będąca wynikiem jednego z parserów w tym module.

    Returns:
        Nazwa kategorii zgrubnej: "Negative", "Neutral" lub "Positive".

    Raises:
        KeyError: Gdy `fine_emotion` nie znajduje się w `COARSE_EMOTION_MAP`
            (nie powinno się zdarzyć dla etykiet pochodzących z funkcji
            `parse_*_filename` zdefiniowanych w tym module).
    """
    return COARSE_EMOTION_MAP[fine_emotion]


def load_dataset_dataframe(dataset_name: str, root_dir: str) -> pd.DataFrame:
    """Buduje niezależną ramkę danych (ścieżka pliku + etykieta) dla
    JEDNEGO, wskazanego zbioru danych.

    Funkcja celowo NIE łączy danych z różnych zbiorów - jest wywoływana
    osobno dla każdego z: TESS, RAVDESS, SAVEE, CREMA-D, zwracając za
    każdym razem oddzielny, niezależny DataFrame z własnym zestawem
    unikalnych klas emocji.

    Args:
        dataset_name: Jedna z wartości: "TESS", "RAVDESS", "SAVEE",
            "CREMA-D".
        root_dir: Katalog główny z plikami .wav danego zbioru
            (przeszukiwany rekurencyjnie, dzięki czemu obsługuje zarówno
            płaską strukturę plików, jak i podział na foldery aktorów).

    Returns:
        DataFrame z kolumnami: ["file_path", "emotion", "gender", "dataset"].

    Raises:
        ValueError: Gdy podano nieobsługiwaną nazwę datasetu.
        FileNotFoundError: Gdy katalog root_dir nie istnieje.
        RuntimeError: Gdy w katalogu nie znaleziono żadnych plików
            odpowiadających znanej konwencji nazewnictwa.
    """
    if dataset_name not in DATASET_PARSERS:
        raise ValueError(
            f"Nieobsługiwany dataset: {dataset_name}. "
            f"Dostępne opcje: {list(DATASET_PARSERS.keys())}"
        )

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Katalog nie istnieje: {root_dir}")

    parser_fn = DATASET_PARSERS[dataset_name]
    gender_parser_fn = GENDER_PARSERS.get(dataset_name)
    records = []
    for wav_path in sorted(root_path.rglob("*.wav")):
        emotion_label = parser_fn(wav_path)
        if emotion_label is None:
            continue  # Pomijamy pliki o nierozpoznanej konwencji nazw.
        gender_label = gender_parser_fn(wav_path) if gender_parser_fn else None
        records.append(
            {
                "file_path": str(wav_path),
                "emotion": emotion_label,
                "gender": gender_label if gender_label else "Unknown",
                "dataset": dataset_name,
            }
        )

    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        raise RuntimeError(
            f"Nie znaleziono żadnych poprawnych plików .wav dla datasetu "
            f"'{dataset_name}' w katalogu '{root_dir}'. Sprawdź ścieżkę "
            f"oraz konwencję nazewnictwa plików."
        )
    return dataframe.reset_index(drop=True)
