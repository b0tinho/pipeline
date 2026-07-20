# SER Pipeline - TESS / RAVDESS / SAVEE / CREMA-D

Modularny pipeline PyTorch do przetwarzania, treningu i ewaluacji modeli
rozpoznawania emocji z mowy (Speech Emotion Recognition, SER).

**Kluczowe założenie projektu:** cztery obsługiwane zbiory danych
(TESS, RAVDESS, SAVEE, CREMA-D) **nigdy nie są łączone**. Pipeline
uruchamia PEŁNĄ KOMBINACJĘ dataset x model - każda z czterech
architektur (`dscnn`, `headfusion_acnn`, `lstm`, `hierarchical_crnn`)
jest trenowana i ewaluowana niezależnie na każdym z czterech zbiorów
(16 eksperymentów łącznie). Każda kombinacja ma własne: parsowanie
nazw plików, podział train/val/test, augmentację, model (z dopasowaną
do liczby klas warstwą wyjściową) oraz komplet wyników zapisywany w
`results/<NAZWA_DATASETU>/<NAZWA_MODELU>/`.

## Struktura projektu

```
pipeline/
├── data/                       # Surowe pliki .wav (do uzupełnienia przez użytkownika)
│   ├── TESS/
│   ├── RAVDESS/
│   ├── SAVEE/
│   └── CREMA-D/
├── results/                    # Generowane automatycznie podczas treningu
│   ├── TESS/<model>/
│   ├── RAVDESS/<model>/
│   ├── SAVEE/<model>/
│   └── CREMA-D/<model>/
├── ser_pipeline/
│   ├── __init__.py
│   ├── data_parsing.py         # Parsowanie nazw plików + mapowanie etykiet (1 na dataset)
│   ├── audio_processing.py     # Ładowanie audio, padding/truncation, MFCC / Mel-spektrogram
│   ├── augmentation.py         # White noise, pitch shift, time stretch (tylko train)
│   ├── dataset.py              # Stratified split, torch Dataset, DataLoadery
│   ├── model.py                # 4 architektury (DSCNN, ACNN, LSTM, hierarchiczny CRNN) + fabryka
│   ├── engine.py                # Pętla treningowa, early stopping, checkpointing, raporty
│   └── main.py                  # Orchestrator - uruchamia pełną kombinację dataset x model
├── run.py                      # Punkt wejścia: `python run.py`
├── requirements.txt
└── README.md
```

## Instalacja

```bash
pip install -r requirements.txt
```

## Przygotowanie danych

Pobierz i rozpakuj oficjalne wersje zbiorów danych, a następnie umieść
je (lub podlinkuj) w katalogu `data/`, zgodnie z domyślną konfiguracją
w `ser_pipeline/main.py` (`DATASET_ROOT_PATHS`). Struktura wewnętrzna
folderów nie ma znaczenia - pliki `.wav` są wyszukiwane rekurencyjnie -
liczy się wyłącznie **konwencja nazewnictwa plików** każdego zbioru:

| Dataset  | Przykładowa nazwa pliku          | Pozycja kodu emocji                          |
|----------|-----------------------------------|-----------------------------------------------|
| TESS     | `OAF_back_angry.wav`             | ostatni token po `_`                          |
| RAVDESS  | `03-01-06-01-02-01-12.wav`       | 3. pole (indeks 2) w numeracji myślnikowej    |
| SAVEE    | `DC_a01.wav`                     | litery bezpośrednio po `_`, przed numerem     |
| CREMA-D  | `1001_DFA_ANG_XX.wav`            | 3. pole (indeks 2) w numeracji podkreślnikowej|

Szczegółowe mapowanie oryginalnych kodów na czytelne nazwy emocji
(`Angry`, `Happy`, `Sad`, `Neutral`, `Fear`, `Disgust`, `Surprise`,
`Calm`) znajduje się w komentarzach i docstringach modułu
`ser_pipeline/data_parsing.py`. Warto zwrócić uwagę na specyficzne
przypadki:

- **TESS**: kod `ps` (pleasant surprise) mapowany jest na `Surprise`.
- **RAVDESS**: jako jedyny zbiór zawiera dodatkową klasę `Calm`. Domyślnie
  (`merge_calm_into_neutral=True`) jest ona łączona z `Neutral`, zgodnie
  z powszechną praktyką w literaturze SER - można to wyłączyć w
  `parse_ravdess_filename`, aby zachować oryginalne 8 klas.
- **SAVEE**: rozróżnienie `a` (anger) od `sa` (sadness) oraz `su`
  (surprise) wymaga dopasowania pełnego prefiksu literowego, nie
  pojedynczego znaku - stąd użycie wyrażenia regularnego.
- **CREMA-D**: jako jedyny zbiór nie zawiera klasy `Surprise` - to jego
  oryginalna, węższa taksonomia emocji (6 klas).

## Uruchomienie

```bash
python run.py
```

Domyślnie `main()` uruchamia PEŁNĄ kombinację: wszystkie 4 modele x
wszystkie 4 datasety (16 eksperymentów). Skrypt automatycznie pominie
(z ostrzeżeniem) dataset, dla którego nie znaleziono katalogu w
`data/`, i będzie kontynuował pracę dla pozostałych kombinacji. Błąd
w trakcie jednej kombinacji (dataset, model) nie przerywa przetwarzania
pozostałych.

Można ograniczyć zakres do wybranych modeli i/lub datasetów:

```python
from ser_pipeline.main import main

main(model_names=["dscnn", "lstm"])              # tylko 2 architektury, wszystkie datasety
main(dataset_names=["SAVEE"])                     # wszystkie modele, tylko SAVEE
main(model_names=["headfusion_acnn"], dataset_names=["TESS", "CREMA-D"])
```

Alternatywnie, pojedynczą kombinację (dataset, model) można
przetworzyć programowo:

```python
from ser_pipeline.audio_processing import AudioProcessingConfig
from ser_pipeline.augmentation import AugmentationConfig
from ser_pipeline.engine import TrainingConfig
from ser_pipeline.main import run_pipeline_for_dataset

run_pipeline_for_dataset(
    dataset_name="RAVDESS",
    root_dir="data/RAVDESS",
    model_name="hierarchical_crnn",
    audio_config=AudioProcessingConfig(feature_type="mfcc"),  # nadpisanie domyślnego Mel-spektrogramu
    augmentation_config=AugmentationConfig(),
    training_config=TrainingConfig(num_epochs=30),
)
```

## Wyniki

Dla każdej kombinacji (dataset, model), w
`results/<NAZWA_DATASETU>/<NAZWA_MODELU>/` zapisywane są:

- `best_model.pt` - wagi modelu z najlepszą epoką (najniższy val_loss).
- `training_log.csv` - historia strat/dokładności per epoka.
- `classification_report.json` - precision/recall/F1 per klasa (zbiór testowy).
- `confusion_matrix.png` - macierz pomyłek na zbiorze testowym.
- `label_mapping.json` - mapowanie indeksów klas na nazwy emocji.

## Konfiguracja

Główne parametry pipeline'u ustawiane są w `ser_pipeline/main.py::main()`:

- `AudioProcessingConfig` - częstotliwość próbkowania, długość nagrania,
  typ cech (`"melspectrogram"` **[domyślnie]** lub `"mfcc"`), liczba
  pasm Mel / współczynników MFCC. Mel-spektrogram jest domyślną
  konfiguracją, ponieważ zachowuje więcej informacji widmowej niż MFCC
  (który dodatkowo kompresuje widmo przez transformatę DCT), co zwykle
  daje lepsze wyniki przy architekturach CNN/CRNN.
- `AugmentationConfig` - prawdopodobieństwo i zakresy augmentacji
  (white noise / pitch shift / time stretch), stosowanej tylko do train.
- `TrainingConfig` - liczba epok, learning rate, cierpliwość early
  stoppingu, urządzenie obliczeniowe.

## Architektury modeli

Wszystkie modele znajdują się w jednym pliku `ser_pipeline/model.py` i
współdzielą ten sam kontrakt: wejście `(batch, 1, n_features, n_frames)`,
wyjście - tensor logitów `(batch, num_classes)`. Dzięki temu są w pełni
wymienne w pętli treningowej `engine.py`, a `num_classes` oraz
`input_freq_dim` są zawsze wyliczane dynamicznie na podstawie bieżącego
datasetu i konfiguracji audio.

| Klucz w `MODEL_REGISTRY` | Klasa                        | Architektura |
|---------------------------|-------------------------------|---------------|
| `"dscnn"`                 | `DSCNNEmotionClassifier`      | Deep Stride CNN (Wani et al., 2020) - 5 warstw konwolucyjnych ze stride=2 zamiast poolingu, zakończone globalnym adaptacyjnym poolingiem. |
| `"headfusion_acnn"`       | `HeadFusionACNN`              | ACNN z równoległymi ścieżkami konwolucyjnymi i mechanizmem uwagi Head Fusion (Xu et al., 2020). |
| `"lstm"`                  | `KumbharLSTMClassifier`       | Lekki model sekwencyjny: 1-warstwowy LSTM + głęboki klasyfikator Dense (Kumbhar et al., 2019). |
| `"hierarchical_crnn"`     | `HierarchicalMultiTaskCRNN`   | Hierarchiczny CRNN z uwagą i fuzją cech coarse/fine (na podst. 3D-HMTL); trenowany z połączoną funkcją straty coarse/fine (patrz niżej), ale `forward()` zwraca logity zadania precyzyjnego, więc pozostaje zgodny z jednozadaniową ewaluacją. |

Projekt NIE definiuje żadnego "domyślnego" modelu - orchestrator z
zasady uruchamia KAżDą z powyższych czterech architektur na KAżDYM z
czterech datasetów, aby porównać ich względną skuteczność (patrz
sekcja "Uruchomienie" powyżej).

Wszystkie cztery architektury (`dscnn`, `headfusion_acnn`, `lstm`,
`hierarchical_crnn`) zostały zaadaptowane z osobnych implementacji do
wspólnego interfejsu pipeline'u - zahardkodowane liczby klas i wymiary
wejściowe (np. obraz 256x256, sekwencja 39 MFCC, mapa 3x300x40)
zastąpiono parametrami dynamicznymi (`num_classes`, `input_freq_dim`)
lub operacjami niezależnymi od rozmiaru wejścia (globalny adaptacyjny
pooling). Szczegóły adaptacji każdego modelu opisane są w docstringach
w `ser_pipeline/model.py`.

### Połączona funkcja straty dla modeli wielozadaniowych (multi-task)

`HierarchicalMultiTaskCRNN` implementuje metodę `forward_multitask()`,
zwracającą parę `(coarse_logits, fine_logits)`. Ponieważ żaden z czterech
datasetów nie definiuje własnej hierarchii emocji, etykieta zgrubna
("coarse") jest wyprowadzana automatycznie z etykiety precyzyjnej według
podziału wg walencji (`data_parsing.COARSE_EMOTION_MAP`):

- **Negative**: Angry, Disgust, Fear, Sad
- **Positive**: Happy, Surprise
- **Neutral**: Neutral, Calm

`dataset.SpeechEmotionDataset` zwraca dla każdej próbki trójkę
`(features, fine_label, coarse_label)`. `engine._run_one_epoch`
automatycznie wykrywa modele z metodą `forward_multitask` (przez
`hasattr`) i podczas treningu/walidacji liczy połączoną funkcję straty:

```
L = multitask_alpha * L_coarse + (1 - multitask_alpha) * L_fine
```

Waga `multitask_alpha` (domyślnie `0.2`, zgodnie z optymalną wartością
wyznaczoną empirycznie w oryginalnej pracy o 3D-HMTL) jest konfigurowana
w `TrainingConfig.multitask_alpha`:

```python
training_config = TrainingConfig(multitask_alpha=0.3)
```

Modele bez metody `forward_multitask` ignorują etykietę zgrubną i
trenowane są zwykłą, jednozadaniową stratą `CrossEntropyLoss` - bez
żadnej zmiany zachowania względem poprzedniej wersji pipeline'u.
Ewaluacja testowa (`evaluate_and_report`) zawsze odbywa się względem
rzeczywistej, precyzyjnej taksonomii klas datasetu.
