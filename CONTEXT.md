# Handoff — Pipeline SER (Speech Emotion Recognition)

## Cel projektu

Modularny pipeline PyTorch do rozpoznawania emocji z mowy na 4 zbiorach danych (**TESS, RAVDESS, SAVEE, CREMA-D**), z żelazną zasadą: **datasety nie są łączone** w treningu per-dataset. Dodatkowo tryb `combined` (mega-zbiór) do oceny generalizacji.

## Stan repozytorium

Ścieżka: `C:\Users\Admin\Documents\Github\pipeline`

```
ser_pipeline/
├── data_parsing.py        # parsery nazw plików → emocja + płeć (per dataset)
├── audio_processing.py    # load, pad/truncate, MFCC/Mel-spektrogram, delta
├── augmentation.py        # white noise, pitch shift, time stretch (tylko train)
├── dataset.py             # stratified split, SpeechEmotionDataset (4-tuple), build_dataloaders + build_combined_dataloaders
├── model.py               # 5 architektur + MODEL_REGISTRY + build_model()
├── engine.py              # pętla treningowa, early stopping, multi-task loss, raporty
├── main.py                # orchestrator: grid dataset×model, tryb combined, --preprocessed, --deltas
├── preprocessing.py       # offline VAD + preemfaza → data_preprocessed/
└── spectrogram_analysis.py# wizualizacja spektrogramów i łańcuchów filtrów
run.py                     # CLI (argparse): --models --datasets --combined --preprocessed --deltas --epochs --batch-size
preprocess_data.py         # CLI do preprocessingu offline
setup.bat / setup.sh / install.bat / install.sh   # instalacja (venv + zależności + CUDA detekcja)
report.tex                 # pełny raport LaTeX (aktualny)
requirements.txt, pyproject.toml, .gitignore
```

Dane: `data/{TESS,RAVDESS,SAVEE,CREMA-D}`. Wyniki: `results/<DATASET>/<MODEL>/` oraz `results/combined*`.

## Architektury (5, w `model.py`, klucze `MODEL_REGISTRY`)

| Klucz | Klasa | Parametry (6 kl., 128 mel, 1 kanał) |
|---|---|---|
| `dscnn` | DSCNNEmotionClassifier | 2.20 M |
| `headfusion_acnn` | HeadFusionACNN | 0.17 M |
| `lstm` | KumbharLSTMClassifier | 0.14 M (0.27 M @3 kanały) |
| `hierarchical_crnn` | HierarchicalMultiTaskCRNN | 13.00 M |
| `gender_aware` | GenderAwareSERModel | 1.15 M |

Wspólny kontrakt: wejście `(B, C, F, T)` (C=1 statyka lub C=3 z delta), wyjście logity `(B, num_classes)`. Modele multi-task (`hierarchical_crnn`, `gender_aware`) mają `forward_multitask()`.

## Kluczowe zaimplementowane funkcje

- **Grid per-dataset**: 4 datasety × 5 modeli = 20 eksperymentów, wyniki w `results/<DATASET>/<MODEL>/`.
- **Tryb combined**: 6 wspólnych klas (bez Surprise, bo CREMA-D jej nie ma), split per-dataset **przed** konkatenacją (ochrona przed speaker leakage).
- **Delta + delta-delta**: flaga `--deltas`, 3 kanały wejściowe, standaryzacja per-kanał.
- **Preprocessing offline**: VAD + preemfaza (`preprocess_data.py` → `data_preprocessed/`, flaga `--preprocessed`).
- **Multi-task loss**: `hierarchical_crnn` → target walencji (Negative/Neutral/Positive); `gender_aware` → target płci (wywodzony z nazwy pliku). Wykrywanie typu aux target w `engine.py` po liczbie neuronów `gender_head`.
- **Płeć z nazw plików**: TESS (OAF/YAF=Female), RAVDESS (aktorzy 1–12=F, 13–24=M), SAVEE (DC=KL=M, JE=JK=F), CREMA-D (parzyste ID=F).
- **GPU**: `num_workers=12`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`, `cudnn.benchmark=True` (w `set_global_seed`).

## Wyniki (accuracy testowe)

**Per-dataset** (7 klas oprócz CREMA-D=6):
- TESS: ~0.995 (dscnn/hier/headfusion) — **nierzetelne** (tylko 2 aktorki, wyciek mówcy)
- RAVDESS: dscnn **0.607**, headfusion 0.551, lstm 0.505
- SAVEE: headfusion **0.653**, dscnn 0.514
- CREMA-D: headfusion **0.609**, dscnn 0.594

**Mega-zbiór** (wszystkie warianty na tym samym teście 1728 próbek):

| Model | surowe | filtry | deltas |
|---|---|---|---|
| hierarchical_crnn | **0.704** | 0.596 | **0.710** |
| dscnn | 0.664 | **0.679** | 0.666 |
| headfusion_acnn | 0.655 | 0.623 | 0.647 |
| gender_aware | 0.653 | 0.629 | 0.584 |
| lstm | 0.573 | 0.573 | 0.549 |

**Wnioski**: (1) surowe dane najlepsze; (2) filtry VAD+preemfaza **szkodzą** (hier −10.8 p.p.); (3) delta/delta-delta **neutralne** (gender_aware −6.9 p.p.); (4) `dscnn` najodporniejszy, `hierarchical_crnn` najlepszy na surowych, `lstm` najsłabszy.

## Napotkane problemy i rozwiązania

1. **Wariancja podziału**: stary `combined` (2088 test) vs nowy (1728) dawał ~6 p.p. różnicy. Weryfikacja starych checkpointów na nowym teście wykazała, że to **efekt podziału** (ten sam model traci 5–7.6 p.p. przy zmianie testu), nie regresja modelu.
2. **Błędne etykiety płci SAVEE** (JE/KL zamienione) — poprawione.
3. **Nietypowy plik TESS** `OA_bite_neutral.wav` (2-literowy prefix) — fallback do nazwy folderu w `parse_tess_gender`.
4. **LSTM bez obsługi 3 kanałów** — dodano `input_channels` i konkatenację kanałów w `forward`.
5. **Colorbar najeżdżał na spektrogram** — dedykowana oś `fig.add_axes([0.87, ...])`.
6. **`setup.bat` błąd `ensurepip`** (Python ze Store) — fallback `--without-pip` + `get-pip.py`.
7. **Niedziałające `--epochs`/`--batch-size` w `run.py`** — ustawiały nieistniejące atrybuty `TrainingConfig`; nie naprawione (obecnie `main()` ma na sztywno `num_epochs=50`, `batch_size=32`).

## Konwencje i decyzje projektowe

- Zawsze 4-tuple z `SpeechEmotionDataset`: `(features, fine_label, coarse_label, gender_label)`.
- Etykieta coarse = walencja (mapa `COARSE_EMOTION_MAP` w `data_parsing.py`).
- `build_model(model_name, num_classes, input_freq_dim, input_channels, **kwargs)`.
- Augmentacja tylko dla train; deterministyczna kolejność: VAD → preemfaza → augmentacja → pad → mel.
- Wyniki: `results/<DATASET>/<MODEL>/{best_model.pt, training_log.csv, classification_report.json, confusion_matrix.png, label_mapping.json}`.

## Kolejne kroki (rekomendowane)

1. **Leave-one-speaker-out split** (zamiast losowego stratyfikowanego) — rzetelna ocena generalizacji; obecnie TESS ~99% to artefakt wycieku mówcy.
2. **Class weighting** dla RAVDESS (Neutral 288 vs reszta 192).
3. **Strojenie `multitask_alpha`** dla `hierarchical_crnn` (grid search).
4. **Eksperyment ablacyjny** VAD vs preemfaza (osobno), bo oba razem szkodzą.
5. **Uśrednianie po wielu seedach** (3–5) — pojedyncze ziarno daje wariancję 3–8 p.p.
6. **Multi-GPU (DDP)** — obecnie jedna karta.
7. **Naprawa `--epochs`/`--batch-size`** w `run.py` (przekazać przez parametry `main()` zamiast atrybutów klasy).

## Jak uruchomić

```bash
python run.py                          # grid: 4 datasety × 5 modeli
python run.py --combined               # mega-zbiór, 6 klas
python run.py --combined --deltas      # + 3 kanały
python run.py --preprocessed           # dane po VAD+preemfazie
python run.py --models dscnn --datasets SAVEE
python preprocess_data.py              # preprocessing offline (raz)
```

Zależności: `requirements.txt`; PyTorch dobierany przez `setup.sh`/`setup.bat` (CUDA 11.8 lub CPU).
