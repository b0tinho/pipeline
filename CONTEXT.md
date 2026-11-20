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

Dane: `data/{TESS,RAVDESS,SAVEE,CREMA-D}` (TESS: 2800 plików — zduplikowany podfolder `TESS Toronto emotional speech set data` usunięty 2026-11-15). Pipeline zapisuje do `results/<DATASET>/<MODEL>/` oraz `results/combined/`; wyniki z maszyny zdalnej są kopiowane do `wyniki/{per_dataset,combined,combined_filtry,combined_deltas}/<MODEL>/`.

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
- **Płeć z nazw plików**: TESS (OAF/YAF=Female), RAVDESS (nieparzysty aktor=M, parzysty=F), SAVEE (wszyscy 4 mówcy=M), CREMA-D (zbiór ID aktorek `CREMAD_FEMALE_ACTOR_IDS` z oficjalnego `VideoDemographics.csv`; 48 M / 43 F).
- **Ocena głowicy płci**: dla modeli z `gender_head` `evaluate_and_report` zapisuje dodatkowo `gender_report.json` (accuracy, macierz pomyłek, raport per klasa płci na teście). W zbiorach jednopłciowych (TESS, SAVEE) druga klasa oznaczona jako „brak w zbiorze”.
- **CLI `--epochs` / `--batch-size`**: przekazywane do `main(num_epochs=..., batch_size=...)` (domyślnie 50 / 32).
- **GPU**: `num_workers=12`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`, `cudnn.benchmark=True` (w `set_global_seed`).

## Wyniki (accuracy testowe)

**Per-dataset** (7 klas oprócz CREMA-D=6; wyniki z 2026-07-08, `gender_aware` z 2026-11-19):
- TESS: ~0.995 (dscnn/hier/headfusion) — **nierzetelne** (tylko 2 aktorki, wyciek mówcy)
- RAVDESS: dscnn **0.607**, headfusion 0.551, lstm 0.505
- SAVEE: headfusion **0.653**, dscnn 0.514
- CREMA-D: headfusion **0.609**, dscnn 0.594
- `gender_aware` (po poprawce płci): TESS 0.983, RAVDESS 0.565, SAVEE 0.458, CREMA-D 0.441 (przed: 0.921 / 0.468 / 0.444 / 0.482). CREMA-D niedouczony: train_acc 0.44, early stopping w ep. 25 (wahania val_loss), Fear recall 0.01.

**Mega-zbiór** (wszystkie warianty na tym samym teście 1728 próbek):

| Model | surowe | filtry | deltas |
|---|---|---|---|
| hierarchical_crnn | **0.704** | 0.596 | **0.710** |
| dscnn | 0.664 | **0.679** | 0.666 |
| headfusion_acnn | 0.655 | 0.623 | 0.647 |
| gender_aware* | 0.663 | 0.660 | 0.635 |
| lstm | 0.573 | 0.573 | 0.549 |

\* `gender_aware` powtórzony 2026-11-19 po poprawce etykiet płci (librosa 1.0.1); przed poprawką: 0.653 / 0.629 / 0.584. We wszystkich 3 wariantach najlepsza val_loss w epoce 47–49 z 50 — model nie zbiegł (limit epok).

**Wnioski**: (1) surowe dane najlepsze; (2) filtry VAD+preemfaza **szkodzą** większości modeli (hier −10.8 p.p.), ale `gender_aware` praktycznie nie (−0.3 p.p.), a `dscnn` zyskuje (+1.5 p.p.); (3) delta/delta-delta **neutralne** (`gender_aware` −2.8 p.p.; dawne −6.9 p.p. było w dużej mierze skutkiem błędnych etykiet płci); (4) `hierarchical_crnn` najlepszy na surowych, `gender_aware` ≈ `dscnn` (0.663 vs 0.664), `lstm` najsłabszy.
**Uwaga o wariancji**: TESS dla `gender_aware` zmienił się o +6.2 p.p., choć etykiety TESS się nie zmieniły — różnice rzędu kilku p.p. między pojedynczymi przebiegami nie są rozstrzygające (patrz kolejne kroki, pkt 5).

## Napotkane problemy i rozwiązania

1. **Wariancja podziału**: stary `combined` (2088 test) vs nowy (1728) dawał ~6 p.p. różnicy. Weryfikacja starych checkpointów na nowym teście wykazała, że to **efekt podziału** (ten sam model traci 5–7.6 p.p. przy zmianie testu), nie regresja modelu.
2. **Błędne etykiety płci** (2026-11-15) — RAVDESS (1–12=F zamiast parzysty=F, ~50% błędnych), SAVEE (JE/JK jako F, a wszyscy to M), CREMA-D (reguła parzystości błędna dla 42/91 aktorów, 46% nagrań). Poprawione w `data_parsing.py`; `gender_aware` powtórzony 2026-11-19 we wszystkich wariantach (pozostałe modele nie używają płci, podział stratyfikowany tylko po emocji).
3. **Nietypowy plik TESS** `OA_bite_neutral.wav` (2-literowy prefix) — fallback do nazwy folderu w `parse_tess_gender`.
4. **LSTM bez obsługi 3 kanałów** — dodano `input_channels` i konkatenację kanałów w `forward`.
5. **Colorbar najeżdżał na spektrogram** — dedykowana oś `fig.add_axes([0.87, ...])`.
6. **`setup.bat` błąd `ensurepip`** (Python ze Store) — fallback `--without-pip` + `get-pip.py`.
7. **Niedziałające `--epochs`/`--batch-size` w `run.py`** — ustawiały atrybuty klasy `TrainingConfig`, a `main()` miał na sztywno 50/32. Naprawione (2026-11-19): wartości idą parametrami `main()`.
8. **Ostrzeżenie numba `invalid value encountered in cast`** (librosa 1.0.1) — pochodzi z `phase_vocoder` → `util.phasor` przy pitch shift / time stretch. Niegroźne: wynik augmentacji skończony, korelacja z librosa 0.11 = 0.9999996, Mel-spektrogram identyczny (różnica ≤ 1.5e-5 dB).
9. **Nieprzypięte wersje** — `librosa>=0.10.0` dało na maszynie zdalnej librosa 1.0.1. Przypięto `librosa==1.0.1`, `numba==0.68.1`, `numpy==2.5.5` w `requirements.txt` i `pyproject.toml` (pozostałe pakiety nadal `>=`).

## Konwencje i decyzje projektowe

- Zawsze 4-tuple z `SpeechEmotionDataset`: `(features, fine_label, coarse_label, gender_label)`.
- Etykieta coarse = walencja (mapa `COARSE_EMOTION_MAP` w `data_parsing.py`).
- `build_model(model_name, num_classes, input_freq_dim, input_channels, **kwargs)`.
- Augmentacja tylko dla train; deterministyczna kolejność: VAD → preemfaza → augmentacja → pad → mel.
- Wyniki: `results/<DATASET>/<MODEL>/{best_model.pt, training_log.csv, classification_report.json, confusion_matrix.png, label_mapping.json}` + `gender_report.json` dla `gender_aware`.

## Kolejne kroki (rekomendowane)

1. **Leave-one-speaker-out split** (zamiast losowego stratyfikowanego) — rzetelna ocena generalizacji; obecnie TESS ~99% to artefakt wycieku mówcy.
2. **Class weighting** dla RAVDESS (Neutral 288 vs reszta 192).
3. **Strojenie `multitask_alpha`** dla `hierarchical_crnn` (grid search).
4. **Eksperyment ablacyjny** VAD vs preemfaza (osobno), bo oba razem szkodzą.
5. **Uśrednianie po wielu seedach** (3–5) — pojedyncze ziarno daje wariancję 3–8 p.p.; priorytet: `gender_aware` na combined, RAVDESS, CREMA-D (seed jest stałą `RANDOM_SEED` w `main.py` — brak flagi CLI).
6. **Multi-GPU (DDP)** — obecnie jedna karta.
7. **Dłuższy trening na combined** (`--epochs 80–100`) dla `gender_aware` i `hierarchical_crnn` — oba dochodzą do limitu 50 epok.
8. **Przypiąć pozostałe pakiety** (soundfile, scikit-learn, pandas, matplotlib) wg `pip freeze` z maszyny zdalnej.

## Jak uruchomić

```bash
python run.py                          # grid: 4 datasety × 5 modeli
python run.py --combined               # mega-zbiór, 6 klas
python run.py --combined --deltas      # + 3 kanały
python run.py --preprocessed           # dane po VAD+preemfazie
python run.py --models dscnn --datasets SAVEE
python run.py --combined --models gender_aware --epochs 100
python preprocess_data.py              # preprocessing offline (raz)
```

Zależności: `requirements.txt`; PyTorch dobierany przez `setup.sh`/`setup.bat` (CUDA 11.8 lub CPU).
