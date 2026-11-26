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
run.py                     # CLI (argparse): --models --datasets --combined --preprocessed --deltas --epochs --batch-size --seed --deterministic --patience
preprocess_data.py         # CLI do preprocessingu offline
aggregate_results.py       # średnia ± odch. std. z wielu seedów (tabela + CSV)
trening_patience15.bat     # pełny przebieg patience 15 / 150 epok na maszynie zdalnej (Windows cmd)
setup.bat / setup.sh / install.bat / install.sh   # instalacja (venv + zależności + CUDA detekcja)
report.tex                 # pełny raport LaTeX (aktualny)
requirements.txt, pyproject.toml, .gitignore
```

Dane: `data/{TESS,RAVDESS,SAVEE,CREMA-D}` (TESS: 2800 plików — zduplikowany podfolder `TESS Toronto emotional speech set data` usunięty 2026-11-15). Pipeline zapisuje do `results/per_dataset<WARIANT>/<DATASET>/<MODEL>/` i `results/combined<WARIANT>/<MODEL>/`, gdzie `<WARIANT>` = `` (surowe), `_filtry` (`--preprocessed`), `_deltas` (`--deltas`), `_filtry_deltas` (obie), a przy `--patience` ≠ 7 dodatkowo `_patience<N>` (np. `combined_filtry_patience15`) — funkcja `results_group_dir` w `main.py` (od 2026-11-20; wcześniej wszystkie warianty szły do jednego katalogu i nadpisywały się). Układ jest taki sam jak w `wyniki/`, do którego kopiowane są wyniki z maszyny zdalnej. Wyniki z 3 seedów (2026-11-21) leżą w `wyniki_seedy/{combined,combined_deltas,combined_filtry}/<MODEL>/seed_{1,2,3}/`.

## Architektury (5, w `model.py`, klucze `MODEL_REGISTRY`)

| Klucz | Klasa | Parametry (6 kl., 128 mel, 1 kanał) |
|---|---|---|
| `dscnn` | DSCNNEmotionClassifier | 2.20 M |
| `headfusion_acnn` | HeadFusionACNN | 0.17 M |
| `lstm` | KumbharLSTMClassifier | 0.14 M (0.27 M @3 kanały) |
| `hierarchical_crnn` | HierarchicalMultiTaskCRNN | 13.00 M |
| `gender_aware` | GenderAwareSERModel | 1.15 M |
| `gender_aware_detach` | GenderAwareDetach | 1.15 M |
| `gender_aware_att` | GenderAwareAtt | 1.17 M |
| `gender_aware_film` | GenderAwareFiLM | 1.17 M |
| `gender_aware_att_nogender` | GenderAwareAttNoGender | 1.16 M |

Warianty `gender_aware_*` (2026-11-25, narastająco): `_detach` — prawdopodobieństwa płci odcięte od gradientu straty emocji (`.detach()`); `_att` — + `AttentionLayer` zamiast ostatniego kroku LSTM; `_film` — + fuzja FiLM (płeć → γ, β po 128, `cechy·(1+γ)+β`, zerowa inicjalizacja); `_att_nogender` — kontrola dla `_att` bez zadania płci i bez fuzji (brak `gender_head`/`forward_multitask`, więc zwykła strata emocji). Wyjściowy `gender_aware` daje przy tym samym seedzie identyczne wagi i wyjścia jak przed zmianą (sprawdzone). Ablacja: `trening_gender_warianty.bat` (combined surowe, seedy 1–3, patience 15, 150 epok, deterministycznie → `results/combined_patience15/`).

**Wyniki ablacji** (2026-11-26, `wyniki_seedy/combined_patience15/`, test acc średnia ± std | średnie val_acc w najlepszej epoce | najlepsza epoka): `gender_aware` 0.659 ± 0.017 | 0.666 | 62; `_detach` 0.668 ± 0.006 | 0.675 | 55; `_att` 0.705 ± 0.015 | 0.695 | 19; `_film` 0.705 ± 0.008 | 0.714 | 23; `_att_nogender` 0.703 ± 0.013 | 0.704 | 27 (dla porównania hier 0.700 ± 0.028 przy 13 M parametrów). Głowica płci 0.97–0.99 we wszystkich wariantach. Wniosek: cały zysk (+4.6 p.p.) daje uwaga zamiast ostatniego kroku LSTM; fuzja płci (concat ani FiLM) nie daje mierzalnej poprawy względem kontroli bez płci (≤1 p.p., w szumie). Detach nie zmienił skuteczności głowicy płci na surowych danych. Uwaga: val_loss modeli wielozadaniowych zawiera stratę płci — między modelami porównywać val_acc.
Następny krok: `trening_gender_film.bat` — `_film` i `_att_nogender` na filtrach i deltach (seedy 1–3) + powtórka na surowych (`gender_aware`, `_film`, `_att_nogender`) dla `emotion_by_gender`.

Wspólny kontrakt: wejście `(B, C, F, T)` (C=1 statyka lub C=3 z delta), wyjście logity `(B, num_classes)`. Modele multi-task (`hierarchical_crnn`, `gender_aware`) mają `forward_multitask()`.

## Kluczowe zaimplementowane funkcje

- **Grid per-dataset**: 4 datasety × 5 modeli = 20 eksperymentów, wyniki w `results/per_dataset/<DATASET>/<MODEL>/`.
- **Tryb combined**: 6 wspólnych klas (bez Surprise, bo CREMA-D jej nie ma), split per-dataset **przed** konkatenacją (ochrona przed speaker leakage).
- **Delta + delta-delta**: flaga `--deltas`, 3 kanały wejściowe, standaryzacja per-kanał.
- **Preprocessing offline**: VAD + preemfaza (`preprocess_data.py` → `data_preprocessed/`, flaga `--preprocessed`).
- **Multi-task loss**: `hierarchical_crnn` → target walencji (Negative/Neutral/Positive); `gender_aware` → target płci (wywodzony z nazwy pliku). Wykrywanie typu aux target w `engine.py` po liczbie neuronów `gender_head`.
- **Płeć z nazw plików**: TESS (OAF/YAF=Female), RAVDESS (nieparzysty aktor=M, parzysty=F), SAVEE (wszyscy 4 mówcy=M), CREMA-D (zbiór ID aktorek `CREMAD_FEMALE_ACTOR_IDS` z oficjalnego `VideoDemographics.csv`; 48 M / 43 F).
- **Ocena głowicy płci**: dla modeli z `gender_head` `evaluate_and_report` zapisuje dodatkowo `gender_report.json` (accuracy, macierz pomyłek, raport per klasa płci na teście). W zbiorach jednopłciowych (TESS, SAVEE) druga klasa oznaczona jako „brak w zbiorze”. Od 2026-11-26 `gender_report.json` zapisywany jest dla **każdego** modelu z sekcją `emotion_by_gender` (accuracy i macro F1 emocji osobno dla nagrań kobiet i mężczyzn, `accuracy_gap` = Female − Male); ocena głowicy płci tylko dla modeli z `gender_head`. `aggregate_results.py` pokazuje to w kolumnach `emo_F`, `emo_M`, `gap_FM`. Starsze przebiegi tej sekcji nie mają.
- **CLI `--epochs` / `--batch-size`**: przekazywane do `main(num_epochs=..., batch_size=...)` (domyślnie 50 / 32).
- **GPU**: `num_workers=12`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`, `cudnn.benchmark=True` (w `set_global_seed`; wyłączane przez `--deterministic`).
- **Seed i powtarzalność** (2026-11-20): dwa ziarna w `main.py` — `SPLIT_SEED=42` (podział train/val/test, **stały**, więc test jest ten sam dla każdego seeda i wariantu) oraz ziarno treningu `--seed` (domyślnie `DEFAULT_TRAINING_SEED=42`: wagi, kolejność batchy, augmentacja). `set_global_seed` wywoływane **przed każdym eksperymentem**, a train loader tworzony od nowa z własnym `torch.Generator` (`_seeded_train_loader`) — wynik modelu nie zależy już od tego, jakie modele trenowano wcześniej w tym samym uruchomieniu. `--deterministic`: `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Seed ≠ 42 → wyniki w podfolderze `<MODEL>/seed_<N>/`. Każdy przebieg zapisuje `run_config.json` (results_group, seed, split_seed, deterministic, epoki, batch, lr, patience). Sprawdzone: dwa przebiegi z tym samym seedem i `--deterministic` dają identyczny `training_log.csv`.
- **Agregacja seedów**: `aggregate_results.py [katalog]` grupuje przebiegi po eksperymencie (folder `seed_<N>` zdejmowany z klucza), liczy średnią ± odch. std. (ddof=1) dla accuracy i macro F1 oraz średnią epokę najlepszą / zatrzymania i accuracy głowicy płci. Domyślnie bierze tylko przebiegi z `run_config.json` (po 2026-11-20); `--include-legacy` dołącza stare (oznaczone `42*`), `--seeds 1,2,3` filtruje, `--csv plik` zapisuje. Ostrzega, gdy seedy w grupie mają różne ustawienia (epoki, batch, deterministic, split_seed, patience). Różny limit epok **nie** wywołuje ostrzeżenia, jeśli każdy przebieg z mniejszym limitem zatrzymał się przez early stopping przed nim (`_epoch_caps_equivalent`) — bez harmonogramu lr i w trybie deterministycznym taki przebieg byłby identyczny przy większym limicie.
- **`--patience N`** (2026-11-21): cierpliwość early stopping przekazywana do `TrainingConfig.early_stopping_patience` (domyślnie `DEFAULT_PATIENCE=7`). Wartość ≠ 7 → osobny katalog `..._patience<N>/`, więc nie miesza się z wynikami głównymi. Przy `--deterministic` i tym samym seedzie przebieg z większą cierpliwością jest dokładnym przedłużeniem przebiegu z mniejszą (identyczne pierwsze epoki) — czyste porównanie „co by było, gdyby trening trwał dalej”.

## Wyniki (accuracy testowe)

**Per-dataset** (7 klas oprócz CREMA-D=6; wyniki z 2026-07-08, `gender_aware` z 2026-11-19, `lstm` z 2026-11-20 przy limicie 100 epok):
- TESS: ~0.995 (dscnn/hier/headfusion) — **nierzetelne** (tylko 2 aktorki, wyciek mówcy)
- RAVDESS: dscnn **0.607**, headfusion 0.551, lstm 0.528
- SAVEE: headfusion **0.653**, dscnn 0.514
- CREMA-D: headfusion **0.609**, dscnn 0.594
- `lstm` (100 epok, acc / macro F1): TESS 0.705 / 0.613, RAVDESS 0.528 / 0.474, SAVEE 0.361 / 0.208, CREMA-D 0.410 / 0.359 (przed: 0.669 / 0.610, 0.505 / 0.427, 0.333 / 0.159, 0.450 / 0.403). SAVEE i CREMA-D zatrzymane w ep. 29 i 24, więc limit epok nie miał tam znaczenia — zmiana to szum między przebiegami. LSTM pomija całe klasy: TESS recall Sad 0.00 / Surprise 0.02; SAVEE Angry, Disgust, Sad = 0.
- `gender_aware` (po poprawce płci): TESS 0.983, RAVDESS 0.565, SAVEE 0.458, CREMA-D 0.441 (przed: 0.921 / 0.468 / 0.444 / 0.482). CREMA-D niedouczony: train_acc 0.44, early stopping w ep. 25 (wahania val_loss), Fear recall 0.01.

**Mega-zbiór — wyniki główne: 3 seedy** (2026-11-21, `wyniki_seedy/`; seedy 1/2/3, `--epochs 100 --deterministic`, patience 7, ten sam test 1728 próbek). Accuracy, średnia ± odch. std.:

| Model | surowe | deltas | filtry |
|---|---|---|---|
| hierarchical_crnn | **0.692 ± 0.018** | **0.688 ± 0.024** | 0.525 ± 0.163 ⚠ |
| dscnn | 0.679 ± 0.004 | 0.661 ± 0.022 | **0.661 ± 0.005** |
| headfusion_acnn | 0.662 ± 0.008 | 0.647 ± 0.013 | 0.635 ± 0.017 |
| gender_aware | 0.642 ± 0.031 | 0.574 ± 0.118 ⚠ | 0.498 ± 0.146 ⚠ |
| lstm | 0.576 ± 0.023 | 0.551 ± 0.011 | 0.580 ± 0.013 |

Macro F1 (surowe / deltas / filtry): hier 0.689 / 0.687 / 0.496; dscnn 0.678 / 0.657 / 0.658; headfusion 0.659 / 0.646 / 0.630; gender_aware 0.641 / 0.554 / 0.477; lstm 0.563 / 0.539 / 0.573. Głowica płci `gender_aware`: 0.965 / 0.847 / 0.721 (spadek przez nieudane przebiegi).

Poszczególne seedy (1 / 2 / 3), surowe: hier 0.702 / 0.671 / 0.701; dscnn 0.682 / 0.681 / 0.674; headfusion 0.653 / 0.667 / 0.666; gender_aware 0.674 / 0.640 / 0.613; lstm 0.556 / 0.571 / 0.601. Deltas: hier 0.661 / 0.698 / 0.706; dscnn 0.635 / 0.672 / 0.675; headfusion 0.635 / 0.644 / 0.661; gender_aware 0.678 / 0.597 / **0.446**; lstm 0.562 / 0.553 / 0.539. Filtry: hier **0.547** / **0.352** / 0.676; dscnn 0.667 / 0.657 / 0.659; headfusion 0.641 / 0.616 / 0.648; gender_aware **0.443** / **0.387** / 0.664; lstm 0.573 / 0.572 / 0.595.

⚠ **Nieudane przebiegi (przedwczesny early stopping)**: filtry/hier s2 (stop ep. 20, train_acc 0.31), filtry/hier s1 (ep. 33, 0.54), filtry/gender_aware s2 (ep. 11, 0.38), filtry/gender_aware s1 (ep. 26, 0.44), deltas/gender_aware s3 (ep. 16, 0.45); udane przebiegi kończą z train_acc ≈ 0.8. Te modele uczą się powoli na początku, a val_loss mocno skacze — przy patience 7 jedna seria gorszych epok przerywa trening, choć model nadal się uczy (filtry/hier s3 przez ~13 epok wyglądał jak s2, przetrwał i doszedł do 0.676). Głowica płci w nieudanych przebiegach ≈ 0.6 — cały model utknął na starcie. Przyczyna potwierdzona eksperymentem z `--patience 15` (niżej).

**Wnioski (3 seedy)**: (1) ranking na surowych: hier 0.692 ≥ dscnn 0.679 > headfusion 0.662 > gender_aware 0.642 > lstm 0.576; różnica hier–dscnn (1.3 p.p.) mieści się w rozrzucie hier — nierozstrzygająca; dscnn najstabilniejszy (±0.004). (2) **`gender_aware` spada na 4. miejsce** (−3.7 p.p. względem dscnn) i jest mniej stabilny; dawne 0.671 było szczęśliwym seedem — adaptacja do płci nie poprawia wyniku względem prostszego CNN. (3) **Delty lekko szkodzą wszystkim 5 modelom** (hier −0.4, headfusion −1.5, dscnn −1.8, lstm −2.5, gender_aware −6.8 p.p. — ten ostatni głównie przez jeden nieudany seed); każda różnica osobno w szumie, ale kierunek wszędzie ten sam. (4) **Filtry (VAD+preemfaza)**: dscnn −1.8, headfusion −2.7, lstm +0.4 p.p.; hier i gender_aware w 2 z 3 seedów nie wystartowały, a najlepsze seedy (0.676 i 0.664) są blisko surowych — filtry raczej **destabilizują trening modeli wielozadaniowych**, niż obniżają osiągalny wynik. Dawny wniosek „filtry szkodzą hier o 11 p.p.” był najpewniej skutkiem przedwczesnego zatrzymania.

**Eksperyment `--patience 15`** (2026-11-22, `wyniki_seedy/combined_{filtry,deltas}_patience15/`; hier i gender_aware, seedy 1–3, limit 100). **Przyczyna potwierdzona**: we wszystkich 12 przebiegach pierwsze epoki identyczne jak przy patience 7 (determinizm działa), a jedyna różnica to dłuższy trening. Wszystkie 5 nieudanych przebiegów wróciło do normy: filtry/hier s1 0.547 → 0.679, s2 0.352 → 0.641; filtry/gender s1 0.443 → 0.644, s2 0.387 → 0.572; deltas/gender s3 0.446 → 0.664. Przebiegi udane bez zmian lub +0–4.3 p.p.

| Model | deltas p7 → p15 | filtry p7 → p15 |
|---|---|---|
| hierarchical_crnn | 0.688 ± 0.024 → **0.694 ± 0.014** | 0.525 ± 0.163 → **0.666 ± 0.021** |
| gender_aware | 0.574 ± 0.118 → **0.661 ± 0.019** | 0.498 ± 0.146 → **0.630 ± 0.053** |

Wnioski: (1) filtry kosztują ok. 1–3 p.p. wszystkie modele splotowe (hier −2.6, gender_aware −1.2 względem surowych p7; dscnn −1.8, headfusion −2.7) — dawne −11 / −17 p.p. to skutek przerwanego treningu; (2) modele wielozadaniowe uczą się wolno przez pierwsze 10–20 epok (val_loss mocno skacze), więc patience 7 to za mało; (3) nadal niedoskonałe: filtry/hier s2 — najlepsza epoka = 100 (trafił w limit, train_acc 0.67), filtry/gender s2 — głowica płci 0.64 (nie nauczyła się). Pozostałe modele przy patience 7: dscnn i headfusion kończą przez przeuczenie (train_acc − val_acc = 0.07–0.18), lstm przez słabe uczenie (różnica ≈ 0, train_acc 0.55–0.6).

**Decyzja (2026-11-22): pełna spójność** — główne wyniki pracy będą z jednej procedury: **patience 15, limit 150 epok**, seedy 1–3, `--deterministic`, dla wszystkich 5 modeli i 3 wariantów mega-zbioru. Limit 150 zamiast 100, bo przy patience 15 przebiegi trwają dłużej (lstm i filtry/hier już przy p7/p15 dochodziły do 80–100). Uruchamia to `trening_patience15.bat` (35 przebiegów, szac. ~20–22 h). Pomija 10 przebiegów z eksperymentu wyżej, które zatrzymały się przed limitem 100 (byłyby identyczne przy 150), i powtarza 2, które w limit trafiły (filtry/hier s2, s3). Do czasu tych wyników tabele z patience 7 traktować jako wstępne.

**Mega-zbiór — wyniki historyczne: pojedynczy przebieg** (seed 42, sprzed poprawki seeda; `wyniki/`). Nie używać do wniosków — tylko do porównania:

| Model | surowe | filtry | deltas |
|---|---|---|---|
| hierarchical_crnn* | 0.707 | 0.596 | 0.679 |
| dscnn | 0.664 | 0.679 | 0.666 |
| headfusion_acnn | 0.655 | 0.623 | 0.647 |
| gender_aware* | 0.671 | 0.668 | 0.625 |
| lstm* | 0.573 | 0.584 | 0.579 |

\* Powtórzone 2026-11-20 z limitem 100 epok (`hier` na filtrach nie powtarzany). Wyniki przy 50 epokach: hier 0.704 / — / 0.710; gender_aware 0.663 / 0.660 / 0.635 (przed poprawką płci: 0.653 / 0.629 / 0.584); lstm 0.573 / 0.573 / 0.549. 100 epok pomogło tylko tam, gdzie model dochodził do limitu (gender_aware surowe +0.8 p.p., hier +0.3 p.p.).

**Uwaga o wariancji (historyczne)**: przebiegi zatrzymane przed 50. epoką też się zmieniały między uruchomieniami (hier deltas 0.710 → 0.679, lstm CREMA-D −4.0 p.p., TESS `gender_aware` +6.2 p.p.). Przyczyny: `cudnn.benchmark=True` oraz — w trybie combined — wspólny train loader z `persistent_workers=True` dla wszystkich modeli (wynik zależał od modeli trenowanych wcześniej w tym samym uruchomieniu). Oba usunięte 2026-11-20; wyniki z `wyniki_seedy/` są już powtarzalne.

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
10. **Niepowtarzalne wyniki przy stałym seedzie** (2026-11-20) — patrz uwaga o wariancji w wynikach. Rozwiązanie: ziarno ustawiane przed każdym eksperymentem, nowy train loader z `torch.Generator` na model, opcjonalny `--deterministic`, rozdzielenie `SPLIT_SEED` (stały) od `--seed` (trening).

## Konwencje i decyzje projektowe

- Zawsze 4-tuple z `SpeechEmotionDataset`: `(features, fine_label, coarse_label, gender_label)`.
- Etykieta coarse = walencja (mapa `COARSE_EMOTION_MAP` w `data_parsing.py`).
- `build_model(model_name, num_classes, input_freq_dim, input_channels, **kwargs)`.
- Augmentacja tylko dla train; deterministyczna kolejność: VAD → preemfaza → augmentacja → pad → mel.
- Wyniki: `results/per_dataset<WARIANT>/<DATASET>/<MODEL>/` lub `results/combined<WARIANT>/<MODEL>/` z plikami `{best_model.pt, training_log.csv, classification_report.json, confusion_matrix.png, label_mapping.json, run_config.json}` + `gender_report.json` dla `gender_aware`; dla seeda ≠ 42 w podfolderze `seed_<N>/`.
- Podział danych zawsze z `SPLIT_SEED=42` — nie zmieniać, inaczej wyniki nie będą porównywalne z dotychczasowymi (test 1728 próbek na combined).

## Kolejne kroki (rekomendowane)

1. **Leave-one-speaker-out split** (zamiast losowego stratyfikowanego) — rzetelna ocena generalizacji; obecnie TESS ~99% to artefakt wycieku mówcy.
2. **Class weighting** dla RAVDESS (Neutral 288 vs reszta 192).
3. **Strojenie `multitask_alpha`** dla `hierarchical_crnn` (grid search).
4. **Eksperyment ablacyjny** VAD vs preemfaza (osobno) — po wynikach z patience 15 spadek przez filtry to tylko ok. 1–3 p.p., więc ablacja ma niski priorytet.
5. ~~Uśrednianie po wielu seedach~~ — combined zrobione 2026-11-21 (`wyniki_seedy/`, wyniki powyżej). Per dataset jeszcze nie (najmniej istotne). Pierwotny plan (2026-11-20): seedy 1, 2, 3, `--epochs 100 --deterministic`, nowe przebiegi (stary seed 42 nie mieszany z nowymi). Kolejność: (1) combined surowe, 5 modeli; (2) combined deltas; (3) combined filtry (min. hier, dscnn, gender_aware); (4) per dataset (najmniej istotne: TESS wyciek mówcy, SAVEE 72 próbki testowe = 1.4 p.p. na próbkę). Stare wyniki nie są błędne — to pojedyncze pomiary z dużym rozrzutem. Raportować średnią ± odch. std. z `aggregate_results.py`.
6. **Multi-GPU (DDP)** — obecnie jedna karta.
7. ~~Dłuższy trening na combined~~ — zrobione 2026-11-20 (`--epochs 100`), patrz wyniki. Dla kolejnych przebiegów na combined używać `--epochs 100` (`gender_aware` zbiega dopiero ok. ep. 69).
8. **Przypiąć pozostałe pakiety** (soundfile, scikit-learn, pandas, matplotlib) wg `pip freeze` z maszyny zdalnej.
9. ~~Eksperyment z `--patience 15`~~ — zrobione 2026-11-22, przyczyna potwierdzona (patrz wyniki).
10. **Pełny przebieg patience 15 / 150 epok** (`trening_patience15.bat`) — w toku od 2026-11-22. Po zakończeniu: skopiować `results/combined*_patience15/` do `wyniki_seedy/`, przy czym filtry/hier seed_2 i seed_3 **zastępują** dotychczasowe; `python aggregate_results.py wyniki_seedy` — grupy `*_patience15` to wyniki główne do pracy.
11. **Ulepszenie i uzasadnienie `gender_aware` (`gender_aware_v2`)** — plan z 2026-11-22, do zrobienia po kroku 10 (nic jeszcze nie zaimplementowane). Stan: na surowych 0.642 ± 0.031, 4. miejsce, −3.7 p.p. względem dscnn; adaptacja do płci obecnie nie daje zysku.
    - **Słabości obecnej architektury** (`model.py`, `GenderAwareSERModel`):
      - (a) Model bierze tylko ostatni krok LSTM (`lstm_out[:, -1, :]`), a nagrania są dopełniane ciszą do 3 s na końcu (`audio_processing.pad_or_truncate`). Przy krótszych plikach (większość CREMA-D i TESS) ostatnie kroki to cisza. Ten sam problem ma `lstm`, czyli najsłabszy model; `hier` zbiera całą wypowiedź uwagą (`AttentionLayer`).
      - (b) Fuzja płci jest prawie pusta: do 128 cech doklejane są tylko 2 prawdopodobieństwa płci. Przy skuteczności głowicy ok. 96% to praktycznie stałe 0/1, więc daje tylko przesunięcie zależne od płci w pierwszej warstwie głowicy emocji. Model nie może przetwarzać głosu K i M inaczej.
      - (c) Mały szkielet: 2 warstwy splotowe i jednokierunkowy LSTM.
      - (d) `multitask_alpha=0.2` nie był strojony.
    - **Proponowane zmiany**:
      - (1) uwaga (`AttentionLayer`) zamiast ostatniego kroku — najtańsza zmiana, prawdopodobnie największy zysk;
      - (2) mocniejsza fuzja: dwie głowice emocji (K i M) mieszane prawdopodobieństwem płci, czyli osobne reguły dla każdej płci, łatwe do opisania w pracy; alternatywa: FiLM (płeć skaluje i przesuwa cechy);
      - (3) opcjonalnie dwukierunkowy LSTM.
      - Implementacja jako **osobny model `gender_aware_v2`**; stary zostaje jako wersja wyjściowa, a w pracy opisujemy drogę v1 → v2.
      - Dodatkowo:
        - flaga `--alpha`;
        - accuracy emocji osobno dla nagrań kobiet i mężczyzn w `gender_report.json` (czy różnica między płciami maleje).
    - **Ablacja do uzasadnienia** (ten sam szkielet, combined surowe, seedy 1–3, patience 15, `--deterministic`, ok. 12 przebiegów, ok. 7–8 h):
      - A: bez zadania płci (sam szkielet, punkt odniesienia);
      - B: płeć tylko jako zadanie pomocnicze, bez fuzji;
      - C: pełny model z fuzją;
      - D: „wyrocznia”, czyli prawdziwa płeć zamiast przewidywanej (górna granica zysku z informacji o płci).
      - Wynik C > B > A uzasadnia model, nawet jeśli dscnn jest ogólnie lepszy. D pokazuje, czy problemem jest przewidywanie płci, czy mała użyteczność płci w tych danych.
    - **Metodyka**: architekturę wybierać po wynikach walidacyjnych, nie testowych, a w pracy wprost napisać, że v2 powstał po analizie v1.

## Jak uruchomić

```bash
python run.py                          # grid: 4 datasety × 5 modeli
python run.py --combined               # mega-zbiór, 6 klas
python run.py --combined --deltas      # + 3 kanały
python run.py --preprocessed           # dane po VAD+preemfazie
python run.py --models dscnn --datasets SAVEE
python run.py --combined --models gender_aware --epochs 100
python run.py --combined --models gender_aware --epochs 100 --seed 7 --deterministic   # → results/combined/gender_aware/seed_7/
python preprocess_data.py              # preprocessing offline (raz)

# wiele seedów (każdy wariant we własnym katalogu, nic się nie nadpisuje)
for s in 1 2 3; do
  python run.py --combined                --epochs 100 --seed $s --deterministic
  python run.py --combined --deltas       --epochs 100 --seed $s --deterministic
  python run.py --combined --preprocessed --epochs 100 --seed $s --deterministic
done
python aggregate_results.py                       # tabela średnia ± std z results/
python aggregate_results.py wyniki --csv wyniki/summary.csv

# wyniki główne: patience 15, limit 150 (Windows cmd)
trening_patience15.bat
```

Zależności: `requirements.txt`; PyTorch dobierany przez `setup.sh`/`setup.bat` (CUDA 11.8 lub CPU).
