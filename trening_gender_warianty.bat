@echo off
rem Warianty gender_aware (ablacja), combined surowe, seedy 1-3.
rem Ta sama procedura co wyniki glowne: patience 15, limit 150 epok,
rem tryb deterministyczny -> results\combined_patience15\<MODEL>\seed_<N>,
rem obok dotychczasowych 5 modeli.
rem
rem   gender_aware_detach        - odciety gradient emocji od glowicy plci
rem   gender_aware_att           - detach + uwaga zamiast ostatniego kroku LSTM
rem   gender_aware_film          - att + fuzja FiLM
rem   gender_aware_att_nogender  - kontrola: att bez zadania plci i bez fuzji

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=trening_gender_warianty.log
set OPTS=--epochs 150 --deterministic --patience 15
set MODELS=gender_aware_detach,gender_aware_att,gender_aware_film,gender_aware_att_nogender

for %%s in (1 2 3) do (
    python run.py --combined --models %MODELS% --seed %%s %OPTS% >> %LOG% 2>&1
)

python aggregate_results.py results >> %LOG% 2>&1
echo Gotowe >> %LOG%
