@echo off
rem Dokonczenie trening_patience15.bat po zawieszeniu na seedzie 3.
rem Gotowe wczesniej: seedy 1-2 (wszystkie warianty), seed 3 combined / dscnn.
rem Seed ustawiany jest od nowa przed kazdym modelem (set_global_seed),
rem wiec osobne uruchomienie daje to samo co ciagly przebieg.
rem
rem PYTHONUNBUFFERED=1 - stdout trafia do logu na biezaco (wczesniej byl
rem buforowany i w logu nie bylo sladu po seedzie 3).

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=trening_patience15_dokoncz.log
set OPTS=--epochs 150 --deterministic --patience 15

python run.py --combined                --models headfusion_acnn,lstm,hierarchical_crnn,gender_aware --seed 3 %OPTS% >> %LOG% 2>&1
python run.py --combined --deltas       --models dscnn,headfusion_acnn,lstm --seed 3 %OPTS% >> %LOG% 2>&1
python run.py --combined --preprocessed --models dscnn,headfusion_acnn,lstm --seed 3 %OPTS% >> %LOG% 2>&1

for %%s in (2 3) do (
    python run.py --combined --preprocessed --models hierarchical_crnn --seed %%s %OPTS% >> %LOG% 2>&1
)

python aggregate_results.py results >> %LOG% 2>&1
echo Gotowe >> %LOG%
