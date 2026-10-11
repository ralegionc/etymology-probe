#!/usr/bin/env bash
# After run_models.sh: the two-class tests, the behavioural test, then the analysis,
# figures and report. The mention part of analysis.py uses whatever corpus counts are
# cached; rerun it after corpus_counts.py finishes.
set -e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONWARNINGS=ignore HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error PYTHONIOENCODING=utf-8
python pairs.py
for size in 410m 1.4b 2.8b; do
  [ -f "results/behave_pythia-$size@main.parquet" ] || python behave.py --threads 4 --model "EleutherAI/pythia-$size"
done
python corpus_counts.py --offline
python analysis.py
python figures.py
python report.py > results/report.md
echo AFTER DONE
