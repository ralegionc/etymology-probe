#!/usr/bin/env bash
# Hidden states and probes for every model and checkpoint, one after another (they
# compete for the CPU otherwise). Safe to re-run: finished extractions are skipped.
set -e
export PYTHONWARNINGS=ignore HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error PYTHONIOENCODING=utf-8
probe() { [ -f "results/probe/$1_$2.csv" ] || python probe.py model --tag "$1" --pool "$2" --jobs 3; }

# final checkpoints of the Pythia suite, then OLMo
for size in 70m 160m 410m 1b 1.4b 2.8b; do
  python extract.py --threads 4 --model "EleutherAI/pythia-$size"
  probe "pythia-$size@main" last
done
for m in allenai/OLMo-1B-0724-hf allenai/OLMo-2-0425-1B; do
  python extract.py --threads 4 --model "$m"
  probe "${m##*/}@main" last
done

# training dynamics: Pythia-410m checkpoints (step143000 is the final model)
for step in 0 512 2000 8000 32000; do
  python extract.py --threads 4 --model EleutherAI/pythia-410m --revision "step$step" --no-mean
  probe "pythia-410m@step$step" last
done

# mean pooling as a sensitivity check
for t in pythia-410m@main pythia-1.4b@main; do probe "$t" mean; done
echo ALL DONE
