# SuperTuxKart comparison

This repository contains the training script, the run launcher, the statistical analysis script, and the raw per-race CSV files. The CSV column `step` counts online training interactions; `ticks` counts steps in a single race. For timeouts, `finish_time_s` is blank. The demonstration and expert rows in separate files repeat the same protocol and are not independent observations.

## Reproduce on Linux with Python 3.12 and uv

```bash
git clone https://github.com/osigaud/pystk_project.git stk
git -C stk checkout ddf2cb6573850f890bba26d4e5f82888d9dac82a
uv venv --python 3.12
uv pip install 'pysupertuxkart2==0.7.4' 'gymnasium==1.4.0' 'numpy==2.5.4' 'torch==2.15.0+cpu' \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
# Initialize the simulator assets in a single process on first use.
mkdir -p runs
PYTHONPATH=stk/src .venv/bin/python experiment.py --budget 100 --output runs/smoke.csv
bash run_all.sh
.venv/bin/python analyze.py
```

The simulator downloads the SuperTuxKart 1.5 assets on first use.

Run the launcher from the repository root. It overwrites the existing CSV and log for each method/seed as that run starts. Use a separate copy to retain the supplied raw measurements. To inspect one run, set `PYTHONPATH=stk/src` and pass `--method`, `--seed`, and `--output` to `experiment.py`.

To check the published CSVs directly, run `python analyze.py` from the repository root with NumPy installed.
