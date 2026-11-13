#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${STK_SRC:-stk/src}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p runs
for seed in {1..8}; do
    for entry in 'dagger|DAgger' 'mix|TD3 50/50' 'online|TD3 online'; do
        printf '%s\0%s\0%s\0' "$seed" "${entry%%|*}" "${entry#*|}"
    done
done | xargs -0 -n 3 -P 6 bash -c '
    set -e
    seed=$1; name=$2; method=$3
    "${PYTHON_BIN:-.venv/bin/python}" -u experiment.py --seed "$seed" --method "$method" --output "runs/${name}_${seed}.csv" > "runs/${name}_${seed}.log" 2>&1
' _
