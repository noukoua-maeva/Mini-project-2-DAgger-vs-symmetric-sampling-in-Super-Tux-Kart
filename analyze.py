"""Summarize the published SuperTuxKart evaluation runs."""
import csv
from itertools import product
from pathlib import Path
import numpy as np

RUNS = Path("runs")
METHODS = ("DAgger", "TD3 50/50", "TD3 online")
PREFIX = dict(zip(METHODS, ("dagger", "mix", "online")))
SEEDS = list(range(1, 9))
STEPS = np.arange(0, 24_001, 4_000)
N = len(SEEDS)

rows = []
for method, seed in product(METHODS, SEEDS):
    with (RUNS / f"{PREFIX[method]}_{seed}.csv").open() as f:
        data = list(csv.DictReader(f))
    expected = {(int(step), race) for step, race in product(STEPS, range(500, 503))}
    observed = [(int(r["step"]), int(r["race_seed"])) for r in data if r["method"] == method]
    assert len(observed) == len(expected) and set(observed) == expected, (method, seed)
    assert all(int(r["seed"]) == seed for r in data if r["method"] == method)
    rows.extend(data if (method, seed) == (METHODS[0], SEEDS[0]) else
                [r for r in data if r["method"] == method])

demos = [r for r in rows if r["method"] == "Demonstration"]
expert = [r for r in rows if r["method"] == "Expert"]
assert len(demos) == 10 and all(int(r["finished"]) for r in demos)
assert len(expert) == 3 and all(int(r["finished"]) for r in expert)
expert_time = np.mean([float(r["finish_time_s"]) for r in expert])
demo_ticks = sum(int(r["ticks"]) for r in demos)

def summarize(method):
    finish, time, distance = [np.zeros((N, len(STEPS))) for _ in range(3)]
    for i, seed in enumerate(SEEDS):
        for j, step in enumerate(STEPS):
            subset = [r for r in rows if (r["method"], int(r["seed"]), int(r["step"])) == (method, seed, step)]
            finish[i, j] = np.mean([int(r["finished"]) for r in subset])
            time[i, j] = np.mean([float(r["finish_time_s"]) if r["finish_time_s"] else 100 for r in subset])
            distance[i, j] = np.mean([float(r["distance"]) for r in subset])
    return finish, time, distance

stats = {m: summarize(m) for m in METHODS}
boot = np.random.default_rng(42).integers(N, size=(20_000, N))
signs = np.array(list(product((-1, 1), repeat=N)))

def band(values):
    return np.percentile(values[boot].mean(axis=1), [2.5, 97.5], axis=0)

def contrast(first, second):
    delta = stats[first][1][:, -1] - stats[second][1][:, -1]
    p = np.mean(np.abs((signs * delta).mean(axis=1)) >= abs(delta.mean()) - 1e-9)
    return delta.mean(), band(delta), p

p_main = contrast("DAgger", "TD3 50/50")
p_control = contrast("TD3 50/50", "TD3 online")

for method, (finish, score, distance) in stats.items():
    lo, hi = band(score[:, -1])
    count = round(finish[:, -1].sum() * 3)
    print(f"{method}: {count}/{3*N} finishes, "
          f"mean capped score {score[:, -1].mean():.1f} s "
          f"(95% CI [{lo:.1f}, {hi:.1f}]), "
          f"mean progress {distance[:, -1].mean():.0f} m")
for label, (mean, (lo, hi), p) in (
    ("DAgger minus TD3 50/50", p_main),
    ("TD3 50/50 minus TD3 online", p_control),
):
    print(f"{label}: {mean:.1f} s (95% CI [{lo:.1f}, {hi:.1f}], p={p:g})")
