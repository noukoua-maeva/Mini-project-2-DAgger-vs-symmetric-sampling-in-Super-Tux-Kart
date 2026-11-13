"""Render the report from the complete, matched set of training runs."""
import csv
from itertools import product
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

RUNS = Path("runs")
OUT = Path("output/pdf")
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ("DAgger", "TD3 50/50", "TD3 online")
PREFIX = dict(zip(METHODS, ("dagger", "mix", "online")))
COLORS = dict(zip(METHODS, ("#126a61", "#9e481c", "#3d5482")))
MARKERS = dict(zip(METHODS, ("o", "s", "^")))
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

def fmt_contrast(res):
    mean, (lo, hi), p = res
    return f"{mean:+.1f} s (95% CI {lo:+.1f} to {hi:+.1f} s, <i>p</i> = {p:.4f})"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, progress_ax) = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
for m in METHODS:
    t = stats[m][1]
    lo, hi = band(t)
    ax.fill_between(STEPS / 1000, lo, hi, color=COLORS[m], alpha=.14)
    ax.plot(STEPS / 1000, t.mean(axis=0), color=COLORS[m], marker=MARKERS[m], label=m, lw=1.6)
ax.axhline(expert_time, color="#444444", ls="--", lw=1, label="Expert")
ax.set(title="Capped lap score during training", xlabel="Online interactions (thousands)", ylabel="Capped lap score (s)")
ax.set_xticks(STEPS / 1000)
ax.set_ylim(min(34, expert_time - 2), 104)
ax.grid(axis="y", alpha=.2)
for m in METHODS:
    progress = stats[m][2]
    lo, hi = band(progress)
    progress_ax.fill_between(STEPS / 1000, lo, hi, color=COLORS[m], alpha=.14)
    progress_ax.plot(STEPS / 1000, progress.mean(axis=0), color=COLORS[m], marker=MARKERS[m], lw=1.6)
progress_ax.set(title="Final progress per race", xlabel="Online interactions (thousands)", ylabel="Distance (m)")
progress_ax.set_xticks(STEPS / 1000)
progress_ax.set_ylim(-30, 930)
progress_ax.grid(axis="y", alpha=.2)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, fontsize=9, loc="outside lower center")
figure = OUT / "learning_curves.png"
fig.savefig(figure, dpi=250)
plt.close(fig)

font_root = Path("/usr/share/fonts/truetype/dejavu")
for name, filename in (("ReportSerif", "DejaVuSerif.ttf"), ("ReportSerifBold", "DejaVuSerif-Bold.ttf"),
                       ("ReportSerifItalic", "DejaVuSerif-Italic.ttf"), ("ReportSans", "DejaVuSans.ttf"),
                       ("ReportSansBold", "DejaVuSans-Bold.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(font_root / filename)))
pdfmetrics.registerFontFamily("ReportSerif", normal="ReportSerif", bold="ReportSerifBold", italic="ReportSerifItalic")

INK = colors.HexColor("#192530")
MUTED = colors.HexColor("#445464")
TEAL = colors.HexColor("#126a61")
BODY = ParagraphStyle("body", fontName="ReportSerif", fontSize=10.2, leading=13.1, textColor=INK, spaceAfter=6)
TITLE = ParagraphStyle("title", fontName="ReportSerifBold", fontSize=25, leading=28, textColor=INK, spaceAfter=10)
HEAD = ParagraphStyle("head", fontName="ReportSansBold", fontSize=12, leading=15, textColor=TEAL, spaceBefore=9, spaceAfter=6)
CAP = ParagraphStyle("caption", parent=BODY, fontSize=9.1, leading=11.2, textColor=MUTED)
META = ParagraphStyle("meta", parent=CAP, fontName="ReportSans", fontSize=10, leading=13, spaceAfter=14)
CELL = ParagraphStyle("cell", parent=CAP, fontName="ReportSans", fontSize=9, leading=11, spaceAfter=0)
CONTENT = 166 * mm
story = []

def add(s, style=BODY):
    story.append(Paragraph(s, style))

def table(data, widths, padding=4):
    cells = [[Paragraph(str(x), CELL) for x in row] for row in data]
    result = Table(cells, colWidths=[w * mm for w in widths], hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f2f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LINEBELOW", (0, -1), (-1, -1), .45, colors.HexColor("#b3c1ca")),
    ]))
    story.append(result)
    story.append(Spacer(1, 7))

add("DAgger vs. symmetric sampling<br/>in SuperTuxKart", TITLE)
add("Ahmed Dargouthy and Maëva Noukoua  ·  Reinforcement Learning Mini-project 2", META)
add("1  Research question", HEAD)
add("With a fixed budget of online interactions, which agent completes a solo lap faster: an imitation learner that can ask an expert how to recover, or an off-policy reinforcement learner that reuses a fixed set of expert laps? We also compare two versions of the same RL learner to measure what changes when half of each batch is drawn from the expert data.")
add("2  Experimental setup", HEAD)
add(f"<b>Task and expert.</b> A single kart drives one lap of Lighthouse in headless SuperTuxKart 1.5 (PySuperTuxKart2 0.7.4). An episode ends when the simulator reports a finish or after 1,000 control steps of 0.1 s each. The expert is Agent3 from the course repository [4] (commit ddf2cb6). It looks ahead on the track and can drift, brake, use nitro and call for rescue. Firing is disabled. Ten expert rollouts (simulator seeds 100–109) all finished in {expert_time:.2f} s and produced {demo_ticks:,} transitions. In this setting their observed lap times match, so the repeated demonstrations offer limited coverage.")
add("<b>State and action.</b> The policy receives eight look-ahead nodes (local x/z coordinates and width), local velocity, lateral distance, steering limit, energy, skid factor, shield time, lap distance, and the first four visible item entries (local x/z and indicators for two hazards). We zero-pad missing entries and use fixed feature scales specified in the source. Both learners emit six values in [−1, 1]. Steering is used directly, throttle maps linearly to [0, 1], and brake, drift, nitro and rescue are thresholded at zero. For TD3, online replay stores the continuous values sent to the decoder. Offline switch labels use −1/+1.")
add("3  Learning methods", HEAD)
add("<b>DAgger [1].</b> Start with 1,000 behaviour cloning updates on the fixed expert dataset. During online training, query the expert on every visited state and add the expert's action to the aggregated dataset even when the learner acts. The probability of executing the expert action is 1/2, 1/3, 1/6 and 0 in successive 4,000-step blocks, remaining at zero thereafter. The supervised loss is the mean squared error for steering and throttle plus 0.25 times the mean squared error for the four switches.")
add("<b>Symmetric sampling [2].</b> Use TD3 [3] with 64 expert and 64 online transitions drawn independently with replacement per update. The online control uses the same TD3 implementation and samples all 128 transitions from online replay. This tests symmetric sampling with a TD3 backbone. The original RLPD recipe also uses critic LayerNorm and a higher update-to-data ratio.")
add("<b>Training settings.</b> Actors and each critic have two 128-unit ReLU layers. The actors use tanh outputs. Adam uses learning rate 0.0003 for both networks. Batch size is 128. DAgger updates every online step after pretraining. TD3 starts at online step 128 and takes one critic update per step, one actor/target update every two critic updates, and uses discount 0.99, Polyak coefficient 0.005, target noise SD 0.2 clipped to ±0.5, and exploration noise SD 0.15. Actions are clipped to [−1, 1]. Replay retains the full applicable history. In the solo setting the course reward is progress in metres divided by 10, minus 0.1 per step, plus 10 at lap completion. Time limits bootstrap in the critic targets.")
story.append(PageBreak())
add("4  Evaluation", HEAD)
add(f"Each method has {N} training seeds (1–{N}) and 24,000 online steps per seed. Training resumes from a new race after each evaluation checkpoint, retaining the networks and replay. Every 4,000 steps, including step zero, we run three noise-free races with fixed simulator seeds 500–502, without expert intervention. The primary outcome is a capped lap score: the simulator finish time in seconds for a completed race and 100 s for a timeout. Lower is better. Completion counts and the median time among completed laps describe the two parts of this score.")
add(f"We first average the three evaluations within each training seed. The seed is the unit for 95% percentile bootstrap intervals (20,000 resamples). The primary comparison uses an exact two-sided paired sign-flip test on the difference in final capped scores, assuming exchangeability within pairs under the null. The 50/50 vs. online-only comparison is exploratory. Identical races under different simulator seeds contribute to the seed average without increasing the inferential sample size.")
add("5  Results", HEAD)
add("Figure 1 shows the capped score and final progress across the training budget. Table 1 reports the final checkpoint. The dashed reference is the expert's mean on the same evaluation seeds.")
story.append(Image(str(figure), width=CONTENT - 10*mm, height=(CONTENT - 10*mm) * 2.8 / 7.0))
add(f"<b>Figure 1.</b> Mean capped lap score and mean final progress across {N} training seeds at each checkpoint. Shading gives pointwise 95% bootstrap intervals across seeds. The expert completed all three evaluation laps in {expert_time:.2f} s each. Final paired DAgger minus TD3 50/50: {fmt_contrast(p_main)}. Negative differences favour DAgger.", CAP)
data = [["Policy", "Completed laps", "Median finish (s)", "Mean capped score (s)", "Final progress (m)"]]
for m in METHODS:
    f, t, d = stats[m]
    final = [float(r["finish_time_s"]) for r in rows if r["method"] == m and int(r["step"]) == STEPS[-1] and r["finish_time_s"]]
    lo, hi = band(t[:, -1])
    med = f"{np.median(final):.2f}" if final else "—"
    data.append([m, f"{int(round(f[:, -1].sum()*3))}/{N*3}", med,
                 f"{t[:, -1].mean():.1f} [{lo:.1f}, {hi:.1f}]", f"{d[:, -1].mean():.0f}"])
data.append(["Expert", "3/3", f"{expert_time:.2f}", f"{expert_time:.1f}", f"{np.mean([float(r['distance']) for r in expert]):.0f}"])
table(data, [29, 29, 31, 42, 35])
add("<b>Table 1.</b> Final evaluations. Brackets show seed-level 95% bootstrap intervals. A dash denotes zero completed laps. Progress is the mean distance reported by the simulator at termination or timeout.", CAP)
add(f"The final difference for TD3 50/50 minus online-only TD3 was {fmt_contrast(p_control)}. The 100 s cap gives equal primary scores to all timeouts, so final progress helps distinguish how far incomplete policies drove.")
if stats["TD3 50/50"][0][:, :-1].any():
    best_ix = int(np.argmax(stats["TD3 50/50"][0][:, :-1].sum(axis=0)))
    best_count = int(round(stats["TD3 50/50"][0][:, best_ix].sum() * 3))
    add(f"Before the final checkpoint, TD3 50/50 reached {best_count}/{N*3} completed evaluations at {STEPS[best_ix]:,} steps. The final comparison uses the fixed budget for all methods.")
story.append(PageBreak())
add("6  Seed-level outcomes and discussion", HEAD)
seed_data = [["Training seed", "DAgger", "TD3 50/50", "TD3 online"]]
for i, seed in enumerate(SEEDS):
    seed_data.append([seed, *[f"{stats[m][1][i, -1]:.2f}" for m in METHODS]])
table(seed_data, [40, 42, 42, 42], padding=2)
add("<b>Table 2.</b> Final capped lap score in seconds by training seed, averaged over the three evaluation races.", CAP)

d_completion = stats["DAgger"][0][:, -1].mean()
m_completion = stats["TD3 50/50"][0][:, -1].mean()
o_completion = stats["TD3 online"][0][:, -1].mean()
if p_main[2] < .05:
    better = "DAgger" if p_main[0] < 0 else "TD3 50/50"
    add(f"<b>Final comparison.</b> The paired test supports a lower capped score for {better} at 24,000 steps in this setup. DAgger completed {d_completion:.0%} of final evaluations and TD3 50/50 completed {m_completion:.0%}. The capped outcome combines the ability to finish with speed, so the difference should be interpreted together with the finish times and completion counts in Table 1.")
else:
    add(f"<b>Final comparison.</b> The paired test did not detect a final difference in capped score. DAgger completed {d_completion:.0%} of final evaluations and TD3 50/50 completed {m_completion:.0%}. The capped outcome combines the ability to finish with speed, so the difference should be interpreted together with the finish times and completion counts in Table 1.")
dagger_times = [float(r["finish_time_s"]) for r in rows if r["method"] == "DAgger" and int(r["step"]) == STEPS[-1] and r["finish_time_s"]]
if dagger_times:
    median = np.median(dagger_times)
    add(f"Among completed DAgger laps, the median was {median:.2f} s, compared with {expert_time:.2f} s for the expert. This comparison is descriptive and conditions on successful laps.")
if m_completion == 0 and o_completion == 0:
    mix_dist, online_dist = [stats[m][2][:, -1].mean() for m in METHODS[1:]]
    add(f"<b>Effect of the fixed demonstrations.</b> Both TD3 agents timed out at the final checkpoint. They reached {mix_dist:.0f} m and {online_dist:.0f} m on average with and without demonstration replay, respectively. Progress is descriptive and can be influenced by individual runs. A cap of 100 s means the primary metric cannot quantify how far either timeout was from finishing.")
else:
    add(f"<b>Effect of the fixed demonstrations.</b> TD3 finished {m_completion:.0%} of final evaluations with demonstration replay and {o_completion:.0%} without it. This comparison keeps the learner and training budget fixed. Its uncertainty is substantial with eight seeds, so the estimate describes the tested configuration.")
add("<b>Why the information differs.</b> When the learner steers too late and ends up near the edge of the track, DAgger can obtain an expert correction at that exact state. Symmetric sampling can repeatedly show TD3 what the expert did on its own path, while corrections on learner-generated states must be inferred from the online rewards and critic estimates. Reward optimisation can in principle discover faster actions than the demonstrations, whereas the DAgger loss rewards action agreement. Neither comparison establishes which mechanism caused a specific failure.")
add("<b>Limits.</b> The training interaction budget is matched, and DAgger additionally uses 24,000 online expert labels plus 1,000 initial supervised updates. This is a practical comparison under different access to supervision. The expert is stateful and uses a broader observation than the policy, so its labels can depend on information the learner cannot recover from one observation. TD3 also models continuous outputs that become discrete switches at execution, which can make critic gradients around thresholds unreliable. The study covers one fixed start, kart and track at a short training budget. Repeating simulator seeds can reproduce the same lap, so the intervals only describe variation across training seeds in this setting. Changing those conditions, collecting more diverse expert trajectories, or allowing a longer budget could change the results.")
add("References", HEAD)
add('[1] Ross, Gordon &amp; Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. <link href="https://proceedings.mlr.press/v15/ross11a.html" color="#126a61">AISTATS</link>.', CAP)
add('[2] Ball, Smith, Kostrikov &amp; Levine (2023). Efficient Online Reinforcement Learning with Offline Data. <link href="https://proceedings.mlr.press/v202/ball23a.html" color="#126a61">ICML</link>.', CAP)
add('[3] Fujimoto, van Hoof &amp; Meger (2018). Addressing Function Approximation Error in Actor-Critic Methods. <link href="https://proceedings.mlr.press/v80/fujimoto18a.html" color="#126a61">ICML</link>.', CAP)
add('[4] SuperTuxKart course project. <link href="https://github.com/osigaud/pystk_project/tree/ddf2cb6573850f890bba26d4e5f82888d9dac82a/src/agents/team3" color="#126a61">Agent3 source and course environment (ddf2cb6)</link>.', CAP)

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#d2dbe2"))
    canvas.line(22*mm, 17*mm, 188*mm, 17*mm)
    canvas.setFont("ReportSans", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(22*mm, 12.5*mm, "REINFORCEMENT LEARNING  /  MINI-PROJECT 2")
    canvas.drawRightString(188*mm, 12.5*mm, str(doc.page))

doc = SimpleDocTemplate(str(OUT / "Dargouthy_Noukoua.pdf"), pagesize=A4,
                        leftMargin=20*mm, rightMargin=20*mm, topMargin=15*mm,
                        bottomMargin=22*mm, title="DAgger vs. symmetric sampling in SuperTuxKart")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(data)
print("Paired contrasts:", p_main, p_control)
