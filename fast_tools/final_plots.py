import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "out_v2/Llama-2-7b-hf"; O = "bench/analysis_v2"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9993"
def g(name):
    d = pd.read_csv(f"{R}/{name}.csv"); return d.groupby(["l_start", "l_end"]).mean(numeric_only=True)
def best(name): return g(name)["steer_to_known"].max()
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
# --- plot 1: bars, best-window success per condition
conds = [("Swap, α=3, 8-shot prompt (best)", "steer_en_first_own_swap3_shots8", BLUE),
         ("Swap, α=2", "steer_en_first_own_swap2", BLUE),
         ("Swap, α=1 (clean prompt)", "steer_en_first_own", BLUE),
         ("Add new direction only", "steer_en_first_own_add1", BLUE),
         ("Patch English 2-d subspace", "steer_en_first_own_patch_en1", BLUE),
         ("Full last-position patch", "steer_en_first_own_patch1", BLUE),
         ("Control: Spanish dirs (best config)", "steer_es_first_own_swap3_shots8_ctl", GRAY),
         ("Control: French dirs (α=2)", "steer_fr_first_own_swap2_ctl", GRAY),
         ("Control: Russian dirs (α=2)", "steer_ru_first_own_swap2_ctl", GRAY),
         ("Control: Chinese dirs (α=2)", "steer_zh_first_own_swap2_ctl", GRAY),
         ("Control: German dirs (α=2)", "steer_de_first_own_swap2_ctl", GRAY)]
vals = [best(n) for _, n, _ in conds]
fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURF); ax.set_facecolor(SURF)
y = np.arange(len(conds))[::-1]
ax.barh(y, vals, color=[c for *_, c in conds], height=0.62)
for yi, v in zip(y, vals): ax.text(v + 0.005, yi, f"{100*v:.1f}%", va="center", color=INK, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([c for c, *_ in conds], color=INK)
ax.set_xlim(0, 0.42); ax.xaxis.set_major_formatter(lambda v, p: f"{100*v:.0f}%")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.set_xlabel("Model outputs the injected word in the target language (best layer window,\nwords the model can translate; Llama-2-7b, 24 or 12 language combinations)")
ax.set_title("English directions steer; other languages' directions don't", color=INK, loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(f"{O}/final_bars.png", dpi=150, facecolor=SURF); plt.close(fig)
# --- plot 2: by end layer
lines = [("Best: swap α=3, 8-shot", "steer_en_first_own_swap3_shots8", BLUE),
         ("Swap α=1, 4-shot", "steer_en_first_own", ORANGE),
         ("Add only, α=1", "steer_en_first_own_add1", AQUA),
         ("Spanish control, best config", "steer_es_first_own_swap3_shots8_ctl", GRAY)]
fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
for lab, n, c in lines:
    s = g(n)["steer_to_known"].groupby(level="l_end").max()
    ax.plot(s.index, s.values, color=c, lw=2, label=lab)
    i = s.values.argmax(); ax.annotate(lab, (s.index[i], s.values[i]), xytext=(6, 4), textcoords="offset points", color=INK2, fontsize=8)
ax.yaxis.set_major_formatter(lambda v, p: f"{100*v:.0f}%"); ax.set_xlim(10, 33)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
ax.set_xlabel("Last steered layer + 1 (best start layer for each)"); ax.set_ylabel("Success (words the model knows)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Steering only works if it ends around layers 18–21", color=INK, loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(f"{O}/final_by_layer.png", dpi=150, facecolor=SURF); plt.close(fig)
print("ok")
