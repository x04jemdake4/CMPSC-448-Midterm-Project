"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

the LAST step: turns the saved results/*.json into all the report figures and results/summary.json
    - dataset response lengths
    - RQ1 accuracy bars, confusion matrices, training curves
    - RQ2 input-vs-output bars + the selection-bias simulation
    - RQ3 held-out target bars
    - RQ4 style features + ablation results

Ryan Broadhurst, Michael Finn
"""
import json         # reading the results files
import os           # folders and file paths

import matplotlib       # plotting
matplotlib.use("Agg")       # draw straight to files, no window pops up
import matplotlib.pyplot as plt
import numpy as np      # using for RNG
import pandas as pd     # using a DataFrame (fancy spreadsheet)

from common import DOMAINS, FAMILIES, FIGURES, MODELS, RESULTS, load_data

os.makedirs(FIGURES, exist_ok=True)     # make the figures folder if its not there yet
# validated categorical palette in fixed order ( colour follows the family, never its rank )
FAM_COL = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a"]))        # family -> colour
MOD_COL = {"tfidf": "#8a8984", "cnn": "#2a78d6", "lstm": "#eb6834"}     # classifier -> colour
MOD_LAB = {"tfidf": "TF-IDF+LR", "cnn": "CNN", "lstm": "BiLSTM"}        # classifier -> label for the plots
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"       # text, secondary text, gridline colours
CH_FAM, CH_MOD = 1 / len(FAMILIES), 1 / len(MODELS)     # chance level for 3 classes and for 6 classes
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})      # one look for every figure


def J(name):
    p = os.path.join(RESULTS, name)     # path to the results file
    return json.load(open(p)) if os.path.exists(p) else None        # None if that experiment hasnt been run yet


def chance_line(ax, c, x_right):
    ax.axhline(c, color=INK2, ls="--", lw=1)        # dashed line at chance
    ax.text(x_right, c + 0.012, f"chance {c:.0%}", ha="right", va="bottom", color=INK2, fontsize=7)     # label it


summary = {}        # all the headline numbers get collected here
df = load_data()        # the 600-row dataset

# ---------------- dataset: response length per model ----------------
df["n_words"] = df.llm_output.str.split().str.len()     # words in each response
fam_of = df.drop_duplicates("llm_name").set_index("llm_name").llm_family        # model -> its family
fig, ax = plt.subplots(figsize=(6.2, 2.3))
bp = ax.boxplot([df[df.llm_name == m].n_words.values for m in MODELS], vert=False, widths=0.6,
                showfliers=False, patch_artist=True, medianprops=dict(color=INK, linewidth=1.5))        # one box per model
for patch, m in zip(bp["boxes"], MODELS):
    patch.set_facecolor(FAM_COL[fam_of[m]]); patch.set_alpha(0.85); patch.set_edgecolor("white")        # colour each box by family
ax.set_yticks(range(1, len(MODELS) + 1), MODELS)
ax.set_xlabel("Response length in words (box = IQR, line = median, whiskers = 1.5 IQR)")
ax.grid(axis="y", visible=False)
fig.savefig(f"{FIGURES}/fig_dataset_length.png"); plt.close(fig)

# ---------------- RQ1 ----------------
rq1 = {lvl: {m: J(f"rq1_{lvl}_{m}.json") for m in ["tfidf", "cnn", "lstm"]} for lvl in ["family", "model"]}     # load every RQ1 result
summary["rq1"] = {lvl: {m: None if r is None else {
    "pooled_acc": r["pooled"]["accuracy"], "pooled_f1": r["pooled"]["macro_f1"],
    "fold_acc_mean": r["acc_mean"], "fold_acc_std": r["acc_std"],
    "per_class_f1": r["pooled"]["per_class_f1"],
    "epochs": [f["epochs_run"] for f in r["folds"]], "time_s": [f["train_time_s"] for f in r["folds"]],
    "n_params": r["folds"][0]["n_params"], "vocab": [f["vocab_size"] for f in r["folds"]]}
    for m, r in d.items()} for lvl, d in rq1.items()}       # the headline numbers for summary.json

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)     # left = 3 families, right = 6 models
for ax, lvl, ch, title in zip(axes, ["family", "model"], [CH_FAM, CH_MOD],
                              ["3 families", "6 individual models"]):
    ms = [m for m in ["tfidf", "cnn", "lstm"] if rq1[lvl][m]]       # only the classifiers that were actually run
    vals = [rq1[lvl][m]["acc_mean"] for m in ms]        # mean accuracy over the 5 folds
    errs = [rq1[lvl][m]["acc_std"] for m in ms]     # error bar = std over the folds
    x = np.arange(len(ms))
    ax.bar(x, vals, yerr=errs, color=[MOD_COL[m] for m in ms], width=0.6, capsize=3,
           error_kw=dict(ecolor=INK2, lw=1))
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.02, f"{v:.1%}", ha="center", color=INK, fontsize=8)       # write the accuracy above each bar
    chance_line(ax, ch, len(ms) - 0.5)
    ax.set_xticks(x, [MOD_LAB[m] for m in ms]); ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Accuracy (mean ± sd over 5 folds)")
fig.savefig(f"{FIGURES}/fig_rq1_accuracy.png"); plt.close(fig)


def conf_plot(ax, conf, labels, title, fs=7.5):
    c = np.array(conf, dtype=float)
    c = c / c.sum(1, keepdims=True)     # row-normalize so each row adds up to 1
    ax.imshow(c, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):        # write the number inside each square
        for j in range(len(labels)):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=fs,
                    color="white" if c[i, j] > 0.55 else INK)       # white text on dark squares
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=fs)
    ax.set_yticks(range(len(labels)), labels, fontsize=fs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8.5)
    ax.grid(False)


for lvl, labels, w, fs in [("family", FAMILIES, 2.6, 8), ("model", MODELS, 3.6, 6.5)]:      # one confusion figure per level
    ms = [m for m in ["tfidf", "cnn", "lstm"] if rq1[lvl][m]]
    if ms:
        fig, axes = plt.subplots(1, len(ms), figsize=(w * len(ms), w * 0.95))
        for ax, m in zip(np.atleast_1d(axes), ms):      # one matrix per classifier
            conf_plot(ax, rq1[lvl][m]["pooled"]["confusion"], labels, MOD_LAB[m], fs)
        fig.tight_layout(); fig.savefig(f"{FIGURES}/fig_rq1_confusion_{lvl}.png"); plt.close(fig)

ms = [m for m in ["cnn", "lstm"] if rq1["family"][m]]       # training curves are only for the neural nets
if ms:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))      # left = train loss, right = validation accuracy
    for m in ms:
        h = pd.DataFrame(rq1["family"][m]["history_fold0"])     # fold 0's epoch-by-epoch history
        axes[0].plot(h.epoch, h.train_loss, color=MOD_COL[m], lw=2, label=MOD_LAB[m])
        axes[1].plot(h.epoch, h.val_acc, color=MOD_COL[m], lw=2, label=MOD_LAB[m])
    axes[0].set_ylabel("Train cross-entropy"); axes[1].set_ylabel("Validation accuracy")
    for a in axes:
        a.set_xlabel("Epoch (fold 0)"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGURES}/fig_training_curves.png"); plt.close(fig)

# ---------------- RQ2 ----------------
rq2 = {(m, v): (rq1["family"][m] if v == "output" else J(f"rq2_{m}_{v}.json"))
       for m in ["tfidf", "cnn", "lstm"] for v in ["input", "output", "both"]}       # output-only comes from RQ1, the others from RQ2
summary["rq2"] = {f"{m}|{v}": None if r is None else {"pooled_acc": r["pooled"]["accuracy"],
                  "fold_acc_mean": r["acc_mean"], "fold_acc_std": r["acc_std"],
                  "pooled_f1": r["pooled"]["macro_f1"]} for (m, v), r in rq2.items()}       # headline numbers for summary.json
fig, ax = plt.subplots(figsize=(5.8, 2.6))
w = 0.26        # bar width
for k, m in enumerate(["tfidf", "cnn", "lstm"]):        # one bar per classifier in each group
    vals = [rq2[(m, v)]["acc_mean"] if rq2[(m, v)] else np.nan for v in ["input", "output", "both"]]
    errs = [rq2[(m, v)]["acc_std"] if rq2[(m, v)] else 0 for v in ["input", "output", "both"]]
    xs = np.arange(3) + (k - 1) * w     # shift the bars so they sit side by side
    ax.bar(xs, vals, yerr=errs, width=w - 0.02, color=MOD_COL[m], label=MOD_LAB[m], capsize=2,
           error_kw=dict(ecolor=INK2, lw=0.8))
    for xi, v, e in zip(xs, vals, errs):
        if not np.isnan(v):     # skip anything that wasnt run
            ax.text(xi, v + e + 0.015, f"{v:.0%}", ha="center", fontsize=7, color=INK)
chance_line(ax, CH_FAM, 2.45)
ax.set_xticks(range(3), ["Input (prompt) only", "Output only", "Input + output"])
ax.set_ylabel("Accuracy (3 families)"); ax.set_ylim(0, 1.0)
ax.legend(ncol=3, fontsize=8, loc="upper left"); ax.grid(axis="x", visible=False)
fig.savefig(f"{FIGURES}/fig_rq2_views.png"); plt.close(fig)

sb = J("rq2_selection_bias.json")       # the selection-bias simulation ( None if not run )
if sb:
    summary["rq2_selection_bias"] = sb
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for k, m in enumerate(["tfidf", "cnn"]):
        vals = [sb[f"{m}_{v}"]["pooled_acc"] for v in ["input", "output", "both"]]
        xs = np.arange(3) + (k - 0.5) * 0.36
        ax.bar(xs, vals, 0.34, color=MOD_COL[m], label=MOD_LAB[m])
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=7)
    ax.axhline(sb["majority_baseline"], color=INK2, ls=":", lw=1)       # dotted line = always guessing the biggest class
    ax.text(2.5, sb["majority_baseline"] + 0.012, f"majority {sb['majority_baseline']:.0%}",
            ha="right", fontsize=7, color=INK2)
    ax.set_xticks(range(3), ["Input only", "Output only", "Input + output"])
    ax.set_ylim(0, 1); ax.set_ylabel("Pooled CV accuracy"); ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(f"{FIGURES}/fig_rq2_selection_bias.png"); plt.close(fig)

# ---------------- RQ3 ----------------
t3 = J("rq3_transfer.json")     # the held-out target results ( None if not run )
if t3:
    keys = [k for k in t3 if ":" in k]      # the target entries ( skips the prompt-count entry )
    rows = []       # one row per ( target, classifier )
    for k in keys:
        r = t3[k]
        for m in ["tfidf", "cnn", "lstm"]:
            if f"{m}_ID" in r:      # only the classifiers that were run
                rows.append({"target": r["target"], "group": r["group"], "model": m,
                             "ID": r[f"{m}_ID"]["accuracy"], "OOD": r[f"{m}_OOD"]["accuracy"],
                             "n_test_rows": r["n_test_rows"], "n_test_prompts": r["n_test_prompts"],
                             "OOD_seeds": r[f"{m}_OOD"].get("accuracy_seeds"),
                             "ID_per_class": r[f"{m}_ID"]["per_class_f1"],
                             "OOD_per_class": r[f"{m}_OOD"]["per_class_f1"]})
    rq3 = pd.DataFrame(rows)        # turn the list into a table
    summary["rq3"] = rows
    targets = [t3[k]["target"] for k in keys]       # target names in order
    models3 = [m for m in ["cnn", "tfidf", "lstm"] if m in set(rq3.model)]      # classifiers that have results
    fig, axes = plt.subplots(1, len(models3), figsize=(3.6 * len(models3), 2.7), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models3):     # one panel per classifier
        sub = rq3[rq3.model == m].set_index("target").reindex(targets)
        x = np.arange(len(targets))
        ax.bar(x - 0.2, sub.ID, 0.38, color="#2a78d6", label="ID (target seen)")
        ax.bar(x + 0.2, sub.OOD, 0.38, color="#eb6834", label="OOD (target held out)")
        lab = [f"{t.replace('_', ' ')}\n(n={n})" for t, n in zip(targets, sub.n_test_prompts)]      # target name + number of test prompts
        ax.set_xticks(x, lab, fontsize=6.5)
        ax.axvline(1.5, color=GRID, lw=1)       # separates the task types from the prompt sources
        ax.set_title(MOD_LAB[m], fontsize=9); chance_line(ax, CH_FAM, len(targets) - 0.5)
        ax.grid(axis="x", visible=False); ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Accuracy on target")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIGURES}/fig_rq3_transfer.png"); plt.close(fig)

# ---------------- RQ4 ----------------
if os.path.exists(f"{RESULTS}/rq4_style_features.csv"):     # only if rq4_features.py has been run
    st = pd.read_csv(f"{RESULTS}/rq4_style_features.csv")
    feats = [("n_words", "Words per response"), ("bold_per_100w", "Bold (**) spans per 100 words"),
             ("frac_header_lines", "Share of lines that are # headers"),
             ("frac_numbered_lines", "Share of lines that are numbered"),
             ("starts_certainly", "Share opening with Certainly/Sure/…"),
             ("mattr", "Lexical diversity (MATTR-50)")]        # the 6 features worth showing
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.7))
    g = st.groupby("llm_family")        # group the rows by family
    for ax, (f, title) in zip(axes.ravel(), feats):
        mu = g[f].mean().reindex(FAMILIES)      # average per family
        se = (g[f].std() / np.sqrt(g[f].count())).reindex(FAMILIES)     # standard error of that average
        ax.bar(range(len(FAMILIES)), mu, yerr=1.96 * se, color=[FAM_COL[x] for x in FAMILIES],
               width=0.65, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))       # error bars = 95% confidence interval
        ax.set_xticks(range(len(FAMILIES)), FAMILIES, fontsize=7.5); ax.set_title(title, fontsize=8)
        ax.grid(axis="x", visible=False)
        if f == "mattr":
            ax.set_ylim(0.7, 0.8)       # zoomed in, the differences are tiny
    fig.tight_layout(); fig.savefig(f"{FIGURES}/fig_rq4_style.png"); plt.close(fig)

ab = J("rq4_ablation.json")     # the ablation results ( None if not run )
if ab and rq1["family"]["cnn"] and rq1["family"]["tfidf"]:
    base = {m: rq1["family"][m]["pooled"]["accuracy"] for m in ["tfidf", "cnn"]}        # the unmodified-text accuracy comes from RQ1
    variants = [("original", "Original"), ("no_open_close", "No first/\nlast line"),
                ("no_format", "No\nformatting"), ("lexical_only", "No format +\nno first/last"),
                ("shuffled", "Words\nshuffled")]        # ( key in the results, label for the plot )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]        # left: accuracy after each text change
    for k, m in enumerate(["tfidf", "cnn"]):
        vals = [base[m] if v == "original" else ab.get(f"{m}_{v}", {}).get("pooled_acc", np.nan)
                for v, _ in variants]
        xs = np.arange(len(variants)) + (k - 0.5) * 0.38
        ax.bar(xs, vals, 0.36, color=MOD_COL[m], label=MOD_LAB[m])
        for xi, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.012, f"{v:.0%}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(variants)), [l for _, l in variants], fontsize=6.8)
    chance_line(ax, CH_FAM, len(variants) - 0.5)
    ax.set_ylabel("Pooled CV accuracy (3 families)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right"); ax.grid(axis="x", visible=False)
    ax = axes[1]        # right: accuracy when only the first N tokens are kept
    ns = [25, 50, 100, 200, 400]        # 400 = the full text ( from RQ1 )
    for m in ["tfidf", "cnn"]:
        pts = [(n, base[m] if n == 400 else ab.get(f"{m}_first{n}", {}).get("pooled_acc")) for n in ns]
        pts = [(a, b) for a, b in pts if b is not None]     # skip the points that werent run
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=MOD_COL[m], marker="o", ms=5, lw=2,
                label=MOD_LAB[m])
    ax.set_xscale("log"); ax.set_xticks(ns, [str(n) for n in ns]); ax.minorticks_off()
    ax.set_xlabel("Tokens kept (first N)"); ax.set_ylabel("Pooled CV accuracy")
    chance_line(ax, CH_FAM, 400); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIGURES}/fig_rq4_ablation.png"); plt.close(fig)
    summary["rq4_ablation"] = ab
    summary["rq4_ablation_base"] = base

feat = J("rq4_features.json")       # the style-only classifier results ( None if not run )
if feat:
    summary["rq4_style_only"] = {"lr": feat["style_lr"]["accuracy"], "rf": feat["style_rf"]["accuracy"],
                                 "rf_importance_top8": dict(list(feat["rf_importance"].items())[:8])}

with open(os.path.join(RESULTS, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)        # save every headline number in one file
print(json.dumps(summary, indent=1, default=str)[:3000])        # print the first part so its easy to eyeball
