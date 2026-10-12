"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

RQ2: does the user's prompt help identify the LLM?

part A ( main ): family-level classification with 3 input views
    input-only | output-only | input + output
    for TF-IDF+LR, CNN and BiLSTM, all with prompt-grouped 5-fold CV
    ( the output-only runs are re-used from RQ1 )

part B ( why could input-only ever work? ): a controlled "selection bias" simulation
    in the main dataset every prompt was answered by all 6 models, so the prompt carries no
    info about the label and input-only should be at chance ( 33% ). real-world logs are
    different because people choose which model to ask. i simulate that by keeping, for each
    prompt, only the answers of ONE family. the family is drawn with a domain-dependent
    preference: 70% the domain's "favourite" family, 15% each other family

Ryan Broadhurst, Michael Finn
"""
import argparse     # lets me pick which parts/models to run from the command line

import numpy as np      # using for RNG

from common import DOMAINS, FAMILIES, load_data, save_json
from train import cv_run

ap = argparse.ArgumentParser()
ap.add_argument("--part", nargs="+", default=["A", "B"])        # which parts to run
ap.add_argument("--models", nargs="+", default=["tfidf", "cnn", "lstm"])        # which models to run in part A
args = ap.parse_args()

df = load_data()        # the 600-row dataset
col = "llm_family"      # both parts predict the family

if "A" in args.part:
    for m in args.models:       # go thru each classifier
        for view in ["input", "both"]:      # output-only already exists from RQ1
            r = cv_run(df, col, FAMILIES, m, view=view)     # 5-fold cross-validation with this view
            print(f"[RQ2] {m} {view}: fold acc {r['acc_mean']:.3f} +- {r['acc_std']:.3f} | "
                  f"pooled {r['pooled']['accuracy']:.3f}", flush=True)
            save_json(r, f"rq2_{m}_{view}.json")        # save for make_figures.py

if "B" in args.part:
    rng = np.random.RandomState(448)        # fixed seed so the simulation is reproducible
    fav = {d: FAMILIES[i % len(FAMILIES)] for i, d in enumerate(DOMAINS)}   # domain -> favourite family
    keep_fam = {}       # prompt id -> the one family whose answers we keep
    for pid, dom in df.drop_duplicates("prompt_id")[["prompt_id", "domain"]].values:        # go thru each prompt once
        p = np.full(len(FAMILIES), 0.15)        # 15% for every family...
        p[FAMILIES.index(fav[dom])] = 0.70      # ...except the domain's favourite gets 70%
        keep_fam[pid] = FAMILIES[rng.choice(len(FAMILIES), p=p)]        # roll the dice
    biased = df[df.apply(lambda r: keep_fam[r.prompt_id] == r.llm_family, axis=1)]      # keep only the chosen family's answers for each prompt
    out = {"favourite_family_per_domain": fav, "n_rows": len(biased),
           "label_dist": biased[col].value_counts().to_dict(),
           "majority_baseline": float(biased[col].value_counts(normalize=True).max())}      # what always guessing the biggest class would score
    for m in ["tfidf", "cnn"]:      # ( skipping the lstm here, its slow )
        for view in ["input", "output", "both"]:
            r = cv_run(biased, col, FAMILIES, m, view=view)     # 5-fold cross-validation on the biased data
            out[f"{m}_{view}"] = {"acc_mean": r["acc_mean"], "acc_std": r["acc_std"],
                                  "pooled_acc": r["pooled"]["accuracy"],
                                  "pooled_macro_f1": r["pooled"]["macro_f1"]}       # keep the numbers we need
            print(f"[RQ2-B biased] {m} {view}: pooled acc {r['pooled']['accuracy']:.3f}", flush=True)
    save_json(out, "rq2_selection_bias.json")       # save for make_figures.py
