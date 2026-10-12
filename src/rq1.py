"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

RQ1: can we identify which LLM generated a response?    ( output-only classification )

two levels, both with prompt-grouped 5-fold cross-validation:
    family-level ( 3 classes: Claude, GPT, Llama )  <- the main task
    model-level  ( 6 classes: two models per family )
models: TextCNN and BiLSTM ( required ) + the TF-IDF / logistic regression reference baseline

Ryan Broadhurst, Michael Finn
"""
import argparse     # lets me pick which models/levels to run from the command line

from common import FAMILIES, MODELS, load_data, save_json
from train import cv_run

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["tfidf", "cnn", "lstm"])        # which models to run
ap.add_argument("--levels", nargs="+", default=["family", "model"])     # which task levels to run
args = ap.parse_args()

df = load_data()        # the 600-row dataset
TASKS = {"family": ("llm_family", FAMILIES), "model": ("llm_name", MODELS)}     # level -> ( label column, class names )

for level in args.levels:       # family first, then model
    col, labels = TASKS[level]
    for m in args.models:       # go thru each classifier
        r = cv_run(df, col, labels, m, view="output")       # 5-fold cross-validation on the response text only
        print(f"[RQ1 {level}] {m}: fold acc {r['acc_mean']:.3f} +- {r['acc_std']:.3f} | "
              f"pooled acc {r['pooled']['accuracy']:.3f} macro-F1 {r['pooled']['macro_f1']:.3f}",
              flush=True)       # mean +- std over the 5 folds, and pooled over all 600 predictions
        save_json(r, f"rq1_{level}_{m}.json")       # save for make_figures.py and the other RQs
