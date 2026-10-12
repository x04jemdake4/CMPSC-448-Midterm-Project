"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

RQ3: do LLM fingerprints generalize across tasks / domains?

for each held-out target T ( a task type or a prompt-source domain ):
    OOD : train on all prompts that are NOT in T ( 85% train / 15% val for early stopping ),
          test on every row of T. the classifier has never seen T
    ID  : the RQ1 cross-validated predictions, only the rows of T. those come from classifiers
          trained on a 60% prompt sample that DID include other prompts of T
    gap = ID - OOD. the OOD models are trained on MORE prompts than the ID models
          ( 85% of the non-T prompts vs 60% of all prompts ), so a drop under OOD cant be
          blamed on less training data. if anything this makes the measured gap conservative

targets:
    task type ( keyword-tagged, as suggested in the assignment ):
        'writing' ( creative/professional writing ), 'tech' ( coding/math )
        -> ex: train on general QA ( + the other type ), test on writing
    source domain ( AlpacaEval prompt sources ): leave-one-domain-out over all 5

Ryan Broadhurst, Michael Finn
"""
import argparse     # lets me pick which models/seeds to run from the command line
import re       # keyword matching for the task types

import numpy as np      # using for RNG

from common import DOMAINS, FAMILIES, load_data, load_json, metrics, save_json
from train import tfidf_lr, train_eval

TECH = re.compile(r"\b(code|python|javascript|java|c\+\+|sql|function|program|algorithm|regex|html|"
                  r"css|bash|script|compute|calculate|solve|equation|math|integral|derivative|probability)\b")     # words that mean a coding / math prompt
WRITE = re.compile(r"\b(write|compose|draft|poem|story|essay|letter|email|lyrics|song|haiku|tweet|"
                   r"slogan|joke|blog|limerick|rap|speech|narrative|dialogue)\b")       # words that mean a writing prompt


def task_type(prompt):
    s = prompt.lower()      # lowercase so the keywords match
    if TECH.search(s):      # tech check goes first
        return "tech"
    if WRITE.search(s):
        return "writing"
    return "qa"     # everything else = general QA / advice


ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["tfidf", "cnn"])        # which models to run
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])      # training seeds for the CNN ( results get averaged )
args = ap.parse_args()

df = load_data()        # the 600-row dataset
df["task"] = df["llm_input"].map(task_type)     # tag every row with its task type
lab2id = {l: i for i, l in enumerate(FAMILIES)}     # class name -> class number
col = "llm_family"      # predicting the family
id_preds = {m: {int(k): v for k, v in load_json(f"rq1_family_{m}.json")["pred"].items()}
            for m in args.models}       # the saved RQ1 predictions ( these are the ID numbers )

targets = [("task", "writing"), ("task", "tech")] + [("domain", d) for d in DOMAINS]        # 2 task types + 5 prompt sources
out = {"task_prompt_counts": df.drop_duplicates("prompt_id").task.value_counts().to_dict()}     # how many prompts of each task type
for group, target in targets:       # go thru each held-out target
    test = df[df[group] == target]      # the target = test set
    rest = df[df[group] != target]      # everything else = train + val
    rng = np.random.RandomState(0)
    rest_prompts = rest.prompt_id.unique()      # prompts available for training
    val_prompts = rng.choice(rest_prompts, size=max(1, int(0.15 * len(rest_prompts))), replace=False)      # 15% of them held back for early stopping
    tr, va = rest[~rest.prompt_id.isin(val_prompts)], rest[rest.prompt_id.isin(val_prompts)]        # split by prompt so all answers to a prompt stay together
    y = test[col].map(lab2id).values        # answer key for the target
    res = {"group": group, "target": target, "n_test_rows": len(test),
           "n_test_prompts": int(test.prompt_id.nunique()),
           "n_train_prompts_OOD": int(tr.prompt_id.nunique())}      # info about this target
    for m in args.models:
        # ID: pooled RQ1 cross-validated predictions on the same rows
        res[f"{m}_ID"] = metrics(y, np.array([id_preds[m][int(i)] for i in test.index]), FAMILIES)
        if m == "tfidf":        # the baseline has no seeds or validation set
            r = tfidf_lr(tr, test, col, FAMILIES)
            r.pop("test_pred")      # dont need the raw guesses
            res[f"{m}_OOD"] = r
        else:       # CNN
            runs = [train_eval(tr, va, test, col, FAMILIES, model_name=m, seed=s, verbose=False)
                    for s in args.seeds]        # train once per seed
            # majority vote over seeds is NOT used; we average the metrics and keep seed 0's confusion
            r = runs[0]
            for k in ("history", "test_pred", "config"):
                r.pop(k, None)      # drop the big stuff we dont need
            r["accuracy_seeds"] = [x["accuracy"] for x in runs]     # accuracy of each seed
            r["accuracy"] = float(np.mean(r["accuracy_seeds"]))     # averaged over seeds
            r["macro_f1"] = float(np.mean([x["macro_f1"] for x in runs]))
            res[f"{m}_OOD"] = r
        print(f"[RQ3 {group}={target}] {m}: ID {res[f'{m}_ID']['accuracy']:.3f}  "
              f"OOD {res[f'{m}_OOD']['accuracy']:.3f}  (n_test={len(test)})", flush=True)
    out[f"{group}:{target}"] = res      # add this target to the results
    save_json(out, "rq3_transfer.json")     # save after every target so progress isnt lost
