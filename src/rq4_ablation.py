"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

RQ4 ( part 2 ): which signals does the classifier rely on?   ablation study

we re-train the CNN ( and the TF-IDF baseline ) on modified versions of the responses, with the
same prompt-grouped 5-fold CV, and measure how much family-level accuracy drops compared to the
unmodified text ( the RQ1 numbers )

    original        : unchanged ( re-used from RQ1 )
    no_format       : markdown removed ( **, #, bullets, numbering, code fences ) and all
                      newlines/whitespace collapsed -> plain running text
    no_open_close   : first and last line removed ( greetings such as "Certainly! Here is..."
                      and sign-offs such as "I hope this helps!" )
    lexical_only    : no_format + no_open_close  ( only the wording of the body is left )
    shuffled        : tokens randomly permuted inside each response ( keeps the vocabulary,
                      destroys word order, n-grams and layout )
    first_N tokens  : only the first N tokens ( N = 25, 50, 100, 200 ) -> how much text is needed

Ryan Broadhurst, Michael Finn
"""
import random       # shuffling the tokens
import re       # stripping markdown

from common import FAMILIES, load_data, save_json, tokenize
from train import cv_run


def no_format(t):
    t = re.sub(r"```[a-zA-Z]*", " ", t)     # code fence markers ( the code itself stays )
    t = t.replace("**", "").replace("__", "")       # bold markers
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)     # markdown headers
    t = re.sub(r"(?m)^\s*([-*•]|\d+[.)])\s+", "", t)        # bullet + numbered list markers
    return re.sub(r"\s+", " ", t).strip()       # collapse ALL whitespace ( kills newlines too )


def no_open_close(t):
    lines = [l for l in t.split("\n") if l.strip()]     # non-empty lines only
    return "\n".join(lines[1:-1]) if len(lines) > 2 else t      # drop first + last line ( unless theres almost nothing there )


def shuffled(t, rng=random.Random(0)):
    toks = ["\n" if x == "<nl>" else x for x in tokenize(t)]        # tokenize, newline tokens turned back into real newlines
    rng.shuffle(toks)       # destroy the word order
    return " ".join(toks)       # glue back into one string


VARIANTS = {
    "no_format": no_format,
    "no_open_close": no_open_close,
    "lexical_only": lambda t: no_format(no_open_close(t)),      # both changes together
    "shuffled": shuffled,
}

df = load_data()        # the 600-row dataset
col = "llm_family"      # predicting the family
out = {}        # all the results get collected here
for name, fn in VARIANTS.items():       # go thru each text change
    for m in ["tfidf", "cnn"]:
        r = cv_run(df, col, FAMILIES, m, transform=fn)      # 5-fold cross-validation on the modified responses
        out[f"{m}_{name}"] = {"pooled_acc": r["pooled"]["accuracy"], "acc_mean": r["acc_mean"],
                              "acc_std": r["acc_std"], "pooled_macro_f1": r["pooled"]["macro_f1"]}      # keep the numbers we need
    print(f"[RQ4 {name}] " + " ".join(f"{k}={v['pooled_acc']:.3f}" for k, v in out.items()
                                      if k.endswith(name)), flush=True)
    save_json(out, "rq4_ablation.json")     # save after every variant so progress isnt lost

for n in [25, 50, 100, 200]:        # how much text is needed?
    for m in (["tfidf", "cnn"] if n in (50, 100) else ["tfidf"]):       # CNN only at 50 and 100 ( saves time )
        r = cv_run(df, col, FAMILIES, m, max_len=n)     # only keep the first n tokens
        out[f"{m}_first{n}"] = {"pooled_acc": r["pooled"]["accuracy"], "acc_mean": r["acc_mean"],
                                "acc_std": r["acc_std"]}
    print(f"[RQ4 first{n}] " + " ".join(f"{k}={v['pooled_acc']:.3f}" for k, v in out.items()
                                        if k.endswith(f"first{n}")), flush=True)
    save_json(out, "rq4_ablation.json")
