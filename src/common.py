"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

shared helpers that every other script imports
    - file paths + constants ( families, models, domains, number of folds )
    - the tokenizer and the vocabulary
    - loading the dataset and the prompt-grouped fold split
    - metrics + saving/loading the results files

Ryan Broadhurst, Michael Finn
"""
import json         # saving + loading the results files
import os           # building file paths
import random       # python RNG seed
import re           # the tokenizer is a regular expression
from collections import Counter     # counting how often each token shows up

import numpy as np      # using for RNG
import pandas as pd     # using a DataFrame (fancy spreadsheet)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score      # metrics for scorekeeping functions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # project folder ( one level above src/ )
DATA = os.path.join(ROOT, "data", "dataset.csv")        # the dataset made by build_dataset.py
RESULTS = os.path.join(ROOT, "results")         # all the .json results get saved here
FIGURES = os.path.join(ROOT, "figures")         # all the figures get saved here

FAMILIES = ["Claude", "GPT", "Llama"]       # the 3 classes for the main task ( keep this order everywhere )
MODELS = ["Claude-3-Opus", "Claude-3.5-Sonnet", "GPT-4", "GPT-4o", "Llama-3-70B", "Llama-3-8B"]     # the 6 classes for the harder task
N_FOLDS = 5         # prompt-grouped cross-validation, 5 rounds
DOMAINS = ["helpful_base", "koala", "oasst", "selfinstruct", "vicuna"]      # the 5 places AlpacaEval's prompts came from

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"       # special tokens: padding, unknown word, prompt/response separator

# tokenizer that keeps formatting info:
#   newlines become their own token, and markdown markers ( **, ###, ```, bullets, numbered lists )
#   stay as separate tokens, because layout is part of an LLM's "fingerprint"
#   words get lowercased
_TOKEN_RE = re.compile(r"\n|\*\*|#{1,6}|`{3}|\d+\.|\w+|[^\w\s]")        # newline | bold | headers | code fence | list number | word | any symbol


def tokenize(text):
    toks = []       # will hold the finished tokens
    for t in _TOKEN_RE.findall(str(text)):      # split the text into pieces
        toks.append("<nl>" if t == "\n" else t.lower())     # newline -> <nl>, everything else lowercased
    return toks


def set_seed(seed):
    random.seed(seed)       # python RNG
    np.random.seed(seed)    # numpy RNG
    try:
        import torch
        torch.manual_seed(seed)     # torch RNG ( weight init, dropout, etc )
    except ImportError:     # torch isnt installed ( ex: only running the tf-idf baseline )
        pass


def load_data():
    return pd.read_csv(DATA, keep_default_na=False)     # keep_default_na so a response that says "NA" doesnt turn into a missing value


def fold_split(df, k):
    """round k of the prompt-grouped cross-validation: test = fold k, val = fold k+1, train = the other 3"""
    v = (k + 1) % N_FOLDS       # validation fold is the next one ( wraps around )
    return df[~df.fold.isin([k, v])], df[df.fold == v], df[df.fold == k]        # train, val, test


class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=30000):
        cnt = Counter(t for toks in token_lists for t in toks)      # count every token in the training texts
        itos = [PAD, UNK, SEP] + [t for t, c in cnt.most_common(max_size) if c >= min_freq and t != SEP]        # specials first, then common tokens ( skip rare ones )
        self.itos = itos        # id -> token
        self.stoi = {t: i for i, t in enumerate(itos)}      # token -> id

    def __len__(self):
        return len(self.itos)       # vocab size

    def encode(self, toks):
        unk = self.stoi[UNK]        # id for unknown words
        return [self.stoi.get(t, unk) for t in toks]        # turn tokens into ids ( unknown -> <unk> )


def make_tokens(df, view, max_in=128, max_len=400):
    """return token lists for one input view

    view = 'output' : the LLM response only
    view = 'input'  : the user prompt only
    view = 'both'   : prompt ( cut to max_in tokens ) + <sep> + response
    """
    outs = []       # one token list per row
    for inp, out in zip(df["llm_input"], df["llm_output"]):     # go thru each prompt/response pair
        if view == "output":        # RQ1: response only
            toks = tokenize(out)
        elif view == "input":       # RQ2: prompt only
            toks = tokenize(inp)
        elif view == "both":        # RQ2: prompt + separator + response
            toks = tokenize(inp)[:max_in] + [SEP] + tokenize(out)
        else:
            raise ValueError(view)      # not a valid view
        outs.append(toks[:max_len])     # cut everything to max_len tokens
    return outs


def metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),      # % of guesses that were correct
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(labels))))),      # f1 averaged over classes equally
        "per_class_f1": {labels[i]: float(v) for i, v in enumerate(
            f1_score(y_true, y_pred, average=None, labels=list(range(len(labels)))))},      # f1 for each class by name
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),        # rows = true label, columns = predicted label
        "labels": labels,       # so the matrix can be read later
    }


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)     # make the results folder if its not there yet
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2)     # indent so its readable


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
