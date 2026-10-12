"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

STEP 0 of the pipeline: build the ( LLM_name, LLM_input, LLM_output ) dataset

where the data comes from:
    the public AlpacaEval leaderboard repo ( tatsu-lab/alpaca_eval on GitHub ), which stores every
    evaluated model's answers to the same 805 prompts in results/<model>/model_outputs.json
    each record has: instruction, output, generator, dataset

what this script does:
    takes 2 models from each of 3 LLM families ( GPT, Claude, Llama ) and a random sample of
    N_PROMPTS = 100 of the 805 prompts. every model answers the same 100 prompts
    6 models x 100 prompts = 600 rows, perfectly balanced ( 200 per family, 100 per model )

usage:
    python src/build_dataset.py

Ryan Broadhurst, Michael Finn
"""
import json         # reading the downloaded files
import os           # folders and file paths
import urllib.request       # downloading the files from GitHub

import numpy as np      # using for RNG
import pandas as pd     # using a DataFrame (fancy spreadsheet)

BASE = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/{}/model_outputs.json"     # where each model's file lives ( {} = model folder name )

# model folder name in the AlpacaEval repo -> ( family, short name for the report )
MODELS = {
    "gpt4": ("GPT", "GPT-4"),
    "gpt-4o-2024-05-13": ("GPT", "GPT-4o"),
    "claude-3-opus-20240229": ("Claude", "Claude-3-Opus"),
    "claude-3-5-sonnet-20240620": ("Claude", "Claude-3.5-Sonnet"),
    "Meta-Llama-3-70B-Instruct": ("Llama", "Llama-3-70B"),
    "Meta-Llama-3-8B-Instruct": ("Llama", "Llama-3-8B"),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # project folder ( one level above src/ )
RAW = os.path.join(ROOT, "data", "raw")     # downloaded copies get saved here
SEED = 448      # fixed seed so the same 100 prompts get picked every time
N_PROMPTS = 100     # how many of the 805 prompts to keep
N_FOLDS = 5     # for the prompt-grouped cross-validation


def download(model_dir):
    path = os.path.join(RAW, f"{model_dir}.json")       # local copy of this model's file
    if not os.path.exists(path):        # only download it once
        urllib.request.urlretrieve(BASE.format(model_dir), path)
    with open(path) as f:
        return json.load(f)     # list of records


def main():
    os.makedirs(RAW, exist_ok=True)     # make the data/raw folder if its not there yet
    rows = []       # will hold every record from every model before it becomes a DataFrame
    for model_dir, (family, name) in MODELS.items():        # go thru each of the 6 models
        recs = download(model_dir)
        assert len(recs) == 805, (model_dir, len(recs))     # sanity check: each model should have all 805 answers
        for r in recs:      # go thru each record
            rows.append({
                "llm_family": family,       # GPT / Claude / Llama
                "llm_name": name,       # the specific model
                "llm_input": r["instruction"],      # the prompt
                "llm_output": r["output"] if isinstance(r["output"], str) else str(r["output"]),        # the response ( forced to be text )
                "domain": r["dataset"],     # which collection the prompt came from
            })
    df = pd.DataFrame(rows)     # turn the list into a table

    # prompt id: identical instructions share an id ( the same 805 prompts for every model )
    all_prompts = sorted(df["llm_input"].unique())      # every distinct prompt
    rng = np.random.RandomState(SEED)
    prompts = sorted(rng.choice(all_prompts, size=N_PROMPTS, replace=False))        # randomly pick the 100 to keep
    df = df[df["llm_input"].isin(prompts)].reset_index(drop=True)       # drop all the other prompts
    pid = {p: i for i, p in enumerate(prompts)}     # prompt text -> id number
    df["prompt_id"] = df["llm_input"].map(pid)      # tag every row with its prompt id

    # prompt-grouped 5-fold assignment: every prompt ( and so all 6 answers to it )
    # belongs to exactly one fold, so no question is ever seen at both train and test time
    # in cross-validation round k: test = fold k, val = fold (k+1) mod 5, train = other 3 folds
    perm = rng.permutation(len(prompts))        # shuffled order of the prompt ids
    fold_of = {int(idx): k % N_FOLDS for k, idx in enumerate(perm)}     # deal them out to the 5 folds like cards
    df["fold"] = df["prompt_id"].map(fold_of)       # tag every row with its fold

    empty = (df["llm_output"].str.strip() == "").sum()      # sanity check: should be 0 empty responses
    print(f"rows={len(df)}  unique prompts={len(prompts)}  empty outputs={empty}")
    print(df.groupby(["llm_family", "llm_name"]).size())        # should be 100 per model
    print(df.groupby("fold").size())        # should be 120 per fold
    print(df.drop_duplicates("prompt_id").groupby("domain").size())     # how many prompts came from each source
    df.to_csv(os.path.join(ROOT, "data", "dataset.csv"), index=False)       # save the finished dataset


if __name__ == "__main__":
    main()
