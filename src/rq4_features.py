"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

RQ4 ( part 1 ): what characteristics distinguish the LLMs?   ( no neural nets needed )

    1. hand-made stylometric features per response ( length, lexical diversity, formatting,
       punctuation, stock phrases ) -> per-family / per-model statistics
    2. a classifier on ONLY these ~20 interpretable features ( how far does style alone get? )
    3. the most discriminative n-grams per family from the TF-IDF + logistic regression model

Ryan Broadhurst, Michael Finn
"""
import re       # counting markdown symbols and phrases

import numpy as np      # using for RNG
import pandas as pd     # using a DataFrame (fancy spreadsheet)
from sklearn.ensemble import RandomForestClassifier     # a second classifier, tells us which features matter most
from sklearn.linear_model import LogisticRegression     # same classifier family as hw1
from sklearn.pipeline import make_pipeline      # chains scaling + the model together
from sklearn.preprocessing import StandardScaler        # shifts and adjusts values for 0 means and 1 standard deviations

from common import FAMILIES, N_FOLDS, RESULTS, load_data, metrics, save_json
from baseline import tfidf_lr

WORD = re.compile(r"[A-Za-z']+")        # a word = letters and apostrophes


def mattr(words, w=50):
    """moving-average type-token ratio ( window 50 ): lexical diversity that isnt
    biased by response length the way plain TTR is"""
    if len(words) < w:      # too short for a full window, so just use plain TTR
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i:i + w])) / w for i in range(0, len(words) - w + 1, 10)]))     # unique words / 50, averaged over sliding windows


def features(text):
    lines = text.split("\n")        # every line
    nonempty = [l for l in lines if l.strip()]      # lines with something on them
    words = [w.lower() for w in WORD.findall(text)]     # lowercase words
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]      # rough sentence split
    nl = max(len(nonempty), 1)      # number of lines ( never 0, so we can divide by it )
    first = text.strip().split("\n")[0].lower()     # the opening line
    return {
        "n_words": len(words),      # --- length ---
        "n_chars": len(text),
        "n_lines": len(nonempty),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,      # --- language style ---
        "avg_sent_len": len(words) / max(len(sents), 1),
        "mattr": mattr(words),      # lexical diversity
        "frac_header_lines": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in nonempty) / nl,     # --- structure / formatting ---
        "frac_bullet_lines": sum(bool(re.match(r"\s*[-*•]\s", l)) for l in nonempty) / nl,
        "frac_numbered_lines": sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in nonempty) / nl,
        "bold_per_100w": 100 * text.count("**") / 2 / max(len(words), 1),       # ** comes in pairs, so divide by 2
        "has_code_block": int("```" in text),
        "blank_line_ratio": (len(lines) - len(nonempty)) / max(len(lines), 1),
        "colon_per_100w": 100 * text.count(":") / max(len(words), 1),       # --- punctuation ---
        "exclam_per_100w": 100 * text.count("!") / max(len(words), 1),
        "comma_per_100w": 100 * text.count(",") / max(len(words), 1),
        "starts_certainly": int(first.startswith(("certainly", "sure", "absolutely", "of course"))),        # --- stock phrases ---
        "starts_heading_or_bold": int(text.lstrip().startswith(("#", "**"))),
        "ends_with_question": int(text.strip().endswith("?")),
        "has_hope_helps": int(bool(re.search(r"i hope (this|that) helps", text.lower()))),
        "has_note_disclaimer": int(bool(re.search(r"\b(note|keep in mind|remember|however)\b", text.lower()))),
        "first_person_I_per_100w": 100 * sum(w == "i" for w in words) / max(len(words), 1),
    }


def main():
    df = load_data()        # the 600-row dataset
    F = pd.DataFrame([features(t) for t in df["llm_output"]])       # one row of features per response
    F[["llm_family", "llm_name", "fold", "domain"]] = df[["llm_family", "llm_name", "fold", "domain"]]      # add the labels back on
    F.to_csv(f"{RESULTS}/rq4_style_features.csv", index=False)      # saved for the figure
    cols = [c for c in F.columns if c not in ("llm_family", "llm_name", "fold", "domain")]      # just the feature columns

    by_fam = F.groupby("llm_family")[cols].mean().round(3)      # average of every feature per family
    by_model = F.groupby("llm_name")[cols].mean().round(3)      # ...and per model
    by_fam.to_csv(f"{RESULTS}/rq4_style_by_family.csv")
    by_model.to_csv(f"{RESULTS}/rq4_style_by_model.csv")
    print(by_fam.T)     # rows = features, columns = families

    # classifier on stylometric features only ( same prompt-grouped 5-fold CV as RQ1;
    # the val fold is not needed and is added to training )
    lab2id = {l: i for i, l in enumerate(FAMILIES)}     # class name -> class number
    y = F.llm_family.map(lab2id).values     # answer key
    out = {}        # results get collected here
    for name, make in [("style_lr", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                       ("style_rf", lambda: RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=2))]:        # 2 classifiers, same features
        pred = np.zeros(len(F), dtype=int)      # one guess per row
        for k in range(N_FOLDS):        # go thru each cross-validation round
            te = (F.fold == k).values       # test rows = fold k
            clf = make().fit(F.loc[~te, cols], y[~te])      # learn from every other fold
            pred[te] = clf.predict(F.loc[te, cols])     # guess the test fold
        out[name] = metrics(y, pred, FAMILIES)      # accuracy / f1 / confusion matrix over all 600
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=2).fit(F[cols], y)     # fit on everything, only to read the feature importances
    out["rf_importance"] = dict(sorted(zip(cols, rf.feature_importances_.round(4).tolist()),
                                       key=lambda kv: -kv[1]))      # most important feature first
    print("style-only LR acc", out["style_lr"]["accuracy"], " RF acc", out["style_rf"]["accuracy"])
    print("RF importances", list(out["rf_importance"].items())[:8])     # the top 8

    # most discriminative n-grams per family from TF-IDF + LR fitted on all 600 responses
    # ( used for interpretation only, not for evaluation )
    _, vec, clf = tfidf_lr(df, df, "llm_family", FAMILIES, return_model=True)
    names = np.array(vec.get_feature_names_out())       # the n-grams
    top = {}        # family -> its top n-grams
    for k, fam in enumerate(FAMILIES):
        order = np.argsort(-clf.coef_[k])[:25]      # the 25 n-grams with the biggest weight for this family
        top[fam] = [(names[j], round(float(clf.coef_[k][j]), 3)) for j in order]
        print(fam, [n for n, _ in top[fam][:15]])
    out["top_ngrams"] = top
    save_json(out, "rq4_features.json")


if __name__ == "__main__":
    main()
