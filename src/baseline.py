"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

the non-neural reference baseline: TF-IDF word n-grams + logistic regression
( same model family as hw1 ) - shows what the CNN / BiLSTM add over plain n-gram counts

Ryan Broadhurst, Michael Finn
"""
from sklearn.feature_extraction.text import TfidfVectorizer     # bag of words / n-grams baseline
from sklearn.linear_model import LogisticRegression     # is the logistic regression model

from common import make_tokens, metrics


def tfidf_lr(train_df, test_df, label_col, labels, view="output", max_len=400, C=10.0,
             return_model=False):
    """bag-of-n-grams baseline: TF-IDF ( 1-2 grams over the same tokens ) + logistic regression"""
    lab2id = {l: i for i, l in enumerate(labels)}       # class name -> class number
    joiner = lambda df: [" ".join(t) for t in make_tokens(df, view, max_len=max_len)]       # tokens back into one space-separated string per row
    vec = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False,
                          token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          max_features=100000)      # split on spaces only ( tokenizing was already done ), 1-2 word phrases, rare ones skipped
    Xtr = vec.fit_transform(joiner(train_df))       # learn the vocab on TRAIN only, then turn train into numbers
    Xte = vec.transform(joiner(test_df))        # apply that same vocab to test ( no re-learning )
    clf = LogisticRegression(C=C, max_iter=2000)        # set up the model
    clf.fit(Xtr, train_df[label_col].map(lab2id).values)        # this is the line that actually learns from the training data
    pred = clf.predict(Xte)     # have the model guess the test labels
    res = metrics(test_df[label_col].map(lab2id).values, pred, labels)      # accuracy / f1 / confusion matrix
    res.update({"model": "tfidf_lr", "view": view, "test_pred": pred.tolist()})     # tag the result and keep the guesses
    if return_model:        # RQ4 needs the fitted vectorizer + model to look at the top n-grams
        return res, vec, clf
    return res
