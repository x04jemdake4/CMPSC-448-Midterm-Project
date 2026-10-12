"""
CMPSC 448 - Midterm Project
Who Wrote It? Identifying LLMs from Their Responses

the training + evaluation loop shared by every experiment
    - train_eval : train ONE model on one train/val/test split, early-stop on validation macro-F1
    - cv_run     : prompt-grouped 5-fold cross-validation ( every row gets tested exactly once )

Ryan Broadhurst, Michael Finn
"""
import copy         # ( not used directly )
import time         # timing how long training takes

import numpy as np      # using for RNG
import torch        # PyTorch, the deep learning library
import torch.nn as nn       # layers + the loss function

from common import N_FOLDS, Vocab, fold_split, make_tokens, metrics, set_seed
from models import build_model
from baseline import tfidf_lr  # noqa: F401  (re-export)     # used in cv_run below, and also re-exported so rq3.py can import it from train.py

import os       # reading settings from environment variables
if os.environ.get("TORCH_THREADS"):            # default: PyTorch uses all CPU cores
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))     # optional: limit how many cores it uses
# device: NVIDIA GPU if there is one, else CPU. apple-silicon GPU is opt-in with DEVICE=mps
DEVICE = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))

# small dataset ( 360 training rows per fold ): smaller batches, more epochs, longer patience
DEFAULTS = dict(max_len=400, batch_size=32, lr=1e-3, epochs=40, patience=8, clip=5.0)       # tokens kept, rows per step, Adam learning rate, epoch cap, early-stop patience, gradient clip
if os.environ.get("QUICK"):                    # smoke test only: 1 epoch per model
    DEFAULTS.update(epochs=1, patience=1)       # just checks the code runs, results mean nothing


def _encode(token_lists, vocab, max_len):
    ids = [vocab.encode(t)[:max_len] or [1] for t in token_lists]       # tokens -> ids, cut to max_len, never empty ( 1 = <unk> )
    return ids


def _batches(ids, y, batch_size, shuffle, rng):
    """yield padded batches. training batches are bucketed by length ( shuffled within
    buckets ) so the LSTM wastes little time on padding"""
    idx = np.arange(len(ids))       # row numbers 0..n-1
    if shuffle:     # training: random order, but similar lengths together
        rng.shuffle(idx)
        chunk = batch_size * 50     # sort inside chunks of 50 batches
        idx = np.concatenate([sorted(idx[i:i + chunk], key=lambda j: len(ids[j]))
                              for i in range(0, len(idx), chunk)])      # sort each chunk by length
        starts = list(range(0, len(idx), batch_size))       # where each batch starts
        rng.shuffle(starts)     # shuffle the batch ORDER so short ones dont all come first
    else:       # validation / test: keep the original order so predictions line up with rows
        starts = list(range(0, len(idx), batch_size))
    for s in starts:        # go thru each batch
        b = idx[s:s + batch_size]       # row numbers in this batch
        lens = [len(ids[j]) for j in b]     # real length of each sequence ( before padding )
        T = max(max(lens), 5)                # >= widest CNN filter
        x = np.zeros((len(b), T), dtype=np.int64)       # 0 = <pad>
        for r, j in enumerate(b):
            x[r, :lens[r]] = ids[j]     # drop the real ids in
        yield torch.from_numpy(x), torch.tensor(lens), torch.tensor(y[b]), b


def _predict(model, ids, y, bs):
    model.eval()        # turns dropout off
    preds = np.zeros(len(ids), dtype=np.int64)      # one guess per row
    with torch.no_grad():       # no learning here, just guessing
        for x, lens, _, b in _batches(ids, y, bs, False, None):
            preds[b] = model(x.to(DEVICE), lens).argmax(1).cpu().numpy()        # highest score = predicted class
    return preds


def train_eval(train_df, val_df, test_df, label_col, labels, view="output", model_name="cnn",
               seed=0, verbose=True, **kw):
    """train `model_name` on train_df, early-stop on val macro-F1, report test metrics"""
    base = dict(DEFAULTS)       # start from the default settings
    if model_name == "lstm" and os.environ.get("FAST_LSTM"):   # optional shorter schedule for slow CPUs
        base.update(epochs=25, patience=5)
    cfg = {**base, **kw}        # any settings passed in override the defaults
    set_seed(seed)      # same seed -> same weights every time
    rng = np.random.RandomState(seed)       # separate RNG for shuffling batches
    lab2id = {l: i for i, l in enumerate(labels)}       # class name -> class number
    y_tr = train_df[label_col].map(lab2id).values       # answer key for train
    y_va = val_df[label_col].map(lab2id).values     # answer key for validation
    y_te = test_df[label_col].map(lab2id).values        # answer key for test

    tok_tr = make_tokens(train_df, view, max_len=cfg["max_len"])        # train tokens
    vocab = Vocab(tok_tr)       # vocab from TRAIN only ( no peeking at val/test )
    ids_tr = _encode(tok_tr, vocab, cfg["max_len"])     # tokens -> ids
    ids_va = _encode(make_tokens(val_df, view, max_len=cfg["max_len"]), vocab, cfg["max_len"])
    ids_te = _encode(make_tokens(test_df, view, max_len=cfg["max_len"]), vocab, cfg["max_len"])

    model = build_model(model_name, len(vocab), len(labels)).to(DEVICE)     # set up the model
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])        # Adam optimizer
    loss_fn = nn.CrossEntropyLoss()     # softmax + negative log likelihood

    best, best_f1, bad, history = None, -1.0, 0, []     # best weights so far, best val f1, epochs without improvement, curve for the figures
    t0 = time.time()        # start the clock
    for ep in range(1, cfg["epochs"] + 1):      # one pass over the training data per epoch
        model.train()       # turns dropout on
        tot, n = 0.0, 0     # running loss + number of rows
        for x, lens, yb, _ in _batches(ids_tr, y_tr, cfg["batch_size"], True, rng):     # go thru each batch
            opt.zero_grad()     # clear old gradients
            loss = loss_fn(model(x.to(DEVICE), lens), yb.to(DEVICE))        # forward pass + loss
            loss.backward()     # backprop
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])       # keeps LSTM gradients from blowing up
            opt.step()      # update the weights
            tot += loss.item() * len(yb)
            n += len(yb)
        va = metrics(y_va, _predict(model, ids_va, y_va, 256), labels)      # check on validation data
        history.append({"epoch": ep, "train_loss": tot / n, "val_acc": va["accuracy"],
                        "val_macro_f1": va["macro_f1"], "time_s": round(time.time() - t0, 1)})      # save this epoch for the training curve
        if verbose:     # print progress
            print(f"  [{model_name}/{view}/seed{seed}] ep{ep} loss={tot/n:.4f} "
                  f"val_acc={va['accuracy']:.4f} val_f1={va['macro_f1']:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if va["macro_f1"] > best_f1:        # new best, save these weights
            best_f1, bad = va["macro_f1"], 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:       # no improvement
            bad += 1
            if bad >= cfg["patience"]:      # early stopping
                break
    model.load_state_dict(best)  # best checkpoint by validation macro-F1
    pred_te = _predict(model, ids_te, y_te, 256)        # test set is only touched ONCE, at the very end
    res = metrics(y_te, pred_te, labels)        # accuracy / f1 / confusion matrix
    res.update({"model": model_name, "view": view, "seed": seed, "vocab_size": len(vocab),
                "device": str(DEVICE),
                "n_params": int(sum(p.numel() for p in model.parameters())),
                "best_val_macro_f1": best_f1, "epochs_run": len(history),
                "train_time_s": round(time.time() - t0, 1), "history": history,
                "config": cfg, "test_pred": pred_te.tolist()})      # add all the extra info about this run
    return res


def cv_run(df, label_col, labels, model_name, view="output", seed=0, transform=None,
           verbose=False, **kw):
    """prompt-grouped 5-fold cross-validation. every row is tested exactly once, so the
    pooled test predictions cover the whole dataset. returns per-fold and pooled metrics"""
    import numpy as np
    if transform is not None:       # optional change to the responses ( RQ4 ablations )
        df = df.copy()      # copy so the original data isnt touched
        df["llm_output"] = df["llm_output"].map(transform)
    lab2id = {l: i for i, l in enumerate(labels)}       # class name -> class number
    pred = {}       # row number -> predicted class
    folds = []      # per-fold results
    hist0 = None        # training curve from fold 0
    for k in range(N_FOLDS):        # go thru each cross-validation round
        tr, va, te = fold_split(df, k)      # train / val / test for this round
        if model_name == "tfidf":       # the baseline doesnt need a validation set
            r = tfidf_lr(tr, te, label_col, labels, view=view, max_len=kw.get("max_len", 400))
            p = r.pop("test_pred")
        else:       # CNN or LSTM
            r = train_eval(tr, va, te, label_col, labels, view=view, model_name=model_name,
                           seed=seed + k, verbose=verbose, **kw)        # round k uses seed + k
            p = r["test_pred"]
            if k == 0:
                hist0 = r["history"]        # keep fold 0's curve for the figure
        print(f"    {model_name}/{view} fold {k + 1}/{N_FOLDS}: acc={r['accuracy']:.3f}", flush=True)
        for i, v in zip(te.index, p):
            pred[int(i)] = int(v)       # remember each test row's guess
        folds.append({"fold": k, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                      "epochs_run": r.get("epochs_run"), "train_time_s": r.get("train_time_s"),
                      "n_params": r.get("n_params"), "vocab_size": r.get("vocab_size")})
    idx = sorted(pred)      # every row, in order
    y = df.loc[idx, label_col].map(lab2id).values       # answer key for the pooled predictions
    pooled = metrics(y, np.array([pred[i] for i in idx]), labels)       # metrics over all rows at once
    accs = [f["accuracy"] for f in folds]       # accuracy of each fold
    return {"model": model_name, "view": view, "label": label_col, "folds": folds,
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
            "pooled": pooled, "pred": pred, "history_fold0": hist0,
            "config": {**DEFAULTS, **kw}}
