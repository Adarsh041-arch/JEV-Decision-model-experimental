"""Train Jev-Lite: encoder pipeline + multi-head training + temperature scaling + plots."""
import csv, json, os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import JevLite, QUEUES, REFUND_LABELS

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

Q2I = {q: i for i, q in enumerate(QUEUES)}
R2I = {r: i for i, r in enumerate(REFUND_LABELS)}

def load_csv(path="data/tickets.csv"):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def ece_score(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    ece, out = 0.0, []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf <= hi)
        if m.sum() == 0:
            out.append(( (lo+hi)/2, 0.0, 0)); continue
        acc = (pred[m] == labels[m]).mean()
        avgc = conf[m].mean()
        ece += abs(acc - avgc) * m.sum() / len(labels)
        out.append(((lo+hi)/2, float(acc), int(m.sum())))
    return float(ece), out

def reliability_plot(bins_before, bins_after, ece_b, ece_a, path, title):
    xs = [b[0] for b in bins_before]; acc_b = [b[1] for b in bins_before]; acc_a = [b[1] for b in bins_after]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(xs, acc_b, "o-", label=f"before (ECE={ece_b:.3f})")
    ax.plot(xs, acc_a, "s-", label=f"after (ECE={ece_a:.3f})")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Actual accuracy")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

def fit_temperature(logits, labels, init=1.0, iters=300, lr=0.05):
    T = torch.tensor([np.log(max(init - 1 + 1e-6, 1e-6)) if False else 0.0], requires_grad=True)  # log-param
    # simpler: optimize raw logT
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=iters)
    nll = nn.CrossEntropyLoss()
    L = torch.tensor(labels); G = torch.tensor(logits)
    def closure():
        opt.zero_grad()
        loss = nll(G / torch.exp(logT), L)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(logT).detach().item())

def main():
    os.makedirs("artifacts", exist_ok=True)
    rows = load_csv()
    for split in ("train", "val", "test"):
        print(f"{split}: {sum(1 for r in rows if r['split']==split)}")

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    Xtr_t = [r["text"] for r in tr]; Xva_t = [r["text"] for r in va]; Xte_t = [r["text"] for r in te]

    def labels(rs):
        return (np.array([int(r["is_urgent"]) for r in rs]),
                np.array([Q2I[r["queue"]] for r in rs]),
                np.array([float(r["severity"]) for r in rs], dtype=np.float32),
                np.array([R2I[r["refund_clear"]] for r in rs]))

    ytr_u, ytr_q, ytr_s, ytr_r = labels(tr)
    yva_u, yva_q, yva_s, yva_r = labels(va)
    yte_u, yte_q, yte_s, yte_r = labels(te)

    # --- Encoder: TF-IDF + SVD (fast; plan allows MiniLM or TF-IDF+SVD) ---
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    Xtr_tf = vec.fit_transform(Xtr_t)
    svd = TruncatedSVD(n_components=128, random_state=SEED)
    Xtr = svd.fit_transform(Xtr_tf).astype(np.float32)
    Xva = svd.transform(vec.transform(Xva_t)).astype(np.float32)
    Xte = svd.transform(vec.transform(Xte_t)).astype(np.float32)
    print(f"encoder: tfidf {Xtr_tf.shape} -> svd {Xtr.shape}; explained var {svd.explained_variance_ratio_.sum():.3f}")

    import pickle
    with open("artifacts/vectorizer.pkl", "wb") as f: pickle.dump(vec, f)
    with open("artifacts/svd.pkl", "wb") as f: pickle.dump(svd, f)

    dev = torch.device("cpu")
    tXtr, tXva, tXte = (torch.tensor(X, device=dev) for X in (Xtr, Xva, Xte))
    ty = {}
    for name, arr in (("tru", ytr_u), ("trq", ytr_q), ("trs", ytr_s), ("trr", ytr_r),
                      ("vau", yva_u), ("vaq", yva_q), ("vas", yva_s), ("var", yva_r),
                      ("teu", yte_u), ("teq", yte_q), ("tes", yte_s), ("ter", yte_r)):
        ty[name] = torch.tensor(arr, device=dev) if arr.dtype != np.float32 else torch.tensor(arr, device=dev)

    model = JevLite(in_dim=128).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
    W_SEV, W_REF = 0.5, 0.5  # loss weights (tune if one head dominates)

    def batch_loss(o, b):
        lu = ce(o["urgent"][b], ty["tru"][b])
        lq = ce(o["queue"][b], ty["trq"][b])
        ls = bce(o["severity"][b], ty["trs"][b])  # BCE on 0-1 severity, sharper than MSE
        lr_ = ce(o["refund"][b], ty["trr"][b])
        return lu + lq + W_SEV * ls + W_REF * lr_, (float(lu), float(lq), float(ls), float(lr_))

    N = len(Xtr); BS = 64; EPOCHS = 20
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(N)
        tot, parts = 0.0, [0.0]*4
        for i in range(0, N, BS):
            b = perm[i:i+BS]
            opt.zero_grad()
            o = model(tXtr)
            loss, p = batch_loss(o, b)
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
            for k in range(4): parts[k] += p[k] * len(b)
        # val metrics
        model.eval()
        with torch.no_grad():
            v = model(tXva)
            vauc = float((v["urgent"].argmax(1) == ty["vau"]).float().mean())
            vaqc = float((v["queue"].argmax(1) == ty["vaq"]).float().mean())
            vam = float(torch.abs(torch.sigmoid(v["severity"]) - ty["vas"]).mean())
            varc = float((v["refund"].argmax(1) == ty["var"]).float().mean())
        print(f"ep {ep:02d} loss={tot/N:.4f} (urg={parts[0]/N:.3f} q={parts[1]/N:.3f} sev={parts[2]/N:.3f} ref={parts[3]/N:.3f}) "
              f"| val urg={vauc:.3f} queue={vaqc:.3f} MAE={vam:.3f} refund={varc:.3f}")
    print(f"training time: {time.time()-t0:.1f}s")

    # --- Temperature scaling on val ---
    model.eval()
    with torch.no_grad():
        lv, lt = model(tXva), model(tXte)
    Tu = fit_temperature(lv["urgent"].cpu().numpy(), yva_u)
    Tq = fit_temperature(lv["queue"].cpu().numpy(), yva_q)
    Tr = fit_temperature(lv["refund"].cpu().numpy(), yva_r)
    temps = {"urgent": Tu, "queue": Tq, "refund": Tr}
    print("temperatures:", {k: round(v, 3) for k, v in temps.items()})

    def probs(logits, T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z); return e / e.sum(axis=1, keepdims=True)

    results = {}
    for head, key, yte, yva, lv_l, lt_l in (
        ("is_urgent", "urgent", yte_u, yva_u, lv["urgent"].cpu().numpy(), lt["urgent"].cpu().numpy()),
        ("queue", "queue", yte_q, yva_q, lv["queue"].cpu().numpy(), lt["queue"].cpu().numpy()),
        ("refund_clear", "refund", yte_r, yva_r, lv["refund"].cpu().numpy(), lt["refund"].cpu().numpy()),
    ):
        T = temps[key]
        pb, pa = probs(lt_l, 1.0), probs(lt_l, T)
        acc = float((pa.argmax(1) == yte).mean())
        ece_b, bins_b = ece_score(pb, yte)
        ece_a, bins_a = ece_score(pa, yte)
        results[head] = {"acc": round(acc, 4), "ece_before": round(ece_b, 4), "ece_after": round(ece_a, 4), "T": round(T, 3)}
        reliability_plot(bins_b, bins_a, ece_b, ece_a,
                         f"artifacts/reliability_{key}.png", f"Jev-Lite reliability: {head} (test)")
        print(f"{head}: acc={acc:.3f} ECE {ece_b:.3f} -> {ece_a:.3f} (T={T:.2f})")

    # severity
    with torch.no_grad():
        sev_pred = torch.sigmoid(lt["severity"]).cpu().numpy()
    mae = float(np.abs(sev_pred - yte_s).mean())
    results["severity"] = {"mae": round(mae, 4)}
    print(f"severity MAE={mae:.3f}")

    # latency: single forward pass per ticket
    model.eval()
    with torch.no_grad():
        _ = model(tXte[:8])  # warmup
        t0 = time.perf_counter()
        for i in range(len(Xte)):
            _ = model(tXte[i:i+1])
        t1 = time.perf_counter()
    lat_ms = (t1 - t0) / len(Xte) * 1000
    # batched throughput too
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(tXte)
        t1 = time.perf_counter()
    thr = len(Xte) / (t1 - t0)
    results["latency_ms_per_ticket"] = round(lat_ms, 3)
    results["throughput_tps"] = round(thr, 1)
    print(f"latency {lat_ms:.2f} ms/ticket (single), throughput {thr:.0f} tickets/s (batched)")

    torch.save({"state": model.state_dict(), "temps": temps}, "artifacts/jev_lite.pt")
    with open("artifacts/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved artifacts/jev_lite.pt, metrics.json, reliability_*.png")

if __name__ == "__main__":
    main()
