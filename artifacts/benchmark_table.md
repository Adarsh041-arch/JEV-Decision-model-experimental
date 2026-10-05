| Metric | Jev-lite | Classical (logreg) | Rule prompt-proxy |
|---|---|---|---|
| acc is_urgent | 0.8867 | 0.8767 | 0.8767 |
| acc queue | 1.0 | 1.0 | 0.88 |
| MAE severity | 0.1779 | 0.1796 | 0.2011 |
| ECE urgent | 0.033 | 0.0803 | n/a (no probs) |
| latency ms/ticket | 1.155 | 2.705 | 0.0294 |
| parse fail rate | 0.0 | 0.0 | ~0.0267 (simulated) |
| LLM API (est.) | — | — | 600-1200ms, 2-5% parse fails (literature) |
