"""Jev-Lite: shared encoder + parallel typed decision heads."""
import torch
import torch.nn as nn

QUEUES = ["billing", "technical", "account", "other"]
REFUND_LABELS = ["yes", "no", "unclear"]

class JevLite(nn.Module):
    """TF-IDF+SVD embedding in, shared MLP trunk, 4 parallel heads out."""
    def __init__(self, in_dim=128, trunk=(256, 128), dropout=0.2):
        super().__init__()
        h1, h2 = trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.urgent_head = nn.Linear(h2, 2)     # choice: yes/no
        self.queue_head = nn.Linear(h2, 4)      # choice: 4 queues
        self.severity_head = nn.Linear(h2, 1)   # score: 0-1
        self.refund_head = nn.Linear(h2, 3)     # null-type: yes/no/unclear

    def forward(self, x):
        h = self.trunk(x)
        return {
            "urgent": self.urgent_head(h),
            "queue": self.queue_head(h),
            "severity": self.severity_head(h).squeeze(-1),  # raw logit; sigmoid applied in loss/infer
            "refund": self.refund_head(h),
        }

    def predict_one(self, x, temps=None):
        """Single forward pass -> typed JSON. x: 1D tensor."""
        temps = temps or {}
        self.eval()
        with torch.no_grad():
            out = self.forward(x.unsqueeze(0))
            def cal(logits, name):
                t = temps.get(name, 1.0)
                return torch.softmax(logits / t, dim=-1).squeeze(0)
            pu = cal(out["urgent"], "urgent")
            pq = cal(out["queue"], "queue")
            pr = cal(out["refund"], "refund")
            iu, iq, ir = int(pu.argmax()), int(pq.argmax()), int(pr.argmax())
            sev = float(torch.sigmoid(out["severity"]).item())
            return {
                "is_urgent": {"answer": "yes" if iu == 1 else "no", "confidence": round(float(pu[iu]), 3)},
                "queue": {"answer": QUEUES[iq], "confidence": round(float(pq[iq]), 3)},
                "severity": {"answer": round(sev, 3)},
                "refund_clear": {"answer": REFUND_LABELS[ir], "confidence": round(float(pr[ir]), 3)},
            }
