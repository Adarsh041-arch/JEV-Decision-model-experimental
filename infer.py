"""Single-pass typed inference demo + confidence-gated escalation."""
import json, pickle, torch
from model import JevLite

with open("artifacts/vectorizer.pkl", "rb") as f: VEC = pickle.load(f)
with open("artifacts/svd.pkl", "rb") as f: SVD = pickle.load(f)
CKPT = torch.load("artifacts/jev_lite.pt", map_location="cpu", weights_only=False)
MODEL = JevLite(); MODEL.load_state_dict(CKPT["state"]); MODEL.eval()
TEMPS = CKPT["temps"]

def encode(texts):
    return torch.tensor(SVD.transform(VEC.transform(texts)).astype("float32"))

def decide(text, gate=0.5):
    x = encode([text])
    out = MODEL.predict_one(x[0], TEMPS)
    low = [k for k in ("is_urgent", "queue", "refund_clear")
           if out[k]["confidence"] < gate]
    out["escalate_to_human"] = bool(low)
    if low: out["escalate_reason"] = f"low confidence: {low}"
    return out

if __name__ == "__main__":
    demos = [
        "App crashes on login. This is URGENT, production is down, need help ASAP!",
        "Hi, I was charged twice this month, please refund order ORD-1234 for $49.00.",
        "Love the product, quick feedback: do you offer SSO on team plan? No rush.",
        "Locked out after enabling 2FA, backup codes don't work. Might need money back, or maybe just fix?",
    ]
    for d in demos:
        print(f"\n> {d}")
        print(json.dumps(decide(d), indent=2))
