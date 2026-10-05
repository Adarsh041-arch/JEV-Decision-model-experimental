"""Tiny FastAPI server mimicking how Jev is actually served: ticket JSON in, typed decisions out."""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle, torch
from model import JevLite

app = FastAPI(title="Jev-Lite", version="0.1.0")

with open("artifacts/vectorizer.pkl", "rb") as f: VEC = pickle.load(f)
with open("artifacts/svd.pkl", "rb") as f: SVD = pickle.load(f)
CKPT = torch.load("artifacts/jev_lite.pt", map_location="cpu", weights_only=False)
MODEL = JevLite(); MODEL.load_state_dict(CKPT["state"]); MODEL.eval()
TEMPS = CKPT["temps"]

class Ticket(BaseModel):
    text: str
    gate: float = 0.5

@app.post("/decide")
def decide(t: Ticket):
    x = torch.tensor(SVD.transform(VEC.transform([t.text])).astype("float32"))
    out = MODEL.predict_one(x[0], TEMPS)
    low = [k for k in ("is_urgent", "queue", "refund_clear") if out[k]["confidence"] < t.gate]
    out["escalate_to_human"] = bool(low)
    if low: out["escalate_reason"] = f"low confidence: {low}"
    return out

@app.get("/health")
def health():
    return {"ok": True, "temps": TEMPS}


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev-Lite triage demo</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#222}
textarea{width:100%;height:110px;font-size:1rem;padding:.6rem}
button{font-size:1rem;padding:.5rem 1.1rem;cursor:pointer}
.ex{margin:.3rem .3rem .3rem 0;font-size:.85rem;padding:.3rem .6rem}
.card{border:1px solid #ddd;border-radius:10px;padding:1rem;margin-top:1rem}
.row{display:flex;justify-content:space-between;padding:.25rem 0}
.bar{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin:.2rem 0 .6rem}
.bar>i{display:block;height:100%;background:#4a8}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:20px;background:#eef;font-weight:600}
.alert{background:#fff3cd;border:1px solid #e6c200;border-radius:8px;padding:.6rem;margin-top:.6rem}
pre{background:#f6f6f6;padding:.6rem;border-radius:8px;overflow:auto}
label{font-size:.9rem;color:#555}
</style></head><body>
<h2>Jev-Lite — ticket triage demo</h2>
<p>Type a support ticket, hit <b>Decide</b>. One forward pass, four typed heads, no text generation.</p>
<textarea id="t">App crashes on login. This is URGENT, production is down, need help ASAP!</textarea><br><br>
<label>Escalation gate (flag heads below this confidence): <b id="gv">0.50</b></label><br>
<input id="g" type="range" min="0" max="0.95" step="0.05" value="0.5" style="width:100%" oninput="gv.textContent=(+g.value).toFixed(2)"><br><br>
<button onclick="go()">Decide</button>
<button class="ex" onclick="fill(0)">urgent crash</button>
<button class="ex" onclick="fill(1)">billing refund</button>
<button class="ex" onclick="fill(2)">low-priority feedback</button>
<button class="ex" onclick="fill(3)">unclear refund</button>
<div id="out"></div>
<script>
const EXS=[
"App crashes on login. This is URGENT, production is down, need help ASAP!",
"Hi, I was charged twice this month, please refund order ORD-1234 for $49.00.",
"Love the product, quick feedback: do you offer SSO on team plan? No rush.",
"Locked out after enabling 2FA, backup codes don't work. Might need money back, or maybe just fix?"];
function fill(i){t.value=EXS[i];go();}
function conf(x){return `<div class="bar"><i style="width:${(x*100).toFixed(0)}%"></i></div>`;}
async function go(){
 out.innerHTML="<p>…deciding…</p>";
 const r=await fetch("/decide",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({text:t.value,gate:+g.value})});
 const d=await r.json();
 out.innerHTML=`<div class="card">
  <div class="row"><span>urgent</span><span class="badge">${d.is_urgent.answer}</span></div>${conf(d.is_urgent.confidence)}<small>conf ${d.is_urgent.confidence}</small>
  <div class="row"><span>queue</span><span class="badge">${d.queue.answer}</span></div>${conf(d.queue.confidence)}<small>conf ${d.queue.confidence}</small>
  <div class="row"><span>severity</span><span class="badge">${d.severity.answer}</span></div>
  <div class="row"><span>refund clarity</span><span class="badge">${d.refund_clear.answer}</span></div>${conf(d.refund_clear.confidence)}<small>conf ${d.refund_clear.confidence}</small>
  ${d.escalate_to_human?`<div class="alert">⚠ Escalate to human — ${d.escalate_reason}</div>`:`<div class="row"><span>escalation</span><b>auto-ok ✓</b></div>`}
  <details><summary>raw JSON</summary><pre>${JSON.stringify(d,null,2)}</pre></details></div>`;
}
go();
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
