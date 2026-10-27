"""Show exactly what call #3 saw, with no API call.

Re-runs emit_constraints.py with the VLM transport replaced by a replay of
the responses recorded in an emissions .log.json, so every stage is rebuilt
and re-rendered as in the original run (same chained entry, same views,
same legend, same compile-retry turns; parse-retry turns are listed, not
reproduced, because the rejected text was not logged), and dumps each
request:

    <out>/index.html          one section per call: system text, every user /
                              assistant turn, images inline
    <out>/call<k>_img<j>.png  the images, as sent

Run from ~/ur5e-manip-sim, with the SAME emit_constraints arguments the
original run used (minus --out):

    MUJOCO_GL=egl PYTHONPATH=. python dump_emit_prompts.py \\
        outputs/emissions/insert_marker.log.json outputs/runs/gt_opus1/prompts -- \\
        --scene scenes/insert_marker.json --grounding outputs/grounding/insert_marker \\
        --stage-plan outputs/stage_plan/insert_marker.marks.json \\
        --selections outputs/selections/insert_marker.json \\
        --render-chained /tmp/dump_frames
"""
import base64
import html
import json
import sys
import tempfile
from pathlib import Path

import manip_sim.vlm as vlm

if "--" not in sys.argv or len(sys.argv) < 4:
    sys.exit(__doc__)
cut = sys.argv.index("--")
log_path, out = Path(sys.argv[1]), Path(sys.argv[2])
emit_args = sys.argv[cut + 1:]
logs = json.loads(log_path.read_text())
# Each log entry keeps only the ACCEPTED response; responses the parser
# rejected were not stored. Replaying the accepted one at the first attempt
# makes every call's FIRST request exact (stage, views, legend, entry line,
# compile-retry turns), and only drops the parse-retry turns, which are
# listed per call on the page instead.
raws = [l["raw"] for l in logs]
rejected = [l.get("rejections") or [] for l in logs]
out.mkdir(parents=True, exist_ok=True)
calls = []


def replay(payload):
    k = len(calls)
    if k >= len(raws):
        raise SystemExit(f"replay ran past the {len(raws)} recorded responses")
    calls.append(payload)
    return raws[k]


vlm._urllib_transport = replay
import scripts.emit_constraints as ec  # noqa: E402

tmp = Path(tempfile.mkdtemp()) / "replayed.json"
sys.argv = ["emit_constraints.py", *emit_args, "--out", str(tmp)]
try:
    ec.main()
except SystemExit as e:          # a FAIL gate exits nonzero; the dump still stands
    if e.code not in (0, None):
        print(f"[dump] emit_constraints exited {e.code}")

parts = ["<html><head><meta charset='utf-8'><title>call #3 prompts</title>"
         "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}"
         "pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;font-size:12px}"
         "img{width:32%;margin:2px;border:1px solid #ccc}h2{border-top:3px solid #333;padding-top:8px}"
         ".assistant pre{background:#eef6ee}</style></head><body>"]
for k, p in enumerate(calls):
    parts.append(f"<h2>call {k}</h2>")
    if rejected[k]:
        parts.append("<p><b>Parse rejections in the original run (turns not reproduced):</b></p><pre>"
                     + html.escape("\n".join(rejected[k])) + "</pre>")
    parts.append(f"<h3>system</h3><pre>{html.escape(p['system'])}</pre>")
    for m in p["messages"]:
        parts.append(f"<div class='{m['role']}'><h3>{m['role']}</h3>")
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for j, b in enumerate(content):
            if b["type"] == "text":
                parts.append(f"<pre>{html.escape(b['text'])}</pre>")
            elif b["type"] == "image":
                f = out / f"call{k}_img{j}.png"
                f.write_bytes(base64.b64decode(b["source"]["data"]))
                parts.append(f"<img src='{f.name}'>")
        parts.append("</div>")
    parts.append(f"<div class='assistant'><h3>response</h3><pre>{html.escape(raws[k])}</pre></div>")
parts.append("</body></html>")
(out / "index.html").write_text("\n".join(parts))
print(f"[dump] {len(calls)} calls -> {out}/index.html")
