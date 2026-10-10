"""Local browser review, persisted only under this task directory."""
import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import ROOT, input_path, output_path, read_csv, write_csv

PAGE = '''<!doctype html><meta charset="utf-8"><title>D 候选审核</title>
<style>body{font:18px sans-serif;margin:24px}img{max-width:48%;image-rendering:pixelated}button,select,input{font:inherit;margin:8px}textarea{width:90%}</style>
<h2>D 候选审核</h2><p>0=确认非D；1=确认真实D（若GT漏标，需另附精确正标注才能补GT）；3=不确定。未审核不进入负监督。GT待复核目标也必须人工判断。</p>
<input id="idx" type="number" min="0" value="0"><button onclick="load()">跳转</button><span id="info"></span><div id="images"></div>
<select id="label"><option value="">未审核</option><option value="0">确认非D</option><option value="1">确认真实D</option><option value="3">不确定</option></select>
<input id="kind" placeholder="负样本类型：TiN/划痕/背景等"><br><textarea id="note" placeholder="审核依据或备注"></textarea><br>
<input id="difficulty" placeholder="真实D难度：weak（低对比度）/typical"><br>
<input id="positive" placeholder="精确正标注mask绝对路径，仅漏标D使用"><br>
<button onclick="save(-1)">保存并上一个</button><button onclick="save(1)">保存并下一个</button><span id="status"></span>
<script>
let total=0; const el=id=>document.getElementById(id);
async function load(){let i=Number(el('idx').value);let r=await fetch('/candidate/'+i);if(!r.ok){el('status').textContent=await r.text();return}let d=await r.json();total=d.total;el('info').textContent=` ${i+1}/${total} ${d.row.candidate_id} area=${d.row.area_px} P(D)=${d.row.mean_confidence}`;el('label').value=d.row.review_label;el('kind').value=d.row.negative_type;el('note').value=d.row.review_note;el('positive').value=d.row.positive_mask_path;el('difficulty').value=d.row.positive_difficulty||'';el('images').innerHTML=d.images.map(v=>`<img src="data:image/png;base64,${v}">`).join('')}
async function save(step){let i=Number(el('idx').value);let r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:i,review_label:el('label').value,negative_type:el('kind').value,positive_difficulty:el('difficulty').value,review_note:el('note').value,positive_mask_path:el('positive').value})});if(!r.ok){el('status').textContent=await r.text();return}el('status').textContent='已保存';el('idx').value=Math.max(0,Math.min(total-1,i+step));load()};load();
</script>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True, help="Task-relative review CSV")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    destination = output_path(args.output)
    rows = read_csv(destination if destination.exists() else input_path(args.candidates))
    if not rows or "review_label" not in rows[0]:
        raise ValueError("No reviewable candidates")
    for row in rows:
        row.setdefault("positive_difficulty", "")
    write_csv(destination, rows)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, kind="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path == "/":
                    return self.respond(200, PAGE, "text/html; charset=utf-8")
                if not path.startswith("/candidate/"):
                    return self.respond(404, "Not found")
                index = int(path.rsplit("/", 1)[1])
                if not 0 <= index < len(rows):
                    raise ValueError("Index outside candidate list")
                row = rows[index]
                images = []
                for key in ("raw_path", "context_path"):
                    asset = Path(row[key]).resolve()
                    if not asset.is_relative_to(ROOT):
                        raise ValueError("Review assets must be inside task directory")
                    images.append(base64.b64encode(asset.read_bytes()).decode())
                zoom = Path(row["raw_path"]).with_name(Path(row["raw_path"]).name.replace("_raw.png", "_zoom.png")).resolve()
                if zoom.is_relative_to(ROOT) and zoom.exists():
                    images.append(base64.b64encode(zoom.read_bytes()).decode())
                return self.respond(200, json.dumps(dict(row=row, images=images, total=len(rows))), "application/json")
            except (ValueError, KeyError, OSError) as exc:
                self.respond(400, str(exc))

        def do_POST(self):
            # JSON-only requests without cross-origin access; bind loopback only.
            if self.path != "/save" or self.headers.get("Content-Type") != "application/json":
                return self.respond(400, "Invalid request")
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{args.port}":
                return self.respond(403, "Origin rejected")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 65536:
                    raise ValueError("Invalid body length")
                data = json.loads(self.rfile.read(length))
                index = int(data["index"])
                if not 0 <= index < len(rows) or data["review_label"] not in {"", "0", "1", "3"}:
                    raise ValueError("Invalid index or label")
                old = rows[index].copy()
                for key in ("review_label", "negative_type", "positive_difficulty", "review_note", "positive_mask_path"):
                    rows[index][key] = str(data.get(key, ""))
                write_csv(destination, rows)
                journal = output_path(destination.with_name(destination.stem+"_history.jsonl"))
                with journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                            before=old, after=rows[index]), ensure_ascii=False)+"\n")
                self.respond(200, "Saved")
            except (ValueError, KeyError, OSError) as exc:
                self.respond(400, str(exc))

    print(f"Open http://127.0.0.1:{args.port} ; Ctrl+C stops review", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
