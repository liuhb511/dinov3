# -*- coding: utf-8 -*-
"""SSH-friendly web reviewer for hard-negative candidates.

Server:
  python inclusion_v2/tools/review_hard_negatives_web.py \
      --csv ./hard_negative_mining/candidates.csv \
      --host 127.0.0.1 --port 7860

Local computer:
  ssh -L 7860:127.0.0.1:7860 user@server
Then open http://127.0.0.1:7860

Keys:
  0 confirmed negative
  1 missed inclusion
  2 known distractor
  3 uncertain
  S skip / next
  B or left arrow previous
  right arrow next
"""

import argparse
import csv
import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LABELS = {
    "0": "确认负样本",
    "1": "漏标夹杂物",
    "2": "已知干扰物",
    "3": "不确定",
}

class Store:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"找不到 candidates.csv: {self.csv_path}")
        self.lock = threading.RLock()
        self.load()

    def load(self):
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.fields = list(reader.fieldnames or [])
            self.rows = list(reader)
        if not self.rows:
            raise RuntimeError("candidates.csv 为空")
        for key in ("review_label", "review_name", "review_note"):
            if key not in self.fields:
                self.fields.append(key)
                for r in self.rows:
                    r[key] = ""

    def save(self):
        tmp = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            w.writeheader(); w.writerows(self.rows)
        os.replace(tmp, self.csv_path)

    def review(self, idx, label):
        with self.lock:
            if label not in LABELS:
                raise ValueError("非法 label")
            r = self.rows[idx]
            r["review_label"] = label
            r["review_name"] = LABELS[label]
            self.save()

    def clear(self, idx):
        with self.lock:
            self.rows[idx]["review_label"] = ""
            self.rows[idx]["review_name"] = ""
            self.save()

    def filtered(self, cls="ALL", state="UNREVIEWED", order="AREA_DESC"):
        ids = list(range(len(self.rows)))
        if cls in {"A", "B", "C"}:
            ids = [i for i in ids if self.rows[i].get("pred_class_name") == cls]
        if state == "UNREVIEWED":
            ids = [i for i in ids if not self.rows[i].get("review_label", "").strip()]
        elif state == "REVIEWED":
            ids = [i for i in ids if self.rows[i].get("review_label", "").strip()]
        elif state in LABELS:
            ids = [i for i in ids if self.rows[i].get("review_label", "").strip() == state]
        def area(i):
            try: return float(self.rows[i].get("area_px", 0) or 0)
            except Exception: return 0.0
        if order == "AREA_DESC": ids.sort(key=area, reverse=True)
        elif order == "AREA_ASC": ids.sort(key=area)
        return ids

    def summary(self):
        out = {"TOTAL": len(self.rows), "UNREVIEWED": 0}
        out.update({v: 0 for v in LABELS.values()})
        out["BY_CLASS"] = {k: {"TOTAL": 0, "UNREVIEWED": 0} for k in "ABC"}
        for r in self.rows:
            lab = r.get("review_label", "").strip()
            if not lab: out["UNREVIEWED"] += 1
            elif lab in LABELS: out[LABELS[lab]] += 1
            c = r.get("pred_class_name", "")
            if c in out["BY_CLASS"]:
                out["BY_CLASS"][c]["TOTAL"] += 1
                if not lab: out["BY_CLASS"][c]["UNREVIEWED"] += 1
        return out

    def asset(self, text):
        p = Path(text)
        probes = [p] if p.is_absolute() else [Path.cwd()/p, self.csv_path.parent.parent/p, self.csv_path.parent/p]
        for q in probes:
            q = q.resolve()
            if q.exists() and q.is_file(): return q
        raise FileNotFoundError(f"找不到图片: {text}")

HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>困难负样本人工审核</title><style>
body{margin:0;background:#171717;color:#eee;font-family:Arial,"Microsoft YaHei",sans-serif}.top{position:sticky;top:0;background:#242424;padding:10px;z-index:5;border-bottom:1px solid #555}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}select,input,button{padding:7px;font-size:14px}.main{padding:14px}.meta{padding:10px;background:#222;border:1px solid #555;border-radius:8px;line-height:1.55}.imgs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.box{background:#111;padding:8px;border:1px solid #444;border-radius:8px}.box img{width:100%;height:62vh;object-fit:contain;background:#000}.act{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}.act button{min-height:60px;font-weight:bold}.n0{background:#285332;color:#fff}.n1{background:#6b332f;color:#fff}.n2{background:#4c3f76;color:#fff}.n3{background:#555;color:#fff}.nav{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap}.small{font-size:13px;color:#ccc;margin-top:8px}@media(max-width:900px){.imgs{grid-template-columns:1fr}.act{grid-template-columns:1fr 1fr}}
</style></head><body>
<div class="top"><div class="row">
<label>类别 <select id="cls"><option value="ALL">全部</option><option>A</option><option>B</option><option>C</option></select></label>
<label>状态 <select id="state"><option value="UNREVIEWED">未审核</option><option value="ALL">全部</option><option value="REVIEWED">已审核</option><option value="0">0 - 确认负样本</option><option value="1">1 - 漏标夹杂物</option><option value="2">2 - 已知干扰物</option><option value="3">3 - 不确定</option></select></label>
<label>排序 <select id="order"><option value="AREA_DESC">面积从大到小</option><option value="AREA_ASC">面积从小到大</option><option value="CSV_ORDER">CSV顺序</option></select></label>
<button onclick="refresh(true)">应用</button><label>跳转 <input id="jump" placeholder="HN_000123" size="12"></label><button onclick="jumpTo()">跳转</button>
</div><div id="summary" class="small">加载中...</div></div>
<div class="main"><div id="meta" class="meta">加载中...</div>
<div class="imgs"><div class="box"><b>原始裁剪图</b><img id="raw"></div><div class="box"><b>候选位置叠加图</b><img id="ov"></div></div>
<div class="act"><button class="n0" onclick="review('0')">0 — 确认负样本<br><small>确定不是夹杂物</small></button><button class="n1" onclick="review('1')">1 — 漏标夹杂物<br><small>是真夹杂，但原 GT 没标</small></button><button class="n2" onclick="review('2')">2 — 已知干扰物<br><small>如 HH / XW / XQL 等</small></button><button class="n3" onclick="review('3')">3 — 不确定<br><small>暂时无法可靠判断</small></button></div>
<div class="nav"><button onclick="prev()">← B / 上一张</button><button onclick="next()">S / 跳过</button><button onclick="clearLab()">清除当前标签</button></div>
<div class="small">快捷键：0=确认负样本，1=漏标夹杂物，2=已知干扰物，3=不确定；S=跳过，B/←=上一张，→=下一张。每次审核都会立即保存 candidates.csv。</div></div>
<script>
let ids=[],pos=0,cur=null; const $=x=>document.getElementById(x);
async function api(u,opt={}){let r=await fetch(u,opt),d=await r.json();if(!r.ok||d.ok===false)throw Error(d.error||r.status);return d}
function esc(x){return String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}
async function refresh(reset=false){let p=new URLSearchParams({cls:$('cls').value,state:$('state').value,order:$('order').value});let d=await api('/api/list?'+p);ids=d.indices;if(reset)pos=0;showSummary(d.summary);await load()}
function showSummary(s){let b=s.BY_CLASS;$('summary').innerHTML=`总候选 ${s.TOTAL} ｜ 未审核 ${s.UNREVIEWED} ｜ 确认负样本 ${s["确认负样本"]} ｜ 漏标夹杂物 ${s["漏标夹杂物"]} ｜ 已知干扰物 ${s["已知干扰物"]} ｜ 不确定 ${s["不确定"]} ｜ A：共${b.A.TOTAL}/未审${b.A.UNREVIEWED} ｜ B：共${b.B.TOTAL}/未审${b.B.UNREVIEWED} ｜ C：共${b.C.TOTAL}/未审${b.C.UNREVIEWED}`}
async function load(){if(!ids.length){cur=null;$('meta').innerHTML='当前筛选下没有候选';$('raw').removeAttribute('src');$('ov').removeAttribute('src');return}pos=Math.max(0,Math.min(pos,ids.length-1));let gi=ids[pos],d=await api('/api/item?index='+gi);cur=d.row;let r=cur;$('meta').innerHTML=`<b>${esc(r.candidate_id)}</b> ｜ 当前 ${pos+1}/${ids.length} ｜ 全局 ${gi+1}/${d.total}<br>预测类别=<b>${esc(r.pred_class_name)}</b> ｜ 候选面积=${esc(r.area_px)} px ｜ 外接框=(${esc(r.x)},${esc(r.y)},${esc(r.w)},${esc(r.h)}) ｜ 长宽比=${esc(r.aspect_ratio)}<br>图像：${esc(r.image_name)}<br>当前审核：<b>${esc(r.review_name||'未审核')}</b>`;let t=Date.now();$('raw').src=`/asset/raw?index=${gi}&t=${t}`;$('ov').src=`/asset/overlay?index=${gi}&t=${t}`}
async function review(l){if(!cur)return;let gi=cur.global_index;await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:gi,label:l})});if($('state').value==='UNREVIEWED')await refresh(false);else next()}
function next(){if(ids.length&&pos<ids.length-1){pos++;load()}} function prev(){if(pos>0){pos--;load()}}
async function clearLab(){if(!cur)return;await api('/api/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:cur.global_index})});await refresh(false)}
async function jumpTo(){let c=$('jump').value.trim();if(!c)return;let d=await api('/api/find?candidate_id='+encodeURIComponent(c));$('state').value='ALL';$('cls').value='ALL';await refresh(true);let k=ids.indexOf(d.index);if(k>=0){pos=k;await load()}}
document.addEventListener('keydown',async e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(['0','1','2','3'].includes(e.key)){e.preventDefault();await review(e.key)}else if(e.key.toLowerCase()==='s'||e.key==='ArrowRight'){e.preventDefault();next()}else if(e.key.toLowerCase()==='b'||e.key==='ArrowLeft'){e.preventDefault();prev()}});refresh(true)
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    store = None
    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)
    def send_json(self, obj, status=200):
        b=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(b))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(b)
    def send_bytes(self,b,typ):
        self.send_response(200); self.send_header("Content-Type",typ); self.send_header("Content-Length",str(len(b))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(b)
    def body(self):
        n=int(self.headers.get("Content-Length","0") or 0); return json.loads(self.rfile.read(n).decode() if n else "{}")
    def do_GET(self):
        try:
            u=urlparse(self.path); q=parse_qs(u.query)
            if u.path=="/": return self.send_bytes(HTML.encode(),"text/html; charset=utf-8")
            if u.path=="/api/list":
                ids=self.store.filtered(q.get("cls",["ALL"])[0],q.get("state",["UNREVIEWED"])[0],q.get("order",["AREA_DESC"])[0]); return self.send_json({"ok":True,"indices":ids,"summary":self.store.summary()})
            if u.path=="/api/item":
                i=int(q["index"][0]); r=dict(self.store.rows[i]); r["global_index"]=i; return self.send_json({"ok":True,"row":r,"total":len(self.store.rows)})
            if u.path=="/api/find":
                cid=q.get("candidate_id",[""])[0];
                for i,r in enumerate(self.store.rows):
                    if r.get("candidate_id")==cid: return self.send_json({"ok":True,"index":i})
                return self.send_json({"ok":False,"error":f"找不到 {cid}"},404)
            if u.path in ("/asset/raw","/asset/overlay"):
                i=int(q["index"][0]); fld="raw_crop_path" if u.path.endswith("raw") else "overlay_path"; p=self.store.asset(self.store.rows[i][fld]); return self.send_bytes(p.read_bytes(),mimetypes.guess_type(str(p))[0] or "application/octet-stream")
            return self.send_json({"ok":False,"error":"Not found"},404)
        except Exception as e: return self.send_json({"ok":False,"error":str(e)},500)
    def do_POST(self):
        try:
            b=self.body()
            if self.path=="/api/review": self.store.review(int(b["index"]),str(b["label"])); return self.send_json({"ok":True})
            if self.path=="/api/clear": self.store.clear(int(b["index"])); return self.send_json({"ok":True})
            return self.send_json({"ok":False,"error":"Not found"},404)
        except Exception as e: return self.send_json({"ok":False,"error":str(e)},500)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",default="./hard_negative_mining/candidates.csv"); ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",type=int,default=7860); a=ap.parse_args()
    s=Store(a.csv); Handler.store=s; srv=ThreadingHTTPServer((a.host,a.port),Handler)
    print("="*70); print("困难负样本 Web 人工审核"); print("CSV:",s.csv_path); print("Total:",len(s.rows)); print(f"Listen: http://{a.host}:{a.port}"); print(f"本机端口转发: ssh -L {a.port}:127.0.0.1:{a.port} user@server"); print(f"浏览器: http://127.0.0.1:{a.port}"); print("每次审核立即保存；停止服务 Ctrl+C"); print("="*70)
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nStopped")
    finally: srv.server_close()

if __name__=="__main__": main()