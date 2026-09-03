import csv
from pathlib import Path
import cv2
import numpy as np

CFG = {
    "image_dir": "D:/lhb/datasets/testsets/HMS/testsets/test_images",
    "gt_dir": "D:/lhb/datasets/testsets/HMS/testsets/test_GT",
    "v1_dir": "D:/lhb/datasets/testsets/HMS/testsets/result/V1/test_masks",
    "v2_dir": "D:/lhb/datasets/testsets/HMS/testsets/result/V2/mask",
    "output_dir": "D:/lhb/datasets/testsets/HMS/testsets/merge/GTV1V2_compare",
    "num_classes": 5,
    "red_class": 4,
    "red_component_min_area": 2,
    "save_visualization": True,
    "visualization_alpha": 0.65,
}

CLASS_NAMES = {0:"background",1:"hd_w",2:"hd_y",3:"hd_t",4:"red"}
VALID_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
EPS = 1e-12

def read_mask(path):
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if m is None: raise RuntimeError(f"无法读取: {path}")
    if m.ndim == 3:
        if m.shape[2] == 1: m = m[:,:,0]
        else: raise ValueError(f"mask必须是单通道类别索引图: {path}, shape={m.shape}")
    return m.astype(np.int64)

def read_image(path):
    if path is None or not path.exists(): return None
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def save_image(path, image):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if not ok: raise RuntimeError(f"保存失败: {path}")
    buf.tofile(str(path))

def stem_map(directory):
    d = Path(directory)
    if not d.exists(): raise FileNotFoundError(f"目录不存在: {d}")
    r = {}
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            if p.stem in r: raise ValueError(f"重复stem: {p.stem}")
            r[p.stem] = p
    return r

def cmatrix(gt, pred, n):
    valid = (gt>=0)&(gt<n)&(pred>=0)&(pred<n)
    x = n*gt[valid]+pred[valid]
    return np.bincount(x, minlength=n*n).reshape(n,n)

def metrics(cm):
    cm=cm.astype(np.float64); tp=np.diag(cm)
    g=cm.sum(1); p=cm.sum(0); fp=p-tp; fn=g-tp
    return {
        "iou":tp/np.maximum(tp+fp+fn,EPS),
        "dice":2*tp/np.maximum(2*tp+fp+fn,EPS),
        "precision":tp/np.maximum(tp+fp,EPS),
        "recall":tp/np.maximum(tp+fn,EPS),
        "gt":g,"pred":p,"valid":(g+p)>0,
        "acc":tp.sum()/max(cm.sum(),EPS)
    }

def fgmean(a, valid):
    ids=[i for i in range(1,len(a)) if valid[i]]
    return float(np.mean(a[ids])) if ids else float("nan")

def red_metrics(gt,pred,r):
    g=gt==r; p=pred==r
    tp=int(np.sum(g&p)); fp=int(np.sum((~g)&p)); fn=int(np.sum(g&(~p)))
    return {"tp":tp,"fp":fp,"fn":fn,
            "iou":tp/max(tp+fp+fn,EPS),
            "dice":2*tp/max(2*tp+fp+fn,EPS),
            "precision":tp/max(tp+fp,EPS),
            "recall":tp/max(tp+fn,EPS)}

def paired(gt,v1,v2,r):
    c1=v1==gt; c2=v2==gt; g=gt==r; p1=v1==r; p2=v2==r
    fixed=(~c1)&c2; reg=c1&(~c2)
    rec=g&(~p1)&p2; redreg=g&p1&(~p2)
    newfp=(~g)&(~p1)&p2; fpfixed=(~g)&p1&(~p2)
    fn1=int(np.sum(g&(~p1)))
    return {
        "fixed":int(np.sum(fixed)),"regressed":int(np.sum(reg)),
        "net":int(np.sum(fixed))-int(np.sum(reg)),
        "red_recovered":int(np.sum(rec)),"red_regressed":int(np.sum(redreg)),
        "red_new_fp":int(np.sum(newfp)),"red_fp_fixed":int(np.sum(fpfixed)),
        "red_both_missed":int(np.sum(g&(~p1)&(~p2))),"v1_red_fn":fn1
    }

def component_rows(stem,gt,v1,v2,r,min_area):
    binary=(gt==r).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(binary,8)
    rows=[]
    for cid in range(1,n):
        area=int(stats[cid,cv2.CC_STAT_AREA])
        if area<min_area: continue
        region=lab==cid
        c1=float(np.sum(region&(v1==r))/area); c2=float(np.sum(region&(v2==r))/area)
        rows.append({"image":stem,"component_id":cid,"area":area,
                     "v1_coverage":c1,"v2_coverage":c2,"delta_coverage":c2-c1})
    return rows

def blend(img, items, alpha):
    out=img.astype(np.float32).copy()
    for mask,color in items:
        if np.any(mask):
            c=np.asarray(color,np.float32); out[mask]=out[mask]*(1-alpha)+c*alpha
    return np.clip(out,0,255).astype(np.uint8)

def visualize(img,gt,v1,v2,r,alpha):
    c1=v1==gt; c2=v2==gt
    overall=blend(img,[((~c1)&(~c2),(0,255,255)),((~c1)&c2,(0,255,0)),(c1&(~c2),(0,0,255))],alpha)
    g=gt==r; p1=v1==r; p2=v2==r
    red=blend(img,[
        (g&(~p1)&(~p2),(255,0,0)),      # 蓝: 都漏
        (g&(~p1)&p2,(0,255,0)),         # 绿: V2找回
        (g&p1&(~p2),(0,0,255)),         # 红: V2退化
        ((~g)&(~p1)&p2,(0,255,255)),    # 黄: V2新增FP
        ((~g)&p1&(~p2),(255,0,255))     # 紫: V2修复V1 FP
    ],alpha)
    return overall,red

def write_csv(path,rows,fields):
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    n=CFG["num_classes"]; r=CFG["red_class"]; out=Path(CFG["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    gm=stem_map(CFG["gt_dir"]); m1=stem_map(CFG["v1_dir"]); m2=stem_map(CFG["v2_dir"])
    idir=Path(CFG["image_dir"]); im=stem_map(idir) if idir.exists() else {}
    names=sorted(set(gm)&set(m1)&set(m2))
    if not names: raise RuntimeError("GT/V1/V2没有同stem文件")
    print(f"GT={len(gm)}, V1={len(m1)}, V2={len(m2)}, common={len(names)}")

    C1=np.zeros((n,n),np.int64); C2=np.zeros((n,n),np.int64)
    totals={k:0 for k in ["fixed","regressed","net","red_recovered","red_regressed","red_new_fp","red_fp_fixed","red_both_missed","v1_red_fn"]}
    per=[]; comps=[]

    for i,s in enumerate(names,1):
        gt=read_mask(gm[s]); v1=read_mask(m1[s]); v2=read_mask(m2[s])
        if gt.shape!=v1.shape or gt.shape!=v2.shape:
            raise ValueError(f"{s}尺寸不一致 GT={gt.shape} V1={v1.shape} V2={v2.shape}")
        for label,m in [("GT",gt),("V1",v1),("V2",v2)]:
            bad=np.unique(m[(m<0)|(m>=n)])
            if bad.size: raise ValueError(f"{s} {label}非法类别: {bad.tolist()}")

        c1=cmatrix(gt,v1,n); c2=cmatrix(gt,v2,n); C1+=c1; C2+=c2
        a=metrics(c1); b=metrics(c2); q1=red_metrics(gt,v1,r); q2=red_metrics(gt,v2,r); p=paired(gt,v1,v2,r)
        for k in totals: totals[k]+=p[k]
        mi1=fgmean(a["iou"],a["valid"]); mi2=fgmean(b["iou"],b["valid"])
        md1=fgmean(a["dice"],a["valid"]); md2=fgmean(b["dice"],b["valid"])
        per.append({
            "image":s,"pixels":gt.size,
            "v1_miou_fg":mi1,"v2_miou_fg":mi2,"delta_miou_fg":mi2-mi1,
            "v1_mdice_fg":md1,"v2_mdice_fg":md2,"delta_mdice_fg":md2-md1,
            "v1_red_iou":q1["iou"],"v2_red_iou":q2["iou"],"delta_red_iou":q2["iou"]-q1["iou"],
            "v1_red_dice":q1["dice"],"v2_red_dice":q2["dice"],"delta_red_dice":q2["dice"]-q1["dice"],
            "v1_red_recall":q1["recall"],"v2_red_recall":q2["recall"],"delta_red_recall":q2["recall"]-q1["recall"],
            "v1_red_precision":q1["precision"],"v2_red_precision":q2["precision"],"delta_red_precision":q2["precision"]-q1["precision"],
            "v1_red_fp":q1["fp"],"v2_red_fp":q2["fp"],"v1_red_fn":q1["fn"],"v2_red_fn":q2["fn"],
            "v2_fixed_pixels":p["fixed"],"v2_regressed_pixels":p["regressed"],"v2_net_fixed_pixels":p["net"],
            "red_recovered_pixels":p["red_recovered"],"red_regressed_pixels":p["red_regressed"],
            "red_new_fp_pixels":p["red_new_fp"],"red_fp_fixed_pixels":p["red_fp_fixed"],
            "red_fn_recovery_rate":p["red_recovered"]/max(p["v1_red_fn"],EPS)
        })
        comps += component_rows(s,gt,v1,v2,r,CFG["red_component_min_area"])

        if CFG["save_visualization"]:
            img=read_image(im.get(s))
            if img is None: img=np.full((*gt.shape,3),128,np.uint8)
            elif img.shape[:2]!=gt.shape: img=cv2.resize(img,(gt.shape[1],gt.shape[0]))
            ov,rv=visualize(img,gt,v1,v2,r,CFG["visualization_alpha"])
            save_image(out/"visualization"/"overall_diff"/f"{s}.png",ov)
            save_image(out/"visualization"/"red_diff"/f"{s}.png",rv)

        print(f"[{i}/{len(names)}] {s}: mIoU {mi1:.4f}->{mi2:.4f}, RED IoU {q1['iou']:.4f}->{q2['iou']:.4f}, Recall {q1['recall']:.4f}->{q2['recall']:.4f}")

    write_csv(out/"per_image.csv",per,list(per[0].keys()))
    if comps: write_csv(out/"red_component.csv",comps,list(comps[0].keys()))

    A=metrics(C1); B=metrics(C2)
    def red_cm(cm):
        tp=int(cm[r,r]); fn=int(cm[r,:].sum()-tp); fp=int(cm[:,r].sum()-tp)
        return {"iou":tp/max(tp+fp+fn,EPS),"dice":2*tp/max(2*tp+fp+fn,EPS),
                "precision":tp/max(tp+fp,EPS),"recall":tp/max(tp+fn,EPS),"fp":fp,"fn":fn}
    R1=red_cm(C1); R2=red_cm(C2)

    pc=[]
    for cid in range(n):
        pc.append({"class_id":cid,"class_name":CLASS_NAMES[cid],
                   "v1_iou":A["iou"][cid],"v2_iou":B["iou"][cid],"delta_iou":B["iou"][cid]-A["iou"][cid],
                   "v1_dice":A["dice"][cid],"v2_dice":B["dice"][cid],"delta_dice":B["dice"][cid]-A["dice"][cid],
                   "v1_precision":A["precision"][cid],"v2_precision":B["precision"][cid],
                   "v1_recall":A["recall"][cid],"v2_recall":B["recall"][cid],
                   "gt_pixels":int(A["gt"][cid]),"v1_pred_pixels":int(A["pred"][cid]),"v2_pred_pixels":int(B["pred"][cid])})
    write_csv(out/"per_class.csv",pc,list(pc[0].keys()))

    mi1=fgmean(A["iou"],A["valid"]); mi2=fgmean(B["iou"],B["valid"])
    md1=fgmean(A["dice"],A["valid"]); md2=fgmean(B["dice"],B["valid"])
    rows=[]
    def add(name,x,y): rows.append({"metric":name,"v1":x,"v2":y,"delta_v2_minus_v1":y-x})
    add("mIoU_fg",mi1,mi2); add("mDice_fg",md1,md2)
    add("RED_IoU",R1["iou"],R2["iou"]); add("RED_Dice",R1["dice"],R2["dice"])
    add("RED_Recall",R1["recall"],R2["recall"]); add("RED_Precision",R1["precision"],R2["precision"])
    add("RED_FP_pixels",R1["fp"],R2["fp"]); add("RED_FN_pixels",R1["fn"],R2["fn"])
    fn_reduce=(R1["fn"]-R2["fn"])/max(R1["fn"],EPS)
    add("RED_FN_reduction_rate",0,fn_reduce)
    for k in ["fixed","regressed","net","red_recovered","red_regressed","red_new_fp","red_fp_fixed","red_both_missed"]:
        add(k,0,totals[k])
    recovery=totals["red_recovered"]/max(totals["v1_red_fn"],EPS)
    add("RED_V1_FN_recovery_rate",0,recovery)

    if comps:
        x=np.array([z["v1_coverage"] for z in comps]); y=np.array([z["v2_coverage"] for z in comps])
        add("RED_component_mean_coverage",float(x.mean()),float(y.mean()))
        add("RED_component_median_coverage",float(np.median(x)),float(np.median(y)))
        for t in [0.5,0.75,0.9]: add(f"RED_component_coverage_ge_{int(t*100)}pct",float(np.mean(x>=t)),float(np.mean(y>=t)))
        add("RED_component_completely_missed_rate",float(np.mean(x==0)),float(np.mean(y==0)))
    write_csv(out/"summary.csv",rows,["metric","v1","v2","delta_v2_minus_v1"])

    print("\n"+"="*78)
    print(f"{'Metric':<24}{'V1':>16}{'V2':>16}{'Delta':>16}")
    print("-"*78)
    for name,x,y in [("mIoU_fg",mi1,mi2),("mDice_fg",md1,md2),("RED IoU",R1["iou"],R2["iou"]),
                     ("RED Dice",R1["dice"],R2["dice"]),("RED Recall",R1["recall"],R2["recall"]),
                     ("RED Precision",R1["precision"],R2["precision"])]:
        print(f"{name:<24}{x:>16.6f}{y:>16.6f}{y-x:>+16.6f}")
    print("-"*78)
    print(f"RED FN: {R1['fn']:,} -> {R2['fn']:,}，减少率={fn_reduce:.2%}")
    print(f"RED FP: {R1['fp']:,} -> {R2['fp']:,}")
    print(f"V2修复V1错误像素: {totals['fixed']:,}")
    print(f"V2新增退化像素:   {totals['regressed']:,}")
    print(f"净修复像素:       {totals['net']:,}")
    print(f"V2找回V1漏掉RED:  {totals['red_recovered']:,}，V1漏检修复率={recovery:.2%}")
    if comps:
        x=np.array([z["v1_coverage"] for z in comps]); y=np.array([z["v2_coverage"] for z in comps])
        print(f"RED连通域平均coverage: {x.mean():.4f} -> {y.mean():.4f}")
        print(f"RED连通域>=90%覆盖率: {np.mean(x>=.9):.2%} -> {np.mean(y>=.9):.2%}")
        print(f"RED连通域完全漏检率:  {np.mean(x==0):.2%} -> {np.mean(y==0):.2%}")
    print("="*78)
    print(f"结果已保存: {out}")

if __name__ == "__main__":
    main()