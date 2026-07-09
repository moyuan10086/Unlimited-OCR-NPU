import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PARQUET = "/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/test-00000-of-00001.parquet"
OUT_DIR = Path("/mnt/model/baidu/Unlimited-OCR/benchmarks/OCRBench/str_eval")
IMG_DIR = OUT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["IIIT5K", "svt", "IC13_857", "IC15_1811", "svtp", "ct80"]
URL = "http://127.0.0.1:10080/ocr"


def norm(s):
    if s is None:
        return ""
    if isinstance(s, list):
        s = " ".join(map(str, s))
    s = str(s).strip()
    s = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def answer_text(ans):
    if isinstance(ans, (list, tuple)):
        return str(ans[0]) if ans else ""
    return str(ans)


def image_bytes_to_path(row, idx):
    img = row["image"]
    b = img.get("bytes") if isinstance(img, dict) else None
    if not b:
        raise RuntimeError(f"no image bytes for row {idx}")
    im = Image.open(io.BytesIO(b)).convert("RGB")
    path = IMG_DIR / f"{idx:04d}_{row['dataset']}.png"
    im.save(path)
    return str(path)


df = pd.read_parquet(PARQUET)
sub = df[df["dataset"].isin(TARGETS)].copy().reset_index(drop=True)
print(f"samples={len(sub)} datasets={sub['dataset'].value_counts().to_dict()}", flush=True)

rows = []
start_all = time.time()
for i, row in sub.iterrows():
    img_path = image_bytes_to_path(row, i)
    gt = answer_text(row["answer"])
    t0 = time.time()
    try:
        r = requests.post(URL, json={
            "image_path": img_path,
            "prompt": "<image>document parsing.",
            "max_length": 1024,
        }, timeout=120)
        r.raise_for_status()
        pred = r.json().get("text", "")
        err = ""
    except Exception as e:
        pred = ""
        err = repr(e)
    sec = time.time() - t0
    ngt = norm(gt)
    npred = norm(pred)
    exact = bool(ngt and npred == ngt)
    contains = bool(ngt and ngt in npred)
    rows.append({
        "idx": i,
        "dataset": row["dataset"],
        "question_type": row["question_type"],
        "answer": gt,
        "prediction": pred,
        "norm_answer": ngt,
        "norm_prediction": npred,
        "exact": exact,
        "contains": contains,
        "seconds": round(sec, 3),
        "image_path": img_path,
        "error": err,
    })
    print(f"[{i+1:03d}/{len(sub)}] {row['dataset']} gt={ngt} pred={npred[:80]} exact={exact} contains={contains} sec={sec:.2f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "results.csv", index=False)
summary = {
    "samples": int(len(res)),
    "exact": int(res["exact"].sum()),
    "exact_acc": round(float(res["exact"].mean()), 4) if len(res) else 0,
    "contains": int(res["contains"].sum()),
    "contains_acc": round(float(res["contains"].mean()), 4) if len(res) else 0,
    "avg_seconds": round(float(res["seconds"].mean()), 3) if len(res) else 0,
    "total_seconds": round(time.time() - start_all, 3),
    "by_dataset": {},
}
for name, g in res.groupby("dataset"):
    summary["by_dataset"][name] = {
        "samples": int(len(g)),
        "exact": int(g["exact"].sum()),
        "exact_acc": round(float(g["exact"].mean()), 4),
        "contains": int(g["contains"].sum()),
        "contains_acc": round(float(g["contains"].mean()), 4),
        "avg_seconds": round(float(g["seconds"].mean()), 3),
    }
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
