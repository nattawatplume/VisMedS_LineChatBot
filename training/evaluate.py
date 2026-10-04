"""วัดความแม่นยำของโมเดลกับชุดภาพทดสอบ และเทียบ normalization ทั้งสองแบบ

โครงสร้างโฟลเดอร์ (ชื่อโฟลเดอร์ต้องตรงกับชื่อใน models/labels.txt):
    dataset/test/ซาร่า/*.jpg
    dataset/test/ยาธาตุน้ำขาวกระต่ายบิน/*.jpg
    dataset/test/ไทลินอล/*.jpg

รัน:  python -m training.evaluate --data dataset/test --norm both
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.vision.classifier import DrugClassifier

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def run(data: Path, norm: str):
    s = get_settings()
    clf = DrugClassifier(s.model_path, s.labels_path, norm, s.conf_threshold, s.margin_threshold)
    idx = {l: i for i, l in enumerate(clf.labels)}
    n = len(clf.labels)
    cm = np.zeros((n, n), dtype=int)
    rejected = defaultdict(int)
    for d in sorted(p for p in data.iterdir() if p.is_dir()):
        if d.name not in idx:
            print(f"  ข้ามโฟลเดอร์ '{d.name}' (ไม่อยู่ใน labels.txt)")
            continue
        for f in d.iterdir():
            if f.suffix.lower() not in EXTS:
                continue
            pred = clf.predict(f.read_bytes())
            cm[idx[d.name], idx[pred.label]] += 1
            if not pred.is_confident:
                rejected[d.name] += 1
    total = cm.sum()
    if total == 0:
        print("  ไม่พบภาพ")
        return
    print(f"\n== norm={norm}  ภาพทั้งหมด {total}  accuracy(อันดับ 1) = {np.trace(cm)/total:.1%}")
    print("   ภาพที่บอทจะตอบว่า 'ไม่แน่ใจ' (ต่ำกว่าเกณฑ์):",
          dict(rejected) or "ไม่มี")
    print("   confusion matrix (แถว=ของจริง, คอลัมน์=ที่โมเดลทาย):")
    print("   labels:", clf.labels)
    print(cm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--norm", choices=["neg1_1", "zero_one", "both"], default="both")
    a = ap.parse_args()
    for nm in (["neg1_1", "zero_one"] if a.norm == "both" else [a.norm]):
        run(a.data, nm)
