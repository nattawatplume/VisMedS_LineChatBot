"""จำแนกยาจากรูปด้วยโมเดล TFLite

จุดที่แก้จากโค้ดเดิม:
- แก้การหมุนภาพจาก EXIF (รูปจากมือถือมักหมุน)
- center-crop เป็นสี่เหลี่ยมจัตุรัสก่อน resize แทนการบีบภาพ (ตรงกับที่ Teachable Machine ทำ)
- เลือก normalization ได้ (neg1_1 / zero_one)
- ตัดสิน "ไม่แน่ใจ" จากทั้งความมั่นใจและส่วนต่างอันดับ 1-2 + รองรับคลาส not_a_drug ในโมเดล
- thread-safe (interpreter ใช้พร้อมกันไม่ได้)
"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # runtime ตัวเบา (ไม่ต้องลง tensorflow เต็ม)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # type: ignore

UNKNOWN_LABELS = {"not_a_drug", "not-a-drug", "unknown", "ไม่ใช่ยา"}


@dataclass
class Prediction:
    label: str                      # อันดับ 1
    confidence: float               # 0..1
    margin: float                   # ส่วนต่างอันดับ 1 - 2
    top: list[tuple[str, float]]    # อันดับสูงสุดไม่เกิน 3
    is_confident: bool              # ผ่านเกณฑ์ และไม่ใช่คลาส unknown


def load_labels(path: Path) -> list[str]:
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, tail = line.partition(" ")
        labels.append(tail.strip() if head.isdigit() and tail else line)
    return labels


def decide(probs: np.ndarray, labels: list[str], conf_threshold: float,
           margin_threshold: float) -> Prediction:
    """แปลงความน่าจะเป็นเป็นผลตัดสิน (แยกออกมาเพื่อทดสอบได้ง่าย)"""
    order = np.argsort(probs)[::-1]
    top = [(labels[i], float(probs[i])) for i in order[:3]]
    p1 = float(probs[order[0]])
    p2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    label = labels[order[0]]
    ok = (p1 >= conf_threshold and (p1 - p2) >= margin_threshold
          and label.lower() not in UNKNOWN_LABELS)
    return Prediction(label, p1, p1 - p2, top, ok)


class DrugClassifier:
    def __init__(self, model_path: Path, labels_path: Path, norm: str = "neg1_1",
                 conf_threshold: float = 0.80, margin_threshold: float = 0.15):
        if norm not in ("neg1_1", "zero_one"):
            raise ValueError("norm ต้องเป็น neg1_1 หรือ zero_one")
        self.labels = load_labels(labels_path)
        self.norm = norm
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self._lock = threading.Lock()
        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        n_out = int(self._out["shape"][-1])
        if n_out != len(self.labels):
            raise ValueError(
                f"โมเดลมี {n_out} คลาส แต่ labels.txt มี {len(self.labels)} รายการ — ต้องตรงกัน")

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        _, h, w, _ = self._in["shape"]
        img = ImageOps.fit(img, (int(w), int(h)), Image.Resampling.LANCZOS)
        arr = np.asarray(img)
        if self._in["dtype"] == np.uint8:
            out = arr.astype(np.uint8)
        elif self.norm == "neg1_1":
            out = arr.astype(np.float32) / 127.5 - 1.0
        else:
            out = arr.astype(np.float32) / 255.0
        return out[np.newaxis, ...]

    def predict_probs(self, image_bytes: bytes) -> np.ndarray:
        x = self._preprocess(image_bytes)
        with self._lock:
            self._interp.set_tensor(self._in["index"], x)
            self._interp.invoke()
            return self._interp.get_tensor(self._out["index"])[0].copy()

    def predict(self, image_bytes: bytes) -> Prediction:
        return decide(self.predict_probs(image_bytes), self.labels,
                      self.conf_threshold, self.margin_threshold)
