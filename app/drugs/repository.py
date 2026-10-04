"""ฐานข้อมูลยา (อ่านจาก data/drugs.json) — แหล่งข้อมูลยาแหล่งเดียวของทั้งระบบ"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Drug:
    id: str
    model_label: str
    name_th: str
    generic: str
    what_is: str
    usage: list[str] = field(default_factory=list)
    indications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audio: str | None = None
    verified: bool = False


class DrugRepository:
    def __init__(self, path: Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.disclaimer: str = raw.get("disclaimer", "")
        self._drugs = {d["id"]: Drug(**d) for d in raw["drugs"]}
        self._by_label = {d.model_label: d for d in self._drugs.values()}

    def all(self) -> list[Drug]:
        return list(self._drugs.values())

    def get(self, drug_id: str) -> Drug | None:
        return self._drugs.get(drug_id)

    def by_label(self, label: str) -> Drug | None:
        return self._by_label.get(label)

    # ---- รูปแบบข้อความ ----
    @staticmethod
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"– {i}" for i in items)

    def format_reply(self, drug: Drug) -> str:
        parts = [
            f"💊 {drug.name_th}",
            f"{drug.what_is}\n(ตัวยา: {drug.generic})",
            f"💡 วิธีใช้\n{self._bullets(drug.usage)}",
            f"📌 สรรพคุณ\n{self._bullets(drug.indications)}",
            f"⚠️ ข้อควรระวัง\n{self._bullets(drug.warnings)}",
            self.disclaimer,
        ]
        return "\n\n".join(p for p in parts if p)

    def to_prompt_context(self) -> str:
        """ข้อมูลยาทั้งหมดในรูปข้อความ ใส่ใน system prompt ของ LLM
        (เมื่อฐานข้อมูลโตขึ้นมาก ค่อยเปลี่ยนเป็นการค้นหาเฉพาะยาที่เกี่ยวข้อง)"""
        blocks = []
        for d in self._drugs.values():
            blocks.append(
                f"[{d.id}] {d.name_th} | ตัวยา: {d.generic} | คือ: {d.what_is}\n"
                f"วิธีใช้: {'; '.join(d.usage)}\n"
                f"สรรพคุณ: {'; '.join(d.indications)}\n"
                f"ข้อควรระวัง: {'; '.join(d.warnings)}"
            )
        return "\n\n".join(blocks)
