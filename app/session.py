"""สถานะการคุยรายผู้ใช้ (ในหน่วยความจำ) — ตอนขึ้น production หลาย instance ให้ย้ายไป Redis/SQLite"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class Session:
    history: list[dict] = field(default_factory=list)   # [{"role","content"}]
    current_drug_id: str | None = None   # ยาที่ผู้ใช้ยืนยันแล้วล่าสุด
    pending_drug_id: str | None = None   # ยาที่โมเดลทาย รอผู้ใช้ยืนยัน
    updated: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_s: int, max_turns: int):
        self._ttl = ttl_s
        self._max_messages = max_turns * 2
        self._data: dict[str, Session] = {}
        self._lock = asyncio.Lock()   # ป้องกัน race condition เมื่อหลาย request มาพร้อมกัน

    def get(self, user_id: str) -> Session:
        now = time.time()
        expired = [u for u, s in self._data.items() if now - s.updated > self._ttl]
        for uid in expired:
            del self._data[uid]
        s = self._data.setdefault(user_id, Session())
        s.updated = now
        return s

    def add_turn(self, user_id: str, user_text: str, bot_text: str) -> None:
        s = self.get(user_id)
        s.history += [{"role": "user", "content": user_text},
                      {"role": "assistant", "content": bot_text}]
        s.history = s.history[-self._max_messages:]

