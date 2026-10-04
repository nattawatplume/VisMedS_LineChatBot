"""ตัวเรียก LLM — รองรับ API แบบ OpenAI-compatible (Typhoon, Gemini, Groq, OpenRouter ฯลฯ)
และ Anthropic (ตัวเลือกเสริม)"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx

from app.config import Settings
from app.drugs.repository import DrugRepository
from app.llm.prompts import CURRENT_DRUG, SYSTEM_PROMPT

log = logging.getLogger(__name__)

FALLBACK = "ขออภัยค่ะ ตอนนี้ตอบไม่ได้ชั่วคราว ลองส่งคำถามอีกครั้ง หรือปรึกษาเภสัชกรได้เลยค่ะ"
NO_KEY = "ระบบแชตยังไม่พร้อมใช้งานค่ะ (ยังไม่ได้ตั้งค่า API key) แต่ส่งรูปยามาให้ตรวจได้เลย"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)  # โมเดลแบบ reasoning บางตัวพ่น <think>


class LLMClient:
    def __init__(self, settings: Settings, drugs: DrugRepository,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._s = settings
        self._drugs = drugs
        self._http: httpx.AsyncClient | None = None
        self._anthropic = None
        if settings.llm_provider == "anthropic":
            if settings.anthropic_api_key:
                from anthropic import AsyncAnthropic  # ติดตั้งเพิ่มเองถ้าจะใช้
                self._anthropic = AsyncAnthropic(api_key=settings.anthropic_api_key,
                                                 timeout=settings.llm_timeout_s, max_retries=1)
        elif settings.llm_api_key:
            self._http = httpx.AsyncClient(
                base_url=settings.llm_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                timeout=settings.llm_timeout_s, transport=transport)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    def build_system(self, current_drug_id: str | None) -> str:
        current = ""
        drug = self._drugs.get(current_drug_id) if current_drug_id else None
        if drug:
            current = CURRENT_DRUG.format(drug_id=drug.id, name=drug.name_th)
        return SYSTEM_PROMPT.format(current=current,
                                    drug_context=self._drugs.to_prompt_context())

    async def reply(self, history: list[dict], user_text: str,
                    current_drug_id: str | None = None) -> str:
        if self._http is None and self._anthropic is None:
            return NO_KEY
        system = self.build_system(current_drug_id)
        messages = [*history, {"role": "user", "content": user_text}]
        try:
            if self._anthropic is not None:
                text = await self._call_anthropic(system, messages)
            else:
                text = await self._call_openai_compat(system, messages)
            text = _THINK.sub("", text).strip()
            return text or FALLBACK
        except Exception:  # noqa: BLE001 — ห้ามให้บอทเงียบเมื่อ LLM ล่ม
            log.exception("LLM call failed")
            return FALLBACK

    async def _call_openai_compat(self, system: str, messages: list[dict]) -> str:
        payload = {
            "model": self._s.llm_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": self._s.llm_max_tokens,
            "temperature": self._s.llm_temperature,
        }
        for attempt in range(3):
            r = await self._http.post("/chat/completions", json=payload)
            if r.status_code in (429, 500, 502, 503):
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))  # โดนจำกัดอัตรา/ล่มชั่วคราว
                    continue
                # หมด retry แล้วยังล้ม — คืนค่าว่างให้ caller จัดการ FALLBACK
                log.warning("LLM retry exhausted, last status=%s", r.status_code)
                return ""
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        return ""

    async def _call_anthropic(self, system: str, messages: list[dict]) -> str:
        model = self._s.llm_model if self._s.llm_provider == "anthropic" and \
            self._s.llm_model.startswith("claude") else "claude-3-5-haiku-20241022"
        resp = await self._anthropic.messages.create(
            model=model, max_tokens=self._s.llm_max_tokens, system=system, messages=messages)
        return "".join(b.text for b in resp.content if b.type == "text")
