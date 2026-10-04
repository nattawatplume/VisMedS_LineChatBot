"""ค่าตั้งค่าทั้งหมดอ่านจากตัวแปรสภาพแวดล้อม / ไฟล์ .env"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LINE (ได้จาก LINE Developers Console > Messaging API channel)
    line_channel_secret: str = ""
    line_channel_access_token: str = ""

    # LLM — ค่าเริ่มต้น: Typhoon (โมเดลไทย ฟรี ใช้ผ่าน OpenAI-compatible API)
    # เปลี่ยนเจ้าอื่นได้แค่แก้ LLM_BASE_URL / LLM_MODEL / LLM_API_KEY ใน .env
    llm_provider: str = "openai_compat"   # openai_compat | anthropic
    llm_api_key: str = ""
    llm_base_url: str = "https://api.opentyphoon.ai/v1"
    llm_model: str = "typhoon-v2.5-30b-a3b-instruct"
    anthropic_api_key: str = ""           # ใช้เมื่อ llm_provider=anthropic เท่านั้น
    llm_max_tokens: int = 600
    llm_temperature: float = 0.3
    llm_timeout_s: float = 25.0

    # Vision
    model_path: Path = ROOT / "models" / "model.tflite"
    labels_path: Path = ROOT / "models" / "labels.txt"
    # neg1_1 = (x/127.5)-1 (ค่าปกติของโมเดล Teachable Machine) | zero_one = x/255
    # ยังไม่ยืนยันกับโมเดลของคุณ -> ใช้ training/evaluate.py เทียบก่อน (ดู docs/CHECKLIST.md)
    input_norm: str = "neg1_1"
    conf_threshold: float = 0.80   # ความมั่นใจขั้นต่ำของอันดับ 1
    margin_threshold: float = 0.15  # อันดับ 1 ต้องนำอันดับ 2 อย่างน้อยเท่านี้

    # Data
    drugs_path: Path = ROOT / "data" / "drugs.json"

    # Session (เก็บในหน่วยความจำ — หายเมื่อรีสตาร์ท)
    session_ttl_s: int = 60 * 30
    session_max_turns: int = 8

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
