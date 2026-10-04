"""ทดลองคุยกับ LLM ในเทอร์มินัล โดยไม่ต้องผ่าน LINE:  python -m scripts.chat_cli"""
import asyncio

from app.config import get_settings
from app.drugs.repository import DrugRepository
from app.llm import guardrails
from app.llm.client import LLMClient


async def main():
    s = get_settings()
    llm = LLMClient(s, DrugRepository(s.drugs_path))
    history: list[dict] = []
    print("พิมพ์ข้อความ (Ctrl+C เพื่อออก)")
    while True:
        text = input("คุณ: ").strip()
        if not text:
            continue
        ans = guardrails.check_emergency(text) or await llm.reply(history, text)
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": ans}]
        print(f"บอท: {ans}\n")


if __name__ == "__main__":
    asyncio.run(main())
