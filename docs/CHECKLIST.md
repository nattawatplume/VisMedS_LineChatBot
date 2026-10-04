# Checklist ของคุณ (Plume) — ทำตามลำดับ

## A. เตรียมก่อนเริ่ม (ทำได้เลยวันนี้)
- [ ] 1. สร้าง **LINE Official Account** + เปิด **Messaging API** ที่ https://developers.line.biz
      (Provider → Create a Messaging API channel) แล้วเก็บ **Channel secret** และ **Channel access token (long-lived)**
- [ ] 2. ใน LINE Official Account Manager: ปิด "ข้อความตอบกลับอัตโนมัติ" และ "ข้อความต้อนรับ" (ไม่งั้นชนกับบอท)
      เปิด Webhook (Use webhook) ในหน้า Messaging API
- [ ] 3. **ตรวจ normalization ของโมเดลเดิม**: เก็บรูปยาจริงอย่างละ ~20 รูป (ถ่ายใหม่ ไม่ใช่รูปที่เคยเทรน)
      วางใน `dataset/test/<ชื่อยา>/` แล้วรัน `python -m training.evaluate --data dataset/test --norm both`
      ดูว่าแบบไหน accuracy สูงกว่า แล้วตั้ง `INPUT_NORM` ใน .env ตามนั้น
- [ ] 4. สมัคร **Typhoon API key** ฟรีที่ https://opentyphoon.ai (โมเดลไทย) ใส่ใน `LLM_API_KEY`
      แล้วลอง `python -m scripts.chat_cli` ถามเรื่องยาหลาย ๆ แบบ ถ้าคุณภาพไม่พอ ค่อยสลับไป Gemini (ดู .env.example)
- [ ] 5. **ตรวจข้อมูลยาใน `data/drugs.json` กับฉลากจริง/แหล่งทางการ** (ทั้ง 3 ตัว โดยเฉพาะตัวยาของยาธาตุน้ำขาว)
      หรือให้เภสัชกร/อาจารย์ช่วยดู แล้วเปลี่ยน `"verified": true` + ใส่แหล่งอ้างอิง
- [ ] 6. เริ่ม **เก็บรูปยา** ตาม docs/DATA_COLLECTION.md (ขั้นตอนนี้ใช้เวลานานสุด ควรเริ่มเลย)

## B. ตัดสินใจ (บอกผมได้ตอนถึงขั้นนั้น)
- [ ] 7. ขอบเขตการรู้จักยา: เฉพาะ "กล่อง/ซอง/ขวด" หรือรวม "เม็ดยาเปล่า" ด้วย (ผมแนะนำเริ่มจากบรรจุภัณฑ์)
- [ ] 8. จะ deploy ที่ไหน (ผมจะเช็กเงื่อนไข free tier ล่าสุดให้ตอนเลือก)
- [ ] 9. ยาที่จะเพิ่มในรอบต่อไป (ชื่อ + ฉลากจริง) เพื่อให้ขยายฐานข้อมูล

## C. รันบนเครื่องตัวเอง (เฟส 1)
- [ ] 10. `python -m venv .venv` → activate → `pip install -r requirements-dev.txt`
- [ ] 11. `cp .env.example .env` แล้วกรอกค่า
- [ ] 12. `pytest` ต้องผ่านทั้งหมด
- [ ] 13. `python -m scripts.chat_cli` ลองคุยกับ LLM (ตรวจว่าตอบตามฐานข้อมูล ไม่มั่ว)
- [ ] 14. `uvicorn app.main:app --port 8080` + เปิด tunnel ชั่วคราวระหว่างพัฒนา (ngrok/cloudflared)
      ตั้ง Webhook URL เป็น `https://<tunnel>/callback` กด Verify แล้วลองส่งข้อความ/รูปในแชต
- [ ] 15. เมื่อ deploy จริงแล้ว เปลี่ยน Webhook URL เป็น URL ถาวร (เลิกพึ่ง tunnel)

## D. Roadmap
| เฟส | งาน | สถานะ |
|---|---|---|
| 0 | โครงสร้างใหม่ + ฐานข้อมูลยา + บอท LINE พื้นฐาน + LLM + ชุดทดสอบ | ✅ อยู่ในโปรเจกต์นี้ |
| 1 | ต่อ LINE จริง ทดสอบ ลองใช้ + deploy ขึ้น URL ถาวร | ← ถัดไป |
| 2 | ปรับ prompt/ฐานข้อมูล, ทดสอบคำถามจริงจากผู้ใช้, เพิ่มยา | |
| 3 | เก็บข้อมูล → เทรนโมเดลใหม่ (transfer learning + คลาส not_a_drug) → ประเมินด้วย `training/evaluate.py` | |
| 4 | สแกนสดใน LIFF | |
| 5 | Flex message, เสียงอ่าน, log/สถิติ, ทดสอบกับผู้ใช้สูงอายุจริง | |
