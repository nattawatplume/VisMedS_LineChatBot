# VisMedS — LINE Bot ให้ข้อมูลยา

ส่งรูปยาในแชต LINE → บอทจำแนกยา → ให้ผู้ใช้ยืนยัน → แสดงข้อมูลยา
และพิมพ์ถามต่อได้ โดยมี LLM (ค่าเริ่มต้น Typhoon) ตอบจากฐานข้อมูลยาเท่านั้น

```
LINE ──► POST /callback (FastAPI, ตรวจลายเซ็น)
            ├─ รูปภาพ  → app/vision  (TFLite)  → ถามยืนยัน → app/drugs → ข้อมูลยา
            ├─ ข้อความ → guardrails (ฉุกเฉิน) → app/llm (Typhoon/Gemini/ฯลฯ) + data/drugs.json
            └─ ตอบด้วย Reply API (ไม่นับโควตาข้อความ)
```

## โครงสร้าง
```
app/main.py            จุดเริ่มต้น FastAPI, /healthz
app/config.py          ตัวแปรตั้งค่า (.env)
app/line/              webhook.py (รับ event) + handlers.py (ตรรกะบอท)
app/vision/            classifier.py (โหลดโมเดล, เตรียมภาพ, ตัดสินความมั่นใจ)
app/llm/               client.py, prompts.py, guardrails.py
app/drugs/             repository.py อ่าน data/drugs.json
app/session.py         สถานะการคุยรายผู้ใช้ (ในหน่วยความจำ)
data/drugs.json        ฐานข้อมูลยา (แหล่งความจริงแหล่งเดียว)   data/audio/  ไฟล์เสียงเดิม
models/                model.tflite + labels.txt
training/evaluate.py   วัด accuracy + confusion matrix
scripts/chat_cli.py    คุยกับ LLM ในเทอร์มินัลโดยไม่ต้องใช้ LINE
liff/                  (เฟส 4) หน้าสแกนสด
docs/                  CHECKLIST.md  DATA_COLLECTION.md
tests/                 pytest
```

## เริ่มใช้งาน
ดู **docs/CHECKLIST.md** (ทำตามลำดับ A → C)

```bash
pip install -r requirements-dev.txt
cp .env.example .env      # กรอกค่า
pytest
uvicorn app.main:app --port 8080
```

## หมายเหตุสำคัญ
- ข้อมูลใน `data/drugs.json` ยก "verified": false ทั้งหมด ต้องตรวจก่อนใช้กับคนจริง
- บอทไม่ใช่เครื่องมือวินิจฉัย/สั่งยา; ข้อความฉุกเฉินจะถูกตอบด้วยข้อความสำเร็จรูปโดยไม่ผ่าน LLM
- Session เก็บในหน่วยความจำ รันได้ 1 instance; ถ้าขยายต้องย้ายไป Redis/SQLite
