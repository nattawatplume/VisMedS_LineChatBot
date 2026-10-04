FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY data data
COPY models models
ENV PYTHONUNBUFFERED=1
# แพลตฟอร์ม cloud ส่วนใหญ่กำหนดพอร์ตผ่านตัวแปร PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
