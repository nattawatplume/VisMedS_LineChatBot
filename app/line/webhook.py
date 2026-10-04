from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from linebot.v3.exceptions import InvalidSignatureError

router = APIRouter()


@router.post("/callback")
async def callback(request: Request, background: BackgroundTasks,
                   x_line_signature: str | None = Header(default=None)):
    """LINE ส่ง event มาที่นี่ — ตรวจลายเซ็น แล้วตอบ 200 ทันที ส่วนการประมวลผลทำเบื้องหลัง"""
    body = (await request.body()).decode("utf-8")
    try:
        events = request.app.state.parser.parse(body, x_line_signature or "")
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="invalid signature")
    for event in events:
        background.add_task(request.app.state.handlers.dispatch, event)
    return "OK"
