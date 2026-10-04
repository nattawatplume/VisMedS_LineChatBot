import base64, hashlib, hmac, io, json, os

import numpy as np
from PIL import Image

os.environ["LINE_CHANNEL_SECRET"] = "test-secret"
os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "test-token"
os.environ["LLM_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.config import ROOT, get_settings  # noqa: E402
from app.drugs.repository import DrugRepository  # noqa: E402
from app.llm.guardrails import check_emergency  # noqa: E402
from app.session import SessionStore  # noqa: E402
from app.vision.classifier import DrugClassifier, decide, load_labels  # noqa: E402

LABELS = ["a", "b", "not_a_drug"]


def test_decide_confident():
    p = decide(np.array([0.9, 0.05, 0.05]), LABELS, 0.8, 0.15)
    assert p.label == "a" and p.is_confident


def test_decide_low_confidence():
    assert not decide(np.array([0.6, 0.3, 0.1]), LABELS, 0.8, 0.15).is_confident


def test_decide_small_margin():
    assert not decide(np.array([0.82, 0.80, 0.0]), LABELS, 0.8, 0.15).is_confident


def test_decide_unknown_class_never_confident():
    assert not decide(np.array([0.0, 0.05, 0.95]), LABELS, 0.8, 0.15).is_confident


def test_labels_match_drugs():
    s = get_settings()
    repo = DrugRepository(s.drugs_path)
    for label in load_labels(s.labels_path):
        assert repo.by_label(label) is not None, label
    for d in repo.all():
        assert (ROOT / "data" / "audio" / d.audio).exists()


def test_classifier_runs_on_any_image():
    s = get_settings()
    clf = DrugClassifier(s.model_path, s.labels_path, "neg1_1")
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (300, 500, 3), dtype=np.uint8)).save(buf, "JPEG")
    probs = clf.predict_probs(buf.getvalue())
    assert probs.shape == (3,) and abs(float(probs.sum()) - 1.0) < 1e-3


def test_emergency_guardrail():
    assert check_emergency("กินยาเกินไปเยอะมาก") is not None
    assert check_emergency("หายใจไม่ออก") is not None
    assert check_emergency("ซาร่ากินตอนไหน") is None


def test_session_trim():
    st = SessionStore(ttl_s=100, max_turns=2)
    for i in range(5):
        st.add_turn("u", f"q{i}", f"a{i}")
    assert len(st.get("u").history) == 4


def test_prompt_contains_drugs():
    from app.llm.client import LLMClient
    s = get_settings()
    repo = DrugRepository(s.drugs_path)
    prompt = LLMClient(s, repo).build_system("sara")
    assert "ซาร่า" in prompt and "ฐานข้อมูลยา" in prompt


def _sig(body: str) -> str:
    return base64.b64encode(
        hmac.new(b"test-secret", body.encode(), hashlib.sha256).digest()).decode()


def test_webhook_rejects_bad_signature():
    from app.main import app
    with TestClient(app) as c:
        r = c.post("/callback", content=json.dumps({"destination": "x", "events": []}),
                   headers={"X-Line-Signature": "bad"})
        assert r.status_code == 400


def test_webhook_accepts_valid_signature():
    from app.main import app
    body = json.dumps({"destination": "x", "events": []})
    with TestClient(app) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        r = c.post("/callback", content=body, headers={"X-Line-Signature": _sig(body)})
        assert r.status_code == 200


# ---------- end-to-end ด้วย LINE ปลอม ----------
class FakeApi:
    def __init__(self):
        self.replies = []

    async def reply_message(self, req):
        self.replies.append(req)

    async def show_loading_animation(self, req):
        pass


class FakeBlob:
    async def get_message_content(self, message_id):
        buf = io.BytesIO()
        Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)).save(buf, "JPEG")
        return bytearray(buf.getvalue())


def _event(message: dict) -> str:
    return json.dumps({"destination": "x", "events": [{
        "type": "message", "mode": "active", "timestamp": 1,
        "source": {"type": "user", "userId": "U1"}, "webhookEventId": "01",
        "deliveryContext": {"isRedelivery": False}, "replyToken": "tok", "message": message}]})


def _post(client, body):
    return client.post("/callback", content=body, headers={"X-Line-Signature": _sig(body)})


def test_e2e_text_and_image():
    from app.main import app
    with TestClient(app) as c:
        api = FakeApi()
        app.state.handlers.api, app.state.handlers.blob = api, FakeBlob()

        _post(c, _event({"type": "text", "id": "1", "quoteToken": "q", "text": "ยาที่รู้จัก"}))
        assert "ซาร่า" in api.replies[-1].messages[0].text

        _post(c, _event({"type": "text", "id": "2", "quoteToken": "q", "text": "หายใจไม่ออก"}))
        assert "1669" in api.replies[-1].messages[0].text

        _post(c, _event({"type": "text", "id": "3", "quoteToken": "q", "text": "ซาร่าคืออะไร"}))
        assert api.replies[-1].messages[0].text  # ไม่มี API key -> ข้อความ fallback

        n = len(api.replies)
        _post(c, _event({"type": "image", "id": "4", "quoteToken": "q",
                         "contentProvider": {"type": "line"}}))
        assert len(api.replies) == n + 1  # ภาพสุ่ม -> ตอบกลับเสมอ (ไม่เงียบ)


# ---------- LLM แบบ OpenAI-compatible (ใช้ transport ปลอม ไม่ยิงเน็ตจริง) ----------
import asyncio  # noqa: E402

import httpx  # noqa: E402


def _llm(handler, **over):
    from app.llm.client import LLMClient
    s = get_settings().model_copy(update={"llm_api_key": "k", **over})
    return LLMClient(s, DrugRepository(s.drugs_path), transport=httpx.MockTransport(handler))


def test_llm_openai_compat_ok_and_strips_think():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        seen["path"] = req.url.path
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "<think>คิดอยู่</think>กินหลังอาหารค่ะ"}}]})

    out = asyncio.run(_llm(handler).reply([], "ยาธาตุกินตอนไหน", "thatu-nam-khao"))
    assert out == "กินหลังอาหารค่ะ"
    assert seen["auth"] == "Bearer k" and seen["path"].endswith("/chat/completions")
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "ยาธาตุกินตอนไหน"}


def test_llm_falls_back_on_error():
    from app.llm.client import FALLBACK
    out = asyncio.run(_llm(lambda req: httpx.Response(400, json={"error": "x"})).reply([], "hi"))
    assert out == FALLBACK


def test_llm_retries_on_429():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert asyncio.run(_llm(handler).reply([], "hi")) == "ok" and calls["n"] == 2
