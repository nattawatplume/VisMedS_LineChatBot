import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    AsyncApiClient, AsyncMessagingApi, AsyncMessagingApiBlob, Configuration,
)

from app.config import get_settings
from app.drugs.repository import DrugRepository
from app.line.handlers import BotHandlers
from app.line.webhook import router
from app.llm.client import LLMClient
from app.session import SessionStore
from app.vision.classifier import DrugClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logging.basicConfig(level=s.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    drugs = DrugRepository(s.drugs_path)
    classifier = DrugClassifier(s.model_path, s.labels_path, s.input_norm,
                                s.conf_threshold, s.margin_threshold)
    api_client = AsyncApiClient(Configuration(access_token=s.line_channel_access_token))
    llm = LLMClient(s, drugs)
    app.state.parser = WebhookParser(s.line_channel_secret)
    app.state.handlers = BotHandlers(
        api=AsyncMessagingApi(api_client),
        blob=AsyncMessagingApiBlob(api_client),
        classifier=classifier,
        drugs=drugs,
        llm=llm,
        sessions=SessionStore(s.session_ttl_s, s.session_max_turns),
        base_url=s.app_base_url,
    )
    yield
    await llm.aclose()
    await api_client.close()


s = get_settings()
app = FastAPI(title="VisMedS", lifespan=lifespan)
app.include_router(router)

# ให้บริการไฟล์เสียงสำหรับ LINE Audio Message
audio_path = s.drugs_path.parent / "audio"
if audio_path.exists():
    app.mount("/static/audio", StaticFiles(directory=audio_path), name="audio")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

