"""ตรรกะของบอท: รับ event จาก LINE แล้วตอบด้วย Reply API (ไม่นับโควตาข้อความ)"""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import parse_qs, urlencode

from linebot.v3.messaging import (
    AsyncMessagingApi, AsyncMessagingApiBlob, AudioMessage, PostbackAction, PushMessageRequest,
    QuickReply, QuickReplyItem, ReplyMessageRequest, ShowLoadingAnimationRequest, TextMessage,
)
from linebot.v3.webhooks import (
    FollowEvent, ImageMessageContent, MessageEvent, PostbackEvent, TextMessageContent,
)

from app.drugs.repository import DrugRepository
from app.llm import guardrails
from app.llm.client import LLMClient
from app.session import SessionStore
from app.vision.classifier import DrugClassifier

log = logging.getLogger(__name__)

WELCOME = (
    "สวัสดีค่ะ ยินดีต้อนรับสู่ VisMedS 💊\n"
    "ผู้ช่วยดูข้อมูลยาเบื้องต้น\n\n"
    "📷 ส่งรูปยา (กล่อง/ซอง/ขวด) มาให้ดู\n"
    "💬 หรือพิมพ์ถามเรื่องยาที่รู้จักได้เลย\n"
    "พิมพ์ \"ยาที่รู้จัก\" เพื่อดูรายชื่อยา\n\n"
    "ข้อมูลเป็นเพียงความรู้เบื้องต้น ไม่ใช่คำแนะนำทางการแพทย์"
)
PHOTO_TIPS = (
    "ขออภัยค่ะ ดูไม่ออกว่าเป็นยาอะไรอย่างมั่นใจ 🙏\n"
    "ลองถ่ายใหม่โดย\n"
    "– ให้ชื่อยาบนกล่องอยู่กลางภาพ\n"
    "– ถ่ายในที่สว่าง ไม่เบลอ\n"
    "– ถ่ายยาชิ้นเดียว พื้นหลังโล่ง\n"
    "หรือพิมพ์ชื่อยามาถามได้เลยค่ะ"
)


def _quick(*pairs: tuple[str, dict]) -> QuickReply:
    return QuickReply(items=[
        QuickReplyItem(action=PostbackAction(label=label, data=urlencode(data),
                                             display_text=label))
        for label, data in pairs
    ])


class BotHandlers:
    def __init__(self, api: AsyncMessagingApi, blob: AsyncMessagingApiBlob,
                 classifier: DrugClassifier, drugs: DrugRepository,
                 llm: LLMClient, sessions: SessionStore,
                 base_url: str = "https://vismeds-bot.onrender.com"):
        self.api, self.blob = api, blob
        self.classifier, self.drugs = classifier, drugs
        self.llm, self.sessions = llm, sessions
        self.base_url = base_url

    # ---------- utilities ----------
    async def _reply(self, target: any, *items: any,
                     quick_reply: QuickReply | None = None,
                     user_id: str | None = None):
        msgs = []
        for item in items:
            if isinstance(item, str):
                msgs.append(TextMessage(text=item[:5000]))
            else:
                msgs.append(item)
        msgs = msgs[:5]
        if quick_reply is not None and msgs:
            msgs[-1].quick_reply = quick_reply


        token = None
        uid = user_id

        if isinstance(target, str):
            token = target
        elif target is not None:
            token = getattr(target, "reply_token", None)
            if not uid and hasattr(target, "source"):
                uid = getattr(target.source, "user_id", None)

        if token:
            try:
                await self.api.reply_message(ReplyMessageRequest(reply_token=token, messages=msgs))
                return
            except Exception:
                log.warning("reply_message failed (token might be expired or invalid), attempting push fallback", exc_info=True)

        if uid:
            log.info("Sending push_message fallback to %s", uid)
            try:
                await self.api.push_message(PushMessageRequest(to=uid, messages=msgs))
            except Exception:
                log.exception("push_message fallback failed")
        else:
            log.error("Cannot reply or push: no valid reply_token or user_id")

    async def _loading(self, user_id: str | None, seconds: int = 10):
        if not user_id:
            return
        try:  # ใช้ได้เฉพาะแชต 1:1 — พลาดก็ไม่เป็นไร
            await self.api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=seconds))
        except Exception:  # noqa: BLE001
            log.debug("loading animation failed", exc_info=True)

    # ---------- dispatch ----------
    async def dispatch(self, event) -> None:
        try:
            if isinstance(event, FollowEvent):
                await self._reply(event, WELCOME)
            elif isinstance(event, MessageEvent):
                if isinstance(event.message, TextMessageContent):
                    await self.on_text(event)
                elif isinstance(event.message, ImageMessageContent):
                    await self.on_image(event)
                else:
                    await self._reply(event, "ตอนนี้รับได้เฉพาะข้อความและรูปภาพค่ะ")
            elif isinstance(event, PostbackEvent):
                await self.on_postback(event)
        except Exception:  # noqa: BLE001
            log.exception("handler failed")
            try:
                await self._reply(event, "ขออภัยค่ะ ระบบขัดข้อง ลองใหม่อีกครั้งนะคะ")
            except Exception:  # noqa: BLE001
                log.exception("fallback reply failed")

    # ---------- text ----------
    async def on_text(self, event: MessageEvent) -> None:
        text = event.message.text.strip()
        user_id = event.source.user_id if event.source else None

        if (urgent := guardrails.check_emergency(text)) is not None:
            await self._reply(event, urgent, user_id=user_id)
            return
        if text in ("ยาที่รู้จัก", "รายชื่อยา", "help", "ช่วยเหลือ", "วิธีใช้"):
            names = "\n".join(f"• {d.name_th}" for d in self.drugs.all())
            await self._reply(event, f"ตอนนี้รู้จักยาเหล่านี้ค่ะ\n{names}\n\n"
                                     "ส่งรูปยามาให้ดู หรือพิมพ์ถามได้เลยค่ะ", user_id=user_id)
            return

        # ผู้ใช้ขอฟังเสียงแนะนำการใช้ยา
        if text in ("ฟังเสียง", "เปิดเสียง", "ขอเสียง", "เล่นเสียง", "ฟังเสียงยา"):
            sess = self.sessions.get(user_id or "anon")
            drug = self.drugs.get(sess.current_drug_id) if sess.current_drug_id else None
            if drug and (audio_info := self.drugs.get_audio_info(drug, self.base_url)):
                audio_url, duration_ms = audio_info
                await self._reply(
                    event,
                    f"🔊 เสียงแนะนำวิธีใช้: {drug.name_th}",
                    AudioMessage(original_content_url=audio_url, duration=duration_ms),
                    user_id=user_id,
                )
                return
            await self._reply(event, "ยังไม่ได้เลือกยาค่ะ ส่งรูปยาหรือบอกชื่อยาก่อนนะคะ เช่น พิมพ์ \"ซาร่า\" แล้วกดขอฟังเสียงได้ค่ะ", user_id=user_id)
            return

        # ถ้าผู้ใช้พิมพ์ชื่อยาที่รู้จักตรงๆ ให้ตอบข้อมูลยาพร้อมเสียงทันที
        matched_drug = next((d for d in self.drugs.all() if text.lower() in (d.name_th.lower(), d.id.lower(), d.model_label.lower())), None)
        if matched_drug:
            sess = self.sessions.get(user_id or "anon")
            sess.current_drug_id = matched_drug.id
            reply_items = [self.drugs.format_reply(matched_drug)]
            if audio_info := self.drugs.get_audio_info(matched_drug, self.base_url):
                audio_url, duration_ms = audio_info
                reply_items.append(AudioMessage(original_content_url=audio_url, duration=duration_ms))
            reply_items.append("🔊 ส่งไฟล์เสียงแนะนำวิธีใช้ให้แล้วค่ะ กดฟังได้เลยนะคะ\nมีอะไรอยากถามเพิ่มเกี่ยวกับยานี้ พิมพ์ถามได้เลยค่ะ")
            await self._reply(
                event,
                *reply_items,
                quick_reply=_quick(
                    ("🔊 ฟังเสียงอีกครั้ง", {"a": "audio", "d": matched_drug.id}),
                    ("💊 ยาที่รู้จัก", {"a": "list"}),
                ),
                user_id=user_id,
            )
            return

        await self._loading(user_id, 20)
        sess = self.sessions.get(user_id or "anon")
        answer = await self.llm.reply(sess.history, text, sess.current_drug_id)
        self.sessions.add_turn(user_id or "anon", text, answer)
        await self._reply(event, answer, user_id=user_id)

    # ---------- image ----------
    async def on_image(self, event: MessageEvent) -> None:
        user_id = event.source.user_id if event.source else None
        await self._loading(user_id, 10)
        # get_message_content คืน iterator ของ chunks ต้องรวมก่อนแปลงเป็น bytes
        raw = await self.blob.get_message_content(event.message.id)
        image_bytes = b"".join([chunk async for chunk in raw] if hasattr(raw, "__aiter__") else [raw])
        pred = await asyncio.to_thread(self.classifier.predict, image_bytes)
        log.info("predict label=%s conf=%.3f margin=%.3f ok=%s",
                 pred.label, pred.confidence, pred.margin, pred.is_confident)

        drug = self.drugs.by_label(pred.label) if pred.is_confident else None
        if drug is None:
            await self._reply(event, PHOTO_TIPS, user_id=user_id)
            return

        self.sessions.get(user_id or "anon").pending_drug_id = drug.id
        await self._reply(
            event,
            f"ดูเหมือนจะเป็น \"{drug.name_th}\" ใช่ไหมคะ?\n"
            "(ช่วยเทียบชื่อกับฉลากบนกล่องอีกครั้งนะคะ)",
            quick_reply=_quick(("ใช่", {"a": "yes", "d": drug.id}),
                               ("ไม่ใช่", {"a": "no"})),
            user_id=user_id,
        )

    # ---------- postback ----------
    async def on_postback(self, event: PostbackEvent) -> None:
        user_id = event.source.user_id if event.source else "anon"
        data = {k: v[0] for k, v in parse_qs(event.postback.data).items()}
        sess = self.sessions.get(user_id)

        if data.get("a") == "yes":
            drug = self.drugs.get(data.get("d", ""))
            if drug is None:
                await self._reply(event, PHOTO_TIPS, user_id=user_id)
                return
            sess.current_drug_id, sess.pending_drug_id = drug.id, None
            reply_items = [self.drugs.format_reply(drug)]
            
            # ส่งเสียงวิธีใช้ยาด้วย
            if audio_info := self.drugs.get_audio_info(drug, self.base_url):
                audio_url, duration_ms = audio_info
                reply_items.append(AudioMessage(original_content_url=audio_url, duration=duration_ms))
            
            reply_items.append("🔊 ส่งไฟล์เสียงแนะนำวิธีใช้ให้แล้วค่ะ กดฟังได้เลยนะคะ\nมีอะไรอยากถามเพิ่มเกี่ยวกับยานี้ พิมพ์ถามได้เลยค่ะ")
            await self._reply(
                event,
                *reply_items,
                quick_reply=_quick(
                    ("🔊 ฟังเสียงอีกครั้ง", {"a": "audio", "d": drug.id}),
                    ("💊 ยาที่รู้จัก", {"a": "list"}),
                ),
                user_id=user_id,
            )
        elif data.get("a") == "audio":
            drug = self.drugs.get(data.get("d", ""))
            if drug and (audio_info := self.drugs.get_audio_info(drug, self.base_url)):
                audio_url, duration_ms = audio_info
                await self._reply(
                    event,
                    f"🔊 เสียงแนะนำวิธีใช้: {drug.name_th}",
                    AudioMessage(original_content_url=audio_url, duration=duration_ms),
                    user_id=user_id,
                )
        elif data.get("a") == "list":
            names = "\n".join(f"• {d.name_th}" for d in self.drugs.all())
            await self._reply(
                event,
                f"ตอนนี้รู้จักยาเหล่านี้ค่ะ\n{names}\n\nส่งรูปยามาให้ดู หรือพิมพ์ถามได้เลยค่ะ",
                user_id=user_id,
            )
        elif data.get("a") == "no":
            sess.pending_drug_id = None
            await self._reply(event,
                              "ขออภัยค่ะ ลองถ่ายรูปใหม่ให้เห็นชื่อยาชัด ๆ "
                              "หรือพิมพ์ชื่อยามาถามได้เลยค่ะ",
                              user_id=user_id)
