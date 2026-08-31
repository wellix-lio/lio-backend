import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .config import OPENAI_API_KEY, LIO_ALLOWED_ORIGINS
from .memory import (
    init_db,
    add_message,
    recent_messages,
    get_profile,
    set_display_name,
    set_preferred_language,
    add_memory,
    saved_memories,
    upsert_smart_memory,
    get_smart_memories,
    delete_smart_memory,
    clear_display_name,
    clear_preferred_language,
    delete_saved_memory,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="Lio API", version="1.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=LIO_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str = Field(default="owner")
    message: str = Field(min_length=1, max_length=12000)

class ChatResponse(BaseModel):
    reply: str
    mode: str

def _clean_value(value: str) -> str:
    return value.strip().strip('"\'“”‘’ ').rstrip(".،,!?؟")[:240]

def _first_match(message: str, patterns):
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            value = _clean_value(match.group(1))
            if value:
                return value
    return None

def _extract_name(message: str):
    return _first_match(
        message,
        [
            r"(?:^|\s)(?:أنا\s+)?اسمي\s+([^\n،,.!?؟]{1,80})",
            r"\bmy\s+name\s+is\s+([^\n,.!?]{1,80})",
            r"\b(?:ich\s+hei(?:ß|ss)e|mein\s+name\s+ist)\s+([^\n,.!?]{1,80})",
        ],
    )

def _extract_explicit_memory(message: str):
    patterns = [
        r"(?:تذكر|تذكّر)\s+(?:أن|ان)\s+(.+)",
        r"\bremember\s+that\s+(.+)",
        r"\b(?:merk|merke)\s+dir,?\s+dass\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = _clean_value(match.group(1))
            if value:
                return value[:1000]
    return None

def _extract_structured_facts(message: str):
    facts = []

    company = _first_match(
        message,
        [
            r"(?:اسم\s+)?شركتي\s+(?:هي|اسمها)?\s*([^\n،,.!?؟]{2,120})",
            r"\bmy\s+company(?:'s\s+name)?\s+is\s+([^\n,.!?]{2,120})",
            r"\bmeine\s+firma\s+(?:heißt|heisst|ist)\s+([^\n,.!?]{2,120})",
        ],
    )
    if company:
        facts.append(("business", "company", company, 9))

    role = _first_match(
        message,
        [
            r"(?:أعمل|اعمل)\s+(?:كـ?|بوظيفة)\s*([^\n،,.!?؟]{2,120})",
            r"(?:وظيفتي|عملي)\s+(?:هي|هو)?\s*([^\n،,.!?؟]{2,120})",
            r"\bi\s+work\s+as\s+(?:an?\s+)?([^\n,.!?]{2,120})",
            r"\bich\s+arbeite\s+als\s+([^\n,.!?]{2,120})",
        ],
    )
    if role:
        facts.append(("work", "role", role, 7))

    project = _first_match(
        message,
        [
            r"(?:مشروعي(?:\s+الحالي)?|المشروع\s+الذي\s+أعمل\s+عليه)\s+(?:اسمه|هو)?\s*([^\n،,.!?؟]{2,160})",
            r"\bmy\s+(?:current\s+)?project\s+(?:is\s+called|is)\s+([^\n,.!?]{2,160})",
            r"\bmein\s+(?:aktuelles\s+)?projekt\s+(?:heißt|heisst|ist)\s+([^\n,.!?]{2,160})",
        ],
    )
    if project:
        facts.append(("project", "current_project", project, 8))

    language = _first_match(
        message,
        [
            r"(?:لغتي\s+المفضلة|أفضل\s+أن\s+تتحدث\s+معي\s+ب(?:ال)?لغة)\s+([^\n،,.!?؟]{2,80})",
            r"\bmy\s+preferred\s+language\s+is\s+([^\n,.!?]{2,80})",
            r"\bmeine\s+bevorzugte\s+sprache\s+ist\s+([^\n,.!?]{2,80})",
        ],
    )
    if language:
        facts.append(("preference", "preferred_language", language, 8))

    return facts


def _extract_memory_control(message: str):
    explicit_forget_patterns = [
        r"(?:انسَ|انس|انسى|احذف|امسح)\s+(?:من\s+ذاكرتك\s+)?(?:أن|ان)\s+(.+)",
        r"\bforget\s+that\s+(.+)",
        r"\b(?:vergiss|lösche)\s+(?:bitte\s+)?dass\s+(.+)",
    ]
    for pattern in explicit_forget_patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = _clean_value(match.group(1))
            if value:
                return ("forget_explicit", None, value)

    forget_targets = [
        ("name", [
            r"(?:انسَ|انس|انسى|احذف|امسح)\s+(?:من\s+ذاكرتك\s+)?(?:اسمي|اسمِي|اسم\s+العرض)(?:\s+المحفوظ)?\s*$",
            r"\b(?:forget|delete|remove)\s+my\s+(?:saved\s+)?name\s*$",
            r"\b(?:vergiss|lösche)\s+(?:meinen\s+)?namen\s*$",
        ]),
        ("company", [
            r"(?:انسَ|انس|انسى|احذف|امسح)\s+(?:من\s+ذاكرتك\s+)?(?:اسم\s+)?شركتي\s*$",
            r"\b(?:forget|delete|remove)\s+my\s+company(?:\s+name)?\s*$",
            r"\b(?:vergiss|lösche)\s+(?:den\s+namen\s+)?meiner\s+firma\s*$",
        ]),
        ("role", [
            r"(?:انسَ|انس|انسى|احذف|امسح)\s+(?:من\s+ذاكرتك\s+)?(?:عملي|وظيفتي)\s*$",
            r"\b(?:forget|delete|remove)\s+my\s+(?:job|role|work)\s*$",
            r"\b(?:vergiss|lösche)\s+(?:meinen\s+)?(?:beruf|job|rolle)\s*$",
        ]),
        ("project", [
            r"(?:انسَ|انس|انسى|احذف|امسح)\s+(?:من\s+ذاكرتك\s+)?(?:مشروعي|مشروعي\s+الحالي)\s*$",
            r"\b(?:forget|delete|remove)\s+my\s+(?:current\s+)?project\s*$",
            r"\b(?:vergiss|lösche)\s+(?:mein\s+)?(?:aktuelles\s+)?projekt\s*$",
        ]),
        ("language", [
            r"(?:انسَ|انس|انسى|احذف|امسح)\s+(?:من\s+ذاكرتك\s+)?(?:لغتي\s+المفضلة|تفضيل\s+اللغة)\s*$",
            r"\b(?:forget|delete|remove)\s+my\s+preferred\s+language\s*$",
            r"\b(?:vergiss|lösche)\s+(?:meine\s+)?bevorzugte\s+sprache\s*$",
        ]),
    ]
    for target, patterns in forget_targets:
        for pattern in patterns:
            if re.search(pattern, message.strip(), flags=re.IGNORECASE):
                return ("forget", target, None)

    correction_targets = [
        ("name", [
            r"(?:صحح|صحّح|غير|غيّر|عدل|عدّل|حدّث|حدث)\s+(?:اسمي|اسم\s+العرض)\s+(?:إلى|الى|ليصبح|إلى\s+أن\s+يصبح)\s+(.+)",
            r"\b(?:change|correct|update)\s+my\s+(?:display\s+)?name\s+to\s+(.+)",
            r"\b(?:ändere|korrigiere|aktualisiere)\s+(?:meinen\s+)?namen\s+(?:zu|auf)\s+(.+)",
        ]),
        ("company", [
            r"(?:صحح|صحّح|غير|غيّر|عدل|عدّل|حدّث|حدث)\s+(?:اسم\s+)?شركتي\s+(?:إلى|الى|ليصبح|إلى\s+أن\s+يصبح)\s+(.+)",
            r"\b(?:change|correct|update)\s+my\s+company(?:\s+name)?\s+to\s+(.+)",
            r"\b(?:ändere|korrigiere|aktualisiere)\s+(?:den\s+namen\s+)?meiner\s+firma\s+(?:zu|auf)\s+(.+)",
        ]),
        ("role", [
            r"(?:صحح|صحّح|غير|غيّر|عدل|عدّل|حدّث|حدث)\s+(?:عملي|وظيفتي)\s+(?:إلى|الى|ليصبح|إلى\s+أن\s+يصبح)\s+(.+)",
            r"\b(?:change|correct|update)\s+my\s+(?:job|role|work)\s+to\s+(.+)",
            r"\b(?:ändere|korrigiere|aktualisiere)\s+(?:meinen\s+)?(?:beruf|job|rolle)\s+(?:zu|auf)\s+(.+)",
        ]),
        ("project", [
            r"(?:صحح|صحّح|غير|غيّر|عدل|عدّل|حدّث|حدث)\s+(?:مشروعي|مشروعي\s+الحالي)\s+(?:إلى|الى|ليصبح|إلى\s+أن\s+يصبح)\s+(.+)",
            r"\b(?:change|correct|update)\s+my\s+(?:current\s+)?project\s+to\s+(.+)",
            r"\b(?:ändere|korrigiere|aktualisiere)\s+(?:mein\s+)?(?:aktuelles\s+)?projekt\s+(?:zu|auf)\s+(.+)",
        ]),
        ("language", [
            r"(?:صحح|صحّح|غير|غيّر|عدل|عدّل|حدّث|حدث)\s+(?:لغتي\s+المفضلة|تفضيل\s+اللغة)\s+(?:إلى|الى|ليصبح|إلى\s+أن\s+يصبح)\s+(.+)",
            r"\b(?:change|correct|update)\s+my\s+preferred\s+language\s+to\s+(.+)",
            r"\b(?:ändere|korrigiere|aktualisiere)\s+(?:meine\s+)?bevorzugte\s+sprache\s+(?:zu|auf)\s+(.+)",
        ]),
    ]
    for target, patterns in correction_targets:
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
            if match:
                value = _clean_value(match.group(1))
                if value:
                    return ("correct", target, value)

    return None

async def _apply_memory_control(user_id: str, control):
    if not control:
        return None

    action, target, value = control

    if action == "forget_explicit":
        deleted = await delete_saved_memory(user_id, value)
        if deleted:
            return f"Memory action completed: removed exact saved memory: {value}"
        return f"Memory action requested, but no exact saved memory matched: {value}"

    if action == "forget":
        if target == "name":
            deleted = await clear_display_name(user_id)
        elif target == "company":
            deleted = await delete_smart_memory(user_id, "business", "company")
        elif target == "role":
            deleted = await delete_smart_memory(user_id, "work", "role")
        elif target == "project":
            deleted = await delete_smart_memory(user_id, "project", "current_project")
        elif target == "language":
            a = await delete_smart_memory(user_id, "preference", "preferred_language")
            b = await clear_preferred_language(user_id)
            deleted = a or b
        else:
            deleted = False

        return (
            f"Memory action completed: removed saved {target}."
            if deleted
            else f"Memory action requested, but no saved {target} was found."
        )

    if action == "correct":
        if target == "name":
            await set_display_name(user_id, value)
        elif target == "company":
            await upsert_smart_memory(user_id, "business", "company", value, 9)
        elif target == "role":
            await upsert_smart_memory(user_id, "work", "role", value, 7)
        elif target == "project":
            await upsert_smart_memory(user_id, "project", "current_project", value, 8)
        elif target == "language":
            await upsert_smart_memory(
                user_id, "preference", "preferred_language", value, 8
            )
            await set_preferred_language(user_id, value)

        return f"Memory action completed: corrected saved {target} to: {value}"

    return None

async def _capture_user_memory(user_id: str, message: str):
    control = _extract_memory_control(message)
    if control:
        return await _apply_memory_control(user_id, control)

    name = _extract_name(message)
    if name:
        await set_display_name(user_id, name)

    explicit = _extract_explicit_memory(message)
    if explicit:
        await add_memory(user_id, explicit)

    for category, key, value, importance in _extract_structured_facts(message):
        await upsert_smart_memory(user_id, category, key, value, importance)
        if key == "preferred_language":
            await set_preferred_language(user_id, value)

    return None

async def _persistent_context(user_id: str) -> str:
    profile = await get_profile(user_id)
    memories = await saved_memories(user_id, 20)
    smart = await get_smart_memories(user_id, 30)

    lines = []
    if profile.get("display_name"):
        lines.append(f"User display name: {profile['display_name']}")
    if profile.get("preferred_language"):
        lines.append(f"Preferred language: {profile['preferred_language']}")

    if smart:
        lines.append("Structured user facts:")
        for item in smart:
            lines.append(
                f"- [{item['category']}] {item['key']}: {item['value']}"
            )

    if memories:
        lines.append("Explicit saved memories:")
        lines.extend(f"- {item}" for item in memories)

    if not lines:
        return ""

    return (
        "Persistent user context. Treat these as saved user-provided facts and use "
        "them only when relevant. Do not invent missing details.\n"
        + "\n".join(lines)
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "name": "Lio",
        "ai_connected": bool(OPENAI_API_KEY),
        "languages": ["ar", "de", "en"],
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    memory_action = await _capture_user_memory(req.user_id, req.message)
    await add_message(req.user_id, "user", req.message)

    history = await recent_messages(req.user_id, 10)
    recent_context = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[:-1]
    )
    persistent_context = await _persistent_context(req.user_id)
    memory_action_context = (
        f"Internal memory status for this turn: {memory_action}"
        if memory_action
        else ""
    )
    context_text = "\n\n".join(
        part
        for part in [persistent_context, memory_action_context, recent_context]
        if part
    )

    if not OPENAI_API_KEY:
        reply = (
            "Lio جاهز من ناحية البنية، لكن اتصال الذكاء الاصطناعي غير مفعّل بعد. "
            "عند إضافة OPENAI_API_KEY إلى خادم Lio سأتمكن من تنفيذ هذه المهمة."
        )
        await add_message(req.user_id, "assistant", reply)
        return ChatResponse(reply=reply, mode="setup")

    try:
        from .agents import run_lio
        reply = await run_lio(req.message, context_text)
        await add_message(req.user_id, "assistant", reply)
        return ChatResponse(reply=reply, mode="live")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lio agent error: {exc}")

@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API is not configured")
    from .voice import transcribe_audio
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    try:
        text = await transcribe_audio(data, file.filename or "speech.m4a")
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")

class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

@app.post("/voice/speak")
async def voice_speak(req: SpeechRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API is not configured")
    from .voice import synthesize_speech
    try:
        audio = await synthesize_speech(req.text)
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech error: {exc}")

class WatchRequest(BaseModel):
    user_id: str = "owner"
    name: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=8, max_length=2000)
    rule: str = Field(min_length=1, max_length=2000)
    frequency_minutes: int = Field(default=360, ge=60, le=43200)

@app.post("/watch")
async def create_watch(req: WatchRequest):
    from .watch import add_watch
    if not (req.url.startswith("https://") or req.url.startswith("http://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    watch_id = await add_watch(
        req.user_id, req.name, req.url, req.rule, req.frequency_minutes
    )
    return {"id": watch_id, "status": "created"}

@app.get("/watch/{user_id}")
async def get_watches(user_id: str):
    from .watch import list_watches
    return {"items": await list_watches(user_id)}

class TaskRequest(BaseModel):
    user_id: str = "owner"
    title: str = Field(min_length=1, max_length=200)
    instruction: str = Field(min_length=1, max_length=8000)
    requires_approval: bool = False

@app.post("/tasks")
async def add_task(req: TaskRequest):
    from .tasks import create_task
    task_id = await create_task(
        req.user_id, req.title, req.instruction, req.requires_approval
    )
    return {"id": task_id, "status": "queued"}

@app.get("/tasks/{user_id}")
async def tasks(user_id: str):
    from .tasks import list_tasks
    return {"items": await list_tasks(user_id)}
