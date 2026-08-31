from io import BytesIO
from openai import AsyncOpenAI
from .config import OPENAI_API_KEY

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TTS_INSTRUCTIONS = """
Speak the supplied text exactly in its own language without translating it.
If the text is Arabic, use a warm, natural conversational Arabic delivery and preserve any colloquial wording that is written.
If the text is German, use natural Austrian Standard German pronunciation (Deutsch in Österreich / de-AT), not a strong regional dialect unless the text itself clearly calls for one.
If the text is English, use natural clear English pronunciation.
Keep names, company names, numbers, and technical terms accurate.
Sound friendly, calm, and professional, with natural pacing.
"""

async def transcribe_audio(data: bytes, filename: str = "speech.m4a") -> str:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    f = BytesIO(data)
    f.name = filename
    result = await client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=f,
    )
    return result.text

async def synthesize_speech(text: str) -> bytes:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    response = await client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="cedar",
        input=text,
        instructions=TTS_INSTRUCTIONS,
        response_format="mp3",
    )
    return response.read()
