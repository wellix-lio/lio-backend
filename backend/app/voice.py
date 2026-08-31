from io import BytesIO
from openai import AsyncOpenAI
from .config import OPENAI_API_KEY

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
        voice="alloy",
        input=text,
        format="mp3",
    )
    return response.read()
