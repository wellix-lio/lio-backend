from io import BytesIO
from openai import AsyncOpenAI
from .config import OPENAI_API_KEY

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TTS_INSTRUCTIONS = """
Speak the supplied text in its own language without translating or paraphrasing it.
For Arabic, sound like a natural educated Arabic speaker rather than a formal announcer or synthetic narrator. Use relaxed conversational pacing for everyday dialogue, natural pauses, sentence melody, and emphasis. Preserve colloquial wording when the text is colloquial. When the text is formal Modern Standard Arabic, pronounce it clearly and fluently without exaggerated classical delivery or artificial case endings that are not written.
For German, use natural Austrian Standard German pronunciation (Deutsch in Österreich / de-AT), not a strong regional dialect unless the text itself clearly calls for one.
For English, use natural, conversational English pronunciation.
Keep names, company names, numbers, units, and technical terms accurate.
Avoid a robotic cadence, excessive solemnity, and identical rhythm across sentences. Sound warm, calm, confident, and human.
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
