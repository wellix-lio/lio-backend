import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.6-sol")
LIO_DB_PATH = os.getenv("LIO_DB_PATH", "lio.db")


LIO_ENV = os.getenv("LIO_ENV", "development")
LIO_ALLOWED_ORIGINS = [
    x.strip() for x in os.getenv("LIO_ALLOWED_ORIGINS", "*").split(",") if x.strip()
]
