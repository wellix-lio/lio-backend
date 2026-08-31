import hashlib
import urllib.request
from .config import LIO_DB_PATH
import aiosqlite

async def add_watch(user_id: str, name: str, url: str, rule: str, frequency_minutes: int = 360):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO watch_items(user_id,name,url,rule,frequency_minutes)
               VALUES(?,?,?,?,?)""",
            (user_id, name, url, rule, frequency_minutes),
        )
        await db.commit()
        return cur.lastrowid

async def list_watches(user_id: str):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """SELECT id,name,url,rule,frequency_minutes,enabled,last_checked_at
               FROM watch_items WHERE user_id=? ORDER BY id DESC""",
            (user_id,),
        )
        rows = await cur.fetchall()
    return [
        {"id": r[0], "name": r[1], "url": r[2], "rule": r[3],
         "frequency_minutes": r[4], "enabled": bool(r[5]), "last_checked_at": r[6]}
        for r in rows
    ]

def fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
