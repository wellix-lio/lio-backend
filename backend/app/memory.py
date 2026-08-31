import aiosqlite
from .config import LIO_DB_PATH

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    display_name TEXT,
    preferred_language TEXT DEFAULT 'ar',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watch_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    rule TEXT NOT NULL,
    frequency_minutes INTEGER NOT NULL DEFAULT 360,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_checked_at DATETIME,
    last_fingerprint TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

async def init_db():
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.executescript(CREATE_SQL)
        await db.commit()

async def add_message(user_id: str, role: str, content: str):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(
            "INSERT INTO conversations(user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        await db.commit()

async def recent_messages(user_id: str, limit: int = 12):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            "SELECT role, content FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
    return list(reversed([{"role": r[0], "content": r[1]} for r in rows]))

async def get_profile(user_id: str):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            "SELECT display_name, preferred_language FROM user_profile WHERE user_id=?",
            (user_id,),
        )
        row = await cur.fetchone()
    if not row:
        return {"display_name": None, "preferred_language": "ar"}
    return {"display_name": row[0], "preferred_language": row[1] or "ar"}

async def set_display_name(user_id: str, display_name: str):
    name = display_name.strip()
    if not name:
        return
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_profile(user_id, display_name)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET display_name=excluded.display_name
            """,
            (user_id, name),
        )
        await db.commit()

async def add_memory(user_id: str, content: str):
    fact = content.strip()
    if not fact:
        return
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            "SELECT 1 FROM memories WHERE user_id=? AND content=? LIMIT 1",
            (user_id, fact),
        )
        if await cur.fetchone() is None:
            await db.execute(
                "INSERT INTO memories(user_id, content) VALUES (?, ?)",
                (user_id, fact),
            )
            await db.commit()

async def saved_memories(user_id: str, limit: int = 20):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            "SELECT content FROM memories WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cur.fetchall()
    return list(reversed([r[0] for r in rows]))
