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
    preferred_language TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS smart_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    value TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, category, memory_key)
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
        return {"display_name": None, "preferred_language": None}
    return {"display_name": row[0], "preferred_language": row[1]}

async def set_display_name(user_id: str, display_name: str):
    name = display_name.strip()
    if not name:
        return
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_profile(user_id, display_name, preferred_language)
            VALUES (?, ?, NULL)
            ON CONFLICT(user_id) DO UPDATE SET display_name=excluded.display_name
            """,
            (user_id, name),
        )
        await db.commit()

async def set_preferred_language(user_id: str, language: str):
    value = language.strip()
    if not value:
        return
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_profile(user_id, preferred_language)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET preferred_language=excluded.preferred_language
            """,
            (user_id, value),
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

async def upsert_smart_memory(
    user_id: str,
    category: str,
    memory_key: str,
    value: str,
    importance: int = 5,
):
    value = value.strip()
    if not value:
        return
    importance = max(1, min(int(importance), 10))
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO smart_memories(user_id, category, memory_key, value, importance)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, category, memory_key)
            DO UPDATE SET
                value=excluded.value,
                importance=excluded.importance,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, category, memory_key, value, importance),
        )
        await db.commit()

async def get_smart_memories(user_id: str, limit: int = 30):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """
            SELECT category, memory_key, value, importance
            FROM smart_memories
            WHERE user_id=?
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = await cur.fetchall()
    return [
        {
            "category": r[0],
            "key": r[1],
            "value": r[2],
            "importance": r[3],
        }
        for r in rows
    ]

async def delete_smart_memory(user_id: str, category: str, memory_key: str) -> bool:
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """
            DELETE FROM smart_memories
            WHERE user_id=? AND category=? AND memory_key=?
            """,
            (user_id, category, memory_key),
        )
        await db.commit()
        return cur.rowcount > 0

async def clear_display_name(user_id: str) -> bool:
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """
            UPDATE user_profile
            SET display_name=NULL
            WHERE user_id=? AND display_name IS NOT NULL
            """,
            (user_id,),
        )
        await db.commit()
        return cur.rowcount > 0

async def clear_preferred_language(user_id: str) -> bool:
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """
            UPDATE user_profile
            SET preferred_language=NULL
            WHERE user_id=? AND preferred_language IS NOT NULL
            """,
            (user_id,),
        )
        await db.commit()
        return cur.rowcount > 0

def _normalize_memory_text(text: str) -> str:
    return " ".join((text or "").split()).strip().casefold()

async def delete_saved_memory(user_id: str, content: str) -> bool:
    target = _normalize_memory_text(content)
    if not target:
        return False

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            "SELECT id, content FROM memories WHERE user_id=?",
            (user_id,),
        )
        rows = await cur.fetchall()
        ids = [
            row_id
            for row_id, saved_content in rows
            if _normalize_memory_text(saved_content) == target
        ]
        if not ids:
            return False

        await db.executemany(
            "DELETE FROM memories WHERE id=?",
            [(row_id,) for row_id in ids],
        )
        await db.commit()
        return True

