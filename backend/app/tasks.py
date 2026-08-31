import json
import aiosqlite
from .config import LIO_DB_PATH

TASK_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    instruction TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    requires_approval INTEGER NOT NULL DEFAULT 0,
    result TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

async def init_tasks():
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.executescript(TASK_SQL)
        await db.commit()

async def create_task(user_id, title, instruction, requires_approval=False):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(TASK_SQL)
        cur = await db.execute(
            """INSERT INTO tasks(user_id,title,instruction,requires_approval)
               VALUES(?,?,?,?)""",
            (user_id, title, instruction, 1 if requires_approval else 0),
        )
        await db.commit()
        return cur.lastrowid

async def list_tasks(user_id):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(TASK_SQL)
        cur = await db.execute(
            """SELECT id,title,instruction,status,requires_approval,result,created_at,updated_at
               FROM tasks WHERE user_id=? ORDER BY id DESC""",
            (user_id,),
        )
        rows = await cur.fetchall()
    return [
        {"id":r[0],"title":r[1],"instruction":r[2],"status":r[3],
         "requires_approval":bool(r[4]),"result":r[5],
         "created_at":r[6],"updated_at":r[7]} for r in rows
    ]
