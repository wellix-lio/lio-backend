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

CREATE TABLE IF NOT EXISTS commercial_suppliers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    country TEXT,
    city TEXT,
    website TEXT,
    contact_name TEXT,
    email TEXT,
    phone TEXT,
    supplier_type TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS commercial_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    supplier_id INTEGER,
    product_name TEXT NOT NULL,
    category TEXT,
    size TEXT,
    thickness_mm REAL,
    finish TEXT,
    color TEXT,
    model TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(supplier_id) REFERENCES commercial_suppliers(id)
);

CREATE TABLE IF NOT EXISTS commercial_offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    supplier_id INTEGER,
    product_id INTEGER,
    price REAL,
    currency TEXT,
    price_unit TEXT,
    quantity REAL,
    moq REAL,
    incoterm TEXT,
    payment_terms TEXT,
    quote_date TEXT,
    valid_until TEXT,
    lead_time_days INTEGER,
    status TEXT NOT NULL DEFAULT 'received',
    source TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(supplier_id) REFERENCES commercial_suppliers(id),
    FOREIGN KEY(product_id) REFERENCES commercial_products(id)
);

CREATE INDEX IF NOT EXISTS idx_commercial_suppliers_user
ON commercial_suppliers(user_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_commercial_products_user
ON commercial_products(user_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_commercial_offers_user
ON commercial_offers(user_id, created_at);

CREATE TABLE IF NOT EXISTS commercial_supplier_languages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    supplier_id INTEGER NOT NULL,
    language TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, supplier_id, language),
    FOREIGN KEY(supplier_id) REFERENCES commercial_suppliers(id)
);

CREATE INDEX IF NOT EXISTS idx_commercial_supplier_languages_user
ON commercial_supplier_languages(user_id, supplier_id);
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

async def upsert_commercial_supplier(
    user_id: str,
    name: str,
    country: str | None = None,
    city: str | None = None,
    website: str | None = None,
    contact_name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    supplier_type: str | None = None,
    status: str = "active",
    notes: str | None = None,
) -> int:
    name = (name or "").strip()
    if not name:
        raise ValueError("Supplier name is required")

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO commercial_suppliers(
                user_id, name, country, city, website, contact_name,
                email, phone, supplier_type, status, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, name)
            DO UPDATE SET
                country=COALESCE(excluded.country, commercial_suppliers.country),
                city=COALESCE(excluded.city, commercial_suppliers.city),
                website=COALESCE(excluded.website, commercial_suppliers.website),
                contact_name=COALESCE(excluded.contact_name, commercial_suppliers.contact_name),
                email=COALESCE(excluded.email, commercial_suppliers.email),
                phone=COALESCE(excluded.phone, commercial_suppliers.phone),
                supplier_type=COALESCE(excluded.supplier_type, commercial_suppliers.supplier_type),
                status=COALESCE(excluded.status, commercial_suppliers.status),
                notes=COALESCE(excluded.notes, commercial_suppliers.notes),
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                user_id, name, country, city, website, contact_name,
                email, phone, supplier_type, status, notes
            ),
        )
        cur = await db.execute(
            "SELECT id FROM commercial_suppliers WHERE user_id=? AND name=?",
            (user_id, name),
        )
        row = await cur.fetchone()
        await db.commit()
    return int(row[0])


async def add_commercial_product(
    user_id: str,
    product_name: str,
    supplier_id: int | None = None,
    category: str | None = None,
    size: str | None = None,
    thickness_mm: float | None = None,
    finish: str | None = None,
    color: str | None = None,
    model: str | None = None,
    notes: str | None = None,
) -> int:
    product_name = (product_name or "").strip()
    if not product_name:
        raise ValueError("Product name is required")

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO commercial_products(
                user_id, supplier_id, product_name, category, size,
                thickness_mm, finish, color, model, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, supplier_id, product_name, category, size,
                thickness_mm, finish, color, model, notes
            ),
        )
        await db.commit()
        return int(cur.lastrowid)


async def add_commercial_offer(
    user_id: str,
    supplier_id: int | None = None,
    product_id: int | None = None,
    price: float | None = None,
    currency: str | None = None,
    price_unit: str | None = None,
    quantity: float | None = None,
    moq: float | None = None,
    incoterm: str | None = None,
    payment_terms: str | None = None,
    quote_date: str | None = None,
    valid_until: str | None = None,
    lead_time_days: int | None = None,
    status: str = "received",
    source: str | None = None,
    notes: str | None = None,
) -> int:
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO commercial_offers(
                user_id, supplier_id, product_id, price, currency, price_unit,
                quantity, moq, incoterm, payment_terms, quote_date, valid_until,
                lead_time_days, status, source, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, supplier_id, product_id, price, currency, price_unit,
                quantity, moq, incoterm, payment_terms, quote_date, valid_until,
                lead_time_days, status, source, notes
            ),
        )
        await db.commit()
        return int(cur.lastrowid)


async def get_commercial_memory(
    user_id: str,
    supplier_limit: int = 10,
    offer_limit: int = 20,
):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        suppliers_cur = await db.execute(
            """
            SELECT id, name, country, city, website, contact_name, email, phone,
                   supplier_type, status, notes, updated_at
            FROM commercial_suppliers
            WHERE user_id=?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, supplier_limit),
        )
        supplier_rows = await suppliers_cur.fetchall()

        offers_cur = await db.execute(
            """
            SELECT
                o.id,
                s.name,
                p.product_name,
                p.size,
                p.thickness_mm,
                o.price,
                o.currency,
                o.price_unit,
                o.quantity,
                o.moq,
                o.incoterm,
                o.payment_terms,
                o.quote_date,
                o.valid_until,
                o.lead_time_days,
                o.status,
                o.source,
                o.notes,
                o.created_at
            FROM commercial_offers o
            LEFT JOIN commercial_suppliers s ON s.id=o.supplier_id
            LEFT JOIN commercial_products p ON p.id=o.product_id
            WHERE o.user_id=?
            ORDER BY
                CASE WHEN o.quote_date IS NULL OR o.quote_date='' THEN 1 ELSE 0 END,
                o.quote_date DESC,
                o.id DESC
            LIMIT ?
            """,
            (user_id, offer_limit),
        )
        offer_rows = await offers_cur.fetchall()

    suppliers = [
        {
            "id": r[0],
            "name": r[1],
            "country": r[2],
            "city": r[3],
            "website": r[4],
            "contact_name": r[5],
            "email": r[6],
            "phone": r[7],
            "supplier_type": r[8],
            "status": r[9],
            "notes": r[10],
            "updated_at": r[11],
        }
        for r in supplier_rows
    ]

    offers = [
        {
            "id": r[0],
            "supplier": r[1],
            "product": r[2],
            "size": r[3],
            "thickness_mm": r[4],
            "price": r[5],
            "currency": r[6],
            "price_unit": r[7],
            "quantity": r[8],
            "moq": r[9],
            "incoterm": r[10],
            "payment_terms": r[11],
            "quote_date": r[12],
            "valid_until": r[13],
            "lead_time_days": r[14],
            "status": r[15],
            "source": r[16],
            "notes": r[17],
            "created_at": r[18],
        }
        for r in offer_rows
    ]

    return {"suppliers": suppliers, "offers": offers}

async def add_commercial_supplier_language(
    user_id: str,
    supplier_id: int,
    language: str,
):
    language = (language or "").strip()
    if not language:
        return
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO commercial_supplier_languages(
                user_id, supplier_id, language
            )
            VALUES (?, ?, ?)
            """,
            (user_id, supplier_id, language),
        )
        await db.commit()


async def get_commercial_supplier_languages(user_id: str, supplier_ids: list[int]):
    if not supplier_ids:
        return {}
    placeholders = ",".join("?" for _ in supplier_ids)
    query = f"""
        SELECT supplier_id, language
        FROM commercial_supplier_languages
        WHERE user_id=? AND supplier_id IN ({placeholders})
        ORDER BY supplier_id, language
    """
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(query, (user_id, *supplier_ids))
        rows = await cur.fetchall()

    result = {}
    for supplier_id, language in rows:
        result.setdefault(int(supplier_id), []).append(language)
    return result

async def find_commercial_suppliers(
    user_id: str,
    name_query: str,
    limit: int = 10,
):
    name_query = (name_query or "").strip()
    if not name_query:
        return []

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cur = await db.execute(
            """
            SELECT id, name, country, city, website, contact_name, email, phone,
                   supplier_type, status, notes, updated_at
            FROM commercial_suppliers
            WHERE user_id=? AND name=?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, name_query, limit),
        )
        rows = await cur.fetchall()

        if not rows:
            cur = await db.execute(
                """
                SELECT id, name, country, city, website, contact_name, email, phone,
                       supplier_type, status, notes, updated_at
                FROM commercial_suppliers
                WHERE user_id=? AND name LIKE ?
                ORDER BY
                    CASE
                        WHEN name LIKE ? THEN 0
                        WHEN name LIKE ? THEN 1
                        ELSE 2
                    END,
                    updated_at DESC,
                    id DESC
                LIMIT ?
                """,
                (
                    user_id,
                    f"%{name_query}%",
                    f"{name_query}%",
                    f"%{name_query}",
                    limit,
                ),
            )
            rows = await cur.fetchall()

    return [dict(row) for row in rows]


async def get_commercial_offer_by_id(user_id: str, offer_id: int):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT
                o.id,
                o.supplier_id,
                o.product_id,
                s.name AS supplier,
                p.product_name AS product,
                p.size,
                p.thickness_mm,
                o.price,
                o.currency,
                o.price_unit,
                o.quantity,
                o.moq,
                o.incoterm,
                o.payment_terms,
                o.quote_date,
                o.valid_until,
                o.lead_time_days,
                o.status,
                o.source,
                o.notes,
                o.created_at
            FROM commercial_offers o
            LEFT JOIN commercial_suppliers s ON s.id=o.supplier_id
            LEFT JOIN commercial_products p ON p.id=o.product_id
            WHERE o.user_id=? AND o.id=?
            """,
            (user_id, int(offer_id)),
        )
        row = await cur.fetchone()
    return dict(row) if row else None


async def get_latest_commercial_offer(user_id: str, supplier_id: int):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT
                o.id,
                o.supplier_id,
                o.product_id,
                s.name AS supplier,
                p.product_name AS product,
                p.size,
                p.thickness_mm,
                o.price,
                o.currency,
                o.price_unit,
                o.quantity,
                o.moq,
                o.incoterm,
                o.payment_terms,
                o.quote_date,
                o.valid_until,
                o.lead_time_days,
                o.status,
                o.source,
                o.notes,
                o.created_at
            FROM commercial_offers o
            LEFT JOIN commercial_suppliers s ON s.id=o.supplier_id
            LEFT JOIN commercial_products p ON p.id=o.product_id
            WHERE o.user_id=? AND o.supplier_id=?
            ORDER BY
                CASE WHEN o.quote_date IS NULL OR o.quote_date='' THEN 1 ELSE 0 END,
                o.quote_date DESC,
                o.id DESC
            LIMIT 1
            """,
            (user_id, int(supplier_id)),
        )
        row = await cur.fetchone()
    return dict(row) if row else None


async def update_commercial_supplier(
    user_id: str,
    supplier_id: int,
    **changes,
):
    allowed = {
        "name",
        "country",
        "city",
        "website",
        "contact_name",
        "email",
        "phone",
        "supplier_type",
        "status",
        "notes",
    }
    clean = {
        key: value.strip() if isinstance(value, str) else value
        for key, value in changes.items()
        if key in allowed and value is not None
    }
    if not clean:
        return False

    assignments = ", ".join(f"{key}=?" for key in clean)
    values = list(clean.values())

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            f"""
            UPDATE commercial_suppliers
            SET {assignments}, updated_at=CURRENT_TIMESTAMP
            WHERE user_id=? AND id=?
            """,
            (*values, user_id, int(supplier_id)),
        )
        changed = cur.rowcount > 0
        if changed:
            await db.execute(
                """
                INSERT INTO audit_log(user_id, event_type, detail)
                VALUES (?, ?, ?)
                """,
                (
                    user_id,
                    "commercial_supplier_updated",
                    f"supplier_id={int(supplier_id)}; changes={clean}",
                ),
            )
        await db.commit()
    return changed


async def update_commercial_product(
    user_id: str,
    product_id: int,
    **changes,
):
    allowed = {
        "product_name",
        "category",
        "size",
        "thickness_mm",
        "finish",
        "color",
        "model",
        "notes",
    }
    clean = {
        key: value.strip() if isinstance(value, str) else value
        for key, value in changes.items()
        if key in allowed and value is not None
    }
    if not clean:
        return False

    assignments = ", ".join(f"{key}=?" for key in clean)
    values = list(clean.values())

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            f"""
            UPDATE commercial_products
            SET {assignments}, updated_at=CURRENT_TIMESTAMP
            WHERE user_id=? AND id=?
            """,
            (*values, user_id, int(product_id)),
        )
        changed = cur.rowcount > 0
        if changed:
            await db.execute(
                """
                INSERT INTO audit_log(user_id, event_type, detail)
                VALUES (?, ?, ?)
                """,
                (
                    user_id,
                    "commercial_product_updated",
                    f"product_id={int(product_id)}; changes={clean}",
                ),
            )
        await db.commit()
    return changed


async def update_commercial_offer(
    user_id: str,
    offer_id: int,
    **changes,
):
    allowed = {
        "price",
        "currency",
        "price_unit",
        "quantity",
        "moq",
        "incoterm",
        "payment_terms",
        "quote_date",
        "valid_until",
        "lead_time_days",
        "status",
        "source",
        "notes",
    }
    clean = {
        key: value.strip() if isinstance(value, str) else value
        for key, value in changes.items()
        if key in allowed and value is not None
    }
    if not clean:
        return False

    assignments = ", ".join(f"{key}=?" for key in clean)
    values = list(clean.values())

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        cur = await db.execute(
            f"""
            UPDATE commercial_offers
            SET {assignments}
            WHERE user_id=? AND id=?
            """,
            (*values, user_id, int(offer_id)),
        )
        changed = cur.rowcount > 0
        if changed:
            await db.execute(
                """
                INSERT INTO audit_log(user_id, event_type, detail)
                VALUES (?, ?, ?)
                """,
                (
                    user_id,
                    "commercial_offer_updated",
                    f"offer_id={int(offer_id)}; changes={clean}",
                ),
            )
        await db.commit()
    return changed


async def delete_commercial_offer(user_id: str, offer_id: int):
    async with aiosqlite.connect(LIO_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            SELECT id, supplier_id, product_id, price, currency, price_unit,
                   quantity, moq, incoterm, payment_terms, quote_date,
                   valid_until, lead_time_days, status, source, notes
            FROM commercial_offers
            WHERE user_id=? AND id=?
            """,
            (user_id, int(offer_id)),
        )
        row = await cur.fetchone()
        if not row:
            return False

        snapshot = dict(row)
        await db.execute(
            "DELETE FROM commercial_offers WHERE user_id=? AND id=?",
            (user_id, int(offer_id)),
        )
        await db.execute(
            """
            INSERT INTO audit_log(user_id, event_type, detail)
            VALUES (?, ?, ?)
            """,
            (
                user_id,
                "commercial_offer_deleted",
                f"offer_id={int(offer_id)}; snapshot={snapshot}",
            ),
        )
        await db.commit()
    return True

def _commercial_size_key(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    for token in ("×", "*", "X"):
        text = text.replace(token, "x")
    text = text.replace(" ", "")
    return text or None


def _commercial_product_terms(value):
    text = (value or "").strip().casefold()
    if not text:
        return []
    aliases = (
        ("porcelain", "بورسلان", "بورسلين", "porzellan", "feinsteinzeug"),
        ("ceramic", "سيراميك", "keramik"),
    )
    for group in aliases:
        if any(term in text for term in group):
            return list(group)
    return [value.strip()]


async def get_commercial_offer_comparison(
    user_id: str,
    product_query: str | None = None,
    size_query: str | None = None,
    supplier_ids: list[int] | None = None,
    latest_per_supplier: bool = True,
    limit: int = 100,
):
    """Prepare saved offers for safe supplier/offer comparison."""
    limit = max(1, min(int(limit or 100), 500))
    product_query = (product_query or "").strip()
    wanted_size = _commercial_size_key(size_query)

    where = ["o.user_id=?"]
    params = [user_id]

    if product_query:
        product_terms = _commercial_product_terms(product_query)
        product_clauses = [
            "LOWER(COALESCE(p.product_name,'')) LIKE LOWER(?)"
            for _ in product_terms
        ]
        where.append("(" + " OR ".join(product_clauses) + ")")
        params.extend(f"%{term}%" for term in product_terms)

    if supplier_ids:
        clean_ids = [int(x) for x in supplier_ids]
        placeholders = ",".join("?" for _ in clean_ids)
        where.append(f"o.supplier_id IN ({placeholders})")
        params.extend(clean_ids)

    query = f"""
        SELECT
            o.id, o.supplier_id, o.product_id,
            s.name AS supplier, s.country, s.city, s.website, s.contact_name,
            s.email, s.phone, s.supplier_type, s.status AS supplier_status,
            p.product_name AS product, p.category, p.size, p.thickness_mm,
            p.finish, p.color, p.model,
            o.price, o.currency, o.price_unit, o.quantity, o.moq, o.incoterm,
            o.payment_terms, o.quote_date, o.valid_until, o.lead_time_days,
            o.status AS offer_status, o.source, o.notes, o.created_at
        FROM commercial_offers o
        LEFT JOIN commercial_suppliers s ON s.id=o.supplier_id
        LEFT JOIN commercial_products p ON p.id=o.product_id
        WHERE {" AND ".join(where)}
        ORDER BY
            o.supplier_id,
            CASE WHEN o.quote_date IS NULL OR o.quote_date='' THEN 1 ELSE 0 END,
            o.quote_date DESC,
            o.id DESC
        LIMIT ?
    """
    params.append(limit)

    async with aiosqlite.connect(LIO_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(query, tuple(params))
        rows = [dict(row) for row in await cur.fetchall()]

        ids = sorted({int(r["supplier_id"]) for r in rows if r.get("supplier_id") is not None})
        language_map = {}
        if ids:
            placeholders = ",".join("?" for _ in ids)
            cur = await db.execute(
                f"""
                SELECT supplier_id, language
                FROM commercial_supplier_languages
                WHERE user_id=? AND supplier_id IN ({placeholders})
                ORDER BY supplier_id, language
                """,
                (user_id, *ids),
            )
            for supplier_id, language in await cur.fetchall():
                language_map.setdefault(int(supplier_id), []).append(language)

    if wanted_size:
        rows = [r for r in rows if _commercial_size_key(r.get("size")) == wanted_size]

    if latest_per_supplier:
        newest = []
        seen = set()
        for row in rows:
            sid = row.get("supplier_id")
            if sid in seen:
                continue
            seen.add(sid)
            newest.append(row)
        rows = newest

    missing_watch = (
        "price", "currency", "price_unit", "incoterm", "payment_terms",
        "quote_date", "valid_until", "lead_time_days", "moq",
    )

    for row in rows:
        sid = row.get("supplier_id")
        row["languages"] = language_map.get(int(sid), []) if sid is not None else []
        row["missing_fields"] = [
            field for field in missing_watch
            if row.get(field) is None or row.get(field) == ""
        ]
        row["price_rank"] = None
        row["comparable_price_count"] = 0
        row["is_lowest_saved_price"] = False

    groups = {}
    for row in rows:
        if row.get("price") is None or not row.get("currency") or not row.get("price_unit"):
            continue
        key = (str(row["currency"]).upper(), str(row["price_unit"]).lower())
        groups.setdefault(key, []).append(row)

    comparable_groups = []
    for (currency, price_unit), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: (float(item["price"]), int(item["id"])))
        for rank, item in enumerate(ordered, start=1):
            item["price_rank"] = rank
            item["comparable_price_count"] = len(ordered)
            item["is_lowest_saved_price"] = rank == 1

        comparable_groups.append({
            "currency": currency,
            "price_unit": price_unit,
            "offer_count": len(ordered),
            "offer_ids_by_price": [int(item["id"]) for item in ordered],
            "lowest_price": float(ordered[0]["price"]) if ordered else None,
        })

    warnings = []
    if len(comparable_groups) > 1:
        warnings.append(
            "Offers use different currency and/or price units. "
            "They were not ranked against each other and no FX conversion was performed."
        )
    if any(
        row.get("price") is None or not row.get("currency") or not row.get("price_unit")
        for row in rows
    ):
        warnings.append(
            "Some offers have incomplete price data and were excluded from price ranking."
        )

    return {
        "product_query": product_query or None,
        "size_query": size_query or None,
        "latest_per_supplier": bool(latest_per_supplier),
        "offers": rows,
        "comparable_price_groups": comparable_groups,
        "warnings": warnings,
    }
