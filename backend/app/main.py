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
    upsert_commercial_supplier,
    add_commercial_product,
    add_commercial_offer,
    get_commercial_memory,
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

def _commercial_number(value: str):
    if value is None:
        return None
    translated = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
    translated = translated.replace(" ", "").replace(",", ".")
    try:
        return float(translated)
    except (TypeError, ValueError):
        return None


def _clean_commercial_supplier_name(value: str | None):
    if not value:
        return value
    value = value.strip().strip(" -,:،")
    value = re.sub(r"\s+(?:هي|هو)\s*$", "", value, flags=re.IGNORECASE)
    return value.strip().strip(" -,:،")


def _commercial_save_intent(message: str) -> bool:
    folded = message.casefold()
    save_words = (
        "احفظ", "سجل", "سجّل", "تذكر", "تذكّر",
        "remember", "save", "record",
        "merke", "merk dir", "speichere", "speicher",
    )
    commercial_words = (
        "مورد", "المورد", "مصنع", "المصنع", "شركة", "الشركة",
        "عرض", "سعر", "بورسلان", "سيراميك",
        "supplier", "factory", "vendor", "quote", "price", "offer",
        "lieferant", "fabrik", "angebot", "preis",
    )
    return any(word in folded for word in save_words) and any(
        word in folded for word in commercial_words
    )


def _extract_commercial_record(message: str):
    if not _commercial_save_intent(message):
        return None

    supplier = _first_match(
        message,
        [
            r"(?:المورد|المصنع|الشركة)\s+(?:اسمه|اسمها|هو|هي)?\s*[:\-]?\s*(.{2,120}?)(?=\s+(?:من|في|بمدينة|عرض|قدّم|قدم|يقدم|يبيع|بسعر|سعر|لديه|لديها)\b|[،,;\n]|$)",
            r"\b(?:supplier|factory|vendor|company)\s+(?:name\s+is\s+|is\s+|named\s+)?(.{2,120}?)(?=\s+(?:from|in|offered|quoted|offers|sells|at|with)\b|[,;\n]|$)",
            r"\b(?:lieferant|fabrik|firma)\s+(?:heißt\s+|heisst\s+|ist\s+)?(.{2,120}?)(?=\s+(?:aus|in|bietet|bot|angebot|preis)\b|[,;\n]|$)",
        ],
    )
    supplier = _clean_commercial_supplier_name(supplier)
    if not supplier:
        return None

    size_match = re.search(
        r"(?<!\d)(\d{2,4}(?:[.,]\d+)?)\s*[xX×*/]\s*(\d{2,4}(?:[.,]\d+)?)(?!\d)",
        message,
    )
    size = None
    if size_match:
        size = f"{size_match.group(1).replace(',', '.')}x{size_match.group(2).replace(',', '.')}"

    thickness_match = re.search(
        r"(?:سماكة|السماكة|بسماكة|thickness|stärke|staerke)\s*[:=]?\s*([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)\s*(?:mm|مم)?",
        message,
        flags=re.IGNORECASE,
    )
    thickness_mm = _commercial_number(thickness_match.group(1)) if thickness_match else None

    product = _first_match(
        message,
        [
            r"(?:عرض|قدّم|قدم|يقدم|يبيع)\s+(?:لي\s+)?(.+?)(?=\s+(?:مقاس|قياس|بمقاس|سماكة|السماكة|بسماكة|بسعر|السعر|سعر|MOQ|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[،,;\n]|$)",
            r"\b(?:offered|quoted|offers|sells)\s+(?:me\s+)?(.+?)(?=\s+(?:size|thickness|at|price|MOQ|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[,;\n]|$)",
            r"\b(?:bietet|bot)\s+(.+?)(?=\s+(?:größe|groesse|stärke|staerke|für|preis|MOQ|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[,;\n]|$)",
        ],
    )
    if product:
        product = re.sub(
            r"^(?:عرض\s+سعر|سعر|price\s+quote|quote|angebotspreis|preis)\s+",
            "",
            product,
            flags=re.IGNORECASE,
        ).strip(" -,:،")

    if product and size_match:
        product = re.sub(
            r"(?<!\d)\d{2,4}(?:[.,]\d+)?\s*[xX×*/]\s*\d{2,4}(?:[.,]\d+)?(?!\d)",
            "",
            product,
        ).strip(" -,:،")
    if not product:
        product_keyword = re.search(
            r"\b(porcelain|ceramic|tiles?|slabs?)\b|(?:بورسلان|بُورسلان|سيراميك|بلاط)",
            message,
            flags=re.IGNORECASE,
        )
        product = product_keyword.group(0) if product_keyword else None

    price_match = re.search(
        r"(?:"
        r"بسعر(?:\s+(?:المتر|المتر\s+المربع|متر|متر\s+مربع|الوحدة))?"
        r"|سعر\s+(?:المتر|المتر\s+المربع|متر|متر\s+مربع|الوحدة)"
        r"|السعر|سعر"
        r"|price(?:\s+per\s+(?:m2|m²|sqm|unit))?"
        r"|at|preis(?:\s+pro\s+(?:m2|m²|qm|stück|stueck))?|für|fuer"
        r")\s*[:=]?\s*"
        r"([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)\s*"
        r"(USD|US\$|\$|EUR|€|CNY|RMB|دولار(?:\s+أمريكي)?|يورو|يوان)?",
        message,
        flags=re.IGNORECASE,
    )
    price = _commercial_number(price_match.group(1)) if price_match else None
    currency_raw = price_match.group(2) if price_match and price_match.group(2) else None

    if not currency_raw:
        currency_search = re.search(
            r"\b(USD|EUR|CNY|RMB)\b|US\$|\$|€|دولار(?:\s+أمريكي)?|يورو|يوان",
            message,
            flags=re.IGNORECASE,
        )
        currency_raw = currency_search.group(0) if currency_search else None

    currency = None
    if currency_raw:
        c = currency_raw.casefold()
        if c in ("usd", "us$", "$") or "دولار" in c:
            currency = "USD"
        elif c in ("eur", "€") or "يورو" in c:
            currency = "EUR"
        elif c in ("cny", "rmb") or "يوان" in c:
            currency = "CNY"

    unit = None
    if re.search(r"(?:للمتر(?:\s+المربع)?|لكل\s+متر(?:\s+مربع)?|/m2|/m²|\bper\s+m2\b|\bper\s+m²\b|\bpro\s+m2\b|\bpro\s+m²\b)", message, flags=re.IGNORECASE):
        unit = "m2"
    elif re.search(r"(?:للقطعة|لكل\s+قطعة|\bper\s+piece\b|\bper\s+pc\b|\bpro\s+stück\b|\bpro\s+stueck\b)", message, flags=re.IGNORECASE):
        unit = "piece"

    incoterm_match = re.search(r"\b(EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b", message, flags=re.IGNORECASE)
    incoterm = incoterm_match.group(1).upper() if incoterm_match else None

    moq_match = re.search(
        r"\bMOQ\b\s*[:=]?\s*([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)",
        message,
        flags=re.IGNORECASE,
    )
    moq = _commercial_number(moq_match.group(1)) if moq_match else None

    qty_match = re.search(
        r"(?:الكمية|كمية|quantity|menge)\s*[:=]?\s*([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)",
        message,
        flags=re.IGNORECASE,
    )
    quantity = _commercial_number(qty_match.group(1)) if qty_match else None

    payment_terms = _first_match(
        message,
        [
            r"(?:شروط\s+الدفع|الدفع)\s*[:=]?\s*(.+?)(?=[،,;\n]|$)",
            r"\bpayment\s+terms?\s*[:=]?\s*(.+?)(?=[,;\n]|$)",
            r"\bzahlungsbedingungen\s*[:=]?\s*(.+?)(?=[,;\n]|$)",
        ],
    )

    date_match = re.search(
        r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b|\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b",
        message,
    )
    quote_date = None
    if date_match:
        if date_match.group(1):
            quote_date = f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
        else:
            quote_date = f"{int(date_match.group(6)):04d}-{int(date_match.group(5)):02d}-{int(date_match.group(4)):02d}"

    country = None
    country_map = [
        (r"(?:\bchina\b|الصين)", "China"),
        (r"(?:\bturkey\b|\btürkei\b|\btuerkei\b|تركيا)", "Turkey"),
        (r"(?:\bspain\b|\bspanien\b|إسبانيا|اسبانيا)", "Spain"),
        (r"(?:\baustria\b|\bösterreich\b|\boesterreich\b|النمسا)", "Austria"),
        (r"(?:\bitaly\b|\bitalien\b|إيطاليا|ايطاليا)", "Italy"),
        (r"(?:\bindia\b|\bindien\b|الهند)", "India"),
    ]
    for pattern, normalized in country_map:
        if re.search(pattern, message, flags=re.IGNORECASE):
            country = normalized
            break

    city = _first_match(
        message,
        [
            r"(?:من\s+مدينة|في\s+مدينة|بمدينة|مدينة)\s+([^\s\n،,.]{2,60})",
            r"(?:من|في)\s+([^\n،,.]{2,60}?)(?=\s+(?:في|بالصين|بتركيا|بإسبانيا|باسبانيا|عرض|قدّم|قدم|يقدم|يبيع|بسعر|سعر)\b|[،,.]|$)",
            r"\b(?:from|in)\s+([A-Za-zÀ-ÿ' -]{2,60}?)(?=\s+(?:in|china|turkey|spain|offered|quoted|offers|sells|at)\b|[,.;]|$)",
            r"\b(?:aus|in)\s+([A-Za-zÀ-ÿ' -]{2,60}?)(?=\s+(?:in|china|türkei|spanien|bietet|bot|preis)\b|[,.;]|$)",
        ],
    )
    if city and country and city.casefold() in {
        "الصين", "china", "تركيا", "turkey", "türkei", "tuerkei",
        "إسبانيا", "اسبانيا", "spain", "spanien",
        "النمسا", "austria", "österreich", "oesterreich",
        "إيطاليا", "ايطاليا", "italy", "italien",
        "الهند", "india", "indien",
    }:
        city = None

    return {
        "supplier": supplier,
        "country": country,
        "city": city,
        "product": product,
        "size": size,
        "thickness_mm": thickness_mm,
        "price": price,
        "currency": currency,
        "price_unit": unit,
        "quantity": quantity,
        "moq": moq,
        "incoterm": incoterm,
        "payment_terms": payment_terms,
        "quote_date": quote_date,
    }


async def _capture_commercial_memory(user_id: str, message: str):
    record = _extract_commercial_record(message)
    if not record:
        return None

    supplier_id = await upsert_commercial_supplier(
        user_id,
        record["supplier"],
        country=record["country"],
        city=record["city"],
    )

    product_id = None
    if record["product"] or record["size"] or record["thickness_mm"] is not None:
        product_id = await add_commercial_product(
            user_id,
            record["product"] or "Commercial product",
            supplier_id=supplier_id,
            size=record["size"],
            thickness_mm=record["thickness_mm"],
        )

    has_offer = any(
        (
            record["price"] is not None,
            record["currency"] is not None,
            record["quantity"] is not None,
            record["moq"] is not None,
            record["incoterm"] is not None,
            record["payment_terms"] is not None,
            record["quote_date"] is not None,
        )
    )
    offer_id = None
    if has_offer:
        offer_id = await add_commercial_offer(
            user_id,
            supplier_id=supplier_id,
            product_id=product_id,
            price=record["price"],
            currency=record["currency"],
            price_unit=record["price_unit"],
            quantity=record["quantity"],
            moq=record["moq"],
            incoterm=record["incoterm"],
            payment_terms=record["payment_terms"],
            quote_date=record["quote_date"],
            source="user_message",
        )

    saved_parts = [f"supplier={record['supplier']}"]
    if product_id is not None:
        saved_parts.append(f"product={record['product'] or 'Commercial product'}")
    if offer_id is not None:
        saved_parts.append(f"offer_id={offer_id}")
    return "Commercial memory saved: " + "; ".join(saved_parts)


async def _capture_user_memory(user_id: str, message: str):
    control = _extract_memory_control(message)
    if control:
        return await _apply_memory_control(user_id, control)

    name = _extract_name(message)
    if name:
        await set_display_name(user_id, name)

    commercial_action = await _capture_commercial_memory(user_id, message)

    explicit = _extract_explicit_memory(message)
    if explicit and not commercial_action:
        await add_memory(user_id, explicit)

    for category, key, value, importance in _extract_structured_facts(message):
        await upsert_smart_memory(user_id, category, key, value, importance)
        if key == "preferred_language":
            await set_preferred_language(user_id, value)

    return commercial_action

async def _persistent_context(user_id: str) -> str:
    profile = await get_profile(user_id)
    memories = await saved_memories(user_id, 20)
    smart = await get_smart_memories(user_id, 30)
    commercial = await get_commercial_memory(user_id, supplier_limit=10, offer_limit=20)

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

    suppliers = commercial.get("suppliers", [])
    offers = commercial.get("offers", [])

    if suppliers:
        lines.append("Commercial supplier memory:")
        for supplier in suppliers:
            details = [
                f"name={supplier.get('name')}",
                f"country={supplier.get('country')}" if supplier.get("country") else None,
                f"city={supplier.get('city')}" if supplier.get("city") else None,
                f"website={supplier.get('website')}" if supplier.get("website") else None,
                f"type={supplier.get('supplier_type')}" if supplier.get("supplier_type") else None,
                f"status={supplier.get('status')}" if supplier.get("status") else None,
                f"contact={supplier.get('contact_name')}" if supplier.get("contact_name") else None,
                f"email={supplier.get('email')}" if supplier.get("email") else None,
                f"phone={supplier.get('phone')}" if supplier.get("phone") else None,
                f"notes={supplier.get('notes')}" if supplier.get("notes") else None,
            ]
            lines.append("- " + "; ".join(item for item in details if item))

    if offers:
        lines.append("Commercial offer history (do not overwrite older offers mentally; compare by date):")
        for offer in offers:
            details = [
                f"supplier={offer.get('supplier')}" if offer.get("supplier") else None,
                f"product={offer.get('product')}" if offer.get("product") else None,
                f"size={offer.get('size')}" if offer.get("size") else None,
                f"thickness_mm={offer.get('thickness_mm')}" if offer.get("thickness_mm") is not None else None,
                f"price={offer.get('price')}" if offer.get("price") is not None else None,
                f"currency={offer.get('currency')}" if offer.get("currency") else None,
                f"unit={offer.get('price_unit')}" if offer.get("price_unit") else None,
                f"quantity={offer.get('quantity')}" if offer.get("quantity") is not None else None,
                f"moq={offer.get('moq')}" if offer.get("moq") is not None else None,
                f"incoterm={offer.get('incoterm')}" if offer.get("incoterm") else None,
                f"payment_terms={offer.get('payment_terms')}" if offer.get("payment_terms") else None,
                f"quote_date={offer.get('quote_date')}" if offer.get("quote_date") else None,
                f"valid_until={offer.get('valid_until')}" if offer.get("valid_until") else None,
                f"lead_time_days={offer.get('lead_time_days')}" if offer.get("lead_time_days") is not None else None,
                f"status={offer.get('status')}" if offer.get("status") else None,
                f"source={offer.get('source')}" if offer.get("source") else None,
                f"notes={offer.get('notes')}" if offer.get("notes") else None,
            ]
            lines.append("- " + "; ".join(item for item in details if item))

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
