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
    add_commercial_supplier_language,
    get_commercial_supplier_languages,
    find_commercial_suppliers,
    get_commercial_offer_by_id,
    get_latest_commercial_offer,
    update_commercial_supplier,
    update_commercial_offer,
    delete_commercial_offer,
    get_commercial_offer_comparison,
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

    # Preserve common legal company suffixes such as "Co., Ltd." that may contain commas.
    if supplier:
        legal_name_match = re.search(
            r"([A-Z][A-Za-z0-9&'().,\- ]{1,150}?"
            r"(?:Co\.\s*,?\s*Ltd\.?|Ltd\.?|Limited|Inc\.?|LLC|L\.L\.C\.|"
            r"Corp\.?|Corporation|GmbH|AG|S\.A\.|S\.L\.|B\.V\.|Pte\.?\s+Ltd\.?))",
            message,
            flags=re.IGNORECASE,
        )
        if legal_name_match:
            legal_name = legal_name_match.group(1).strip(" \t-,:;")
            if supplier.casefold() in legal_name.casefold():
                supplier = legal_name
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

    languages = []
    language_map = [
        (r"(?:الانكليزية|الإنكليزية|الانجليزية|الإنجليزية|\benglish\b|\benglisch\b)", "English"),
        (r"(?:الصينية|\bchinese\b|\bchinesisch\b)", "Chinese"),
        (r"(?:العربية|\barabic\b|\barabisch\b)", "Arabic"),
        (r"(?:الألمانية|الالمانية|\bgerman\b|\bdeutsch\b)", "German"),
        (r"(?:الفرنسية|\bfrench\b|\bfranzösisch\b|\bfranzoesisch\b)", "French"),
        (r"(?:الإسبانية|الاسبانية|\bspanish\b|\bspanisch\b)", "Spanish"),
        (r"(?:الإيطالية|الايطالية|\bitalian\b|\bitalienisch\b)", "Italian"),
        (r"(?:التركية|\bturkish\b|\btürkisch\b|\btuerkisch\b)", "Turkish"),
    ]
    for pattern, normalized in language_map:
        if re.search(pattern, message, flags=re.IGNORECASE):
            languages.append(normalized)

    website = None
    website_match = re.search(
        r"(?<![@\w])("
        r"(?:https?://)?(?:www\.)?"
        r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,62}\.)+[A-Za-z]{2,}"
        r"(?:/[^\s،,;]*)?"
        r")",
        message,
        flags=re.IGNORECASE,
    )
    if website_match:
        website = website_match.group(1).rstrip(".,;،")

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
    if city and country:
        country_suffixes = {
            "China": ("China", "الصين"),
            "Turkey": ("Turkey", "Türkiye", "تركيا"),
            "Spain": ("Spain", "إسبانيا", "اسبانيا"),
            "Austria": ("Austria", "Österreich", "النمسا"),
            "Italy": ("Italy", "إيطاليا", "ايطاليا"),
            "India": ("India", "الهند"),
        }
        for country_token in country_suffixes.get(country, ()):
            city = re.sub(
                rf"(?:\s+|,\s*){re.escape(country_token)}\s*$",
                "",
                city,
                flags=re.IGNORECASE,
            ).strip(" \t-،,.;")

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
        "website": website,
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
        "languages": languages,
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
        website=record["website"],
    )

    for language in record.get("languages", []):
        await add_commercial_supplier_language(user_id, supplier_id, language)

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


def _normalize_management_supplier_query(value: str | None):
    if not value:
        return None
    value = _clean_value(value).strip()
    replacements = (
        ("للسيدة ", "السيدة "),
        ("للسيد ", "السيد "),
        ("لشركة ", "شركة "),
        ("للمورد ", ""),
        ("للمصنع ", ""),
        ("للشركة ", ""),
    )
    for prefix, replacement in replacements:
        if value.startswith(prefix):
            value = replacement + value[len(prefix):]
            break
    return value.strip(" -,:،")


def _extract_management_currency(message: str):
    match = re.search(
        r"\b(USD|EUR|CNY|RMB)\b|US\$|\$|€|دولار(?:\s+أمريكي)?|يورو|يوان",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw = match.group(0).casefold()
    if raw in ("usd", "us$", "$") or "دولار" in raw:
        return "USD"
    if raw in ("eur", "€") or "يورو" in raw:
        return "EUR"
    if raw in ("cny", "rmb") or "يوان" in raw:
        return "CNY"
    return None


def _extract_management_price(message: str):
    patterns = [
        r"(?:إلى|الى|ليصبح|بسعر|سعر|price\s+to|to|auf|preis)\s*[:=]?\s*([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)",
        r"([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)\s*(?:USD|US\$|\$|EUR|€|CNY|RMB|دولار|يورو|يوان)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return _commercial_number(match.group(1))
    return None


async def _resolve_management_supplier(user_id: str, supplier_query: str | None):
    supplier_query = _normalize_management_supplier_query(supplier_query)
    if not supplier_query:
        return None, "Commercial management not applied: supplier name was not clear."

    matches = await find_commercial_suppliers(user_id, supplier_query, 10)
    if not matches:
        return None, f"Commercial management not applied: no saved supplier matched: {supplier_query}"
    if len(matches) > 1:
        names = ", ".join(item["name"] for item in matches[:5])
        return None, (
            "Commercial management not applied: supplier name is ambiguous. "
            f"Matches: {names}"
        )
    return matches[0], None


def _extract_latest_offer_supplier_query(message: str):
    return _first_match(
        message,
        [
            r"(?:العرض\s+(?:الأخير|الاخير|الأحدث|الاحدث))\s+(.+?)(?=\s+(?:إلى|الى|ليصبح|بسعر|سعر|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[،,;]|$)",
            r"\b(?:latest|most\s+recent)\s+offer\s+(?:for|from)\s+(.+?)(?=\s+(?:to|at|price|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[,;]|$)",
            r"\b(?:letztes|neueste[sr]?)\s+angebot\s+(?:von|für|fuer)\s+(.+?)(?=\s+(?:auf|zu|preis|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[,;]|$)",
        ],
    )


async def _capture_commercial_management(user_id: str, message: str):
    folded = message.casefold()

    delete_words = (
        "احذف", "امسح", "أزل", "ازل",
        "delete", "remove",
        "lösche", "loesche", "entferne",
    )
    correction_words = (
        "صحح", "صحّح", "غير", "غيّر", "عدل", "عدّل", "حدث", "حدّث",
        "correct", "change", "update",
        "korrigiere", "ändere", "aendere", "aktualisiere",
    )
    add_offer_words = (
        "أضف عرض", "اضف عرض",
        "add offer", "add a new offer", "add new offer",
        "neues angebot", "angebot hinzufügen", "angebot hinzufuegen",
    )

    if any(word in folded for word in delete_words) and (
        "عرض" in folded or "offer" in folded or "angebot" in folded
    ):
        id_match = re.search(
            r"(?:عرض|offer|angebot)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
            message,
            flags=re.IGNORECASE,
        )
        if id_match:
            offer_id = int(id_match.group(1))
            offer = await get_commercial_offer_by_id(user_id, offer_id)
            if not offer:
                return f"Commercial management not applied: offer #{offer_id} was not found."
            deleted = await delete_commercial_offer(user_id, offer_id)
            return (
                f"Commercial offer deleted: offer_id={offer_id}; supplier={offer.get('supplier')}"
                if deleted
                else f"Commercial management not applied: offer #{offer_id} could not be deleted."
            )

        if any(token in folded for token in ("الأخير", "الاخير", "latest", "most recent", "letztes", "neueste")):
            supplier_query = _extract_latest_offer_supplier_query(message)
            supplier, error = await _resolve_management_supplier(user_id, supplier_query)
            if error:
                return error
            offer = await get_latest_commercial_offer(user_id, supplier["id"])
            if not offer:
                return f"Commercial management not applied: no saved offer exists for supplier {supplier['name']}."
            deleted = await delete_commercial_offer(user_id, offer["id"])
            return (
                f"Commercial offer deleted: offer_id={offer['id']}; supplier={supplier['name']}"
                if deleted
                else "Commercial management not applied: deletion failed."
            )

        return (
            "Commercial management not applied: deletion requires a specific offer ID "
            "or the explicit latest offer of one supplier."
        )

    if any(word in folded for word in correction_words) and (
        "عرض" in folded or "offer" in folded or "angebot" in folded
    ):
        id_match = re.search(
            r"(?:عرض|offer|angebot)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
            message,
            flags=re.IGNORECASE,
        )
        if id_match:
            offer = await get_commercial_offer_by_id(user_id, int(id_match.group(1)))
            if not offer:
                return f"Commercial management not applied: offer #{int(id_match.group(1))} was not found."
        else:
            supplier_query = _extract_latest_offer_supplier_query(message)
            supplier, error = await _resolve_management_supplier(user_id, supplier_query)
            if error:
                return error
            offer = await get_latest_commercial_offer(user_id, supplier["id"])
            if not offer:
                return f"Commercial management not applied: no saved offer exists for supplier {supplier['name']}."

        changes = {}
        if any(token in folded for token in ("سعر", "price", "preis")):
            price = _extract_management_price(message)
            if price is None:
                return "Commercial management not applied: the new price was not clear."
            changes["price"] = price
            currency = _extract_management_currency(message)
            if currency:
                changes["currency"] = currency

        incoterm_match = re.search(
            r"\b(EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b",
            message,
            flags=re.IGNORECASE,
        )
        if any(token in folded for token in ("incoterm", "شرط التسليم", "شروط التسليم")) and incoterm_match:
            changes["incoterm"] = incoterm_match.group(1).upper()

        if not changes:
            return (
                "Commercial management not applied: this v1 update supports "
                "offer price/currency and Incoterm."
            )

        changed = await update_commercial_offer(user_id, offer["id"], **changes)
        return (
            f"Commercial offer updated: offer_id={offer['id']}; changes={changes}"
            if changed
            else "Commercial management not applied: offer update failed."
        )

    if any(word in folded for word in add_offer_words):
        supplier_query = _first_match(
            message,
            [
                r"(?:أضف|اضف)\s+عرض(?:اً|ا)?\s+جديد(?:اً|ا)?\s+(.+?)(?=\s+(?:بسعر|سعر|بقيمة|بمبلغ|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[،,;]|$)",
                r"\badd\s+(?:a\s+)?new\s+offer\s+(?:for|from)\s+(.+?)(?=\s+(?:at|price|for|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[,;]|$)",
                r"\bneues\s+angebot\s+(?:von|für|fuer)\s+(.+?)(?=\s+(?:für|fuer|preis|EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b|[,;]|$)",
            ],
        )
        supplier, error = await _resolve_management_supplier(user_id, supplier_query)
        if error:
            return error

        latest = await get_latest_commercial_offer(user_id, supplier["id"])
        price = _extract_management_price(message)
        currency = _extract_management_currency(message)
        incoterm_match = re.search(
            r"\b(EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b",
            message,
            flags=re.IGNORECASE,
        )

        if price is None and not incoterm_match:
            return "Commercial management not applied: the new offer needs at least a price or Incoterm."

        offer_id = await add_commercial_offer(
            user_id,
            supplier_id=supplier["id"],
            product_id=latest.get("product_id") if latest else None,
            price=price,
            currency=currency or (latest.get("currency") if latest else None),
            price_unit=latest.get("price_unit") if latest else None,
            incoterm=(
                incoterm_match.group(1).upper()
                if incoterm_match
                else (latest.get("incoterm") if latest else None)
            ),
            source="user_management_command",
        )
        return f"Commercial offer added: offer_id={offer_id}; supplier={supplier['name']}; price={price}"

    if any(word in folded for word in correction_words) and (
        "مورد" in folded or "supplier" in folded or "lieferant" in folded
    ):
        field_patterns = [
            ("city", [
                r"(?:مدينة|المدينة)\s+(?:المورد\s+)?(.+?)\s+(?:إلى|الى|ليصبح)\s+(.+)$",
                r"\bcity\s+(?:of\s+)?(?:supplier\s+)?(.+?)\s+to\s+(.+)$",
                r"\bstadt\s+(?:des\s+)?lieferanten\s+(.+?)\s+(?:auf|zu)\s+(.+)$",
            ]),
            ("country", [
                r"(?:بلد|الدولة|دولة)\s+(?:المورد\s+)?(.+?)\s+(?:إلى|الى|ليصبح)\s+(.+)$",
                r"\bcountry\s+(?:of\s+)?(?:supplier\s+)?(.+?)\s+to\s+(.+)$",
                r"\bland\s+(?:des\s+)?lieferanten\s+(.+?)\s+(?:auf|zu)\s+(.+)$",
            ]),
            ("email", [
                r"(?:ايميل|إيميل|البريد\s+الإلكتروني)\s+(?:للمورد\s+)?(.+?)\s+(?:إلى|الى|ليصبح)\s+(\S+)$",
                r"\bemail\s+(?:of\s+)?(?:supplier\s+)?(.+?)\s+to\s+(\S+)$",
                r"\be-?mail\s+(?:des\s+)?lieferanten\s+(.+?)\s+(?:auf|zu)\s+(\S+)$",
            ]),
            ("phone", [
                r"(?:هاتف|رقم\s+هاتف|تلفون)\s+(?:المورد\s+)?(.+?)\s+(?:إلى|الى|ليصبح)\s+([+\d][\d\s().-]+)$",
                r"\bphone\s+(?:of\s+)?(?:supplier\s+)?(.+?)\s+to\s+([+\d][\d\s().-]+)$",
                r"\btelefon\s+(?:des\s+)?lieferanten\s+(.+?)\s+(?:auf|zu)\s+([+\d][\d\s().-]+)$",
            ]),
        ]

        for field, patterns in field_patterns:
            for pattern in patterns:
                match = re.search(pattern, message, flags=re.IGNORECASE)
                if not match:
                    continue
                supplier, error = await _resolve_management_supplier(user_id, match.group(1))
                if error:
                    return error
                value = _clean_value(match.group(2))
                if not value:
                    return "Commercial management not applied: new value was empty."
                changed = await update_commercial_supplier(
                    user_id, supplier["id"], **{field: value}
                )
                return (
                    f"Commercial supplier updated: supplier={supplier['name']}; {field}={value}"
                    if changed
                    else "Commercial management not applied: supplier update failed."
                )

    return None


async def _capture_user_memory(user_id: str, message: str):
    control = _extract_memory_control(message)
    if control:
        return await _apply_memory_control(user_id, control)

    name = _extract_name(message)
    if name:
        await set_display_name(user_id, name)

    management_action = await _capture_commercial_management(user_id, message)
    if management_action:
        return management_action

    commercial_action = await _capture_commercial_memory(user_id, message)

    explicit = _extract_explicit_memory(message)
    if explicit and not commercial_action:
        await add_memory(user_id, explicit)

    for category, key, value, importance in _extract_structured_facts(message):
        await upsert_smart_memory(user_id, category, key, value, importance)
        if key == "preferred_language":
            await set_preferred_language(user_id, value)

    return commercial_action

def _is_commercial_comparison_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "قارن", "مقارنة", "الأفضل", "الافضل", "أفضل مورد", "افضل مورد",
        "أرخص", "ارخص", "أقل سعر", "اقل سعر",
        "compare", "comparison", "best supplier", "cheapest", "lowest price",
        "vergleichen", "vergleich", "bester lieferant", "beste lieferant",
        "günstigste", "guenstigste", "niedrigster preis",
    )
    commercial_words = (
        "عرض", "عروض", "مورد", "موردين", "سعر", "أسعار", "اسعار",
        "offer", "offers", "supplier", "suppliers", "price", "prices",
        "angebot", "angebote", "lieferant", "lieferanten", "preis", "preise",
        "بورسلان", "بورسلين", "سيراميك", "porcelain", "ceramic",
        "porzellan", "feinsteinzeug", "keramik",
    )
    return any(x in folded for x in signals) and any(x in folded for x in commercial_words)


def _extract_comparison_size(message: str):
    match = re.search(r"\b(\d{2,4})\s*[xX×*/]\s*(\d{2,4})\b", message or "")
    if not match:
        return None
    return f"{match.group(1)}x{match.group(2)}"


def _extract_comparison_product(message: str):
    folded = (message or "").casefold()
    groups = (
        ("porcelain", ("porcelain", "بورسلان", "بورسلين", "porzellan", "feinsteinzeug")),
        ("ceramic", ("ceramic", "سيراميك", "keramik")),
    )
    for canonical, terms in groups:
        if any(term in folded for term in terms):
            return canonical
    return None


async def _commercial_comparison_context(user_id: str, message: str) -> str:
    if not _is_commercial_comparison_request(message):
        return ""

    product_query = _extract_comparison_product(message)
    size_query = _extract_comparison_size(message)

    data = await get_commercial_offer_comparison(
        user_id,
        product_query=product_query,
        size_query=size_query,
        latest_per_supplier=True,
        limit=100,
    )
    offers = data.get("offers", [])

    lines = [
        "CURRENT COMMERCIAL COMPARISON SNAPSHOT.",
        "Use this saved-data snapshot as the authoritative basis for this turn's comparison.",
        "Do not invent missing commercial facts.",
        "Do not compare prices across different currencies or price units unless the user explicitly asks for conversion and a reliable conversion is separately available.",
        "A lowest price means lowest only inside the same currency + price-unit group.",
        "Different Incoterms are not directly equivalent; mention that when relevant.",
        "Do not describe one Incoterm as inherently better than another. EXW, FOB, CIF, DDP and others allocate cost, risk, transport and customs responsibilities differently.",
        "When Incoterms differ, do not treat the quoted prices as a like-for-like total-cost comparison. State what each term includes and say that landed or equivalent-basis cost is needed for a definitive price winner.",
        "Do not call a supplier 'best overall' solely because its saved price is lower. Consider thickness, Incoterm, MOQ, payment terms, lead time, quote date/validity, and missing data when those fields exist.",
        "Historical offers remain saved; this snapshot shows the latest matching saved offer per supplier.",
    ]

    if product_query:
        lines.append(f"Requested product filter: {product_query}")
    if size_query:
        lines.append(f"Requested size filter: {size_query}")

    if not offers:
        lines.append("No matching saved offers were found.")
        return "\n".join(lines)

    lines.append(f"Matching latest offers: {len(offers)}")
    for item in offers:
        details = [
            f"offer_id={item.get('id')}",
            f"supplier={item.get('supplier')}",
            f"country={item.get('country')}" if item.get("country") else None,
            f"city={item.get('city')}" if item.get("city") else None,
            f"languages={','.join(item.get('languages') or [])}" if item.get("languages") else None,
            f"product={item.get('product')}" if item.get("product") else None,
            f"size={item.get('size')}" if item.get("size") else None,
            f"thickness_mm={item.get('thickness_mm')}" if item.get("thickness_mm") is not None else None,
            f"finish={item.get('finish')}" if item.get("finish") else None,
            f"color={item.get('color')}" if item.get("color") else None,
            f"price={item.get('price')}" if item.get("price") is not None else None,
            f"currency={item.get('currency')}" if item.get("currency") else None,
            f"unit={item.get('price_unit')}" if item.get("price_unit") else None,
            f"price_rank={item.get('price_rank')}" if item.get("price_rank") is not None else None,
            f"comparable_price_count={item.get('comparable_price_count')}" if item.get("comparable_price_count") else None,
            f"quantity={item.get('quantity')}" if item.get("quantity") is not None else None,
            f"moq={item.get('moq')}" if item.get("moq") is not None else None,
            f"incoterm={item.get('incoterm')}" if item.get("incoterm") else None,
            f"payment_terms={item.get('payment_terms')}" if item.get("payment_terms") else None,
            f"quote_date={item.get('quote_date')}" if item.get("quote_date") else None,
            f"valid_until={item.get('valid_until')}" if item.get("valid_until") else None,
            f"lead_time_days={item.get('lead_time_days')}" if item.get("lead_time_days") is not None else None,
            f"missing={','.join(item.get('missing_fields') or [])}" if item.get("missing_fields") else None,
        ]
        lines.append("- " + "; ".join(x for x in details if x))

    groups = data.get("comparable_price_groups", [])
    if groups:
        lines.append("Comparable saved-price groups:")
        for group in groups:
            lines.append(
                "- "
                f"currency={group.get('currency')}; "
                f"unit={group.get('price_unit')}; "
                f"offer_count={group.get('offer_count')}; "
                f"lowest_price={group.get('lowest_price')}; "
                f"offer_ids_by_price={group.get('offer_ids_by_price')}"
            )

    for warning in data.get("warnings", []):
        lines.append(f"Warning: {warning}")

    incoterms = sorted({str(x.get("incoterm")).upper() for x in offers if x.get("incoterm")})
    if len(incoterms) > 1:
        lines.append(
            "Warning: matching offers use different Incoterms: "
            + ", ".join(incoterms)
            + ". Price alone is not a like-for-like comparison."
        )

    return "\n".join(lines)


async def _persistent_context(user_id: str) -> str:
    profile = await get_profile(user_id)
    memories = await saved_memories(user_id, 20)
    smart = await get_smart_memories(user_id, 30)
    commercial = await get_commercial_memory(user_id, supplier_limit=10, offer_limit=20)
    supplier_ids = [item["id"] for item in commercial.get("suppliers", [])]
    commercial_languages = await get_commercial_supplier_languages(user_id, supplier_ids)

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
                (
                    "languages=" + ",".join(commercial_languages.get(supplier.get("id"), []))
                    if commercial_languages.get(supplier.get("id"))
                    else None
                ),
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
    comparison_context = await _commercial_comparison_context(req.user_id, req.message)
    memory_action_context = (
        f"Internal memory status for this turn: {memory_action}"
        if memory_action
        else ""
    )
    context_text = "\n\n".join(
        part
        for part in [persistent_context, comparison_context, memory_action_context, recent_context]
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
