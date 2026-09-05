import base64
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
from pathlib import Path
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
    find_exact_commercial_product,
    add_commercial_offer,
    find_exact_commercial_offer,
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
    create_commercial_deal,
    get_commercial_deals,
    get_commercial_deal_events,
    add_commercial_deal_event,
    get_commercial_deal_by_id,
    update_commercial_deal,
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

    # Remove explicitly negated save/record phrases before checking positive save intent.
    # This prevents analysis requests such as "analyze this offer and do not save it"
    # from becoming persistence commands, while still allowing mixed instructions such
    # as "save the supplier, but do not save a price".
    positive_scan = folded
    negative_save_phrases = (
        "لا تحفظ", "لا تسجل", "لا تسجّل", "بدون حفظ", "دون حفظ",
        "do not save", "don't save", "do not record", "don't record",
        "do not remember", "don't remember",
        "nicht speichern", "nicht merken", "nicht aufzeichnen",
    )
    for phrase in negative_save_phrases:
        positive_scan = positive_scan.replace(phrase, " ")
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
    return any(word in positive_scan for word in save_words) and any(
        word in folded for word in commercial_words
    )


def _extract_commercial_record(
    message: str,
    *,
    supplier_override: str | None = None,
):
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

    if supplier and re.match(
        r"^(?:replied|responded|has\s+replied|has\s+responded)\b",
        supplier,
        flags=re.IGNORECASE,
    ):
        supplier = None

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

            # The surrounding sentence may begin with a role label such as
            # "Supplier Test Ceramics Co., Ltd.". The label is not part of
            # the legal company name. Remove it only when the already-parsed
            # supplier name is still preserved in the remaining text.
            label_match = re.match(
                r"^(?:supplier|factory|vendor|company|lieferant|fabrik|firma)\s+(.+)$",
                legal_name,
                flags=re.IGNORECASE,
            )
            if label_match:
                without_label = label_match.group(1).strip(" \t-,:;")
                if supplier.casefold() in without_label.casefold():
                    legal_name = without_label

            if supplier.casefold() in legal_name.casefold():
                supplier = legal_name
    # Prefer a clearly labelled legal company name when the message also
    # contains meta-instructions such as "analyze this supplier offer".
    # The company-body pattern deliberately excludes sentence-ending dots,
    # so a meta phrase cannot swallow a later real company name.
    labelled_legal_supplier_matches = list(re.finditer(
        r"\b(?:supplier|factory|vendor|company|lieferant|fabrik|firma)\s+("
        r"(?:[A-Za-z0-9&'()+\-]+\s+){0,12}"
        r"(?:Co\.\s*,?\s*Ltd\.?|Ltd\.?|Limited|LLC|Inc\.?|Corporation|Corp\.?|GmbH|AG)"
        r")",
        message,
        flags=re.IGNORECASE,
    ))
    if labelled_legal_supplier_matches:
        supplier = labelled_legal_supplier_matches[-1].group(1).strip(" \t-,:;")

    if supplier_override:
        supplier = _clean_commercial_supplier_name(supplier_override)

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
    if thickness_mm is None:
        value_first_thickness_match = re.search(
            r"([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)\s*mm\s*"
            r"(?:thickness|thick|stärke|staerke|سماكة|السماكة)",
            message,
            flags=re.IGNORECASE,
        )
        if value_first_thickness_match:
            thickness_mm = _commercial_number(value_first_thickness_match.group(1))

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

    if price is None:
        currency_first_price_match = re.search(
            r"(?:\bat\b|\bprice\s*[:=]?\s*)?"
            r"(USD|US\$|\$|EUR|€|CNY|RMB|دولار(?:\s+أمريكي)?|يورو|يوان)\s*"
            r"([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)"
            r"(?=\s*(?:/|per\b|pro\b|للمتر|لكل|m2\b|m²\b|sqm\b|piece\b|pc\b|قطعة\b|[,;.]|$))",
            message,
            flags=re.IGNORECASE,
        )
        if currency_first_price_match:
            currency_raw = currency_first_price_match.group(1)
            price = _commercial_number(currency_first_price_match.group(2))

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

    if payment_terms:
        payment_terms = re.split(
            r"\.\s+(?=(?:lead\s*time|delivery\s*time|valid(?:ity)?|quote\s+valid|"
            r"packing|port|origin|مدة\s+التجهيز|مدة\s+التسليم|صلاحية|التعبئة|"
            r"lieferzeit|gültig|gueltig)\b)",
            payment_terms,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

    lead_time_days = None
    lead_time_match = re.search(
        r"(?:lead\s*time|production\s+lead\s*time|delivery\s*time|"
        r"مدة\s+التجهيز|مدة\s+التسليم|lieferzeit)\s*[:=]?\s*"
        r"([0-9٠-٩۰-۹]+)\s*(?:days?|يوم|يوماً|يوما|tage?n?)\b",
        message,
        flags=re.IGNORECASE,
    )
    if lead_time_match:
        lead_time_value = _commercial_number(lead_time_match.group(1))
        if lead_time_value is not None:
            lead_time_days = int(lead_time_value)

    quote_date = None
    quote_date_match = re.search(
        r"(?:quote\s+date|quotation\s+date|offer\s+date|dated|"
        r"تاريخ\s+(?:عرض\s+السعر|العرض)|بتاريخ|"
        r"angebotsdatum|angebot\s+vom)\s*[:=]?\s*"
        r"((?:20\d{2})[-/.]\d{1,2}[-/.]\d{1,2}|"
        r"\d{1,2}[./]\d{1,2}[./](?:20\d{2}))",
        message,
        flags=re.IGNORECASE,
    )
    if quote_date_match:
        raw_quote_date = quote_date_match.group(1)
        quote_date_parts = re.match(
            r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})|"
            r"(\d{1,2})[./](\d{1,2})[./](20\d{2})",
            raw_quote_date,
        )
        if quote_date_parts:
            if quote_date_parts.group(1):
                quote_date = (
                    f"{int(quote_date_parts.group(1)):04d}-"
                    f"{int(quote_date_parts.group(2)):02d}-"
                    f"{int(quote_date_parts.group(3)):02d}"
                )
            else:
                quote_date = (
                    f"{int(quote_date_parts.group(6)):04d}-"
                    f"{int(quote_date_parts.group(5)):02d}-"
                    f"{int(quote_date_parts.group(4)):02d}"
                )

    valid_until = None
    valid_until_match = re.search(
        r"(?:valid\s+until|quote\s+valid\s+until|validity\s+until|"
        r"صالح\s+حتى|صالح\s+لغاية|صلاحية\s+العرض\s+حتى|"
        r"gültig\s+bis|gueltig\s+bis)\s*[:=]?\s*"
        r"((?:20\d{2})[-/.]\d{1,2}[-/.]\d{1,2}|"
        r"\d{1,2}[./]\d{1,2}[./](?:20\d{2}))",
        message,
        flags=re.IGNORECASE,
    )
    if valid_until_match:
        raw_valid_until = valid_until_match.group(1)
        valid_date_match = re.match(
            r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})|"
            r"(\d{1,2})[./](\d{1,2})[./](20\d{2})",
            raw_valid_until,
        )
        if valid_date_match:
            if valid_date_match.group(1):
                valid_until = (
                    f"{int(valid_date_match.group(1)):04d}-"
                    f"{int(valid_date_match.group(2)):02d}-"
                    f"{int(valid_date_match.group(3)):02d}"
                )
            else:
                valid_until = (
                    f"{int(valid_date_match.group(6)):04d}-"
                    f"{int(valid_date_match.group(5)):02d}-"
                    f"{int(valid_date_match.group(4)):02d}"
                )

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

    if city:
        city_folded = city.casefold().strip()
        bad_city_tokens = (
            "missing", "information", "shipment", "payment", "offer", "quote",
            "supplier", "product", "price", "terms", "lead time", "delivery",
            "analysis", "facts", "details",
        )
        if any(token in city_folded for token in bad_city_tokens) or len(city.split()) > 4:
            city = None

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
        "valid_until": valid_until,
        "lead_time_days": lead_time_days,
        "languages": languages,
    }


def _is_referential_offer_save(message: str) -> bool:
    if not _commercial_save_intent(message):
        return False
    folded = (message or "").casefold()
    references = (
        "هذا العرض", "العرض السابق", "العرض أعلاه", "العرض اعلاه",
        "العرض الذي أرسلته", "العرض الذي ارسلته",
        "this offer", "the offer above", "previous offer", "that offer",
        "dieses angebot", "das angebot oben", "vorheriges angebot",
    )
    return any(ref in folded for ref in references)


def _looks_like_supplier_offer_text(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "سعر", "دولار", "يورو", "يوان", "عرض", "عرضت", "قال",
        "price", "usd", "eur", "cny", "rmb", "quote", "quoted", "offer",
        "preis", "angebot",
        "moq", "exw", "fob", "cif", "cfr", "ddp", "dap", "fca",
        "payment", "lead time", "delivery", "packing",
        "الدفع", "مدة", "التجهيز", "التسليم", "التعبئة",
    )
    return any(signal in folded for signal in signals) and any(ch.isdigit() for ch in message)


async def _capture_commercial_memory(
    user_id: str,
    message: str,
    *,
    forced_supplier_id: int | None = None,
    forced_supplier_name: str | None = None,
    forced_product_id: int | None = None,
):
    record = _extract_commercial_record(
        message,
        supplier_override=(
            forced_supplier_name
            if forced_supplier_id is not None
            else None
        ),
    )

    # Review-then-save workflow:
    # If the user says "save this offer" after first pasting/analyzing it,
    # recover only the most recent USER-authored offer-like text.
    # Never use assistant-generated analysis as the persistence source.
    if not record and _is_referential_offer_save(message):
        history = await recent_messages(user_id, 12)
        for item in reversed(history):
            if item.get("role") != "user":
                continue
            prior = (item.get("content") or "").strip()
            if not prior or not _looks_like_supplier_offer_text(prior):
                continue
            candidate = prior + "\n" + message
            candidate_record = _extract_commercial_record(
                candidate,
                supplier_override=(
                    forced_supplier_name
                    if forced_supplier_id is not None
                    else None
                ),
            )
            if candidate_record:
                record = candidate_record
                break

    if not record:
        return None

    if forced_supplier_id is not None:
        supplier_id = int(forced_supplier_id)
        supplier_display_name = (
            forced_supplier_name
            or record.get("supplier")
            or f"Supplier #{supplier_id}"
        )
    else:
        supplier_id = await upsert_commercial_supplier(
            user_id,
            record["supplier"],
            country=record["country"],
            city=record["city"],
            website=record["website"],
        )
        supplier_display_name = record["supplier"]

    for language in record.get("languages", []):
        await add_commercial_supplier_language(user_id, supplier_id, language)

    product_id = int(forced_product_id) if forced_product_id is not None else None
    if (
        forced_product_id is None
        and (record["product"] or record["size"] or record["thickness_mm"] is not None)
    ):
        product_name = record["product"] or "Commercial product"
        existing_product = await find_exact_commercial_product(
            user_id,
            product_name,
            supplier_id=supplier_id,
            category=None,
            size=record["size"],
            thickness_mm=record["thickness_mm"],
            finish=None,
            color=None,
            model=None,
            notes=None,
        )
        if existing_product:
            product_id = existing_product["id"]
        else:
            product_id = await add_commercial_product(
                user_id,
                product_name,
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
            record["valid_until"] is not None,
            record["lead_time_days"] is not None,
        )
    )
    offer_id = None
    duplicate_offer = None
    if has_offer:
        duplicate_offer = await find_exact_commercial_offer(
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
            valid_until=record["valid_until"],
            lead_time_days=record["lead_time_days"],
            status="received",
            source="user_message",
            notes=None,
        )

        if duplicate_offer:
            offer_id = duplicate_offer["id"]
        else:
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
                valid_until=record["valid_until"],
                lead_time_days=record["lead_time_days"],
                source="user_message",
            )

    saved_parts = [f"supplier={supplier_display_name}"]
    if product_id is not None:
        saved_parts.append(f"product={record['product'] or 'Commercial product'}")
    if offer_id is not None:
        saved_parts.append(f"offer_id={offer_id}")

    if duplicate_offer:
        return "Commercial memory unchanged: exact offer already saved: " + "; ".join(saved_parts)

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



_DEAL_STATUS_PATTERNS = [
    (
        "waiting_supplier",
        [
            r"(?:نحن\s+)?(?:بانتظار|ننتظر)\s+(?:رد\s+)?(?:من\s+)?(?:المورد\s+)?(.+?)[.،,!؟?]*$",
            r"\b(?:we\s+are\s+)?waiting\s+(?:for\s+)?(?:a\s+)?reply\s+(?:from\s+)?(?:supplier\s+)?(.+?)[.!?]*$",
            r"\b(?:wir\s+)?warten\s+auf\s+(?:eine\s+)?antwort\s+(?:von\s+)?(?:lieferant\s+)?(.+?)[.!?]*$",
        ],
        "supplier",
        "Wait for supplier reply",
    ),
    (
        "awaiting_sample",
        [
            r"(?:نحن\s+)?(?:بانتظار|ننتظر)\s+(?:العينة|عينة)\s+(?:من\s+)?(?:المورد\s+)?(.+?)[.،,!؟?]*$",
            r"\b(?:we\s+are\s+)?waiting\s+for\s+(?:the\s+)?sample\s+(?:from\s+)?(?:supplier\s+)?(.+?)[.!?]*$",
            r"\b(?:wir\s+)?warten\s+auf\s+(?:das\s+)?muster\s+(?:von\s+)?(?:lieferant\s+)?(.+?)[.!?]*$",
        ],
        "supplier",
        "Receive and evaluate the supplier sample",
    ),
    (
        "awaiting_pi",
        [
            r"(?:نحن\s+)?(?:بانتظار|ننتظر)\s+(?:الـ?\s*)?(?:PI|proforma|الفاتورة\s+(?:المبدئية|الأولية|الاولية))\s+(?:من\s+)?(?:المورد\s+)?(.+?)[.،,!؟?]*$",
            r"\b(?:we\s+are\s+)?waiting\s+for\s+(?:the\s+)?(?:PI|proforma(?:\s+invoice)?)\s+(?:from\s+)?(?:supplier\s+)?(.+?)[.!?]*$",
            r"\b(?:wir\s+)?warten\s+auf\s+(?:die\s+)?proforma(?:-rechnung)?\s+(?:von\s+)?(?:lieferant\s+)?(.+?)[.!?]*$",
        ],
        "supplier",
        "Review the proforma invoice when received",
    ),
    (
        "negotiating",
        [
            r"(?:نحن\s+)?(?:نتفاوض|في\s+مفاوضات)\s+(?:مع\s+)?(?:المورد\s+)?(.+?)[.،,!؟?]*$",
            r"\b(?:we\s+are\s+)?negotiating\s+with\s+(?:supplier\s+)?(.+?)[.!?]*$",
            r"\b(?:wir\s+)?verhandeln\s+mit\s+(?:lieferant\s+)?(.+?)[.!?]*$",
        ],
        "supplier",
        "Continue negotiation and resolve the open commercial points",
    ),
    (
        "production",
        [
            r"(?:بدأ|بدأت|دخل|دخلت)\s+(?:المورد\s+)?(.+?)\s+(?:مرحلة\s+)?(?:الإنتاج|الانتاج)[.،,!؟?]*$",
            r"\b(?:supplier\s+)?(.+?)\s+(?:has\s+)?started\s+production[.!?]*$",
            r"\b(?:lieferant\s+)?(.+?)\s+hat\s+die\s+produktion\s+begonnen[.!?]*$",
        ],
        "supplier",
        "Monitor production progress and the agreed lead time",
    ),
    (
        "inspection",
        [
            r"(?:صفقة|طلب|شحنة)\s+(.+?)\s+(?:في\s+)?(?:مرحلة\s+)?(?:الفحص|التفتيش)[.،,!؟?]*$",
            r"\b(?:deal|order|shipment)\s+(?:with\s+)?(.+?)\s+is\s+(?:in\s+)?inspection[.!?]*$",
            r"\b(?:auftrag|sendung)\s+(?:mit\s+)?(.+?)\s+ist\s+in\s+der\s+inspektion[.!?]*$",
        ],
        "inspection",
        "Complete inspection and record the result",
    ),
    (
        "ready_to_ship",
        [
            r"(?:المورد\s+)?(.+?)\s+(?:جاهز|جاهزة)\s+للشحن[.،,!؟?]*$",
            r"\b(?:supplier\s+)?(.+?)\s+is\s+ready\s+to\s+ship[.!?]*$",
            r"\b(?:lieferant\s+)?(.+?)\s+ist\s+versandbereit[.!?]*$",
        ],
        "supplier",
        "Confirm shipping documents and shipment release",
    ),
    (
        "completed",
        [
            r"(?:اكتملت|انتهت|أغلق|اغلق)\s+(?:الصفقة\s+)?(?:مع\s+)?(?:المورد\s+)?(.+?)[.،,!؟?]*$",
            r"\b(?:the\s+deal\s+with\s+)?(?:supplier\s+)?(.+?)\s+is\s+completed[.!?]*$",
            r"\b(?:der\s+deal\s+mit\s+)?(?:lieferant\s+)?(.+?)\s+ist\s+abgeschlossen[.!?]*$",
        ],
        "none",
        "No further action",
    ),
]


def _extract_deal_tracking_status(message: str):
    for status, patterns, waiting_on, next_action in _DEAL_STATUS_PATTERNS:
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if not match:
                continue
            supplier_query = _clean_value(match.group(1))
            if supplier_query:
                return {
                    "status": status,
                    "supplier_query": supplier_query,
                    "waiting_on": waiting_on,
                    "next_action": next_action,
                }
    return None



def _deal_followup_today():
    # Prefer the user's business timezone; fall back safely if tzdata is unavailable.
    try:
        return datetime.now(ZoneInfo("Europe/Vienna")).date()
    except Exception:
        return datetime.now(timezone.utc).date()


def _deal_followup_ascii_digits(value: str) -> str:
    return (value or "").translate(str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789",
    ))


_DEAL_FOLLOWUP_MONTHS = {
    # Arabic
    "يناير": 1, "كانون الثاني": 1,
    "فبراير": 2, "شباط": 2,
    "مارس": 3, "آذار": 3, "اذار": 3,
    "أبريل": 4, "ابريل": 4, "نيسان": 4,
    "مايو": 5, "أيار": 5, "ايار": 5,
    "يونيو": 6, "حزيران": 6,
    "يوليو": 7, "تموز": 7,
    "أغسطس": 8, "اغسطس": 8, "آب": 8, "اب": 8,
    "سبتمبر": 9, "أيلول": 9, "ايلول": 9,
    "أكتوبر": 10, "اكتوبر": 10, "تشرين الأول": 10, "تشرين الاول": 10,
    "نوفمبر": 11, "تشرين الثاني": 11,
    "ديسمبر": 12, "كانون الأول": 12, "كانون الاول": 12,
    # English
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # German
    "januar": 1,
    "februar": 2,
    "märz": 3, "maerz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}


def _deal_followup_build_date(year: int, month: int, day: int):
    try:
        return datetime(int(year), int(month), int(day)).date()
    except (TypeError, ValueError):
        return None


def _extract_deal_followup_date(message: str):
    raw = _deal_followup_ascii_digits(message)
    folded = raw.casefold()
    today = _deal_followup_today()

    if re.search(r"(?:\bغد(?:اً|ًا|ا)?\b|\btomorrow\b|\bmorgen\b)", folded):
        return today + timedelta(days=1)

    relative_patterns = (
        r"بعد\s+(\d{1,4})\s*(?:يوم|يوماً|يومًا|ايام|أيام)",
        r"\bin\s+(\d{1,4})\s+days?\b",
        r"\bafter\s+(\d{1,4})\s+days?\b",
        r"\bin\s+(\d{1,4})\s+tagen?\b",
        r"\bnach\s+(\d{1,4})\s+tagen?\b",
    )
    for pattern in relative_patterns:
        m = re.search(pattern, folded, flags=re.IGNORECASE)
        if m:
            days = int(m.group(1))
            if 0 <= days <= 3650:
                return today + timedelta(days=days)

    m = re.search(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", folded)
    if m:
        return _deal_followup_build_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b", folded)
    if m:
        return _deal_followup_build_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))

    month_names = sorted(_DEAL_FOLLOWUP_MONTHS, key=len, reverse=True)
    month_alt = "|".join(re.escape(x) for x in month_names)
    m = re.search(
        rf"(?<!\d)(\d{{1,2}})\.?\s+({month_alt})(?:\s+(20\d{{2}}))?",
        folded,
        flags=re.IGNORECASE,
    )
    if m:
        day = int(m.group(1))
        month = _DEAL_FOLLOWUP_MONTHS.get(m.group(2).casefold())
        year = int(m.group(3)) if m.group(3) else today.year
        candidate = _deal_followup_build_date(year, month, day)
        if candidate and not m.group(3) and candidate < today:
            candidate = _deal_followup_build_date(year + 1, month, day)
        return candidate

    return None


def _is_deal_followup_due_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "تابع", "متابعة", "المتابعة", "ذكّرني", "ذكرني", "موعد متابعة",
        "follow up", "follow-up", "followup", "remind me", "check back", "due date",
        "nachfassen", "nachverfolgen", "erinner", "wiedervorlage",
    )
    return any(signal in folded for signal in signals)


def _extract_deal_followup_supplier_query(message: str):
    patterns = (
        r"(?:المورد|مع\s+المورد|مع)\s+(.+?)(?=\s+(?:بعد|في|بتاريخ|يوم)\b|[.،,!؟?]*$)",
        r"\bsupplier\s+(.+?)(?=\s+(?:in|after|on)\b|[.!?]*$)",
        r"\bwith\s+(.+?)(?=\s+(?:in|after|on)\b|[.!?]*$)",
        r"\blieferant(?:en)?\s+(.+?)(?=\s+(?:in|nach|am)\b|[.!?]*$)",
        r"\bmit\s+(.+?)(?=\s+(?:in|nach|am)\b|[.!?]*$)",
    )
    for pattern in patterns:
        m = re.search(pattern, message, flags=re.IGNORECASE)
        if m:
            value = _clean_value(m.group(1))
            if value and value.casefold() not in {
                "هذا", "هذه", "هالمورد", "the supplier", "this supplier",
                "diesem lieferanten", "dem lieferanten",
            }:
                return value
    return None


def _extract_deal_followup_request(message: str):
    if not _is_deal_followup_due_request(message):
        return None

    due_date = _extract_deal_followup_date(message)
    if not due_date:
        return None

    id_match = re.search(
        r"(?:الصفقة|صفقة|deal)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
        _deal_followup_ascii_digits(message),
        flags=re.IGNORECASE,
    )
    return {
        "deal_id": int(id_match.group(1)) if id_match else None,
        "supplier_query": _extract_deal_followup_supplier_query(message),
        "next_action_due": due_date.isoformat(),
    }




def _is_supplier_reply_handoff_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        # Arabic
        "المورد رد", "رد المورد", "وصل رد المورد", "وصلني رد المورد",
        "استلمت رد المورد", "استلمنا رد المورد", "أرسل المورد رده", "ارسل المورد رده",
        # English
        "supplier replied", "supplier responded", "supplier has replied",
        "supplier has responded", "received the supplier reply",
        "received a reply from the supplier", "got the supplier reply",
        "reply from the supplier",
        # German
        "lieferant hat geantwortet", "der lieferant hat geantwortet",
        "antwort vom lieferanten erhalten", "antwort des lieferanten erhalten",
        "lieferantenantwort erhalten",
    )
    return any(signal in folded for signal in signals)


def _extract_supplier_reply_handoff_supplier_query(message: str):
    patterns = (
        # English: "Supplier ABC replied", "reply from supplier ABC"
        r"\bsupplier\s+(.+?)\s+(?:replied|responded|has\s+replied|has\s+responded)\b",
        r"\breply\s+from\s+(?:supplier\s+)?(.+?)(?=[:;,.\n]|$)",
        r"\bresponse\s+from\s+(?:supplier\s+)?(.+?)(?=[:;,.\n]|$)",
        # German
        r"\blieferant\s+(.+?)\s+hat\s+geantwortet\b",
        r"\bantwort\s+(?:vom|von\s+dem)\s+lieferanten\s+(.+?)(?=[:;,.\n]|$)",
        # Arabic
        r"(?:المورد|مورد)\s+(.+?)\s+(?:رد|أرسل\s+رده|ارسل\s+رده)(?=[،,:;.\n]|$)",
        r"(?:رد|جواب)\s+(?:من\s+)?(?:المورد\s+)?(.+?)(?=[،,:;.\n]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, message or "", flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean_value(match.group(1))
        if value and value.casefold() not in {
            "the supplier", "supplier", "المورد", "مورد",
            "der lieferant", "lieferant",
        }:
            return value
    return None


def _extract_supplier_reply_handoff_request(message: str):
    if not _is_supplier_reply_handoff_request(message):
        return None

    ascii_message = _deal_followup_ascii_digits(message or "")
    id_match = re.search(
        r"(?:الصفقة|صفقة|deal)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
        ascii_message,
        flags=re.IGNORECASE,
    )

    cleaned = " ".join((message or "").strip().split())
    if len(cleaned) > 700:
        cleaned = cleaned[:697] + "..."

    return {
        "deal_id": int(id_match.group(1)) if id_match else None,
        "supplier_query": _extract_supplier_reply_handoff_supplier_query(message),
        "event_summary": f"User-stated supplier reply received: {cleaned}",
        "explicit_commercial_save": _commercial_save_intent(message),
    }




def _clarification_normalize_date(raw):
    if not raw:
        return None
    raw = str(raw).strip()
    match = re.match(
        r"^(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})$|^(\d{1,2})[./](\d{1,2})[./](20\d{2})$",
        raw,
    )
    if not match:
        return None
    if match.group(1):
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return f"{int(match.group(6)):04d}-{int(match.group(5)):02d}-{int(match.group(4)):02d}"


def _extract_clarification_terms(message: str) -> dict:
    values = {}
    try:
        record = _extract_commercial_record(message)
    except Exception:
        record = None

    tracked = (
        "product", "size", "thickness_mm", "price", "currency", "price_unit",
        "quantity", "moq", "incoterm", "payment_terms", "quote_date",
        "valid_until", "lead_time_days",
    )
    if record:
        for field in tracked:
            value = record.get(field)
            if value not in (None, ""):
                values[field] = value

    if "moq" not in values:
        m = re.search(
            r"\bMOQ\b\s*[:=]?\s*([0-9٠-٩۰-۹]+(?:[.,][0-9٠-٩۰-۹]+)?)",
            message or "",
            flags=re.IGNORECASE,
        )
        if m:
            values["moq"] = _commercial_number(m.group(1))

    if "payment_terms" not in values:
        payment = _first_match(
            message or "",
            [
                r"(?:شروط\s+الدفع|الدفع)\s*[:=]?\s*(.+?)(?=[،,;\n]|$)",
                r"\bpayment\s+terms?\s*[:=]?\s*(.+?)(?=[,;\n]|$)",
                r"\bzahlungsbedingungen\s*[:=]?\s*(.+?)(?=[,;\n]|$)",
            ],
        )
        if payment:
            values["payment_terms"] = payment.strip()

    if "lead_time_days" not in values:
        m = re.search(
            r"(?:lead\s*time|production\s+lead\s*time|delivery\s*time|"
            r"مدة\s+التجهيز|مدة\s+التسليم|lieferzeit)\s*[:=]?\s*"
            r"([0-9٠-٩۰-۹]+)\s*(?:days?|يوم|يوما|يومًا|tage?n?)\b",
            message or "",
            flags=re.IGNORECASE,
        )
        if m:
            number = _commercial_number(m.group(1))
            if number is not None:
                values["lead_time_days"] = int(number)

    if "valid_until" not in values:
        m = re.search(
            r"(?:valid\s+until|quote\s+valid\s+until|validity\s+until|"
            r"صالح\s+حتى|صالح\s+لغاية|صلاحية\s+العرض\s+حتى|"
            r"gültig\s+bis|gueltig\s+bis)\s*[:=]?\s*"
            r"((?:20\d{2})[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[./]\d{1,2}[./](?:20\d{2}))",
            message or "",
            flags=re.IGNORECASE,
        )
        if m:
            parsed = _clarification_normalize_date(m.group(1))
            if parsed:
                values["valid_until"] = parsed

    if "incoterm" not in values:
        m = re.search(r"\b(EXW|FOB|CIF|CFR|DDP|DAP|FCA)\b", message or "", flags=re.IGNORECASE)
        if m:
            values["incoterm"] = m.group(1).upper()

    return {k: v for k, v in values.items() if v not in (None, "")}


def _clarification_values_equal(a, b) -> bool:
    if a in (None, "") and b in (None, ""):
        return True
    if a in (None, "") or b in (None, ""):
        return False
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return str(a).strip().casefold() == str(b).strip().casefold()


async def _commercial_clarification_resolution_plan(user_id: str, deal: dict, message: str):
    supplier_id = deal.get("supplier_id")
    if supplier_id is None:
        return None

    guidance = await _commercial_offer_decision_guidance(
        user_id,
        supplier_id=int(supplier_id),
    )
    if not guidance or guidance.get("decision") != "REQUEST_CLARIFICATION":
        return None

    missing_before = list(guidance.get("missing_new") or [])
    if not missing_before:
        return None

    data = await get_commercial_offer_comparison(
        user_id,
        supplier_ids=[int(supplier_id)],
        latest_per_supplier=False,
        limit=100,
    )
    offers = data.get("offers", [])
    base_offer_id = guidance.get("new_offer_id")
    base = next((x for x in offers if x.get("id") == base_offer_id), None)
    if not base:
        return None

    extracted = _extract_clarification_terms(message)
    resolved = [
        field for field in missing_before
        if field in extracted and extracted.get(field) not in (None, "")
    ]

    changed_existing = []
    for field, value in extracted.items():
        if field in missing_before:
            continue
        if field in base and base.get(field) not in (None, ""):
            if not _clarification_values_equal(base.get(field), value):
                changed_existing.append(field)

    remaining = [field for field in missing_before if field not in resolved]
    return {
        "base": base,
        "guidance_before": guidance,
        "missing_before": missing_before,
        "extracted": extracted,
        "resolved": resolved,
        "changed_existing": changed_existing,
        "remaining": remaining,
    }


async def _save_commercial_clarification_resolution(user_id: str, deal: dict, plan: dict):
    base = plan["base"]
    extracted = plan.get("extracted") or {}
    if not extracted:
        return None

    merged = {}
    for field in (
        "product", "size", "thickness_mm", "price", "currency", "price_unit",
        "quantity", "moq", "incoterm", "payment_terms", "quote_date",
        "valid_until", "lead_time_days",
    ):
        merged[field] = extracted.get(field, base.get(field))

    supplier_id = int(deal["supplier_id"])
    product_id = base.get("product_id")

    if product_id is None and (
        merged.get("product") or merged.get("size") or merged.get("thickness_mm") is not None
    ):
        product_name = merged.get("product") or "Commercial product"
        existing_product = await find_exact_commercial_product(
            user_id,
            product_name,
            supplier_id=supplier_id,
            category=None,
            size=merged.get("size"),
            thickness_mm=merged.get("thickness_mm"),
            finish=None,
            color=None,
            model=None,
            notes=None,
        )
        if existing_product:
            product_id = existing_product["id"]
        else:
            product_id = await add_commercial_product(
                user_id,
                product_name,
                supplier_id=supplier_id,
                size=merged.get("size"),
                thickness_mm=merged.get("thickness_mm"),
            )

    notes = f"clarification_base_offer_id={base.get('id')}"

    duplicate = await find_exact_commercial_offer(
        user_id,
        supplier_id=supplier_id,
        product_id=product_id,
        price=merged.get("price"),
        currency=merged.get("currency"),
        price_unit=merged.get("price_unit"),
        quantity=merged.get("quantity"),
        moq=merged.get("moq"),
        incoterm=merged.get("incoterm"),
        payment_terms=merged.get("payment_terms"),
        quote_date=merged.get("quote_date"),
        valid_until=merged.get("valid_until"),
        lead_time_days=merged.get("lead_time_days"),
        status="received",
        source="clarification_resolution",
        notes=notes,
    )

    if duplicate:
        offer_id = duplicate["id"]
        created = False
    else:
        offer_id = await add_commercial_offer(
            user_id,
            supplier_id=supplier_id,
            product_id=product_id,
            price=merged.get("price"),
            currency=merged.get("currency"),
            price_unit=merged.get("price_unit"),
            quantity=merged.get("quantity"),
            moq=merged.get("moq"),
            incoterm=merged.get("incoterm"),
            payment_terms=merged.get("payment_terms"),
            quote_date=merged.get("quote_date"),
            valid_until=merged.get("valid_until"),
            lead_time_days=merged.get("lead_time_days"),
            source="clarification_resolution",
            notes=notes,
        )
        created = True

    guidance_after = await _commercial_offer_decision_guidance(
        user_id,
        supplier_id=supplier_id,
        newest_offer_id=offer_id,
    )

    return {
        "offer_id": offer_id,
        "created": created,
        "guidance_after": guidance_after,
        "merged": merged,
    }


def _commercial_clarification_resolution_text(plan: dict | None, result: dict | None = None):
    if not plan:
        return None

    parts = [
        f"base_offer_id={plan.get('base', {}).get('id')}",
        "resolved=" + (",".join(plan.get("resolved") or []) or "none"),
        "remaining=" + (",".join(plan.get("remaining") or []) or "none"),
    ]
    if plan.get("changed_existing"):
        parts.append("changed_existing=" + ",".join(plan["changed_existing"]))

    if result and result.get("offer_id") is not None:
        parts.append(f"offer_id={result['offer_id']}")
        parts.append("saved=true")
        parts.append(f"created={'true' if result.get('created') else 'false'}")
    else:
        parts.append("saved=false")

    return "|".join(parts)




def _negotiation_cycle_clean_value(value):
    if value in (None, ""):
        return None
    return str(value).strip().replace("|", "/").replace(";", ",")


def _negotiation_cycle_offer_matches(current: dict, candidate: dict) -> bool:
    if current.get("supplier_id") != candidate.get("supplier_id"):
        return False

    current_product = (current.get("product") or "").strip().casefold()
    candidate_product = (candidate.get("product") or "").strip().casefold()
    current_size = (current.get("size") or "").strip().casefold()
    candidate_size = (candidate.get("size") or "").strip().casefold()

    product_ok = (
        not current_product
        or not candidate_product
        or current_product == candidate_product
    )
    size_ok = (
        not current_size
        or not candidate_size
        or current_size == candidate_size
    )
    return product_ok and size_ok


def _negotiation_cycle_term_changes(current: dict, previous: dict | None):
    if not previous:
        return []

    fields = (
        "price", "currency", "price_unit", "incoterm", "quantity", "moq",
        "payment_terms", "lead_time_days", "valid_until",
        "product", "size", "thickness_mm",
    )

    changes = []
    for field in fields:
        old = previous.get(field)
        new = current.get(field)
        if _clarification_values_equal(old, new):
            continue

        if old in (None, "") and new not in (None, ""):
            kind = "added"
        elif old not in (None, "") and new in (None, ""):
            kind = "missing_now"
        else:
            kind = "changed"

        changes.append({
            "field": field,
            "kind": kind,
            "from": old,
            "to": new,
        })

    return changes


async def _commercial_negotiation_cycle_snapshot(
    user_id: str,
    deal: dict,
    newest_offer_id: int | None = None,
):
    supplier_id = deal.get("supplier_id")
    if supplier_id is None:
        return None

    data = await get_commercial_offer_comparison(
        user_id,
        supplier_ids=[int(supplier_id)],
        latest_per_supplier=False,
        limit=100,
    )
    offers = data.get("offers", [])
    if not offers:
        return None

    current = None
    if newest_offer_id is not None:
        current = next(
            (item for item in offers if item.get("id") == newest_offer_id),
            None,
        )

    if current is None:
        # Clarification-resolution snapshots refine an already saved commercial
        # offer; they are not counted as new negotiation price rounds.
        current = next(
            (
                item for item in offers
                if str(item.get("source") or "") != "clarification_resolution"
            ),
            offers[0],
        )

    negotiation_offers = [
        item for item in offers
        if str(item.get("source") or "") != "clarification_resolution"
        and _negotiation_cycle_offer_matches(current, item)
    ]
    negotiation_offers = sorted(
        negotiation_offers,
        key=lambda item: int(item.get("id") or 0),
    )

    if str(current.get("source") or "") == "clarification_resolution":
        # Clarification snapshots refine a saved offer. Reuse the existing
        # clarification-chain-aware selector so comparison goes back to the
        # prior real commercial offer.
        previous = _offer_decision_pick_previous(offers, current)
    else:
        # For a normal negotiation round, never let a later clarification
        # snapshot or later offer become its "previous" offer.
        current_id = int(current.get("id") or 0)
        earlier_negotiation_offers = [
            item for item in negotiation_offers
            if int(item.get("id") or 0) < current_id
        ]
        previous = (
            earlier_negotiation_offers[-1]
            if earlier_negotiation_offers
            else None
        )

    round_number = None
    for index, item in enumerate(negotiation_offers, start=1):
        if item.get("id") == current.get("id"):
            round_number = index
            break
    if round_number is None:
        round_number = max(1, len(negotiation_offers))

    price_change = None
    price_change_pct = None
    price_trend = "not_comparable"

    if previous:
        currency_same = _offer_decision_same_text(
            current.get("currency"), previous.get("currency")
        )
        unit_same = _offer_decision_same_text(
            current.get("price_unit"), previous.get("price_unit")
        )
        incoterm_same = _offer_decision_same_text(
            current.get("incoterm"), previous.get("incoterm")
        )

        if (
            currency_same is True
            and unit_same is True
            and incoterm_same is True
            and current.get("price") is not None
            and previous.get("price") is not None
        ):
            try:
                price_change = float(current["price"]) - float(previous["price"])
                if float(previous["price"]) != 0:
                    price_change_pct = (
                        price_change / float(previous["price"])
                    ) * 100.0

                if price_change < -1e-9:
                    price_trend = "improved"
                elif price_change > 1e-9:
                    price_trend = "worsened"
                else:
                    price_trend = "unchanged"
            except (TypeError, ValueError):
                price_change = None
                price_change_pct = None
                price_trend = "not_comparable"
    else:
        price_trend = "baseline"

    guidance = await _commercial_offer_decision_guidance(
        user_id,
        supplier_id=int(supplier_id),
        newest_offer_id=current.get("id"),
    )

    events = await get_commercial_deal_events(
        user_id,
        int(deal["id"]),
        limit=200,
    )
    recorded_rounds = [
        event for event in events
        if event.get("event_type") == "negotiation_round_recorded"
    ]

    return {
        "deal_id": deal.get("id"),
        "supplier_id": supplier_id,
        "supplier": deal.get("supplier"),
        "round_number": round_number,
        "recorded_round_count": len(recorded_rounds),
        "current_offer_id": current.get("id"),
        "previous_offer_id": previous.get("id") if previous else None,
        "price": current.get("price"),
        "currency": current.get("currency"),
        "price_unit": current.get("price_unit"),
        "incoterm": current.get("incoterm"),
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "price_trend": price_trend,
        "term_changes": _negotiation_cycle_term_changes(current, previous),
        "guidance": guidance,
    }


def _commercial_negotiation_cycle_text(snapshot: dict | None):
    if not snapshot:
        return None

    parts = [
        f"deal_id={snapshot.get('deal_id')}",
        f"round={snapshot.get('round_number')}",
        f"offer_id={snapshot.get('current_offer_id')}",
        f"price_trend={snapshot.get('price_trend')}",
    ]

    if snapshot.get("previous_offer_id") is not None:
        parts.append(f"previous_offer_id={snapshot.get('previous_offer_id')}")
    if snapshot.get("price") is not None:
        parts.append(f"price={snapshot.get('price')}")
    if snapshot.get("currency"):
        parts.append(f"currency={snapshot.get('currency')}")
    if snapshot.get("price_unit"):
        parts.append(f"price_unit={snapshot.get('price_unit')}")
    if snapshot.get("incoterm"):
        parts.append(f"incoterm={snapshot.get('incoterm')}")
    if snapshot.get("price_change") is not None:
        parts.append(f"price_change={snapshot.get('price_change'):.6g}")
    if snapshot.get("price_change_pct") is not None:
        parts.append(
            f"price_change_pct={snapshot.get('price_change_pct'):.4g}"
        )

    changes = snapshot.get("term_changes") or []
    if changes:
        encoded = []
        for change in changes:
            field = change.get("field")
            kind = change.get("kind")
            old = _negotiation_cycle_clean_value(change.get("from"))
            new = _negotiation_cycle_clean_value(change.get("to"))
            encoded.append(
                f"{field}:{kind}:{old if old is not None else 'none'}"
                f"->{new if new is not None else 'none'}"
            )
        parts.append("changes=" + ",".join(encoded))
    else:
        parts.append("changes=none")

    guidance = snapshot.get("guidance") or {}
    if guidance.get("decision"):
        parts.append(f"recommendation={guidance.get('decision')}")
    if guidance.get("confidence"):
        parts.append(f"confidence={guidance.get('confidence')}")

    return "|".join(parts)


async def _record_commercial_negotiation_round(
    user_id: str,
    deal: dict,
    newest_offer_id: int,
):
    events = await get_commercial_deal_events(
        user_id,
        int(deal["id"]),
        limit=200,
    )
    offer_marker = f"offer_id={int(newest_offer_id)}"

    for event in events:
        if (
            event.get("event_type") == "negotiation_round_recorded"
            and offer_marker in str(event.get("summary") or "")
        ):
            snapshot = await _commercial_negotiation_cycle_snapshot(
                user_id,
                deal,
                newest_offer_id=int(newest_offer_id),
            )
            return {
                "snapshot": snapshot,
                "event_created": False,
            }

    snapshot = await _commercial_negotiation_cycle_snapshot(
        user_id,
        deal,
        newest_offer_id=int(newest_offer_id),
    )
    if not snapshot:
        return None

    summary = _commercial_negotiation_cycle_text(snapshot)
    await add_commercial_deal_event(
        user_id,
        int(deal["id"]),
        "negotiation_round_recorded",
        summary,
        source="system",
    )
    return {
        "snapshot": snapshot,
        "event_created": True,
    }


def _is_negotiation_cycle_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "negotiation history", "negotiation cycle", "negotiation rounds",
        "summarize negotiation", "negotiation summary", "how has the negotiation",
        "price rounds", "offer rounds",
        "verhandlungsverlauf", "verhandlungsrunden", "verhandlung zusammenfassen",
        "جولات التفاوض", "دورة التفاوض", "تاريخ التفاوض", "ملخص التفاوض",
        "لخص التفاوض", "لخّص التفاوض", "كيف تطور التفاوض", "تطور السعر",
    )
    commercial = (
        "deal", "supplier", "offer", "price", "negotiat",
        "صفقة", "مورد", "عرض", "سعر", "تفاوض",
        "lieferant", "angebot", "preis", "verhandlung",
    )
    return any(signal in folded for signal in signals) and any(
        signal in folded for signal in commercial
    )


async def _commercial_negotiation_cycle_context(
    user_id: str,
    message: str,
) -> str:
    if not _is_negotiation_cycle_request(message):
        return ""

    deal_match = re.search(
        r"(?:الصفقة|صفقة|deal)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
        _deal_followup_ascii_digits(message or ""),
        flags=re.IGNORECASE,
    )

    deal = None
    if deal_match:
        deal = await get_commercial_deal_by_id(
            user_id,
            int(deal_match.group(1)),
        )
    else:
        deals = await get_commercial_deals(
            user_id,
            active_only=True,
            limit=20,
        )
        if len(deals) == 1:
            deal = deals[0]

    if not deal:
        return (
            "SMART NEGOTIATION CYCLE.\n"
            "No unique active deal could be resolved. Ask for the deal ID.\n"
            "Do not invent a target price, concession, commitment, or supplier statement."
        )

    data = await get_commercial_offer_comparison(
        user_id,
        supplier_ids=[int(deal["supplier_id"])],
        latest_per_supplier=False,
        limit=100,
    )
    offers = [
        item for item in data.get("offers", [])
        if str(item.get("source") or "") != "clarification_resolution"
    ]
    offers = [
        item for item in offers
        if not deal.get("product_id")
        or item.get("product_id") == deal.get("product_id")
        or _negotiation_cycle_offer_matches(
            {
                "supplier_id": deal.get("supplier_id"),
                "product": deal.get("product"),
                "size": deal.get("size"),
            },
            item,
        )
    ]
    offers = sorted(
        offers,
        key=lambda item: int(item.get("id") or 0),
    )

    events = await get_commercial_deal_events(
        user_id,
        int(deal["id"]),
        limit=200,
    )
    round_events = [
        event for event in events
        if event.get("event_type") == "negotiation_round_recorded"
    ]

    latest_snapshot = await _commercial_negotiation_cycle_snapshot(
        user_id,
        deal,
        newest_offer_id=(offers[-1].get("id") if offers else None),
    )

    lines = [
        "SMART NEGOTIATION CYCLE.",
        "Read-only negotiation history. Do not send, accept, reject, save, close, or change the deal automatically.",
        "Use only saved commercial facts. Never invent a target price, competitor quote, concession, deadline, or volume commitment.",
        f"Deal ID: {deal.get('id')}",
        f"Supplier: {deal.get('supplier')}",
        f"Recorded negotiation round events: {len(round_events)}",
    ]

    previous = None
    for index, offer in enumerate(offers, start=1):
        details = [
            f"round={index}",
            f"offer_id={offer.get('id')}",
            f"price={offer.get('price')}" if offer.get("price") is not None else None,
            f"currency={offer.get('currency')}" if offer.get("currency") else None,
            f"unit={offer.get('price_unit')}" if offer.get("price_unit") else None,
            f"incoterm={offer.get('incoterm')}" if offer.get("incoterm") else None,
            f"moq={offer.get('moq')}" if offer.get("moq") is not None else None,
            f"payment_terms={offer.get('payment_terms')}" if offer.get("payment_terms") else None,
            f"lead_time_days={offer.get('lead_time_days')}" if offer.get("lead_time_days") is not None else None,
            f"valid_until={offer.get('valid_until')}" if offer.get("valid_until") else None,
        ]

        if previous:
            same_currency = _offer_decision_same_text(
                offer.get("currency"), previous.get("currency")
            )
            same_unit = _offer_decision_same_text(
                offer.get("price_unit"), previous.get("price_unit")
            )
            same_incoterm = _offer_decision_same_text(
                offer.get("incoterm"), previous.get("incoterm")
            )
            if (
                same_currency is True
                and same_unit is True
                and same_incoterm is True
                and offer.get("price") is not None
                and previous.get("price") is not None
            ):
                delta = float(offer["price"]) - float(previous["price"])
                details.append(f"price_delta_from_prior={delta:.6g}")

        lines.append("- " + "; ".join(x for x in details if x))
        previous = offer

    if latest_snapshot:
        guidance = latest_snapshot.get("guidance") or {}
        lines.append(
            f"Latest price trend: {latest_snapshot.get('price_trend')}"
        )
        if guidance.get("decision"):
            lines.append(
                f"Latest advisory recommendation: {guidance.get('decision')}"
            )
        if guidance.get("reason"):
            lines.append(
                "Latest recommendation reason: " + str(guidance.get("reason"))
            )

    return "\n".join(lines)


async def _capture_supplier_reply_handoff(user_id: str, message: str):
    request = _extract_supplier_reply_handoff_request(message)
    if not request:
        return None

    deal = None

    if request["deal_id"] is not None:
        deal = await get_commercial_deal_by_id(user_id, request["deal_id"])
        if not deal:
            return "Supplier reply handoff not recorded: deal #%s was not found." % request["deal_id"]
        if not deal.get("is_active"):
            return "Supplier reply handoff not recorded: deal #%s is not active." % request["deal_id"]

    elif request["supplier_query"]:
        supplier, error = await _resolve_management_supplier(
            user_id, request["supplier_query"]
        )
        if error:
            return error.replace("Commercial management", "Supplier reply handoff")

        deals = await get_commercial_deals(
            user_id,
            active_only=True,
            supplier_id=supplier["id"],
            limit=10,
        )
        if not deals:
            return (
                "Supplier reply handoff not recorded: no active deal exists for supplier "
                f"{supplier['name']}."
            )
        if len(deals) > 1:
            return (
                "Supplier reply handoff not recorded: multiple active deals exist for supplier "
                f"{supplier['name']}; specify the deal ID."
            )
        deal = deals[0]

    else:
        deals = await get_commercial_deals(user_id, active_only=True, limit=20)
        if not deals:
            return "Supplier reply handoff not recorded: there is no active deal."
        if len(deals) > 1:
            return (
                "Supplier reply handoff not recorded: multiple active deals exist; "
                "specify the deal ID or supplier."
            )
        deal = deals[0]

    # A supplier reply ends the old "follow up if no reply" waiting period.
    # If the deal is already in the correct handoff state, that is not a failure:
    # continue so the reply event and any explicit commercial save can still run.
    handoff_state_changed = (
        deal.get("waiting_on") != "user"
        or deal.get("next_action") != "Review supplier response and reply"
        or deal.get("next_action_due") is not None
    )

    if handoff_state_changed:
        changed = await update_commercial_deal(
            user_id,
            deal["id"],
            waiting_on="user",
            next_action="Review supplier response and reply",
            clear_next_action_due=bool(deal.get("next_action_due")),
        )
        if not changed:
            return "Supplier reply handoff not recorded: deal update failed."

    await add_commercial_deal_event(
        user_id,
        deal["id"],
        "supplier_reply_received",
        request["event_summary"],
        source="user",
    )

    commercial_save_status = None
    decision_guidance_status = None
    clarification_resolution_status = None
    negotiation_cycle_status = None

    clarification_plan = await _commercial_clarification_resolution_plan(
        user_id, deal, message
    )
    if clarification_plan:
        clarification_resolution_status = _commercial_clarification_resolution_text(
            clarification_plan
        )

    if request["explicit_commercial_save"]:
        clarification_result = None

        if clarification_plan and clarification_plan.get("extracted"):
            clarification_result = await _save_commercial_clarification_resolution(
                user_id, deal, clarification_plan
            )

        if clarification_result:
            merged_product = (
                clarification_result.get("merged", {}).get("product")
                or "Commercial product"
            )
            if clarification_result.get("created"):
                commercial_save_status = (
                    "Commercial memory saved: "
                    f"supplier={deal.get('supplier')}; product={merged_product}; "
                    f"offer_id={clarification_result['offer_id']}"
                )
            else:
                commercial_save_status = (
                    "Commercial memory unchanged: exact offer already saved: "
                    f"supplier={deal.get('supplier')}; product={merged_product}; "
                    f"offer_id={clarification_result['offer_id']}"
                )

            clarification_resolution_status = _commercial_clarification_resolution_text(
                clarification_plan, clarification_result
            )
            decision_guidance_status = _commercial_offer_decision_guidance_text(
                clarification_result.get("guidance_after")
            )

        elif clarification_plan:
            commercial_save_status = (
                "Commercial clarification save requested, but the supplier reply did not "
                "resolve any tracked missing commercial field."
            )

        else:
            commercial_save_status = await _capture_commercial_memory(
                user_id,
                message,
                forced_supplier_id=int(deal["supplier_id"]),
                forced_supplier_name=deal.get("supplier"),
                forced_product_id=(
                    int(deal["product_id"])
                    if deal.get("product_id") is not None
                    else None
                ),
            )
            if not commercial_save_status:
                commercial_save_status = (
                    "Commercial save was explicitly requested but no complete commercial "
                    "record could be extracted from this message."
                )
            elif commercial_save_status.startswith("Commercial memory saved:"):
                offer_id_match = re.search(r"offer_id=(\d+)", commercial_save_status)
                newest_offer_id = int(offer_id_match.group(1)) if offer_id_match else None
                guidance = await _commercial_offer_decision_guidance(
                    user_id,
                    supplier_id=int(deal["supplier_id"]),
                    newest_offer_id=newest_offer_id,
                )
                decision_guidance_status = _commercial_offer_decision_guidance_text(guidance)

        if (
            commercial_save_status
            and commercial_save_status.startswith("Commercial memory saved:")
        ):
            saved_offer_match = re.search(
                r"offer_id=(\d+)",
                commercial_save_status,
            )
            if saved_offer_match:
                saved_offer_id = int(saved_offer_match.group(1))
                if deal.get("offer_id") != saved_offer_id:
                    pointer_updated = await update_commercial_deal(
                        user_id,
                        int(deal["id"]),
                        offer_id=saved_offer_id,
                    )
                    if pointer_updated:
                        deal["offer_id"] = saved_offer_id

        if (
            commercial_save_status
            and commercial_save_status.startswith("Commercial memory saved:")
            and clarification_result is None
        ):
            negotiation_offer_match = re.search(
                r"offer_id=(\d+)",
                commercial_save_status,
            )
            if negotiation_offer_match:
                negotiation_result = await _record_commercial_negotiation_round(
                    user_id,
                    deal,
                    int(negotiation_offer_match.group(1)),
                )
                if negotiation_result:
                    negotiation_cycle_status = _commercial_negotiation_cycle_text(
                        negotiation_result.get("snapshot")
                    )

    parts = [
        f"Supplier reply handoff recorded: deal_id={deal['id']}",
        f"supplier={deal.get('supplier')}",
        "waiting_on=user",
        "next_action=Review supplier response and reply",
    ]
    if deal.get("next_action_due"):
        parts.append("previous internal no-reply follow-up due date cleared")
    parts.append("event=supplier_reply_received")
    parts.append("No new deal was created")

    if commercial_save_status:
        parts.append(f"commercial_save_status={commercial_save_status}")
        if clarification_resolution_status:
            parts.append(f"clarification_resolution={clarification_resolution_status}")
        if negotiation_cycle_status:
            parts.append(f"negotiation_cycle={negotiation_cycle_status}")
        if decision_guidance_status:
            parts.append(f"decision_guidance={decision_guidance_status}")
    else:
        if clarification_resolution_status:
            parts.append(f"clarification_resolution={clarification_resolution_status}")
        parts.append(
            "Commercial terms in the reply were not saved because no explicit commercial save was requested"
        )

    return "; ".join(parts) + "."
def _is_deal_followup_outcome_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        # Arabic
        "أرسلت المتابعة", "ارسلت المتابعة", "تم إرسال المتابعة", "تم ارسال المتابعة",
        "أرسلت رسالة متابعة", "ارسلت رسالة متابعة", "تابعت مع المورد",
        "بانتظار المورد", "ننتظر المورد", "انتظار المورد",
        "المورد طلب", "طلب المورد", "المورد رد", "رد المورد", "وصل رد المورد",
        "المورد ينتظرنا", "بانتظارنا",
        # English
        "sent the follow-up", "sent a follow-up", "follow-up sent", "follow up sent",
        "followed up with", "waiting for the supplier", "waiting on the supplier",
        "awaiting the supplier", "supplier asked for", "supplier requested",
        "supplier replied", "supplier responded", "received the supplier reply",
        "supplier is waiting for us", "waiting on us", "we need to reply",
        # German
        "nachfassnachricht gesendet", "nachricht nachgefasst", "nachgefasst bei",
        "warten auf den lieferanten", "warte auf den lieferanten",
        "lieferant hat um", "lieferant bat um", "lieferant hat geantwortet",
        "antwort vom lieferanten erhalten", "lieferant wartet auf uns",
        "wir müssen antworten", "wir muessen antworten",
    )
    return any(signal in folded for signal in signals)


def _extract_deal_followup_outcome_supplier_query(message: str):
    patterns = (
        r"\bwith\s+(?:supplier\s+)?(.+?)(?=[,;.]\s*|\s+(?:and|then|who|which)\b|$)",
        r"\bfor\s+supplier\s+(.+?)(?=[,;.]\s*|\s+(?:and|then|who|which)\b|$)",
        r"\bmit\s+(?:dem\s+)?lieferanten\s+(.+?)(?=[,;.]\s*|\s+(?:und|dann|der|die)\b|$)",
        r"(?:مع\s+المورد|مع)\s+(.+?)(?=[،,;.]\s*|\s+(?:ثم|وهو|والذي|و)\b|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, message or "", flags=re.IGNORECASE)
        if match:
            value = _clean_value(match.group(1))
            if value and value.casefold() not in {
                "the supplier", "supplier", "المورد", "هذا المورد",
                "dem lieferanten", "lieferant",
            }:
                return value
    return None


def _extract_deal_followup_outcome_request(message: str):
    if not _is_deal_followup_outcome_request(message):
        return None

    folded = (message or "").casefold()
    ascii_message = _deal_followup_ascii_digits(message or "")
    id_match = re.search(
        r"(?:الصفقة|صفقة|deal)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
        ascii_message,
        flags=re.IGNORECASE,
    )

    due = _extract_deal_followup_date(message)

    sent = any(x in folded for x in (
        "أرسلت المتابعة", "ارسلت المتابعة", "تم إرسال المتابعة", "تم ارسال المتابعة",
        "أرسلت رسالة متابعة", "ارسلت رسالة متابعة", "تابعت مع المورد",
        "sent the follow-up", "sent a follow-up", "follow-up sent", "follow up sent",
        "followed up with", "nachfassnachricht gesendet", "nachricht nachgefasst",
        "nachgefasst bei",
    ))

    waiting_supplier = any(x in folded for x in (
        "بانتظار المورد", "ننتظر المورد", "انتظار المورد",
        "waiting for the supplier", "waiting on the supplier", "awaiting the supplier",
        "warten auf den lieferanten", "warte auf den lieferanten",
    ))

    waiting_user = any(x in folded for x in (
        "المورد ينتظرنا", "بانتظارنا",
        "supplier is waiting for us", "waiting on us", "we need to reply",
        "lieferant wartet auf uns", "wir müssen antworten", "wir muessen antworten",
    ))

    supplier_replied = any(x in folded for x in (
        "المورد رد", "رد المورد", "وصل رد المورد",
        "supplier replied", "supplier responded", "received the supplier reply",
        "lieferant hat geantwortet", "antwort vom lieferanten erhalten",
    ))

    supplier_requested_time = any(x in folded for x in (
        "المورد طلب", "طلب المورد",
        "supplier asked for", "supplier requested",
        "lieferant hat um", "lieferant bat um",
    ))

    waiting_on = None
    if waiting_user:
        waiting_on = "user"
    elif waiting_supplier or sent or supplier_requested_time:
        waiting_on = "supplier"
    elif supplier_replied:
        waiting_on = "user"

    next_action = None
    if waiting_user or supplier_replied:
        next_action = "Review supplier response and reply"
    elif due and (sent or waiting_supplier or supplier_requested_time):
        next_action = "Follow up with supplier if no reply"
    elif sent or waiting_supplier:
        next_action = "Wait for supplier reply"

    if sent:
        event_type = "followup_sent"
    elif supplier_requested_time:
        event_type = "supplier_requested_time"
    elif supplier_replied:
        event_type = "supplier_reply_received"
    elif waiting_supplier:
        event_type = "waiting_supplier"
    elif waiting_user:
        event_type = "waiting_user"
    elif due:
        event_type = "followup_rescheduled"
    else:
        event_type = "followup_outcome"

    cleaned = " ".join((message or "").strip().split())
    if len(cleaned) > 500:
        cleaned = cleaned[:497] + "..."

    return {
        "deal_id": int(id_match.group(1)) if id_match else None,
        "supplier_query": _extract_deal_followup_outcome_supplier_query(message),
        "next_action_due": due.isoformat() if due else None,
        "waiting_on": waiting_on,
        "next_action": next_action,
        "event_type": event_type,
        "event_summary": f"User-stated follow-up outcome: {cleaned}",
    }


async def _capture_deal_followup_outcome(user_id: str, message: str):
    request = _extract_deal_followup_outcome_request(message)
    if not request:
        return None

    deal = None

    if request["deal_id"] is not None:
        deal = await get_commercial_deal_by_id(user_id, request["deal_id"])
        if not deal:
            return "Follow-up outcome not recorded: deal #%s was not found." % request["deal_id"]
        if not deal.get("is_active"):
            return "Follow-up outcome not recorded: deal #%s is not active." % request["deal_id"]

    elif request["supplier_query"]:
        supplier, error = await _resolve_management_supplier(
            user_id, request["supplier_query"]
        )
        if error:
            return error.replace("Commercial management", "Follow-up outcome")

        deals = await get_commercial_deals(
            user_id,
            active_only=True,
            supplier_id=supplier["id"],
            limit=10,
        )
        if not deals:
            return (
                "Follow-up outcome not recorded: no active deal exists for supplier "
                f"{supplier['name']}."
            )
        if len(deals) > 1:
            return (
                "Follow-up outcome not recorded: multiple active deals exist for supplier "
                f"{supplier['name']}; specify the deal ID."
            )
        deal = deals[0]

    else:
        deals = await get_commercial_deals(user_id, active_only=True, limit=20)
        if not deals:
            return "Follow-up outcome not recorded: there is no active deal."
        if len(deals) > 1:
            return (
                "Follow-up outcome not recorded: multiple active deals exist; "
                "specify the deal ID or supplier."
            )
        deal = deals[0]

    changes = {}

    if (
        request["waiting_on"] is not None
        and request["waiting_on"] != deal.get("waiting_on")
    ):
        changes["waiting_on"] = request["waiting_on"]

    if (
        request["next_action"] is not None
        and request["next_action"] != deal.get("next_action")
    ):
        changes["next_action"] = request["next_action"]

    if (
        request["next_action_due"] is not None
        and request["next_action_due"] != deal.get("next_action_due")
    ):
        changes["next_action_due"] = request["next_action_due"]

    if changes:
        changed = await update_commercial_deal(
            user_id,
            deal["id"],
            waiting_on=changes.get("waiting_on"),
            next_action=changes.get("next_action"),
            next_action_due=changes.get("next_action_due"),
        )
        if not changed:
            return "Follow-up outcome not recorded: deal update failed."

    await add_commercial_deal_event(
        user_id,
        deal["id"],
        request["event_type"],
        request["event_summary"],
        source="user",
    )

    saved_parts = [f"event={request['event_type']}"]
    for key in ("waiting_on", "next_action", "next_action_due"):
        if key in changes:
            saved_parts.append(f"{key}={changes[key]}")

    return (
        f"Follow-up outcome recorded: deal_id={deal['id']}; "
        f"supplier={deal.get('supplier')}; "
        + "; ".join(saved_parts)
        + ". No new deal was created. "
        "A saved next_action_due is internal deal tracking only; it does not schedule an external notification."
    )


async def _capture_deal_followup_due(user_id: str, message: str):
    request = _extract_deal_followup_request(message)
    if not request:
        return None

    deal = None

    if request["deal_id"] is not None:
        deal = await get_commercial_deal_by_id(user_id, request["deal_id"])
        if not deal:
            return "Deal follow-up date not saved: deal #%s was not found." % request["deal_id"]
        if not deal.get("is_active"):
            return "Deal follow-up date not saved: deal #%s is not active." % request["deal_id"]

    elif request["supplier_query"]:
        supplier, error = await _resolve_management_supplier(user_id, request["supplier_query"])
        if error:
            return error.replace("Commercial management", "Deal follow-up")
        deals = await get_commercial_deals(
            user_id,
            active_only=True,
            supplier_id=supplier["id"],
            limit=10,
        )
        if not deals:
            return "Deal follow-up date not saved: no active deal exists for supplier %s." % supplier["name"]
        if len(deals) > 1:
            return (
                "Deal follow-up date not saved: multiple active deals exist for "
                "supplier %s; specify the deal ID." % supplier["name"]
            )
        deal = deals[0]

    else:
        deals = await get_commercial_deals(user_id, active_only=True, limit=20)
        if not deals:
            return "Deal follow-up date not saved: there is no active deal."
        if len(deals) > 1:
            return (
                "Deal follow-up date not saved: multiple active deals exist; "
                "specify the deal ID or supplier."
            )
        deal = deals[0]

    changed = await update_commercial_deal(
        user_id,
        deal["id"],
        next_action_due=request["next_action_due"],
    )
    if not changed:
        return "Deal follow-up date not saved: deal update failed."

    return (
        f"Deal follow-up due date saved: deal_id={deal['id']}; "
        f"supplier={deal.get('supplier')}; "
        f"next_action_due={request['next_action_due']}. "
        "This records an internal deal-tracking due date; "
        "it does not schedule an external notification."
    )




def _acceptance_guard_deal_id(message: str):
    m = re.search(
        r"(?:deal|الصفقة|صفقة)\s*(?:id|رقم|#)?\s*[:#]?\s*(\d+)",
        _deal_followup_ascii_digits(message or ""),
        flags=re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


def _acceptance_guard_offer_id(message: str):
    m = re.search(
        r"(?:offer|العرض|عرض)\s*(?:id|رقم|#)?\s*[:#]?\s*(\d+)",
        _deal_followup_ascii_digits(message or ""),
        flags=re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


def _is_explicit_offer_acceptance_request(message: str) -> bool:
    folded = (message or "").casefold()

    strong = (
        "accept the offer", "accept offer", "approve the offer", "approve offer",
        "i accept the offer", "we accept the offer", "record my acceptance",
        "اقبل العرض", "أقبل العرض", "وافق على العرض", "أوافق على العرض",
        "اعتمد العرض", "أعتمد العرض", "سجل قبولي", "سجّل قبولي",
        "angebot annehmen", "ich nehme das angebot an", "angebot akzeptieren",
        "annahme bestätigen", "annahme bestaetigen",
    )
    return any(x in folded for x in strong)


def _is_explicit_deal_close_request(message: str) -> bool:
    folded = (message or "").casefold()
    strong = (
        "close deal", "close the deal", "mark deal", "mark the deal",
        "complete deal", "complete the deal", "mark as completed",
        "close as completed", "record the deal as completed",
        "اغلق الصفقة", "أغلق الصفقة", "اقفل الصفقة", "أقفل الصفقة",
        "انه الصفقة", "أنهِ الصفقة", "انهي الصفقة", "أنهي الصفقة",
        "سجل الصفقة مكتملة", "سجّل الصفقة مكتملة", "اعتبر الصفقة مكتملة",
        "deal abschließen", "deal abschliessen", "deal als abgeschlossen markieren",
    )
    return any(x in folded for x in strong)


async def _resolve_acceptance_guard_deal(
    user_id: str,
    message: str,
    *,
    require_explicit_id: bool = False,
):
    deal_id = _acceptance_guard_deal_id(message)

    if deal_id is not None:
        deal = await get_commercial_deal_by_id(user_id, deal_id)
        if not deal:
            return None, f"Commercial approval not applied: deal #{deal_id} was not found."
        return deal, None

    if require_explicit_id:
        return None, (
            "Deal closure not applied: specify the deal ID explicitly, for example "
            "'Close deal #4 as completed'."
        )

    deals = await get_commercial_deals(user_id, active_only=True, limit=20)
    if not deals:
        return None, "Commercial approval not applied: there is no active deal."
    if len(deals) > 1:
        return None, (
            "Commercial approval not applied: multiple active deals exist; specify the deal ID."
        )
    return deals[0], None


async def _has_current_offer_acceptance_event(user_id: str, deal: dict) -> bool:
    offer_id = deal.get("offer_id")
    if offer_id is None:
        return False

    marker = f"offer_id={int(offer_id)}"
    events = await get_commercial_deal_events(
        user_id,
        int(deal["id"]),
        limit=200,
    )
    return any(
        event.get("event_type") == "offer_acceptance_approved"
        and marker in str(event.get("summary") or "")
        for event in events
    )


async def _capture_acceptance_and_closing_guardrails(user_id: str, message: str):
    explicit_accept = _is_explicit_offer_acceptance_request(message)
    explicit_close = _is_explicit_deal_close_request(message)

    if not explicit_accept and not explicit_close:
        return None

    if explicit_close:
        deal, error = await _resolve_acceptance_guard_deal(
            user_id,
            message,
            require_explicit_id=True,
        )
        if error:
            return "Deal closing guardrail: " + error

        if not deal.get("is_active"):
            return (
                "Deal closing guardrail: deal #%s is already inactive/closed."
                % deal["id"]
            )

        if not await _has_current_offer_acceptance_event(user_id, deal):
            return (
                "Deal closing guardrail: closure blocked. The current saved offer "
                f"ID {deal.get('offer_id')} does not have an explicit recorded acceptance. "
                "Accept the current offer explicitly first; the deal remains open."
            )

        changed = await update_commercial_deal(
            user_id,
            int(deal["id"]),
            status="completed",
            waiting_on="none",
            next_action="No further action",
            clear_next_action_due=bool(deal.get("next_action_due")),
        )
        if not changed:
            return "Deal closing guardrail: deal closure failed; no status change was recorded."

        await add_commercial_deal_event(
            user_id,
            int(deal["id"]),
            "deal_closed_explicitly",
            (
                f"Explicit deal closure approved by user; "
                f"offer_id={deal.get('offer_id')}; status=completed"
            ),
            source="user",
        )
        return (
            "Deal closing guardrail: deal closed explicitly; "
            f"deal_id={deal['id']}; offer_id={deal.get('offer_id')}; status=completed. "
            "This was an internal deal-status change only; no supplier message was sent."
        )

    deal, error = await _resolve_acceptance_guard_deal(
        user_id,
        message,
        require_explicit_id=False,
    )
    if error:
        return "Offer acceptance guardrail: " + error

    if not deal.get("is_active"):
        return (
            "Offer acceptance guardrail: acceptance not recorded because deal "
            f"#{deal['id']} is inactive/closed."
        )

    latest_offer_id = deal.get("offer_id")
    if latest_offer_id is None:
        return (
            "Offer acceptance guardrail: acceptance not recorded because the deal "
            "does not point to a saved current offer."
        )

    requested_offer_id = _acceptance_guard_offer_id(message)
    if requested_offer_id is not None and int(requested_offer_id) != int(latest_offer_id):
        return (
            "Offer acceptance guardrail: acceptance blocked because the requested "
            f"offer ID {requested_offer_id} is not the deal's current saved offer "
            f"ID {latest_offer_id}."
        )

    guidance = await _commercial_offer_decision_guidance(
        user_id,
        supplier_id=int(deal["supplier_id"]),
        newest_offer_id=int(latest_offer_id),
    )
    if not guidance or guidance.get("decision") != "ACCEPT":
        decision = (guidance or {}).get("decision") or "UNAVAILABLE"
        return (
            "Offer acceptance guardrail: acceptance blocked. The current saved offer "
            f"is not in ACCEPT guidance state (current guidance={decision}). "
            "Review the missing/comparability issues first."
        )

    if await _has_current_offer_acceptance_event(user_id, deal):
        return (
            "Offer acceptance guardrail: the current offer is already explicitly "
            f"accepted internally; deal_id={deal['id']}; offer_id={latest_offer_id}. "
            "The deal is still not closed."
        )

    await add_commercial_deal_event(
        user_id,
        int(deal["id"]),
        "offer_acceptance_approved",
        (
            f"Explicit user acceptance recorded; offer_id={int(latest_offer_id)}; "
            f"decision=ACCEPT; previous_offer_id={guidance.get('previous_offer_id')}"
        ),
        source="user",
    )

    await update_commercial_deal(
        user_id,
        int(deal["id"]),
        waiting_on="user",
        next_action="Proceed to order / execution handoff",
        clear_next_action_due=bool(deal.get("next_action_due")),
    )

    return (
        "Offer acceptance guardrail: explicit acceptance recorded internally; "
        f"deal_id={deal['id']}; offer_id={latest_offer_id}. "
        "The deal remains open and is not marked completed. "
        "No supplier message was sent. Next step: order / execution handoff."
    )



def _is_explicit_order_execution_handoff_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "start order handoff",
        "start the order handoff",
        "proceed to order handoff",
        "proceed with order handoff",
        "start order execution",
        "start the order execution",
        "proceed to order execution",
        "proceed with order execution",
        "begin order execution",
        "begin the order execution",
        "move to awaiting pi",
        "move the deal to awaiting pi",
        "start awaiting pi",
        "ابدأ تسليم الطلب للتنفيذ",
        "ابدأ تحويل الطلب للتنفيذ",
        "ابدأ تنفيذ الطلب",
        "ابدأ مرحلة التنفيذ",
        "انتقل إلى تنفيذ الطلب",
        "انتقل الى تنفيذ الطلب",
        "انتقل إلى انتظار الفاتورة المبدئية",
        "انتقل الى انتظار الفاتورة المبدئية",
        "starte die bestellübergabe",
        "starte die bestelluebergabe",
        "mit der bestellabwicklung beginnen",
        "bestellabwicklung starten",
        "zur auftragsausführung übergehen",
        "zur auftragsausfuehrung uebergehen",
    )
    return any(x in folded for x in signals)


async def _has_order_execution_handoff_event(user_id: str, deal_id: int) -> bool:
    events = await get_commercial_deal_events(user_id, int(deal_id), limit=200)
    return any(
        event.get("event_type") == "order_execution_handoff_started"
        for event in events
    )


async def _capture_order_execution_handoff(user_id: str, message: str):
    if not _is_explicit_order_execution_handoff_request(message):
        return None

    deal_id = _acceptance_guard_deal_id(message)
    if deal_id is None:
        return (
            "Order execution handoff guardrail: handoff not started. "
            "Specify the deal ID explicitly, for example "
            "'Start order execution for deal #5'."
        )

    deal = await get_commercial_deal_by_id(user_id, int(deal_id))
    if not deal:
        return (
            "Order execution handoff guardrail: handoff not started because "
            f"deal #{deal_id} was not found."
        )

    if not deal.get("is_active"):
        return (
            "Order execution handoff guardrail: handoff not started because "
            f"deal #{deal_id} is inactive/closed."
        )

    offer_id = deal.get("offer_id")
    if offer_id is None:
        return (
            "Order execution handoff guardrail: handoff not started because "
            "the deal does not point to a saved current offer."
        )

    if not await _has_current_offer_acceptance_event(user_id, deal):
        return (
            "Order execution handoff guardrail: handoff blocked. "
            f"Current offer ID {offer_id} does not have an explicit recorded acceptance. "
            "Accept the current offer explicitly first."
        )

    if await _has_order_execution_handoff_event(user_id, int(deal_id)):
        return (
            "Order execution handoff guardrail: order / execution handoff "
            f"is already started for deal #{deal_id}; current status={deal.get('status')}. "
            "No duplicate handoff event was created."
        )

    changed = await update_commercial_deal(
        user_id,
        int(deal_id),
        status="awaiting_pi",
        waiting_on="supplier",
        next_action="Receive and review the supplier proforma invoice",
        clear_next_action_due=bool(deal.get("next_action_due")),
    )
    if not changed:
        return (
            "Order execution handoff guardrail: handoff could not be started "
            "because the deal update failed."
        )

    await add_commercial_deal_event(
        user_id,
        int(deal_id),
        "order_execution_handoff_started",
        (
            f"Order / execution handoff started explicitly by user; "
            f"offer_id={int(offer_id)}; status=awaiting_pi; "
            "waiting_on=supplier; next_action=Receive and review the supplier proforma invoice"
        ),
        source="user",
    )

    return (
        "Order execution handoff guardrail: order / execution handoff started internally; "
        f"deal_id={deal_id}; offer_id={offer_id}; status=awaiting_pi. "
        "Waiting on the supplier proforma invoice (PI). "
        "No purchase order, supplier message, payment, shipment release, or other external action was executed."
    )


async def _execution_stage_transition_allowed(user_id: str, deal: dict, target_status: str):
    execution_statuses = {"awaiting_pi", "production", "inspection", "ready_to_ship"}
    if target_status not in execution_statuses:
        return True, None

    if target_status == "awaiting_pi":
        return False, (
            "Execution-stage guardrail: awaiting_pi must be entered through the explicit "
            "order / execution handoff after the current offer has an explicit recorded acceptance."
        )

    if not await _has_order_execution_handoff_event(user_id, int(deal["id"])):
        return False, (
            "Execution-stage guardrail: execution stage change blocked because the "
            "order / execution handoff has not been started explicitly."
        )

    order = {
        "awaiting_pi": 0,
        "production": 1,
        "inspection": 2,
        "ready_to_ship": 3,
    }
    current_status = str(deal.get("status") or "")
    if current_status not in order:
        return False, (
            "Execution-stage guardrail: execution stage change blocked because the "
            f"current deal status is {current_status or 'unknown'}, not an active execution stage."
        )

    current_rank = order[current_status]
    target_rank = order[target_status]

    if target_rank > current_rank + 1:
        return False, (
            "Execution-stage guardrail: stage jump blocked. "
            f"Current status={current_status}; requested status={target_status}. "
            "Record the intermediate execution stage first."
        )

    if target_status == "production" and current_status == "awaiting_pi":
        if not await _has_current_pi_approval_event(user_id, deal):
            return False, (
                "Execution-stage guardrail: production transition blocked because the "
                "latest MATCH PI for the current accepted offer does not have an explicit "
                "recorded PI approval."
            )

    if target_status == "ready_to_ship" and current_status == "inspection":
        inspection_result = await _latest_current_inspection_result(user_id, deal)
        if inspection_result != "PASS":
            return False, (
                "Execution-stage guardrail: ready-to-ship transition blocked because the current "
                "offer and PI do not have a latest recorded inspection result of PASS."
            )

    if target_rank < current_rank:
        return False, (
            "Execution-stage guardrail: backward execution-stage change blocked. "
            f"Current status={current_status}; requested status={target_status}."
        )

    return True, None


def _extract_deal_status_by_id_command(message: str):
    raw = _deal_followup_ascii_digits(message or "")
    folded = raw.casefold()

    deal_match = re.search(
        r"(?:الصفقة|صفقة|deal)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
        raw,
        flags=re.IGNORECASE,
    )
    if not deal_match:
        return None

    explicit_change_signals = (
        "set deal", "change deal", "update deal", "status to", "status=",
        "غيّر حالة", "غير حالة", "تغيير حالة", "حدّث حالة", "حدث حالة",
        "حالة الصفقة", "status ändern", "status aendern", "status setzen",
    )
    if not any(signal in folded for signal in explicit_change_signals):
        return None

    status_aliases = (
        ("awaiting_pi", ("awaiting_pi", "awaiting pi")),
        ("production", ("production", "الإنتاج", "الانتاج", "produktion")),
        ("inspection", ("inspection", "الفحص", "التفتيش", "inspektion")),
        ("ready_to_ship", ("ready_to_ship", "ready to ship", "جاهز للشحن", "جاهزة للشحن", "versandbereit")),
        ("completed", ("completed", "complete", "مكتملة", "مكتمل", "abgeschlossen")),
        ("cancelled", ("cancelled", "canceled", "ملغاة", "ملغى", "storniert")),
    )

    target_status = None
    for canonical, aliases in status_aliases:
        if any(alias.casefold() in folded for alias in aliases):
            target_status = canonical
            break

    if target_status is None:
        return None

    return {"deal_id": int(deal_match.group(1)), "status": target_status}


async def _capture_deal_tracking_by_id(user_id: str, message: str):
    command = _extract_deal_status_by_id_command(message)
    if not command:
        return None

    deal_id = int(command["deal_id"])
    status = command["status"]

    deal = await get_commercial_deal_by_id(user_id, deal_id)
    if not deal:
        return f"Deal tracking not applied: deal #{deal_id} was not found."
    if not deal.get("is_active"):
        return f"Deal tracking not applied: deal #{deal_id} is inactive/closed."

    if status == "completed":
        return (
            "Deal closing guardrail: completion not applied through status tracking. "
            "Use the explicit deal-closing command after the current offer has an "
            "explicit recorded acceptance."
        )

    if status == "cancelled":
        return (
            "Deal tracking not applied: cancellation requires a separate explicit "
            "cancellation workflow; no status change was made."
        )

    execution_allowed, execution_error = await _execution_stage_transition_allowed(
        user_id, deal, status
    )
    if not execution_allowed:
        return execution_error

    waiting_next = {
        "production": ("supplier", "Monitor production progress and the agreed lead time"),
        "inspection": ("inspection", "Complete inspection and record the result"),
        "ready_to_ship": ("supplier", "Confirm shipping documents and shipment release"),
    }
    waiting_on, next_action = waiting_next.get(
        status, (deal.get("waiting_on"), deal.get("next_action"))
    )

    changed = await update_commercial_deal(
        user_id,
        deal_id,
        status=status,
        waiting_on=waiting_on,
        next_action=next_action,
    )
    if not changed:
        return (
            f"Deal tracking not applied: deal #{deal_id} could not be updated. "
            "No external action was executed."
        )

    return (
        f"Deal tracking updated: deal_id={deal_id}; status={status}; "
        f"waiting_on={waiting_on}; next_action={next_action}. "
        "No payment, supplier message, production command, shipment release, "
        "or other external action was executed."
    )

async def _capture_deal_tracking(user_id: str, message: str):
    command = _extract_deal_tracking_status(message)
    if not command:
        return None

    supplier, error = await _resolve_management_supplier(
        user_id, command["supplier_query"]
    )
    if error:
        return error.replace("Commercial management", "Deal tracking")

    active_deals = await get_commercial_deals(
        user_id,
        active_only=True,
        supplier_id=supplier["id"],
        limit=10,
    )
    status = command["status"]

    if status == "completed":
        return (
            "Deal closing guardrail: legacy completion wording was not applied. "
            "To close a deal, use an explicit command with the deal ID after the "
            "current offer has an explicit recorded acceptance, for example "
            "'Close deal #4 as completed'."
        )

    if active_deals:
        deal = active_deals[0]

        execution_allowed, execution_error = await _execution_stage_transition_allowed(
            user_id,
            deal,
            status,
        )
        if not execution_allowed:
            return execution_error

        await update_commercial_deal(
            user_id,
            deal["id"],
            status=status,
            waiting_on=command["waiting_on"],
            next_action=command["next_action"],
        )
        return (
            f"Deal tracking updated: deal_id={deal['id']}; "
            f"supplier={supplier['name']}; status={status}; "
            f"waiting_on={command['waiting_on']}; "
            f"next_action={command['next_action']}"
        )

    if status in {"completed", "cancelled"}:
        return (
            "Deal tracking not applied: no active deal exists for "
            f"supplier {supplier['name']}."
        )

    if status in {"awaiting_pi", "production", "inspection", "ready_to_ship"}:
        return (
            "Execution-stage guardrail: no active deal exists for this supplier. "
            "An execution-stage deal cannot be created directly; start from an existing deal "
            "and follow acceptance, order handoff, PI review, and PI approval in sequence."
        )

    latest = await get_latest_commercial_offer(user_id, supplier["id"])
    deal_id = await create_commercial_deal(
        user_id,
        supplier_id=supplier["id"],
        product_id=latest.get("product_id") if latest else None,
        offer_id=latest.get("id") if latest else None,
        title=(
            f"{latest.get('product')} - {supplier['name']}"
            if latest and latest.get("product")
            else f"Deal with {supplier['name']}"
        ),
        status=status,
        waiting_on=command["waiting_on"],
        next_action=command["next_action"],
    )
    return (
        f"Deal tracking created: deal_id={deal_id}; "
        f"supplier={supplier['name']}; status={status}; "
        f"waiting_on={command['waiting_on']}; "
        f"next_action={command['next_action']}"
    )



def _is_pi_review_text_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "review pi", "review the pi", "check pi", "check the pi", "verify pi", "verify the pi",
        "pi received", "received the pi", "received pi", "supplier sent the pi", "supplier sent pi",
        "proforma invoice received", "received the proforma invoice", "review proforma invoice",
        "check proforma invoice", "verify proforma invoice",
        "راجع pi", "راجع الـ pi", "راجع ال pi", "راجع الفاتورة المبدئية", "راجع الفاتورة الأولية",
        "استلمنا pi", "استلمت pi", "وصل pi", "وصلت الفاتورة المبدئية", "وصلت الفاتورة الأولية",
        "راجع proforma", "pi prüfen", "pi pruefen", "proforma-rechnung prüfen", "proforma-rechnung pruefen",
        "proforma rechnung prüfen", "proforma rechnung pruefen",
    )
    return any(signal in folded for signal in signals)


def _pi_normalize_text(value):
    if value is None:
        return None
    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w.%/+ -]+", "", text, flags=re.UNICODE)
    return text or None


def _pi_values_equal(saved, pi_value, *, numeric: bool = False) -> bool:
    if saved is None or pi_value is None:
        return False
    if numeric:
        try:
            left = float(saved)
            right = float(pi_value)
            tolerance = max(1e-6, abs(left) * 1e-6)
            return abs(left - right) <= tolerance
        except (TypeError, ValueError):
            pass
    return _pi_normalize_text(saved) == _pi_normalize_text(pi_value)


def _compare_pi_to_offer(offer: dict, pi: dict) -> dict:
    checks = (
        ("supplier", "supplier", False),
        ("product", "product", False),
        ("size", "size", False),
        ("thickness_mm", "thickness_mm", True),
        ("quantity", "quantity", True),
        ("price", "unit_price", True),
        ("currency", "currency", False),
        ("price_unit", "price_unit", False),
        ("incoterm", "incoterm", False),
        ("payment_terms", "payment_terms", False),
        ("valid_until", "valid_until", False),
        ("lead_time_days", "lead_time_days", True),
    )
    matched, mismatches, missing = [], [], []
    for offer_field, pi_field, numeric in checks:
        saved_value = offer.get(offer_field)
        if saved_value is None or str(saved_value).strip() == "":
            continue
        pi_value = pi.get(pi_field)
        if pi_value is None or str(pi_value).strip() == "":
            missing.append(pi_field)
            continue
        if _pi_values_equal(saved_value, pi_value, numeric=numeric):
            matched.append(pi_field)
        else:
            mismatches.append({"field": pi_field, "accepted_offer": saved_value, "pi": pi_value})
    result = "DISCREPANCIES" if mismatches else ("INCOMPLETE" if missing else "MATCH")
    return {"result": result, "matched": matched, "mismatches": mismatches, "missing": missing}


def _pi_review_next_action(result: str) -> str:
    if result == "MATCH":
        return "Review and explicitly approve the PI before any payment or production authorization"
    if result == "DISCREPANCIES":
        return "Resolve PI discrepancies before approval, payment, or production authorization"
    return "Resolve missing PI details before approval, payment, or production authorization"


async def _has_pi_review_fingerprint(user_id: str, deal_id: int, fingerprint: str) -> bool:
    events = await get_commercial_deal_events(user_id, int(deal_id), limit=200)
    marker = chr(34) + "fingerprint" + chr(34) + ":" + chr(34) + fingerprint + chr(34)
    return any(
        event.get("event_type") == "pi_review_recorded"
        and marker in str(event.get("summary") or "")
        for event in events
    )


async def _validate_pi_review_deal(user_id: str, deal_id: int):
    deal = await get_commercial_deal_by_id(user_id, int(deal_id))
    if not deal:
        return None, None, f"PI review guardrail: review not performed because deal #{deal_id} was not found."
    if not deal.get("is_active"):
        return None, None, f"PI review guardrail: review not performed because deal #{deal_id} is inactive/closed."
    if str(deal.get("status") or "") != "awaiting_pi":
        return None, None, (
            "PI review guardrail: review blocked because the deal is not currently in "
            f"awaiting_pi (current status={deal.get('status') or 'unknown'})."
        )
    if not await _has_order_execution_handoff_event(user_id, int(deal_id)):
        return None, None, (
            "PI review guardrail: review blocked because the explicit order / execution handoff has not been recorded."
        )
    if not await _has_current_offer_acceptance_event(user_id, deal):
        return None, None, (
            "PI review guardrail: review blocked because the deal's current saved offer does not have an explicit recorded acceptance."
        )
    offer_id = deal.get("offer_id")
    if offer_id is None:
        return None, None, "PI review guardrail: review blocked because the deal does not point to a saved current offer."
    offer = await get_commercial_offer_by_id(user_id, int(offer_id))
    if not offer:
        return None, None, f"PI review guardrail: review blocked because current offer #{offer_id} was not found."
    return deal, offer, None


def _pi_review_reply(deal: dict, offer: dict, pi: dict, comparison: dict) -> str:
    result = comparison["result"]
    parts = [
        "PI review guardrail: PI reviewed internally",
        f"for deal #{deal['id']} against accepted offer #{offer['id']}.",
        f"Result={result}.",
    ]
    if pi.get("pi_number"):
        parts.append(f"PI number: {pi['pi_number']}.")
    if comparison["mismatches"]:
        details = "; ".join(
            f"{item['field']} accepted={item['accepted_offer']} PI={item['pi']}"
            for item in comparison["mismatches"]
        )
        parts.append("Discrepancies: " + details + ".")
    if comparison["missing"]:
        parts.append("Missing/unverified in the PI: " + ", ".join(comparison["missing"]) + ".")
    if result == "MATCH":
        parts.append("The compared saved terms match, but this is not PI approval.")
    parts.append("Deal status remains awaiting_pi and is now waiting on you for the next decision.")
    parts.append(
        "No PI approval, payment, supplier message, production authorization, shipment release, or other external action was executed."
    )
    return " ".join(parts)


async def _record_pi_review(user_id: str, deal: dict, offer: dict, pi: dict, *, fingerprint: str, source: str):
    if await _has_pi_review_fingerprint(user_id, int(deal["id"]), fingerprint):
        return (
            "PI review guardrail: this exact PI input was already reviewed for "
            f"deal #{deal['id']}; no duplicate PI review event was created."
        )
    if not pi.get("document_is_pi"):
        return (
            "PI review guardrail: review not recorded because the supplied content could not be reliably identified as a proforma invoice (PI)."
        )
    comparison = _compare_pi_to_offer(offer, pi)
    next_action = _pi_review_next_action(comparison["result"])
    changed = await update_commercial_deal(
        user_id,
        int(deal["id"]),
        waiting_on="user",
        next_action=next_action,
        clear_next_action_due=bool(deal.get("next_action_due")),
    )
    if not changed:
        return (
            "PI review guardrail: the PI was extracted, but the deal tracking update failed; the PI review was not recorded."
        )
    event_payload = {
        "offer_id": int(offer["id"]),
        "fingerprint": fingerprint,
        "result": comparison["result"],
        "pi_number": pi.get("pi_number"),
        "mismatches": comparison["mismatches"],
        "missing": comparison["missing"],
    }
    await add_commercial_deal_event(
        user_id,
        int(deal["id"]),
        "pi_review_recorded",
        "PI review recorded; " + json.dumps(event_payload, ensure_ascii=False, separators=(",", ":")),
        source=source,
    )
    if comparison["result"] == "DISCREPANCIES":
        await add_commercial_deal_event(
            user_id,
            int(deal["id"]),
            "pi_discrepancy_detected",
            "PI discrepancies detected; " + json.dumps(
                {"offer_id": int(offer["id"]), "fingerprint": fingerprint, "mismatches": comparison["mismatches"]},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            source=source,
        )
    return _pi_review_reply(deal, offer, pi, comparison)


async def _capture_pi_review_text(user_id: str, message: str):
    if not _is_pi_review_text_request(message):
        return None
    deal_id = _acceptance_guard_deal_id(message)
    if deal_id is None:
        return (
            "PI review guardrail: review not performed. Specify the deal ID explicitly, for example 'Review the PI for deal #5'."
        )
    deal, offer, error = await _validate_pi_review_deal(user_id, int(deal_id))
    if error:
        return error
    if not OPENAI_API_KEY:
        return "PI review guardrail: review not performed because the OpenAI connection is not configured."
    fingerprint = "text:" + hashlib.sha256((message or "").encode("utf-8")).hexdigest()
    if await _has_pi_review_fingerprint(user_id, int(deal_id), fingerprint):
        return (
            "PI review guardrail: this exact PI input was already reviewed for "
            f"deal #{deal_id}; no duplicate PI review event was created."
        )
    from .agents import run_pi_review_text
    extracted = await run_pi_review_text(message)
    return await _record_pi_review(
        user_id, deal, offer, extracted.model_dump(), fingerprint=fingerprint, source="user_text"
    )



def _is_explicit_pi_approval_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "approve pi", "approve the pi", "approve proforma", "approve the proforma",
        "approve proforma invoice", "approve the proforma invoice", "confirm pi approval",
        "اعتمد pi", "اعتمد الـ pi", "اعتمد ال pi", "اعتمد الفاتورة المبدئية",
        "اعتمد الفاتورة الأولية", "وافق على pi", "وافق على الـ pi",
        "وافق على الفاتورة المبدئية", "وافق على الفاتورة الأولية",
        "pi freigeben", "pi genehmigen", "proforma-rechnung freigeben",
        "proforma-rechnung genehmigen", "proforma rechnung freigeben",
        "proforma rechnung genehmigen",
    )
    return any(signal in folded for signal in signals)


def _is_payment_execution_request(message: str) -> bool:
    folded = (message or "").casefold()
    signals = (
        "pay supplier", "pay the supplier", "make payment", "make the payment",
        "send payment", "send the payment", "pay deposit", "pay the deposit",
        "send deposit", "send the deposit", "transfer payment", "transfer the payment",
        "wire payment", "wire the payment", "wire the deposit",
        "تحويل الدفعة", "حول الدفعة", "حوّل الدفعة", "ادفع للمورد", "ادفع المورد",
        "ادفع العربون", "حول العربون", "حوّل العربون", "ارسل الدفعة", "أرسل الدفعة",
        "zahlung senden", "zahlung überweisen", "zahlung ueberweisen",
        "anzahlung senden", "anzahlung überweisen", "anzahlung ueberweisen",
        "lieferanten bezahlen",
    )
    return any(signal in folded for signal in signals)


def _parse_pi_review_event_summary(summary: str | None):
    prefix = "PI review recorded; "
    text = str(summary or "")
    if not text.startswith(prefix):
        return None
    try:
        payload = json.loads(text[len(prefix):])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


async def _latest_pi_review_record(user_id: str, deal_id: int):
    events = await get_commercial_deal_events(user_id, int(deal_id), limit=200)
    for event in reversed(events):
        if event.get("event_type") != "pi_review_recorded":
            continue
        payload = _parse_pi_review_event_summary(event.get("summary"))
        if payload:
            return event, payload
    return None, None


async def _has_pi_approval_for_fingerprint(user_id: str, deal_id: int, offer_id: int, fingerprint: str) -> bool:
    events = await get_commercial_deal_events(user_id, int(deal_id), limit=200)
    offer_marker = f"offer_id={int(offer_id)}"
    fingerprint_marker = f"fingerprint={fingerprint}"
    return any(
        event.get("event_type") == "pi_approval_approved"
        and offer_marker in str(event.get("summary") or "")
        and fingerprint_marker in str(event.get("summary") or "")
        for event in events
    )


async def _has_current_pi_approval_event(user_id: str, deal: dict) -> bool:
    offer_id = deal.get("offer_id")
    if offer_id is None:
        return False
    _, payload = await _latest_pi_review_record(user_id, int(deal["id"]))
    if not payload:
        return False
    try:
        reviewed_offer_id = int(payload.get("offer_id"))
    except (TypeError, ValueError):
        return False
    fingerprint = str(payload.get("fingerprint") or "").strip()
    if reviewed_offer_id != int(offer_id) or not fingerprint:
        return False
    if str(payload.get("result") or "").upper() != "MATCH":
        return False
    if payload.get("mismatches") or payload.get("missing"):
        return False
    return await _has_pi_approval_for_fingerprint(
        user_id, int(deal["id"]), int(offer_id), fingerprint
    )



def _extract_inspection_shipping_intent(message: str):
    raw = _deal_followup_ascii_digits(message or "")
    folded = raw.casefold()
    deal_id = _acceptance_guard_deal_id(raw)

    shipment_commands = (
        "release shipment", "release the shipment", "authorize shipment",
        "authorize shipping", "ship the goods", "ship the order",
        "tell the supplier to ship", "tell supplier to ship",
        "send shipment release", "shipment release",
        "versand freigeben", "sendung freigeben",
    )
    inspection_pass = (
        "inspection passed", "passed inspection", "inspection result pass",
        "inspection result passed", "qc passed", "quality inspection passed",
        "quality control passed", "inspektion bestanden",
    )
    inspection_fail = (
        "inspection failed", "failed inspection", "inspection result fail",
        "inspection result failed", "qc failed", "quality inspection failed",
        "quality control failed", "inspektion fehlgeschlagen",
    )
    ready_signals = (
        "supplier is ready to ship", "supplier ready to ship",
        "goods are ready to ship", "order is ready to ship",
        "shipment is ready to ship", "ready for shipment",
        "versandbereit",
    )

    if any(signal in folded for signal in shipment_commands):
        return {"kind": "shipment_command", "deal_id": int(deal_id) if deal_id is not None else None}
    if any(signal in folded for signal in inspection_pass):
        return {"kind": "inspection_result", "result": "PASS", "deal_id": int(deal_id) if deal_id is not None else None}
    if any(signal in folded for signal in inspection_fail):
        return {"kind": "inspection_result", "result": "FAIL", "deal_id": int(deal_id) if deal_id is not None else None}
    if any(signal in folded for signal in ready_signals):
        return {"kind": "ready_to_ship", "deal_id": int(deal_id) if deal_id is not None else None}
    return None


async def _latest_current_inspection_result(user_id: str, deal: dict):
    offer_id = deal.get("offer_id")
    if offer_id is None:
        return None
    _, pi_payload = await _latest_pi_review_record(user_id, int(deal["id"]))
    if not pi_payload:
        return None
    pi_fingerprint = str(pi_payload.get("fingerprint") or "").strip()
    if not pi_fingerprint:
        return None
    offer_marker = f"offer_id={int(offer_id)}"
    pi_marker = f"pi_fingerprint={pi_fingerprint}"
    events = await get_commercial_deal_events(user_id, int(deal["id"]), limit=200)
    for event in reversed(events):
        if event.get("event_type") != "inspection_result_recorded":
            continue
        summary = str(event.get("summary") or "")
        if offer_marker not in summary or pi_marker not in summary:
            continue
        if "result=PASS" in summary:
            return "PASS"
        if "result=FAIL" in summary:
            return "FAIL"
    return None


async def _has_current_ready_to_ship_event(user_id: str, deal: dict) -> bool:
    offer_id = deal.get("offer_id")
    if offer_id is None:
        return False
    _, pi_payload = await _latest_pi_review_record(user_id, int(deal["id"]))
    if not pi_payload:
        return False
    pi_fingerprint = str(pi_payload.get("fingerprint") or "").strip()
    if not pi_fingerprint:
        return False
    offer_marker = f"offer_id={int(offer_id)}"
    pi_marker = f"pi_fingerprint={pi_fingerprint}"
    events = await get_commercial_deal_events(user_id, int(deal["id"]), limit=200)
    return any(
        event.get("event_type") == "ready_to_ship_recorded"
        and offer_marker in str(event.get("summary") or "")
        and pi_marker in str(event.get("summary") or "")
        for event in events
    )


def _normalize_shipping_document_type(value) -> str:
    folded = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "commercial_invoice": "commercial_invoice",
        "invoice": "commercial_invoice",
        "packing_list": "packing_list",
        "packinglist": "packing_list",
        "bill_of_lading": "bill_of_lading",
        "billoflading": "bill_of_lading",
        "bl": "bill_of_lading",
        "b_l": "bill_of_lading",
        "certificate_of_origin": "certificate_of_origin",
        "certificateoforigin": "certificate_of_origin",
        "coo": "certificate_of_origin",
        "other_shipping_document": "other_shipping_document",
        "unknown": "unknown",
    }
    return aliases.get(folded, "unknown")


def _compare_shipping_document_to_offer(offer: dict, document: dict) -> dict:
    document_type = _normalize_shipping_document_type(document.get("document_type"))
    checks = (
        ("supplier", "supplier", False),
        ("product", "product", False),
        ("size", "size", False),
        ("thickness_mm", "thickness_mm", True),
        ("quantity", "quantity", True),
        ("price", "unit_price", True),
        ("currency", "currency", False),
        ("incoterm", "incoterm", False),
    )
    mismatches = []
    matched = []
    for offer_field, document_field, numeric in checks:
        saved_value = offer.get(offer_field)
        document_value = document.get(document_field)
        if saved_value is None or str(saved_value).strip() == "":
            continue
        if document_value is None or str(document_value).strip() == "":
            continue
        if _pi_values_equal(saved_value, document_value, numeric=numeric):
            matched.append(document_field)
        else:
            mismatches.append(
                {
                    "field": document_field,
                    "accepted_offer": saved_value,
                    "shipping_document": document_value,
                }
            )

    required_by_type = {
        "commercial_invoice": ("supplier", "product", "quantity", "unit_price", "currency"),
        "packing_list": ("product", "quantity"),
        "bill_of_lading": ("bill_of_lading_number",),
        "certificate_of_origin": ("product", "country_of_origin"),
        "other_shipping_document": ("document_number",),
    }
    required = required_by_type.get(document_type, ())
    missing = [
        field
        for field in required
        if document.get(field) is None or str(document.get(field)).strip() == ""
    ]
    result = "DISCREPANCIES" if mismatches else ("INCOMPLETE" if missing else "MATCH")
    return {
        "result": result,
        "document_type": document_type,
        "matched": matched,
        "mismatches": mismatches,
        "missing": missing,
    }


def _parse_shipping_document_review_event_summary(summary: str | None):
    prefix = "Shipping document review recorded; "
    text = str(summary or "")
    if not text.startswith(prefix):
        return None
    try:
        payload = json.loads(text[len(prefix):])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


async def _has_shipping_document_fingerprint(user_id: str, deal_id: int, fingerprint: str) -> bool:
    events = await get_commercial_deal_events(user_id, int(deal_id), limit=200)
    for event in events:
        if event.get("event_type") != "shipping_document_review_recorded":
            continue
        payload = _parse_shipping_document_review_event_summary(event.get("summary"))
        if payload and str(payload.get("fingerprint") or "") == str(fingerprint or ""):
            return True
    return False


async def _validate_shipping_document_review_deal(user_id: str, deal_id: int):
    deal = await get_commercial_deal_by_id(user_id, int(deal_id))
    if not deal:
        return None, None, None, f"Shipping documents: review not performed because deal #{deal_id} was not found."
    if not deal.get("is_active"):
        return None, None, None, f"Shipping documents: review not performed because deal #{deal_id} is inactive/closed."
    if str(deal.get("status") or "") != "ready_to_ship":
        return None, None, None, (
            "Shipping documents: review blocked because the deal is not currently ready_to_ship "
            f"(current status={deal.get('status') or 'unknown'})."
        )
    if not await _has_current_ready_to_ship_event(user_id, deal):
        return None, None, None, (
            "Shipping documents: review blocked because current ready-to-ship evidence is not recorded "
            "for this offer and PI."
        )
    if await _latest_current_inspection_result(user_id, deal) != "PASS":
        return None, None, None, (
            "Shipping documents: review blocked because the latest inspection result for the current "
            "offer and PI is not PASS."
        )
    if not await _has_current_pi_approval_event(user_id, deal):
        return None, None, None, (
            "Shipping documents: review blocked because the latest MATCH PI for the current accepted "
            "offer does not have explicit recorded approval."
        )
    offer_id = deal.get("offer_id")
    if offer_id is None:
        return None, None, None, "Shipping documents: review blocked because the deal has no current saved offer."
    offer = await get_commercial_offer_by_id(user_id, int(offer_id))
    if not offer:
        return None, None, None, f"Shipping documents: review blocked because current offer #{offer_id} was not found."
    _, pi_payload = await _latest_pi_review_record(user_id, int(deal_id))
    pi_fingerprint = str((pi_payload or {}).get("fingerprint") or "").strip()
    try:
        reviewed_offer_id = int((pi_payload or {}).get("offer_id"))
    except (TypeError, ValueError):
        reviewed_offer_id = None
    if reviewed_offer_id != int(offer_id) or not pi_fingerprint:
        return None, None, None, (
            "Shipping documents: review blocked because current offer / PI binding is unavailable."
        )
    return deal, offer, pi_fingerprint, None


def _shipping_document_next_action(result: str) -> str:
    if result == "MATCH":
        return "Review remaining shipping documents and keep shipment release as a separate explicit workflow"
    if result == "DISCREPANCIES":
        return "Resolve shipping document discrepancies before any shipment release"
    return "Resolve missing shipping document details before any shipment release"


async def _record_shipping_document_review(
    user_id: str,
    deal: dict,
    offer: dict,
    document: dict,
    *,
    pi_fingerprint: str,
    fingerprint: str,
    source: str,
):
    if await _has_shipping_document_fingerprint(user_id, int(deal["id"]), fingerprint):
        return (
            "Shipping documents: this exact shipping document file was already reviewed for "
            f"deal #{deal['id']}; no duplicate event was created."
        )
    if not document.get("document_is_shipping_document"):
        return (
            "Shipping documents: review not recorded because the supplied file could not be reliably "
            "identified as a shipping document."
        )
    document_type = _normalize_shipping_document_type(document.get("document_type"))
    if document_type == "unknown":
        return (
            "Shipping documents: review not recorded because the shipping document type could not be "
            "reliably identified."
        )
    comparison = _compare_shipping_document_to_offer(offer, document)
    next_action = _shipping_document_next_action(comparison["result"])
    changed = await update_commercial_deal(
        user_id,
        int(deal["id"]),
        waiting_on="user",
        next_action=next_action,
        clear_next_action_due=bool(deal.get("next_action_due")),
    )
    if not changed:
        return (
            "Shipping documents: the document was extracted, but deal tracking could not be updated; "
            "the document review was not recorded. No shipment release or other external action was executed."
        )
    event_payload = {
        "offer_id": int(offer["id"]),
        "pi_fingerprint": pi_fingerprint,
        "fingerprint": fingerprint,
        "document_type": document_type,
        "document_number": document.get("document_number"),
        "result": comparison["result"],
        "mismatches": comparison["mismatches"],
        "missing": comparison["missing"],
        "bill_of_lading_number": document.get("bill_of_lading_number"),
        "container_number": document.get("container_number"),
        "seal_number": document.get("seal_number"),
        "country_of_origin": document.get("country_of_origin"),
    }
    await add_commercial_deal_event(
        user_id,
        int(deal["id"]),
        "shipping_document_review_recorded",
        "Shipping document review recorded; "
        + json.dumps(event_payload, ensure_ascii=False, separators=(",", ":")),
        source=source,
    )
    parts = [
        f"Shipping documents: {document_type} reviewed internally for deal #{deal['id']}.",
        f"Result={comparison['result']}.",
    ]
    if document.get("document_number"):
        parts.append(f"Document number: {document['document_number']}.")
    if comparison["mismatches"]:
        details = "; ".join(
            f"{item['field']} accepted={item['accepted_offer']} document={item['shipping_document']}"
            for item in comparison["mismatches"]
        )
        parts.append("Discrepancies: " + details + ".")
    if comparison["missing"]:
        parts.append("Missing/unverified in this document: " + ", ".join(comparison["missing"]) + ".")
    parts.append("Deal status remains ready_to_ship.")
    parts.append(
        "This review does not release the shipment, contact the supplier, make a payment, or execute any external action."
    )
    return " ".join(parts)


def _extract_production_execution_intent(message: str):
    raw = _deal_followup_ascii_digits(message or "")
    folded = raw.casefold()
    deal_id = _acceptance_guard_deal_id(raw)

    external_commands = (
        "authorize production", "approve production", "start production", "begin production",
        "tell the supplier to start production", "tell supplier to start production",
        "send production authorization", "production authorization",
        "produktion freigeben", "produktion starten",
    )
    factual_signals = (
        "supplier started production", "supplier has started production",
        "production started", "production has started",
        "supplier confirmed production started", "supplier confirmed production has started",
        "produktion hat begonnen", "produktion wurde gestartet",
    )

    if any(signal in folded for signal in external_commands):
        return {"kind": "command", "deal_id": int(deal_id) if deal_id is not None else None}
    if any(signal in folded for signal in factual_signals):
        return {"kind": "record", "deal_id": int(deal_id) if deal_id is not None else None}
    return None


async def _has_current_production_started_event(user_id: str, deal: dict) -> bool:
    offer_id = deal.get("offer_id")
    if offer_id is None:
        return False
    _, pi_payload = await _latest_pi_review_record(user_id, int(deal["id"]))
    if not pi_payload:
        return False
    pi_fingerprint = str(pi_payload.get("fingerprint") or "").strip()
    if not pi_fingerprint:
        return False
    events = await get_commercial_deal_events(user_id, int(deal["id"]), limit=200)
    offer_marker = f"offer_id={int(offer_id)}"
    pi_marker = f"pi_fingerprint={pi_fingerprint}"
    return any(
        event.get("event_type") == "production_started_recorded"
        and offer_marker in str(event.get("summary") or "")
        and pi_marker in str(event.get("summary") or "")
        for event in events
    )


def _payment_record_number(value: str):
    if value is None:
        return None
    text = _deal_followup_ascii_digits(str(value)).replace(' ', '')
    if ',' in text and '.' in text:
        if text.rfind(',') > text.rfind('.'):
            text = text.replace('.', '').replace(',', '.')
        else:
            text = text.replace(',', '')
    elif text.count(',') == 1:
        left, right = text.split(',', 1)
        text = left + right if len(right) == 3 else left + '.' + right
    elif text.count('.') == 1:
        left, right = text.split('.', 1)
        text = left + right if len(right) == 3 else text
    elif text.count(',') > 1:
        text = text.replace(',', '')
    elif text.count('.') > 1:
        text = text.replace('.', '')
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _extract_external_payment_record(message: str):
    raw = _deal_followup_ascii_digits(message or '')
    folded = raw.casefold()
    past_signals = (
        'i paid', 'we paid', 'payment made', 'payment was made',
        'paid the supplier', 'paid supplier', 'deposit paid',
        'i sent the payment', 'we sent the payment',
        'دفعت', 'دفعنا', 'تم الدفع', 'تم دفع', 'قمنا بالدفع',
        'حولت', 'حوّلت', 'تم التحويل', 'تم تحويل',
        'ich habe bezahlt', 'wir haben bezahlt', 'zahlung erfolgt',
        'zahlung wurde', 'anzahlung bezahlt', 'überwiesen', 'ueberwiesen',
    )
    if not any(x in folded for x in past_signals):
        return None

    deal_id = _acceptance_guard_deal_id(raw)
    if deal_id is None:
        return {'error': 'Payment record: not recorded. Specify the deal ID explicitly.'}

    percent = None
    pm = re.search(
        r'([0-9]+(?:[.,][0-9]+)?)\s*(?:%|percent\b|prozent\b|بالمئة|بالمائة|في\s+المئة)',
        raw, flags=re.IGNORECASE,
    )
    if pm:
        percent = _commercial_number(pm.group(1))
        if percent is None or not (0 < percent <= 100):
            return {'error': 'Payment record: not recorded because the stated payment percentage is outside 0-100%.'}

    amount = None
    currency = None
    currency_token = r'USD|US\$|\$|EUR|€|CNY|RMB|دولار(?:\s+أمريكي)?|يورو|يوان'
    number_token = r'[0-9]+(?:[.,][0-9]+)*'
    m1 = re.search(rf'({currency_token})\s*({number_token})', raw, flags=re.IGNORECASE)
    m2 = re.search(rf'({number_token})\s*({currency_token})', raw, flags=re.IGNORECASE)
    if m1 or m2:
        if m1:
            currency_raw, amount_raw = m1.group(1), m1.group(2)
        else:
            amount_raw, currency_raw = m2.group(1), m2.group(2)
        amount = _payment_record_number(amount_raw)
        currency = _extract_management_currency(currency_raw)
        if amount is None or amount <= 0:
            return {'error': 'Payment record: not recorded because the stated payment amount is invalid.'}

    if amount is None and percent is None:
        return {'error': ('Payment record: not recorded. State the payment amount and currency '
                          'and/or the paid percentage explicitly; nothing will be inferred from payment terms.')}

    fingerprint = 'text:' + hashlib.sha256((message or '').strip().encode('utf-8')).hexdigest()
    return {'deal_id': int(deal_id), 'amount': amount, 'currency': currency,
            'percent': percent, 'fingerprint': fingerprint}


async def _has_payment_record_fingerprint(user_id: str, deal_id: int, fingerprint: str) -> bool:
    events = await get_commercial_deal_events(user_id, int(deal_id), limit=200)
    marker = f'fingerprint={fingerprint}'
    return any(event.get('event_type') == 'payment_recorded'
               and marker in str(event.get('summary') or '') for event in events)


async def _capture_inspection_shipping(user_id: str, message: str):
    intent = _extract_inspection_shipping_intent(message)
    if not intent:
        return None

    deal_id = intent.get("deal_id")
    if deal_id is None:
        return "Inspection & shipping: action not applied. Specify the deal ID explicitly."

    deal = await get_commercial_deal_by_id(user_id, int(deal_id))
    if not deal:
        return f"Inspection & shipping: action not applied because deal #{deal_id} was not found."
    if not deal.get("is_active"):
        return f"Inspection & shipping: action not applied because deal #{deal_id} is inactive/closed."

    kind = intent.get("kind")

    if kind == "shipment_command":
        inspection_result = await _latest_current_inspection_result(user_id, deal)
        if inspection_result != "PASS":
            return (
                "Inspection & shipping: shipment release blocked because the current offer and PI "
                "do not have a latest recorded inspection result of PASS. "
                "No shipment release or supplier message was sent."
            )
        current_status = str(deal.get("status") or "")
        if current_status != "ready_to_ship":
            return (
                "Inspection & shipping: shipment release blocked because the deal is not currently "
                f"ready_to_ship (current status={current_status or 'unknown'}). "
                "No external action was executed."
            )
        return (
            "Inspection & shipping: no external shipment release was executed. "
            "Lio does not release shipments or send supplier shipping commands in this stage; "
            "a separately verified shipment-release workflow is required."
        )

    if kind == "inspection_result":
        result = str(intent.get("result") or "").upper()
        current_status = str(deal.get("status") or "")
        if current_status not in {"production", "inspection"}:
            return (
                "Inspection & shipping: inspection result not recorded because the deal is not "
                f"in production or inspection (current status={current_status or 'unknown'})."
            )

        latest_result = await _latest_current_inspection_result(user_id, deal)
        if latest_result == result:
            return (
                f"Inspection & shipping: inspection result {result} is already the latest recorded "
                f"result for deal #{deal_id}; no duplicate event was created."
            )

        allowed, error = await _execution_stage_transition_allowed(user_id, deal, "inspection")
        if not allowed:
            detail = str(error or "").replace("Execution-stage guardrail:", "").strip()
            return "Inspection & shipping: inspection result not recorded. " + detail

        _, pi_payload = await _latest_pi_review_record(user_id, int(deal_id))
        pi_fingerprint = str((pi_payload or {}).get("fingerprint") or "").strip()
        offer_id = deal.get("offer_id")
        if offer_id is None or not pi_fingerprint:
            return (
                "Inspection & shipping: inspection result not recorded because current "
                "offer / PI binding is unavailable."
            )

        next_action = (
            "Confirm goods are ready to ship and verify shipping documents"
            if result == "PASS"
            else "Resolve inspection failures / corrective action before shipment readiness"
        )
        changed = await update_commercial_deal(
            user_id,
            int(deal_id),
            status="inspection",
            waiting_on="supplier",
            next_action=next_action,
            clear_next_action_due=bool(deal.get("next_action_due")),
        )
        if not changed:
            return (
                "Inspection & shipping: inspection result not recorded because deal tracking "
                "could not be updated. No supplier message, shipment release, payment, or "
                "other external action was executed."
            )

        await add_commercial_deal_event(
            user_id,
            int(deal_id),
            "inspection_result_recorded",
            (
                "Inspection result recorded; "
                f"offer_id={int(offer_id)}; pi_fingerprint={pi_fingerprint}; "
                f"result={result}; status=inspection"
            ),
            source="user",
        )
        return (
            f"Inspection & shipping: inspection result {result} recorded internally for "
            f"deal #{deal_id}; status=inspection. This records the reported result only. "
            "Lio did not contact the supplier, release a shipment, or execute any external action."
        )

    if kind == "ready_to_ship":
        if await _has_current_ready_to_ship_event(user_id, deal):
            return (
                f"Inspection & shipping: ready-to-ship status is already recorded for "
                f"deal #{deal_id} for the current offer and PI; no duplicate event was created."
            )

        inspection_result = await _latest_current_inspection_result(user_id, deal)
        if inspection_result != "PASS":
            return (
                "Inspection & shipping: ready-to-ship status not recorded because the latest "
                "inspection result for the current offer and PI is not PASS."
            )

        allowed, error = await _execution_stage_transition_allowed(
            user_id, deal, "ready_to_ship"
        )
        if not allowed:
            detail = str(error or "").replace("Execution-stage guardrail:", "").strip()
            return "Inspection & shipping: ready-to-ship status not recorded. " + detail

        _, pi_payload = await _latest_pi_review_record(user_id, int(deal_id))
        pi_fingerprint = str((pi_payload or {}).get("fingerprint") or "").strip()
        offer_id = deal.get("offer_id")
        if offer_id is None or not pi_fingerprint:
            return (
                "Inspection & shipping: ready-to-ship status not recorded because current "
                "offer / PI binding is unavailable."
            )

        changed = await update_commercial_deal(
            user_id,
            int(deal_id),
            status="ready_to_ship",
            waiting_on="user",
            next_action=(
                "Verify shipping documents and keep shipment release as a separate explicit workflow"
            ),
            clear_next_action_due=bool(deal.get("next_action_due")),
        )
        if not changed:
            return (
                "Inspection & shipping: ready-to-ship status not recorded because deal tracking "
                "could not be updated. No shipment release, supplier message, payment, or other "
                "external action was executed."
            )

        await add_commercial_deal_event(
            user_id,
            int(deal_id),
            "ready_to_ship_recorded",
            (
                "Supplier-reported ready-to-ship status recorded; "
                f"offer_id={int(offer_id)}; pi_fingerprint={pi_fingerprint}; "
                "inspection_result=PASS; status=ready_to_ship"
            ),
            source="user",
        )
        return (
            f"Inspection & shipping: supplier-reported ready-to-ship status recorded internally "
            f"for deal #{deal_id}; status=ready_to_ship. No shipment was released and no "
            "supplier shipping command was sent."
        )

    return None


async def _capture_production_execution(user_id: str, message: str):
    intent = _extract_production_execution_intent(message)
    if not intent:
        return None

    deal_id = intent.get("deal_id")
    if deal_id is None:
        return "Production execution: action not applied. Specify the deal ID explicitly."

    deal = await get_commercial_deal_by_id(user_id, int(deal_id))
    if not deal:
        return f"Production execution: action not applied because deal #{deal_id} was not found."
    if not deal.get("is_active"):
        return f"Production execution: action not applied because deal #{deal_id} is inactive/closed."

    if intent.get("kind") == "command":
        if not await _has_current_pi_approval_event(user_id, deal):
            return (
                "Production execution: external production authorization blocked because the latest "
                "MATCH PI for the current accepted offer does not have an explicit recorded PI approval. "
                "No supplier message or production command was sent."
            )
        return (
            "Production execution: no external production authorization was executed. "
            "Lio does not send supplier production commands in this stage; a separately verified "
            "external production-authorization workflow is required."
        )

    if await _has_current_production_started_event(user_id, deal):
        return (
            f"Production execution: production start is already recorded for deal #{deal_id} "
            "for the current offer and PI; no duplicate event was created."
        )

    allowed, error = await _execution_stage_transition_allowed(user_id, deal, "production")
    if not allowed:
        detail = str(error or "").replace("Execution-stage guardrail:", "").strip()
        return "Production execution: production start not recorded. " + detail

    _, pi_payload = await _latest_pi_review_record(user_id, int(deal_id))
    pi_fingerprint = str((pi_payload or {}).get("fingerprint") or "").strip()
    offer_id = deal.get("offer_id")

    changed = await update_commercial_deal(
        user_id,
        int(deal_id),
        status="production",
        waiting_on="supplier",
        next_action="Monitor production progress and the agreed lead time",
        clear_next_action_due=bool(deal.get("next_action_due")),
    )
    if not changed:
        return (
            "Production execution: production start not recorded because deal tracking could not be updated. "
            "No supplier message, production command, payment, or other external action was executed."
        )

    await add_commercial_deal_event(
        user_id,
        int(deal_id),
        "production_started_recorded",
        (
            "Supplier-reported production start recorded; "
            f"offer_id={int(offer_id)}; pi_fingerprint={pi_fingerprint}; status=production"
        ),
        source="user",
    )

    return (
        f"Production execution: supplier-reported production start recorded internally for deal #{deal_id}; "
        "status=production. This records the reported external fact only. Lio did not send a supplier "
        "message, authorize or start production externally, execute a payment, or release a shipment."
    )


async def _capture_external_payment_record(user_id: str, message: str):
    record = _extract_external_payment_record(message)
    if not record:
        return None
    if record.get('error'):
        return record['error']

    deal_id = int(record['deal_id'])
    deal = await get_commercial_deal_by_id(user_id, deal_id)
    if not deal:
        return f'Payment record: not recorded because deal #{deal_id} was not found.'
    if not deal.get('is_active'):
        return f'Payment record: not recorded because deal #{deal_id} is inactive/closed.'

    if str(deal.get('status') or '') not in {'awaiting_pi', 'production', 'inspection', 'ready_to_ship'}:
        return ('Payment record: not recorded because the deal is not in an active execution stage '
                f"(current status={deal.get('status') or 'unknown'}).")

    if not await _has_current_pi_approval_event(user_id, deal):
        return ('Payment record: not recorded because the latest MATCH PI for the current '
                'accepted offer does not have an explicit recorded PI approval.')

    fingerprint = record['fingerprint']
    if await _has_payment_record_fingerprint(user_id, deal_id, fingerprint):
        return ('Payment record: this exact payment statement was already recorded for '
                f'deal #{deal_id}; no duplicate payment event was created.')

    changed = await update_commercial_deal(
        user_id, deal_id, waiting_on='user',
        next_action='Verify payment evidence / supplier receipt separately and decide the next execution step',
        clear_next_action_due=bool(deal.get('next_action_due')),
    )
    if not changed:
        return ('Payment record: not recorded because deal tracking could not be updated. '
                'No payment or external action was executed.')

    _, pi_payload = await _latest_pi_review_record(user_id, deal_id)
    parts = [
        'External payment reported by user',
        f"offer_id={int(deal['offer_id'])}",
        f"pi_fingerprint={pi_payload.get('fingerprint') if pi_payload else 'unknown'}",
        f'fingerprint={fingerprint}',
    ]
    if record.get('amount') is not None:
        parts.append(f"amount={record['amount']}")
    if record.get('currency'):
        parts.append(f"currency={record['currency']}")
    if record.get('percent') is not None:
        parts.append(f"percent={record['percent']}")

    await add_commercial_deal_event(user_id, deal_id, 'payment_recorded', '; '.join(parts), source='user')

    details = []
    if record.get('amount') is not None:
        details.append(f"amount={record['amount']} {record.get('currency') or 'currency-not-stated'}")
    if record.get('percent') is not None:
        details.append(f"percent={record['percent']}%")
    return (f'Payment record: external payment reported by you was recorded internally for deal #{deal_id}; '
            + '; '.join(details)
            + '. This records your statement only; Lio did not execute or verify a bank transfer, '
              'supplier receipt, production authorization, shipment release, or other external action.')

async def _capture_pi_approval_payment_guardrails(user_id: str, message: str):
    approve_pi = _is_explicit_pi_approval_request(message)
    payment_request = _is_payment_execution_request(message)
    if not approve_pi and not payment_request:
        return None

    deal_id = _acceptance_guard_deal_id(message)
    if deal_id is None:
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action not applied. Specify the deal ID explicitly."

    deal = await get_commercial_deal_by_id(user_id, int(deal_id))
    if not deal:
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action not applied because deal #{deal_id} was not found."
    if not deal.get("is_active"):
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action not applied because deal #{deal_id} is inactive/closed."

    offer_id = deal.get("offer_id")
    if offer_id is None:
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action blocked because the deal does not point to a current saved offer."

    if not await _has_current_offer_acceptance_event(user_id, deal):
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action blocked because current offer #{offer_id} does not have an explicit recorded acceptance."

    if not await _has_order_execution_handoff_event(user_id, int(deal_id)):
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action blocked because the order / execution handoff has not been started."

    _, payload = await _latest_pi_review_record(user_id, int(deal_id))
    if not payload:
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action blocked because no recorded PI review exists for deal #{deal_id}."

    try:
        reviewed_offer_id = int(payload.get("offer_id"))
    except (TypeError, ValueError):
        reviewed_offer_id = None
    fingerprint = str(payload.get("fingerprint") or "").strip()
    result = str(payload.get("result") or "").upper()
    mismatches = payload.get("mismatches") or []
    missing = payload.get("missing") or []
    pi_number = payload.get("pi_number")

    if reviewed_offer_id != int(offer_id):
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        return f"{prefix} action blocked because the latest PI review is tied to offer #{reviewed_offer_id}, while the deal's current offer is #{offer_id}."

    if result != "MATCH" or mismatches or missing or not fingerprint:
        prefix = "PI approval guardrail:" if approve_pi else "Payment guardrail:"
        details = []
        if result:
            details.append(f"latest_result={result}")
        if mismatches:
            details.append("discrepancies_present")
        if missing:
            details.append("missing_fields_present")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{prefix} action blocked because the latest PI is not an approval-safe MATCH{suffix}. Resolve and re-review the PI first."

    already_approved = await _has_pi_approval_for_fingerprint(
        user_id, int(deal_id), int(offer_id), fingerprint
    )

    if payment_request:
        if not already_approved:
            return "Payment guardrail: payment blocked because the latest MATCH PI has not been explicitly approved yet."
        return (
            "Payment guardrail: the latest MATCH PI is explicitly approved internally, "
            "but no payment was executed. Lio does not execute supplier payments in this "
            "stage; a separate verified payment workflow is required."
        )

    if str(deal.get("status") or "") != "awaiting_pi":
        return (
            "PI approval guardrail: approval blocked because the deal is not currently "
            f"in awaiting_pi (current status={deal.get('status') or 'unknown'})."
        )

    if already_approved:
        return f"PI approval guardrail: the latest reviewed PI is already explicitly approved internally for deal #{deal_id}; no duplicate approval event was created."

    summary_parts = [
        "Explicit user PI approval recorded",
        f"offer_id={int(offer_id)}",
        f"fingerprint={fingerprint}",
        "result=MATCH",
    ]
    if pi_number:
        summary_parts.append(f"pi_number={pi_number}")

    changed = await update_commercial_deal(
        user_id,
        int(deal_id),
        waiting_on="user",
        next_action="Proceed to payment verification / production authorization workflow",
        clear_next_action_due=bool(deal.get("next_action_due")),
    )
    if not changed:
        return (
            "PI approval guardrail: approval not recorded because the deal tracking "
            "update failed. No payment or production authorization was executed."
        )

    await add_commercial_deal_event(
        user_id,
        int(deal_id),
        "pi_approval_approved",
        "; ".join(summary_parts),
        source="user",
    )

    return (
        "PI approval guardrail: explicit PI approval recorded internally; "
        f"deal_id={deal_id}; offer_id={offer_id}"
        + (f"; pi_number={pi_number}" if pi_number else "")
        + ". Deal status remains awaiting_pi. No payment, supplier message, production "
          "authorization, shipment release, or other external action was executed."
    )

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

    pi_review_action = await _capture_pi_review_text(user_id, message)
    if pi_review_action:
        return pi_review_action

    inspection_shipping_action = await _capture_inspection_shipping(user_id, message)
    if inspection_shipping_action:
        return inspection_shipping_action

    production_execution_action = await _capture_production_execution(user_id, message)
    if production_execution_action:
        return production_execution_action

    payment_record_action = await _capture_external_payment_record(user_id, message)
    if payment_record_action:
        return payment_record_action

    pi_approval_payment_action = await _capture_pi_approval_payment_guardrails(
        user_id, message
    )
    if pi_approval_payment_action:
        return pi_approval_payment_action

    supplier_reply_action = await _capture_supplier_reply_handoff(user_id, message)
    if supplier_reply_action:
        return supplier_reply_action

    followup_outcome_action = await _capture_deal_followup_outcome(user_id, message)
    if followup_outcome_action:
        return followup_outcome_action

    followup_due_action = await _capture_deal_followup_due(user_id, message)
    if followup_due_action:
        return followup_due_action

    acceptance_closing_action = await _capture_acceptance_and_closing_guardrails(
        user_id, message
    )
    if acceptance_closing_action:
        return acceptance_closing_action

    order_execution_action = await _capture_order_execution_handoff(
        user_id, message
    )
    if order_execution_action:
        return order_execution_action

    deal_by_id_action = await _capture_deal_tracking_by_id(user_id, message)
    if deal_by_id_action:
        return deal_by_id_action

    deal_action = await _capture_deal_tracking(user_id, message)
    if deal_action:
        return deal_action

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




def _offer_decision_missing_fields(offer: dict):
    watch = (
        "price", "currency", "price_unit", "incoterm",
        "moq", "payment_terms", "lead_time_days", "valid_until",
    )
    return [field for field in watch if offer.get(field) in (None, "")]


def _offer_decision_same_text(a, b):
    if a in (None, "") or b in (None, ""):
        return None
    return str(a).strip().casefold() == str(b).strip().casefold()


def _offer_decision_same_number(a, b, tolerance=1e-9):
    if a is None or b is None:
        return None
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def _offer_decision_pick_previous(offers: list[dict], newest: dict):
    # Clarification snapshots form a chain:
    # final clarification -> partial clarification -> original incomplete offer.
    # None of those superseded snapshots should become the "previous commercial offer"
    # used for price decision guidance.
    offers_by_id = {
        item.get("id"): item
        for item in offers
        if item.get("id") is not None
    }

    superseded_offer_ids = set()
    current = newest
    visited = set()

    while current and current.get("id") not in visited:
        current_id = current.get("id")
        if current_id is not None:
            visited.add(current_id)

        notes = str(current.get("notes") or "")
        superseded_match = re.search(
            r"clarification_base_offer_id=(\d+)",
            notes,
        )
        if not superseded_match:
            break

        superseded_id = int(superseded_match.group(1))
        superseded_offer_ids.add(superseded_id)
        current = offers_by_id.get(superseded_id)

    candidates = [
        item for item in offers
        if item.get("id") != newest.get("id")
        and item.get("id") not in superseded_offer_ids
        and item.get("supplier_id") == newest.get("supplier_id")
    ]
    if not candidates:
        return None

    newest_product = (newest.get("product") or "").strip().casefold()
    newest_size = (newest.get("size") or "").strip().casefold()
    same_product_size = []
    for item in candidates:
        product = (item.get("product") or "").strip().casefold()
        size = (item.get("size") or "").strip().casefold()
        product_ok = not newest_product or not product or product == newest_product
        size_ok = not newest_size or not size or size == newest_size
        if product_ok and size_ok:
            same_product_size.append(item)

    return (same_product_size or candidates)[0]


async def _commercial_offer_decision_guidance(
    user_id: str,
    *,
    supplier_id: int,
    newest_offer_id: int | None = None,
):
    data = await get_commercial_offer_comparison(
        user_id,
        supplier_ids=[supplier_id],
        latest_per_supplier=False,
        limit=100,
    )
    offers = data.get("offers", [])
    if not offers:
        return None

    newest = None
    if newest_offer_id is not None:
        newest = next((x for x in offers if x.get("id") == newest_offer_id), None)
    if newest is None:
        newest = offers[0]

    previous = _offer_decision_pick_previous(offers, newest)
    missing_new = _offer_decision_missing_fields(newest)

    if previous is None:
        decision = "REQUEST_CLARIFICATION" if missing_new else "FOLLOW_UP"
        reason = (
            "No earlier comparable saved offer exists for this supplier. "
            + (
                "Clarify the missing commercial terms before deciding."
                if missing_new
                else "Use this as the baseline and continue normal commercial follow-up."
            )
        )
        return {
            "decision": decision,
            "confidence": "medium",
            "new_offer_id": newest.get("id"),
            "previous_offer_id": None,
            "reason": reason,
            "price_change": None,
            "price_change_pct": None,
            "missing_new": missing_new,
        }

    currency_same = _offer_decision_same_text(newest.get("currency"), previous.get("currency"))
    unit_same = _offer_decision_same_text(newest.get("price_unit"), previous.get("price_unit"))
    incoterm_same = _offer_decision_same_text(newest.get("incoterm"), previous.get("incoterm"))
    product_same = _offer_decision_same_text(newest.get("product"), previous.get("product"))
    size_same = _offer_decision_same_text(newest.get("size"), previous.get("size"))
    thickness_same = _offer_decision_same_number(newest.get("thickness_mm"), previous.get("thickness_mm"))

    spec_conflict = any(v is False for v in (product_same, size_same, thickness_same))

    price_change = None
    price_change_pct = None
    if newest.get("price") is not None and previous.get("price") is not None:
        try:
            price_change = float(newest["price"]) - float(previous["price"])
            if float(previous["price"]) != 0:
                price_change_pct = (price_change / float(previous["price"])) * 100.0
        except (TypeError, ValueError):
            price_change = None
            price_change_pct = None

    if currency_same is not True or unit_same is not True:
        decision = "REQUEST_CLARIFICATION"
        confidence = "high"
        reason = (
            "The two saved prices are not on the same currency and price-unit basis, "
            "so a direct price decision would be unreliable."
        )
    elif incoterm_same is not True:
        decision = "REQUEST_CLARIFICATION"
        confidence = "high"
        reason = (
            "The Incoterm basis is different or incomplete, so the quoted prices are "
            "not safely comparable as equivalent commercial costs."
        )
    elif spec_conflict:
        decision = "REQUEST_CLARIFICATION"
        confidence = "high"
        reason = (
            "The saved product specifications differ between the two offers. "
            "Confirm that the new quotation is for the same specification before deciding."
        )
    elif missing_new:
        decision = "REQUEST_CLARIFICATION"
        confidence = "high"
        reason = (
            "The new offer is missing material commercial terms: "
            + ", ".join(missing_new)
            + ". Clarify these before accepting or negotiating from the new offer."
        )
    elif price_change is None:
        decision = "REQUEST_CLARIFICATION"
        confidence = "medium"
        reason = "A reliable saved price comparison is not available."
    elif price_change < -1e-9:
        decision = "ACCEPT"
        confidence = "medium"
        reason = (
            "The new saved offer is lower on the same saved commercial basis and no tracked "
            "critical offer field is missing. This is a recommendation only; it does not "
            "accept the offer automatically."
        )
    elif price_change > 1e-9:
        decision = "NEGOTIATE"
        confidence = "high"
        reason = (
            "The new saved offer is higher on the same saved commercial basis. "
            "Negotiate the price or request justification for the increase."
        )
    else:
        decision = "NEGOTIATE"
        confidence = "medium"
        reason = (
            "The saved price is unchanged on the same basis. Negotiate for an improvement "
            "or another commercial concession before accepting."
        )

    return {
        "decision": decision,
        "confidence": confidence,
        "new_offer_id": newest.get("id"),
        "previous_offer_id": previous.get("id"),
        "reason": reason,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "missing_new": missing_new,
    }


def _commercial_offer_decision_guidance_text(guidance: dict | None):
    if not guidance:
        return None
    parts = [
        f"decision={guidance.get('decision')}",
        f"confidence={guidance.get('confidence')}",
        f"new_offer_id={guidance.get('new_offer_id')}",
    ]
    if guidance.get("previous_offer_id") is not None:
        parts.append(f"previous_offer_id={guidance.get('previous_offer_id')}")
    if guidance.get("price_change") is not None:
        parts.append(f"price_change={guidance.get('price_change'):.6g}")
    if guidance.get("price_change_pct") is not None:
        parts.append(f"price_change_pct={guidance.get('price_change_pct'):.4g}")
    if guidance.get("missing_new"):
        parts.append("missing_new=" + ",".join(guidance["missing_new"]))
    parts.append("reason=" + str(guidance.get("reason") or "").replace(";", ","))
    return "|".join(parts)


def _is_offer_decision_guidance_request(message: str) -> bool:
    folded = (message or "").casefold()
    decision_signals = (
        "what should we do", "what do you recommend", "recommendation",
        "accept", "negotiate", "clarify", "decision", "latest offer", "new offer",
        "should we accept", "should i accept",
        "was sollen wir tun", "empfehlung", "akzeptieren", "verhandeln",
        "entscheidung", "neues angebot", "letztes angebot",
        "ماذا تنصح", "ماذا نفعل", "ما القرار", "هل نقبل", "اقبل", "قبول",
        "تفاوض", "نتفاوض", "توضيح", "العرض الجديد", "العرض الأخير", "العرض الاخير",
    )
    commercial_signals = (
        "offer", "supplier", "price", "angebot", "lieferant", "preis",
        "عرض", "مورد", "سعر",
    )
    return any(x in folded for x in decision_signals) and any(x in folded for x in commercial_signals)


async def _commercial_offer_decision_guidance_context(user_id: str, message: str) -> str:
    if not _is_offer_decision_guidance_request(message):
        return ""

    deal_id_match = re.search(
        r"(?:الصفقة|صفقة|deal)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
        _deal_followup_ascii_digits(message or ""),
        flags=re.IGNORECASE,
    )

    supplier_id = None
    if deal_id_match:
        deal = await get_commercial_deal_by_id(user_id, int(deal_id_match.group(1)))
        if deal:
            supplier_id = deal.get("supplier_id")

    if supplier_id is None:
        deals = await get_commercial_deals(user_id, active_only=True, limit=20)
        if len(deals) == 1:
            supplier_id = deals[0].get("supplier_id")

    if supplier_id is None:
        return (
            "SMART OFFER DECISION GUIDANCE.\n"
            "No unique active supplier/deal could be resolved. Ask the user to specify the deal ID or supplier."
        )

    guidance = await _commercial_offer_decision_guidance(
        user_id,
        supplier_id=int(supplier_id),
    )
    if not guidance:
        return "SMART OFFER DECISION GUIDANCE.\nNo saved offer history was found."

    lines = [
        "SMART OFFER DECISION GUIDANCE.",
        "This is advisory analysis only. Do not accept, reject, send, save, or change a deal automatically.",
        "Use only saved commercial facts. Do not invent missing terms or user approval thresholds.",
        f"Recommended action: {guidance['decision']}",
        f"Confidence: {guidance['confidence']}",
        f"New offer ID: {guidance['new_offer_id']}",
    ]
    if guidance.get("previous_offer_id") is not None:
        lines.append(f"Previous offer ID: {guidance['previous_offer_id']}")
    if guidance.get("price_change") is not None:
        lines.append(f"Saved price change: {guidance['price_change']:.6g}")
    if guidance.get("price_change_pct") is not None:
        lines.append(f"Saved price change percent: {guidance['price_change_pct']:.4g}%")
    if guidance.get("missing_new"):
        lines.append("Missing in new saved offer: " + ", ".join(guidance["missing_new"]))
    lines.append("Reason: " + guidance["reason"])
    return "\n".join(lines)


def _commercial_followup_action_queue_items(deals):
    """Build a deterministic read-only queue from saved active deals."""
    today = _deal_followup_today()
    items = []

    for deal in deals or []:
        raw_due = deal.get("next_action_due")
        due = None
        bucket = "UNSCHEDULED"
        timing = "no follow-up date saved"
        rank = 3

        if raw_due:
            try:
                due = datetime.strptime(str(raw_due)[:10], "%Y-%m-%d").date()
            except (TypeError, ValueError):
                bucket = "UNKNOWN_DUE"
                timing = f"unreadable saved due date={raw_due}"
                rank = 4
            else:
                delta = (due - today).days
                if delta < 0:
                    bucket = "OVERDUE"
                    timing = f"{abs(delta)} day(s) overdue"
                    rank = 0
                elif delta == 0:
                    bucket = "DUE_TODAY"
                    timing = "due today"
                    rank = 1
                else:
                    bucket = "UPCOMING"
                    timing = f"due in {delta} day(s)"
                    rank = 2

        items.append({
            "rank": rank,
            "bucket": bucket,
            "due": due,
            "timing": timing,
            "deal": deal,
        })

    items.sort(key=lambda item: (
        item["rank"],
        item["due"].isoformat() if item["due"] else "9999-12-31",
        int(item["deal"].get("id") or 0),
    ))
    return items


def _commercial_followup_action_queue_context(deals) -> str:
    items = _commercial_followup_action_queue_items(deals)
    if not items:
        return ""

    lines = [
        "Commercial follow-up action queue (read-only; derived from saved active deals):"
    ]

    for item in items:
        deal = item["deal"]
        details = [
            f"priority={item['bucket']}",
            f"deal_id={deal.get('id')}",
            f"supplier={deal.get('supplier')}" if deal.get("supplier") else None,
            f"product={deal.get('product')}" if deal.get("product") else None,
            f"status={deal.get('status')}" if deal.get("status") else None,
            f"waiting_on={deal.get('waiting_on')}" if deal.get("waiting_on") else None,
            f"next_action={deal.get('next_action')}" if deal.get("next_action") else None,
            f"next_action_due={deal.get('next_action_due')}" if deal.get("next_action_due") else None,
            f"timing={item['timing']}",
        ]
        lines.append("- " + "; ".join(x for x in details if x))

    return "\n".join(lines)


async def _persistent_context(user_id: str) -> str:
    profile = await get_profile(user_id)
    memories = await saved_memories(user_id, 20)
    smart = await get_smart_memories(user_id, 30)
    commercial = await get_commercial_memory(user_id, supplier_limit=10, offer_limit=20)
    deals = await get_commercial_deals(user_id, active_only=True, limit=30)
    followup_queue_context = _commercial_followup_action_queue_context(deals)
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

    if followup_queue_context:
        lines.extend(followup_queue_context.splitlines())

    if deals:
        lines.append("Active commercial deal tracking:")
        for deal in deals:
            details = [
                f"deal_id={deal.get('id')}",
                f"supplier={deal.get('supplier')}",
                f"product={deal.get('product')}" if deal.get("product") else None,
                f"offer_id={deal.get('offer_id')}" if deal.get("offer_id") else None,
                f"status={deal.get('status')}",
                f"waiting_on={deal.get('waiting_on')}" if deal.get("waiting_on") else None,
                f"next_action={deal.get('next_action')}" if deal.get("next_action") else None,
                f"next_action_due={deal.get('next_action_due')}" if deal.get("next_action_due") else None,
            ]
            lines.append("- " + "; ".join(item for item in details if item))

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




def _offer_decision_action_handoff(guidance: dict | None) -> dict | None:
    # Map advisory offer guidance to a safe, non-executing next action.
    if not guidance:
        return None

    decision = str(guidance.get("decision") or "").strip().upper()
    missing = [str(x) for x in (guidance.get("missing_new") or []) if str(x).strip()]

    if decision == "REQUEST_CLARIFICATION":
        action = "PREPARE_CLARIFICATION"
        instruction = (
            "Prepare a concise supplier clarification reply. Ask only for missing or unclear "
            "commercial facts supported by the saved guidance. Do not invent target prices, "
            "deadlines, commitments, concessions, or specifications."
        )
        if missing:
            instruction += " Missing saved fields to clarify: " + ", ".join(missing) + "."
    elif decision == "NEGOTIATE":
        action = "PREPARE_NEGOTIATION"
        instruction = (
            "Prepare an evidence-based negotiation reply using only saved commercial facts. "
            "If the user has not supplied a target price, ask for the supplier's best/improved "
            "price or justification rather than inventing a numeric target. Do not send anything."
        )
    elif decision == "FOLLOW_UP":
        action = "PREPARE_FOLLOW_UP"
        instruction = (
            "Prepare a concise supplier follow-up using the existing follow-up drafting rules. "
            "Do not invent a deadline or claim that any message was sent."
        )
    elif decision == "ACCEPT":
        action = "PREPARE_ACCEPTANCE_REVIEW"
        instruction = (
            "Prepare an acceptance-review step and, if useful, a draft acceptance message for "
            "the user's review only. Explicit user approval is still required before any acceptance, "
            "sending, deal-status change, or other external action."
        )
    else:
        return None

    return {
        "decision": decision,
        "action": action,
        "instruction": instruction,
        "new_offer_id": guidance.get("new_offer_id"),
        "previous_offer_id": guidance.get("previous_offer_id"),
        "missing_new": missing,
        "reason": guidance.get("reason"),
        "price_change": guidance.get("price_change"),
        "price_change_pct": guidance.get("price_change_pct"),
    }


def _is_offer_decision_action_request(message: str) -> bool:
    folded = (message or "").casefold()

    action_signals = (
        "prepare the next step", "prepare next step", "next action", "what should i send",
        "what should we send", "draft reply", "draft a reply", "write a reply",
        "prepare a reply", "prepare reply", "draft message", "prepare message",
        "reply to the supplier", "respond to the supplier", "what do i send",
        "nächster schritt", "naechster schritt", "antwort formulieren",
        "antwort vorbereiten", "nachricht vorbereiten", "lieferant antworten",
        "الخطوة التالية", "ما الخطوة التالية", "جهز الرد", "جهّز الرد", "حضر الرد",
        "حضّر الرد", "اكتب الرد", "صياغة الرد", "جهز رسالة", "جهّز رسالة",
        "اكتب رسالة", "ماذا نرسل", "ماذا أرسل", "ماذا ارسل", "الرد على المورد",
    )
    commercial_signals = (
        "offer", "supplier", "deal", "quotation", "price",
        "angebot", "lieferant", "preis",
        "عرض", "مورد", "صفقة", "سعر",
    )

    return any(x in folded for x in action_signals) and any(
        x in folded for x in commercial_signals
    )


async def _commercial_offer_decision_action_handoff_context(user_id: str, message: str) -> str:
    # Build read-only action-preparation context from the latest saved offer guidance.
    if not _is_offer_decision_action_request(message):
        return ""

    deal_id_match = re.search(
        r"(?:الصفقة|صفقة|deal)\s*(?:رقم|#|id)?\s*[:#]?\s*(\d+)",
        _deal_followup_ascii_digits(message or ""),
        flags=re.IGNORECASE,
    )

    supplier_id = None
    deal_id = None
    if deal_id_match:
        deal_id = int(deal_id_match.group(1))
        deal = await get_commercial_deal_by_id(user_id, deal_id)
        if deal:
            supplier_id = deal.get("supplier_id")

    if supplier_id is None:
        deals = await get_commercial_deals(user_id, active_only=True, limit=20)
        if len(deals) == 1:
            deal_id = deals[0].get("id")
            supplier_id = deals[0].get("supplier_id")

    if supplier_id is None:
        return (
            "SMART DECISION ACTION HANDOFF.\n"
            "No unique active supplier/deal could be resolved. Ask the user to specify the deal ID or supplier.\n"
            "Do not send, accept, reject, save, or change any deal automatically."
        )

    guidance = await _commercial_offer_decision_guidance(
        user_id,
        supplier_id=int(supplier_id),
    )
    handoff = _offer_decision_action_handoff(guidance)
    if not handoff:
        return (
            "SMART DECISION ACTION HANDOFF.\n"
            "No actionable saved offer guidance is available yet.\n"
            "Do not invent commercial facts or execute any external action."
        )

    lines = [
        "SMART DECISION ACTION HANDOFF.",
        "This is preparation only. Do not send, accept, reject, save, close, or change a deal automatically.",
        "Use only saved commercial facts and explicit user instructions.",
        f"Deal ID: {deal_id}" if deal_id is not None else "Deal ID: unresolved",
        f"Advisory decision: {handoff['decision']}",
        f"Safe handoff action: {handoff['action']}",
        f"Instruction: {handoff['instruction']}",
    ]

    if handoff.get("new_offer_id") is not None:
        lines.append(f"New offer ID: {handoff['new_offer_id']}")
    if handoff.get("previous_offer_id") is not None:
        lines.append(f"Previous offer ID: {handoff['previous_offer_id']}")
    if handoff.get("price_change") is not None:
        lines.append(f"Saved price change: {handoff['price_change']:.6g}")
    if handoff.get("price_change_pct") is not None:
        lines.append(f"Saved price change percent: {handoff['price_change_pct']:.4g}%")
    if handoff.get("missing_new"):
        lines.append("Missing saved fields: " + ", ".join(handoff["missing_new"]))
    if handoff.get("reason"):
        lines.append("Decision reason: " + str(handoff["reason"]))

    if handoff["action"] == "PREPARE_CLARIFICATION":
        lines.append(
            "Output preference: briefly explain what is missing, then provide a send-ready supplier clarification draft."
        )
    elif handoff["action"] == "PREPARE_NEGOTIATION":
        lines.append(
            "Output preference: briefly state the evidence-based negotiation angle, then provide a send-ready negotiation draft."
        )
    elif handoff["action"] == "PREPARE_FOLLOW_UP":
        lines.append(
            "Output preference: provide a concise send-ready follow-up draft and clearly state that it has not been sent."
        )
    elif handoff["action"] == "PREPARE_ACCEPTANCE_REVIEW":
        lines.append(
            "Output preference: summarize why acceptance is being considered and provide a review-only draft. "
            "State clearly that explicit user approval is still required and nothing has been accepted or sent."
        )

    return "\n".join(lines)


def _authoritative_memory_action_reply(memory_action: str | None):
    if not memory_action:
        return None

    if memory_action.startswith("Supplier reply handoff recorded:"):
        save_status = None
        marker = "commercial_save_status="
        if marker in memory_action:
            save_status = memory_action.split(marker, 1)[1].rstrip(".")

        parts = ["Supplier reply recorded for the deal."]
        if "waiting_on=user" in memory_action:
            parts.append("The deal is now waiting on you.")
        if "next_action=Review supplier response and reply" in memory_action:
            parts.append("Next action: review the supplier response and reply.")

        if save_status:
            if save_status.startswith("Commercial memory saved:"):
                details = save_status.split("Commercial memory saved:", 1)[1].strip()
                parts.append("The commercial offer was saved successfully.")
                if "offer_id=" in details:
                    offer_id = details.split("offer_id=", 1)[1].split(";", 1)[0].strip()
                    parts.append(f"Saved offer ID: {offer_id}.")
            elif save_status.startswith("Commercial memory unchanged: exact offer already saved:"):
                details = save_status.split(
                    "Commercial memory unchanged: exact offer already saved:", 1
                )[1].strip()
                parts.append("That exact commercial offer was already saved, so no duplicate offer was created.")
                if "offer_id=" in details:
                    offer_id = details.split("offer_id=", 1)[1].split(";", 1)[0].strip()
                    parts.append(f"Existing offer ID: {offer_id}.")
            else:
                parts.append(save_status)
        elif "Commercial terms in the reply were not saved" in memory_action:
            parts.append(
                "Commercial terms from the reply were not saved because you did not explicitly request that."
            )

        negotiation_marker = "negotiation_cycle="
        if negotiation_marker in memory_action:
            raw_negotiation = memory_action.split(negotiation_marker, 1)[1]
            raw_negotiation = raw_negotiation.split(
                "; decision_guidance=", 1
            )[0].rstrip(".")

            negotiation_values = {}
            for token in raw_negotiation.split("|"):
                if "=" in token:
                    key, value = token.split("=", 1)
                    negotiation_values[key] = value

            round_value = negotiation_values.get("round")
            offer_value = negotiation_values.get("offer_id")
            trend_value = negotiation_values.get("price_trend")
            previous_value = negotiation_values.get("previous_offer_id")

            if round_value and offer_value:
                text = (
                    f"Negotiation round {round_value} recorded for saved offer ID: "
                    f"{offer_value}."
                )
                if previous_value:
                    text += f" Previous comparable offer ID: {previous_value}."
                if trend_value:
                    text += f" Saved price trend: {trend_value}."
                parts.append(text)

        clarification_marker = "clarification_resolution="
        if clarification_marker in memory_action:
            raw_clarification = memory_action.split(clarification_marker, 1)[1]
            raw_clarification = raw_clarification.split(
                "; decision_guidance=", 1
            )[0].rstrip(".")

            clarification_values = {}
            for token in raw_clarification.split("|"):
                if "=" in token:
                    key, value = token.split("=", 1)
                    clarification_values[key] = value

            resolved = clarification_values.get("resolved")
            remaining = clarification_values.get("remaining")
            saved = clarification_values.get("saved") == "true"

            if resolved and resolved != "none":
                if saved:
                    parts.append(
                        "Supplier clarification was merged with the saved offer for: "
                        + resolved.replace(",", ", ")
                        + "."
                    )
                else:
                    parts.append(
                        "Supplier clarification appears to resolve: "
                        + resolved.replace(",", ", ")
                        + ", but those commercial facts were not saved because no explicit save was requested."
                    )

            if remaining and remaining != "none":
                parts.append(
                    "Still missing after this clarification: "
                    + remaining.replace(",", ", ")
                    + "."
                )
            elif saved and resolved and resolved != "none":
                parts.append(
                    "All previously tracked missing offer fields are now resolved."
                )

        decision_marker = "decision_guidance="
        if decision_marker in memory_action:
            raw = memory_action.split(decision_marker, 1)[1].rstrip(".")
            values = {}
            for token in raw.split("|"):
                if "=" in token:
                    key, value = token.split("=", 1)
                    values[key] = value
            if values.get("decision"):
                parts.append(f"Advisory recommendation: {values['decision']}.")
            if values.get("confidence"):
                parts.append(f"Confidence: {values['confidence']}.")
            if values.get("previous_offer_id"):
                parts.append(f"Compared with saved offer ID: {values['previous_offer_id']}.")
            if values.get("price_change") is not None:
                change_text = f"Saved price change: {values['price_change']}"
                if values.get("price_change_pct") is not None:
                    change_text += f" ({values['price_change_pct']}%)."
                else:
                    change_text += "."
                parts.append(change_text)
            if values.get("reason"):
                parts.append(values["reason"])

            decision = str(values.get("decision") or "").upper()
            if decision == "REQUEST_CLARIFICATION":
                parts.append(
                    "Next step: prepare a clarification reply for the missing or unclear commercial terms before any acceptance."
                )
            elif decision == "NEGOTIATE":
                parts.append(
                    "Next step: prepare an evidence-based negotiation reply; no target price is assumed unless you supplied one."
                )
            elif decision == "FOLLOW_UP":
                parts.append(
                    "Next step: prepare a supplier follow-up. Nothing has been sent automatically."
                )
            elif decision == "ACCEPT":
                parts.append(
                    "Next step: review and explicitly approve an acceptance reply. Nothing has been accepted or sent automatically."
                )

        return " ".join(parts)

    if memory_action.startswith("Offer acceptance guardrail:"):
        return memory_action.split("Offer acceptance guardrail:", 1)[1].strip()

    if memory_action.startswith("Deal closing guardrail:"):
        return memory_action.split("Deal closing guardrail:", 1)[1].strip()

    if memory_action.startswith("Order execution handoff guardrail:"):
        return memory_action.split("Order execution handoff guardrail:", 1)[1].strip()

    if memory_action.startswith("Execution-stage guardrail:"):
        return memory_action.split("Execution-stage guardrail:", 1)[1].strip()

    if memory_action.startswith("PI review guardrail:"):
        return memory_action.split("PI review guardrail:", 1)[1].strip()

    if memory_action.startswith("PI approval guardrail:"):
        return memory_action.split("PI approval guardrail:", 1)[1].strip()

    if memory_action.startswith("Payment guardrail:"):
        return memory_action.split("Payment guardrail:", 1)[1].strip()

    if memory_action.startswith("Inspection & shipping:"):
        return memory_action.split("Inspection & shipping:", 1)[1].strip()

    if memory_action.startswith("Production execution:"):
        return memory_action.split("Production execution:", 1)[1].strip()

    if memory_action.startswith("Payment record:"):
        return memory_action.split("Payment record:", 1)[1].strip()

    if memory_action.startswith("Commercial memory saved:"):
        details = memory_action.split("Commercial memory saved:", 1)[1].strip()
        reply = "Commercial data saved successfully."
        if "offer_id=" in details:
            offer_id = details.split("offer_id=", 1)[1].split(";", 1)[0].strip()
            reply += f" Saved offer ID: {offer_id}."
        return reply

    if memory_action.startswith("Commercial memory unchanged: exact offer already saved:"):
        details = memory_action.split(
            "Commercial memory unchanged: exact offer already saved:", 1
        )[1].strip()
        reply = "That exact commercial offer is already saved; no duplicate offer was created."
        if "offer_id=" in details:
            offer_id = details.split("offer_id=", 1)[1].split(";", 1)[0].strip()
            reply += f" Existing offer ID: {offer_id}."
        return reply

    return None


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

    authoritative_reply = _authoritative_memory_action_reply(memory_action)
    if authoritative_reply:
        await add_message(req.user_id, "assistant", authoritative_reply)
        return ChatResponse(reply=authoritative_reply, mode="live")

    history = await recent_messages(req.user_id, 10)
    recent_context = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[:-1]
    )
    persistent_context = await _persistent_context(req.user_id)
    comparison_context = await _commercial_comparison_context(req.user_id, req.message)
    decision_guidance_context = await _commercial_offer_decision_guidance_context(
        req.user_id, req.message
    )
    decision_action_handoff_context = await _commercial_offer_decision_action_handoff_context(
        req.user_id, req.message
    )
    negotiation_cycle_context = await _commercial_negotiation_cycle_context(
        req.user_id, req.message
    )
    memory_action_context = (
        f"Internal memory status for this turn: {memory_action}"
        if memory_action
        else ""
    )
    context_text = "\n\n".join(
        part
        for part in [
            persistent_context,
            comparison_context,
            decision_guidance_context,
            decision_action_handoff_context,
            negotiation_cycle_context,
            memory_action_context,
            recent_context,
        ]
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


@app.post("/pi/review", response_model=ChatResponse)
async def review_pi_file(user_id: str, deal_id: int, file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API is not configured")
    deal, offer, error = await _validate_pi_review_deal(user_id, int(deal_id))
    if error:
        reply = _authoritative_memory_action_reply(error) or error
        return ChatResponse(reply=reply, mode="live")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty PI file")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PI file is too large; maximum size is 20 MB")
    filename = (file.filename or "pi").strip() or "pi"
    mime_type = (file.content_type or "").lower().strip()
    suffix = Path(filename).suffix.lower()
    allowed_images = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if mime_type == "application/pdf" or suffix == ".pdf":
        mime_type = "application/pdf"
    elif mime_type in allowed_images or suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        if mime_type not in allowed_images:
            mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[suffix]
    else:
        raise HTTPException(status_code=415, detail="Unsupported PI file type. Use PDF, PNG, JPG/JPEG, or WEBP.")
    fingerprint = "file:" + hashlib.sha256(data).hexdigest()
    if await _has_pi_review_fingerprint(user_id, int(deal_id), fingerprint):
        reply = (
            "PI review guardrail: this exact PI file was already reviewed for "
            f"deal #{deal_id}; no duplicate PI review event was created."
        )
        return ChatResponse(reply=_authoritative_memory_action_reply(reply) or reply, mode="live")
    encoded = base64.b64encode(data).decode("ascii")
    file_data_url = f"data:{mime_type};base64,{encoded}"
    try:
        from .agents import run_pi_review_file
        extracted = await run_pi_review_file(file_data_url, filename, mime_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PI extraction error: {exc}")
    reply = await _record_pi_review(
        user_id, deal, offer, extracted.model_dump(), fingerprint=fingerprint, source="uploaded_file"
    )
    authoritative_reply = _authoritative_memory_action_reply(reply) or reply
    await add_message(user_id, "user", f"PI uploaded for review; deal_id={deal_id}; filename={filename}")
    await add_message(user_id, "assistant", authoritative_reply)
    return ChatResponse(reply=authoritative_reply, mode="live")

@app.post("/shipping-documents/review", response_model=ChatResponse)
async def review_shipping_document_file(user_id: str, deal_id: int, file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API is not configured")
    deal, offer, pi_fingerprint, error = await _validate_shipping_document_review_deal(
        user_id, int(deal_id)
    )
    if error:
        reply = _authoritative_memory_action_reply(error) or error
        return ChatResponse(reply=reply, mode="live")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty shipping document file")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Shipping document file is too large; maximum size is 20 MB",
        )

    filename = (file.filename or "shipping-document").strip() or "shipping-document"
    mime_type = (file.content_type or "").lower().strip()
    suffix = Path(filename).suffix.lower()
    allowed_images = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if mime_type == "application/pdf" or suffix == ".pdf":
        mime_type = "application/pdf"
    elif mime_type in allowed_images or suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        if mime_type not in allowed_images:
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }[suffix]
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported shipping document file type. Use PDF, PNG, JPG/JPEG, or WEBP.",
        )

    fingerprint = "file:" + hashlib.sha256(data).hexdigest()
    if await _has_shipping_document_fingerprint(user_id, int(deal_id), fingerprint):
        reply = (
            "Shipping documents: this exact shipping document file was already reviewed for "
            f"deal #{deal_id}; no duplicate event was created."
        )
        return ChatResponse(reply=_authoritative_memory_action_reply(reply) or reply, mode="live")

    encoded = base64.b64encode(data).decode("ascii")
    file_data_url = f"data:{mime_type};base64,{encoded}"
    try:
        from .agents import run_shipping_document_review_file

        extracted = await run_shipping_document_review_file(
            file_data_url, filename, mime_type
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Shipping document extraction error: {exc}")

    reply = await _record_shipping_document_review(
        user_id,
        deal,
        offer,
        extracted.model_dump(),
        pi_fingerprint=pi_fingerprint,
        fingerprint=fingerprint,
        source="uploaded_file",
    )
    authoritative_reply = _authoritative_memory_action_reply(reply) or reply
    await add_message(
        user_id,
        "user",
        f"Shipping document uploaded for review; deal_id={deal_id}; filename={filename}",
    )
    await add_message(user_id, "assistant", authoritative_reply)
    return ChatResponse(reply=authoritative_reply, mode="live")


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
