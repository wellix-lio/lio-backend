import hashlib
import hmac
from typing import Any


def verify_whatsapp_subscription(
    mode: str,
    provided_token: str,
    expected_token: str,
) -> bool:
    if mode != "subscribe" or not expected_token:
        return False
    return hmac.compare_digest(provided_token or "", expected_token)


def verify_whatsapp_signature(
    raw_body: bytes,
    signature_header: str,
    app_secret: str,
) -> bool:
    if not app_secret or not signature_header:
        return False

    expected = "sha256=" + hmac.new(
        app_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature_header)


def extract_whatsapp_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    # Extract inbound WhatsApp messages without performing side effects.
    extracted: list[dict[str, Any]] = []

    for entry in payload.get("entry", []) or []:
        if not isinstance(entry, dict):
            continue

        for change in entry.get("changes", []) or []:
            if not isinstance(change, dict):
                continue

            value = change.get("value") or {}
            if not isinstance(value, dict):
                continue

            metadata = value.get("metadata") or {}
            phone_number_id = (
                metadata.get("phone_number_id")
                if isinstance(metadata, dict)
                else None
            )

            for message in value.get("messages", []) or []:
                if not isinstance(message, dict):
                    continue

                message_type = str(message.get("type") or "")
                item: dict[str, Any] = {
                    "id": message.get("id"),
                    "from": message.get("from"),
                    "timestamp": message.get("timestamp"),
                    "type": message_type,
                    "phone_number_id": phone_number_id,
                }

                if message_type == "text":
                    text_data = message.get("text") or {}
                    if isinstance(text_data, dict):
                        item["text"] = str(text_data.get("body") or "")

                elif message_type == "audio":
                    audio_data = message.get("audio") or {}
                    if isinstance(audio_data, dict):
                        item["media_id"] = audio_data.get("id")
                        item["mime_type"] = audio_data.get("mime_type")
                        item["voice"] = bool(audio_data.get("voice"))

                extracted.append(item)

    return extracted
