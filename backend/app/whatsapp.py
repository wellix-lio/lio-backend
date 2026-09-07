import hashlib
import hmac
import httpx
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

def _graph_url(graph_api_version: str, resource: str) -> str:
    version = (graph_api_version or "").strip().strip("/")
    if not version:
        raise RuntimeError("WHATSAPP_GRAPH_API_VERSION is not configured")
    return f"https://graph.facebook.com/{version}/{resource.lstrip('/')}"


def _require_whatsapp_credentials(
    access_token: str,
    phone_number_id: str | None = None,
) -> None:
    if not (access_token or "").strip():
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN is not configured")
    if phone_number_id is not None and not (phone_number_id or "").strip():
        raise RuntimeError("WHATSAPP_PHONE_NUMBER_ID is not configured")


async def send_whatsapp_text(
    *,
    to: str,
    text: str,
    access_token: str,
    phone_number_id: str,
    graph_api_version: str,
) -> dict[str, Any]:
    # Send one WhatsApp text message. No retry and no automatic invocation.
    _require_whatsapp_credentials(access_token, phone_number_id)
    recipient = (to or "").strip()
    body = (text or "").strip()
    if not recipient:
        raise ValueError("WhatsApp recipient is required")
    if not body:
        raise ValueError("WhatsApp text is required")

    url = _graph_url(graph_api_version, f"{phone_number_id}/messages")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {"preview_url": False, "body": body},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected WhatsApp send response")
    return data


async def get_whatsapp_media_info(
    *,
    media_id: str,
    access_token: str,
    graph_api_version: str,
) -> dict[str, Any]:
    # Fetch metadata and temporary download URL for inbound WhatsApp media.
    _require_whatsapp_credentials(access_token)
    media = (media_id or "").strip()
    if not media:
        raise ValueError("WhatsApp media_id is required")

    url = _graph_url(graph_api_version, media)
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected WhatsApp media metadata response")
    return data


async def download_whatsapp_media(
    *,
    media_id: str,
    access_token: str,
    graph_api_version: str,
) -> tuple[bytes, str]:
    # Download authenticated WhatsApp media bytes and return content plus MIME type.
    info = await get_whatsapp_media_info(
        media_id=media_id,
        access_token=access_token,
        graph_api_version=graph_api_version,
    )
    media_url = str(info.get("url") or "").strip()
    if not media_url:
        raise RuntimeError("WhatsApp media metadata did not contain a download URL")

    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(media_url, headers=headers)
        response.raise_for_status()
        content = response.content
        mime_type = response.headers.get("content-type") or str(info.get("mime_type") or "")

    if not content:
        raise RuntimeError("WhatsApp media download returned empty content")
    return content, mime_type
