import re
from dataclasses import dataclass


PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\-\.\s\(\)]{6,}\d)(?!\w)")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b")
CHINA_ID_RE = re.compile(r"\b\d{17}[\dXx]\b")
GENERIC_SSN_RE = re.compile(r"\b\d{3}[.\-]?\d{2,4}[.\-]?\d{4}\b")
DRIVER_LICENSE_RE = re.compile(
    r"(?i)(?:driver(?:'s)? license(?: number)?|dl)\s*[:#]?\s*([A-Z0-9\-]{6,20})"
)
PASSPORT_RE = re.compile(r"\b[A-Z]\d{7,8}\b")
VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b")
ACCOUNT_RE = re.compile(r"\b\d{8,20}\b")
PASSWORD_INLINE_RE = re.compile(
    r"(?i)(?:password|pin)\s*[:=]\s*([A-Za-z0-9!@#$%^&*()_+\-=\[\]{};':\",.<>/?\\|`~]{4,64})"
)


@dataclass(frozen=True)
class PIISample:
    pii_type: str
    secret: str
    prompt: str
    full_text: str
    char_start: int | None = None
    char_end: int | None = None


def normalize_pii_type(pii_type: str | None) -> str | None:
    if pii_type is None:
        return None
    value = str(pii_type).strip().upper()
    if not value:
        return None
    aliases = {
        "PHONENUMBER": "TEL",
        "PHONE": "TEL",
        "EMAILADDRESS": "EMAIL",
        "SSN": "ID_CARD",
        "IDNUMBER": "ID_CARD",
        "ID_CARD_NUMBER": "ID_CARD",
        "DRIVERLICENSENUMBER": "DRIVER_LICENSE",
        "VEHICLEVIN": "VEHICLE_VIN",
        "VIN": "VEHICLE_VIN",
        "ACCOUNTNUMBER": "ACCOUNT_NUMBER",
        "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
        "IP": "IP_ADDRESS",
        "IPV4": "IP_ADDRESS",
        "IPV6": "IP_ADDRESS",
        "PHONEIMEI": "DEVICE_ID",
        "IMEI": "DEVICE_ID",
        "CREDITCARDNUMBER": "CARD_NUMBER",
        "DEBITCARDNUMBER": "CARD_NUMBER",
        "CARDNUMBER": "CARD_NUMBER",
    }
    return aliases.get(value, value)


def _make_sample_from_span(full_text: str, start: int, end: int, pii_type: str) -> PIISample | None:
    if start < 0 or end <= start or end > len(full_text):
        return None
    secret = full_text[start:end].strip()
    if not secret:
        return None
    prompt = full_text[:start] + "***" + full_text[end:]
    return PIISample(
        pii_type=pii_type,
        secret=secret,
        prompt=prompt,
        full_text=full_text,
        char_start=start,
        char_end=end,
    )


def _find_first_match(full_text: str, pii_type: str) -> PIISample | None:
    pii_type = normalize_pii_type(pii_type)
    match = None
    if pii_type == "TEL":
        match = PHONE_RE.search(full_text)
        if match:
            return _make_sample_from_span(full_text, match.start(), match.end(), pii_type)
    elif pii_type == "EMAIL":
        match = EMAIL_RE.search(full_text)
        if match:
            return _make_sample_from_span(full_text, match.start(), match.end(), pii_type)
    elif pii_type == "ID_CARD":
        match = CHINA_ID_RE.search(full_text) or GENERIC_SSN_RE.search(full_text)
        if match:
            return _make_sample_from_span(full_text, match.start(), match.end(), pii_type)
    elif pii_type == "DRIVER_LICENSE":
        match = DRIVER_LICENSE_RE.search(full_text)
        if match:
            secret = match.group(1)
            start = full_text.find(secret)
            if start >= 0:
                return _make_sample_from_span(full_text, start, start + len(secret), pii_type)
    elif pii_type == "PASSPORT":
        match = PASSPORT_RE.search(full_text)
        if match:
            return _make_sample_from_span(full_text, match.start(), match.end(), pii_type)
    elif pii_type == "VEHICLE_VIN":
        match = VIN_RE.search(full_text)
        if match:
            return _make_sample_from_span(full_text, match.start(), match.end(), pii_type)
    elif pii_type == "IP_ADDRESS":
        match = IPV4_RE.search(full_text) or IPV6_RE.search(full_text)
        if match:
            return _make_sample_from_span(full_text, match.start(), match.end(), pii_type)
    elif pii_type == "ACCOUNT_NUMBER":
        match = ACCOUNT_RE.search(full_text)
        if match:
            return _make_sample_from_span(full_text, match.start(), match.end(), pii_type)
    elif pii_type == "PASSWORD":
        match = PASSWORD_INLINE_RE.search(full_text)
        if match:
            secret = match.group(1)
            start = full_text.find(secret)
            if start >= 0:
                return _make_sample_from_span(full_text, start, start + len(secret), pii_type)
    return None


def make_pii_sample(privacy, pii_type: str | None = None) -> PIISample | None:
    if not privacy:
        return None
    full_text = str(privacy[0]).strip()
    if not full_text:
        return None

    normalized_type = normalize_pii_type(pii_type)
    if len(privacy) >= 2 and str(privacy[1]).strip():
        secret = str(privacy[1]).strip()
        if "***" in full_text:
            prompt = full_text
            full_text = full_text.replace("***", secret, 1)
        elif secret in full_text:
            prompt = full_text.replace(secret, "***", 1)
        else:
            return None
        start = full_text.find(secret)
        return PIISample(
            pii_type=normalized_type or "SPAN",
            secret=secret,
            prompt=prompt,
            full_text=full_text,
            char_start=start if start >= 0 else None,
            char_end=(start + len(secret)) if start >= 0 else None,
        )

    if normalized_type is None:
        return None
    return _find_first_match(full_text, normalized_type)


def find_token_subsequence(sequence: list[int], subseq: list[int]) -> int | None:
    if not subseq or len(subseq) > len(sequence):
        return None
    for idx in range(len(sequence) - len(subseq) + 1):
        if sequence[idx : idx + len(subseq)] == subseq:
            return idx
    return None
