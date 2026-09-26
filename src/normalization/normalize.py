from __future__ import annotations

import re
import unicodedata
from typing import Optional


try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


# ---------------------------------------------------------
# Basic text normalization
# ---------------------------------------------------------

def normalize_unicode(value: Optional[str]) -> str:
    """
    Unicode-safe normalization.

    Steps:
    1. Handle nulls
    2. Unicode NFKC normalization
    3. Case folding
    4. Replace punctuation/symbols with spaces
    5. Collapse repeated whitespace
    """

    if value is None:
        return ""

    value = str(value)

    if not value:
        return ""

    # Unicode compatibility normalization
    value = unicodedata.normalize("NFKC", value)

    # Case-insensitive normalization
    value = value.casefold()

    # Convert punctuation/symbols into spaces.
    # Keep Unicode letters and digits.
    chars = []

    for ch in value:
        category = unicodedata.category(ch)

        if category.startswith(("L", "N")):
            chars.append(ch)
        else:
            chars.append(" ")

    value = "".join(chars)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ---------------------------------------------------------
# ASCII transliteration
# ---------------------------------------------------------

def normalize_ascii(value: Optional[str]) -> str:
    """
    Unicode-safe normalization followed by optional
    transliteration to ASCII.

    We keep this as a SECOND representation rather
    than replacing the Unicode representation.
    """

    normalized = normalize_unicode(value)

    if not normalized:
        return ""

    if unidecode is not None:
        normalized = unidecode(normalized)

    normalized = normalized.casefold()

    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


# ---------------------------------------------------------
# Compact representation
# ---------------------------------------------------------

def compact(value: Optional[str]) -> str:
    """
    Remove spaces from an already normalized string.

    Example:

        "abc private limited"
        ->
        "abcprivatelimited"
    """

    if not value:
        return ""

    return re.sub(r"[^a-z0-9]", "", value)


# ---------------------------------------------------------
# Token extraction
# ---------------------------------------------------------

def tokenize(value: Optional[str]) -> list[str]:
    """
    Split normalized text into tokens.
    """

    if not value:
        return []

    return [
        token
        for token in value.split()
        if token
    ]


# ---------------------------------------------------------
# Numeric token extraction
# ---------------------------------------------------------

def extract_numeric_tokens(value: Optional[str]) -> list[str]:
    """
    Extract digit-containing tokens.

    Useful for:
        house numbers
        postal codes
        unit numbers
        building numbers
    """

    tokens = tokenize(value)

    return [
        token
        for token in tokens
        if any(ch.isdigit() for ch in token)
    ]


# ---------------------------------------------------------
# Country normalization
# ---------------------------------------------------------

def normalize_country(value: Optional[str]) -> str:
    """
    Normalize country without applying country-specific
    assumptions.
    """

    return normalize_ascii(value)


# ---------------------------------------------------------
# Business-name normalization
# ---------------------------------------------------------

def normalize_name(value: Optional[str]) -> dict[str, object]:
    """
    Produce multiple normalized views for business names.
    """

    unicode_norm = normalize_unicode(value)
    ascii_norm = normalize_ascii(value)

    return {
        "name_norm": unicode_norm,
        "name_ascii": ascii_norm,
        "name_compact": compact(ascii_norm),
        "name_tokens": tokenize(ascii_norm),
        "name_numeric_tokens": extract_numeric_tokens(ascii_norm),
    }


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_address(value: Optional[str]) -> dict[str, object]:
    """
    Produce multiple normalized views for addresses.
    """

    unicode_norm = normalize_unicode(value)
    ascii_norm = normalize_ascii(value)

    return {
        "address_norm": unicode_norm,
        "address_ascii": ascii_norm,
        "address_compact": compact(ascii_norm),
        "address_tokens": tokenize(ascii_norm),
        "address_numeric_tokens": extract_numeric_tokens(ascii_norm),
    }


# ---------------------------------------------------------
# Complete record normalization
# ---------------------------------------------------------

def normalize_record(
    business_name: Optional[str],
    business_address: Optional[str],
    country: Optional[str],
) -> dict[str, object]:

    name = normalize_name(business_name)
    address = normalize_address(business_address)

    return {
        **name,
        **address,
        "country_norm": normalize_country(country),
    }