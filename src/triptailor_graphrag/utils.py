from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
TIME_RANGE_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*[\-–—~]\s*(\d{1,2}:\d{2})")
BUDGET_PATTERN = re.compile(r"budget of [¥$]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
MEAL_RANGE_PATTERN = re.compile(
    r"meal costs?\s*(?:ranging from)?\s*[¥$]?\s*([0-9]+)\s*(?:to|-)\s*[¥$]?\s*([0-9]+)",
    re.IGNORECASE,
)
MEAL_OVER_PATTERN = re.compile(r"meal costs?\s*(?:over|above)\s*[¥$]?\s*([0-9]+)", re.IGNORECASE)

HOTEL_CATEGORY_KEYWORDS = {
    "economy": "Economy",
    "midscale": "Midscale",
    "luxury": "Luxury",
    "high-end": "Luxury",
    "budget": "Economy",
}

INTENSITY_KEYWORDS = {
    "relaxed": "low",
    "low": "low",
    "moderate": "moderate",
    "balanced": "moderate",
    "intense": "high",
    "high": "high",
    "packed": "high",
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def slugify(text: str | None) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or "na"


def tokenize(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    return TOKEN_PATTERN.findall(normalized)


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            cleaned = value.strip().replace("/5", "").replace("/10", "")
            return float(cleaned)
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_budget(query_text: str) -> float | None:
    match = BUDGET_PATTERN.search(query_text or "")
    if not match:
        return None
    return float(match.group(1))


def parse_meal_price_range(query_text: str) -> tuple[float, float] | None:
    query = query_text or ""
    match = MEAL_RANGE_PATTERN.search(query)
    if match:
        lo, hi = float(match.group(1)), float(match.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    match = MEAL_OVER_PATTERN.search(query)
    if match:
        return float(match.group(1)), 2_147_483_647.0
    return None


def infer_hotel_category(query_text: str) -> str | None:
    normalized = normalize_text(query_text)
    for key, value in HOTEL_CATEGORY_KEYWORDS.items():
        if key in normalized:
            return value
    return None


def infer_intensity(query_text: str) -> str | None:
    normalized = normalize_text(query_text)
    for key, value in INTENSITY_KEYWORDS.items():
        if key in normalized:
            return value
    return None


def extract_interest_tags(query_text: str) -> list[str]:
    query = normalize_text(query_text)
    tags: list[str] = []
    anchor = "interested in"
    if anchor in query:
        segment = query.split(anchor, 1)[1]
        for boundary in ["along with", "the itinerary", "itinerary should"]:
            if boundary in segment:
                segment = segment.split(boundary, 1)[0]
                break
        parts = re.split(r",| and |/", segment)
        tags.extend(part.strip(" .") for part in parts if part.strip())
    return [t for t in dict.fromkeys(tags) if len(t) > 2]


def parse_time_to_minutes(hhmm: str) -> int | None:
    if not hhmm:
        return None
    try:
        hh, mm = hhmm.split(":", 1)
        hh_i, mm_i = int(hh), int(mm)
        if 0 <= hh_i < 24 and 0 <= mm_i < 60:
            return hh_i * 60 + mm_i
    except (ValueError, AttributeError):
        return None
    return None


def parse_time_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = TIME_RANGE_PATTERN.search(value)
    if not match:
        return None
    start = parse_time_to_minutes(match.group(1))
    end = parse_time_to_minutes(match.group(2))
    if start is None or end is None:
        return None
    return start, end


def intervals_overlap(interval_a: tuple[int, int], interval_b: tuple[int, int]) -> bool:
    return interval_a[0] < interval_b[1] and interval_b[0] < interval_a[1]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def unique_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
