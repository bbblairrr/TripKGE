from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable

DURATION_RANGE_PATTERN = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:to|-)\s*([0-9]+(?:\.[0-9]+)?)\s*(minutes?|hours?|days?)",
    re.IGNORECASE,
)
DURATION_SINGLE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(minutes?|hours?|days?)", re.IGNORECASE)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
TIME_RANGE_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*(?:-|to)\s*(\d{1,2}:\d{2})", re.IGNORECASE)
BUDGET_PATTERN = re.compile(r"budget of [楼$]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
ANSI_ESCAPE_PATTERN = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07\x1B]*(?:\x07|\x1B\\))"
)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
MEAL_RANGE_PATTERN = re.compile(
    r"meal costs?\s*(?:ranging from)?\s*[楼$]?\s*([0-9]+)\s*(?:to|-)\s*[楼$]?\s*([0-9]+)",
    re.IGNORECASE,
)
MEAL_OVER_PATTERN = re.compile(r"meal costs?\s*(?:over|above)\s*[楼$]?\s*([0-9]+)", re.IGNORECASE)

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


def sanitize_llm_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = ANSI_ESCAPE_PATTERN.sub("", cleaned)
    cleaned = ZERO_WIDTH_PATTERN.sub("", cleaned)
    cleaned = CONTROL_CHAR_PATTERN.sub("", cleaned)
    return cleaned.strip()


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
    if end < start:
        end += 24 * 60
    return start, end


def format_minutes_hhmm(total_minutes: int) -> str:
    minutes = max(0, total_minutes)
    hh = (minutes // 60) % 24
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"


def format_time_range(start_minutes: int, end_minutes: int) -> str:
    return f"{format_minutes_hhmm(start_minutes)}-{format_minutes_hhmm(end_minutes)}"


def parse_opening_hours(value: str | None) -> tuple[int, int] | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    if "24/7" in normalized or "24 hours" in normalized:
        return 0, 24 * 60
    return parse_time_range(value)


def parse_duration_minutes(value: str | None) -> int | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    match = DURATION_RANGE_PATTERN.search(normalized)
    if match:
        return _duration_value_to_minutes(float(match.group(1)), match.group(3))
    match = DURATION_SINGLE_PATTERN.search(normalized)
    if match:
        return _duration_value_to_minutes(float(match.group(1)), match.group(2))
    return None


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


def _duration_value_to_minutes(value: float, unit: str) -> int:
    unit_norm = normalize_text(unit)
    if unit_norm.startswith("minute"):
        return int(round(value))
    if unit_norm.startswith("hour"):
        return int(round(value * 60))
    if unit_norm.startswith("day"):
        return int(round(value * 8 * 60))
    return int(round(value))
