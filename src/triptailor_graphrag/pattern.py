from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .utils import parse_time_range

HEADER_PATTERN = re.compile(r"^###\s*([^|]+?)\|\s*(.+)$")


@dataclass
class ItineraryPattern:
    day_count: int
    signature: tuple[tuple[str, ...], ...]
    support: int


class PatternMiner:
    def __init__(self) -> None:
        self.patterns_by_day: dict[int, list[ItineraryPattern]] = {}

    def fit(self, samples: list[dict[str, Any]]) -> None:
        grouped: dict[int, Counter[tuple[tuple[str, ...], ...]]] = defaultdict(Counter)
        for sample in samples:
            day_count = int(sample.get("day") or 0)
            if day_count <= 0:
                continue
            plans = sample.get("final_plan")
            if not isinstance(plans, list) or not plans:
                continue
            signature = self._extract_signature(plans, day_count)
            grouped[day_count][signature] += 1

        patterns_by_day: dict[int, list[ItineraryPattern]] = {}
        for day_count, counter in grouped.items():
            ranked = counter.most_common(8)
            patterns_by_day[day_count] = [
                ItineraryPattern(day_count=day_count, signature=sig, support=sup)
                for sig, sup in ranked
            ]
        self.patterns_by_day = patterns_by_day

    def get_pattern(self, day_count: int) -> ItineraryPattern:
        if day_count in self.patterns_by_day and self.patterns_by_day[day_count]:
            return self.patterns_by_day[day_count][0]
        return self._default_pattern(day_count)

    def _extract_signature(self, day_texts: list[str], day_count: int) -> tuple[tuple[str, ...], ...]:
        day_slots: list[tuple[str, ...]] = []
        for day_idx in range(day_count):
            text = day_texts[day_idx] if day_idx < len(day_texts) else ""
            parsed: list[tuple[int, str]] = []
            for line in text.splitlines():
                line = line.strip()
                match = HEADER_PATTERN.match(line)
                if not match:
                    continue
                time_part = match.group(1).strip()
                title = match.group(2).strip()
                timerange = parse_time_range(time_part)
                start = timerange[0] if timerange else 24 * 60
                slot = self._time_to_slot(start)
                action = self._classify_action(title)
                parsed.append((start, f"{slot}:{action}"))

            parsed.sort(key=lambda x: x[0])
            if parsed:
                tokens = tuple(x[1] for x in parsed[:8])
            else:
                tokens = self._default_day_tokens(day_idx, day_count)
            day_slots.append(tokens)

        return tuple(day_slots)

    def _default_pattern(self, day_count: int) -> ItineraryPattern:
        signature = tuple(self._default_day_tokens(i, day_count) for i in range(day_count))
        return ItineraryPattern(day_count=day_count, signature=signature, support=0)

    def _default_day_tokens(self, day_idx: int, day_count: int) -> tuple[str, ...]:
        if day_idx == 0:
            return ("afternoon:transport", "afternoon:checkin", "evening:dining", "evening:sightseeing")
        if day_idx == day_count - 1:
            return ("morning:sightseeing", "afternoon:dining", "afternoon:transport")
        return ("morning:sightseeing", "noon:dining", "afternoon:sightseeing", "evening:dining")

    def _classify_action(self, title: str) -> str:
        lower = title.lower()
        if any(k in lower for k in ["check-in", "check in", "check out"]):
            return "checkin"
        if any(k in lower for k in ["flight", "train", "travel", "airport", "station", "return"]):
            return "transport"
        if any(k in lower for k in ["lunch", "dinner", "breakfast", "dining", "restaurant", "meal"]):
            return "dining"
        return "sightseeing"

    def _time_to_slot(self, minute: int) -> str:
        if minute < 0:
            return "unknown"
        if minute < 11 * 60:
            return "morning"
        if minute < 14 * 60:
            return "noon"
        if minute < 18 * 60:
            return "afternoon"
        return "evening"
