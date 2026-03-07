from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import Candidate, QuerySpec
from .utils import (
    extract_interest_tags,
    infer_hotel_category,
    infer_intensity,
    normalize_text,
    parse_budget,
    parse_meal_price_range,
    safe_float,
    safe_int,
    slugify,
)


@dataclass
class DatasetBundle:
    train_samples: list[dict[str, Any]]
    test_samples: list[dict[str, Any]]
    info_by_pid: dict[str, dict[str, Any]]
    query_specs: list[QuerySpec]
    candidates_by_pid: dict[int, list[Candidate]]
    candidates_global: dict[str, Candidate]
    city_transport: dict[str, list[dict[str, Any]]]


class DataLoader:
    def __init__(
        self,
        data_dir: Path,
        train_file: str = "train.json",
        eval_file: str = "test.json",
        info_file: str = "infomation.json",
    ):
        self.data_dir = Path(data_dir)
        self.train_file = train_file
        self.eval_file = eval_file
        self.info_file = info_file

    def load(self) -> DatasetBundle:
        train_samples = self._load_json(self.train_file)
        test_samples = self._load_json(self.eval_file)
        info_by_pid = self._load_json(self.info_file)

        query_specs = [self._build_query_spec(item) for item in test_samples]

        candidates_by_pid: dict[int, list[Candidate]] = {}
        candidates_global: dict[str, Candidate] = {}
        for sample in test_samples:
            pid = safe_int(sample.get("pid"))
            info = info_by_pid.get(str(pid), {})
            candidates = self._build_candidates_from_info(sample, info)
            candidates_by_pid[pid] = candidates
            for c in candidates:
                candidates_global[c.candidate_id] = c

        city_transport = self._build_city_transport()

        return DatasetBundle(
            train_samples=train_samples,
            test_samples=test_samples,
            info_by_pid=info_by_pid,
            query_specs=query_specs,
            candidates_by_pid=candidates_by_pid,
            candidates_global=candidates_global,
            city_transport=city_transport,
        )

    def _load_json(self, name: str) -> Any:
        path = self.data_dir / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_csv(self, name: str) -> list[dict[str, str]]:
        path = self.data_dir / name
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _build_query_spec(self, item: dict[str, Any]) -> QuerySpec:
        query = item.get("query", "")
        meal_raw = item.get("meal_price_range")
        meal: tuple[float, float] | None = None
        if isinstance(meal_raw, list) and len(meal_raw) == 2:
            meal = (float(meal_raw[0]), float(meal_raw[1]))
        if meal is None:
            meal = parse_meal_price_range(query)

        budget = item.get("budget")
        if budget is None:
            budget = parse_budget(query)

        return QuerySpec(
            pid=safe_int(item.get("pid")),
            departure_city=str(item.get("departure_city") or "").strip(),
            destination_city=str(item.get("destination_city") or "").strip(),
            day=max(1, safe_int(item.get("day"), 1)),
            budget=safe_float(budget, default=0.0) if budget is not None else None,
            meal_price_range=meal,
            query_text=query,
            hotel_category_pref=infer_hotel_category(query),
            intensity_pref=infer_intensity(query),
            interest_tags=extract_interest_tags(query),
        )

    def _build_candidates_from_info(self, sample: dict[str, Any], info: dict[str, Any]) -> list[Candidate]:
        city = str(sample.get("destination_city") or "").strip()
        candidates: list[Candidate] = []

        for hotel in info.get("hotel", []):
            name = str(hotel.get("Hotel Name") or hotel.get("name") or "Unknown Hotel")
            category = str(hotel.get("Category") or "")
            rating = str(hotel.get("Rating") or "")
            product = str(hotel.get("Product Rating") or "")
            env = str(hotel.get("Environment Rating") or "")
            service = str(hotel.get("Service Rating") or "")
            price = safe_float(hotel.get("Average Price"))
            lat = hotel.get("latitude")
            lon = hotel.get("longitude")
            cid = f"hotel:{slugify(city)}:{slugify(name)}"
            text = (
                f"hotel {name} in {city}, category {category}, avg price {price}, "
                f"rating {rating}, product {product}, environment {env}, service {service}"
            )
            candidates.append(
                Candidate(
                    candidate_id=cid,
                    entity_type="hotel",
                    city=city,
                    name=name,
                    price=price,
                    latitude=safe_float(lat, default=0.0) if lat is not None else None,
                    longitude=safe_float(lon, default=0.0) if lon is not None else None,
                    text=text,
                    tags=[category] if category else [],
                    meta={"category": category, "rating": rating},
                )
            )

        for attr in info.get("attractions", []):
            poiid = str(attr.get("poiid") or attr.get("poiId") or slugify(attr.get("name_en") or attr.get("name")))
            name = str(attr.get("name_en") or attr.get("name") or "Unknown Attraction")
            tag = str(attr.get("tag") or "")
            summary = str(attr.get("summary") or "")
            features = str(attr.get("shortFeatures") or "")
            opening = str(attr.get("opening_hours") or "")
            duration = str(attr.get("recommended_duration") or "")
            price = safe_float(attr.get("price"))
            lat = attr.get("latitude")
            lon = attr.get("longitude")
            cid = f"attraction:{poiid}"
            text = (
                f"attraction {name} in {city}, price {price}, tag {tag}, "
                f"features {features}, summary {summary}, opening {opening}, duration {duration}"
            )
            tags = [t.strip() for t in tag.split(";") if t.strip()]
            candidates.append(
                Candidate(
                    candidate_id=cid,
                    entity_type="attraction",
                    city=city,
                    name=name,
                    price=price,
                    latitude=safe_float(lat, default=0.0) if lat is not None else None,
                    longitude=safe_float(lon, default=0.0) if lon is not None else None,
                    text=text,
                    tags=tags,
                    meta={
                        "opening_hours": opening,
                        "recommended_duration": duration,
                        "name_raw": attr.get("name"),
                    },
                )
            )

        for rest in info.get("restaurants", []):
            name = str(rest.get("name_en") or rest.get("name") or "Unknown Restaurant")
            tag = str(rest.get("tag") or rest.get("small_cate") or "")
            price = safe_float(rest.get("price") or rest.get("avg_price"))
            rating = str(rest.get("Rating") or rest.get("stars") or "")
            product = str(rest.get("Product Rating") or rest.get("product_rating") or "")
            env = str(rest.get("Environment Rating") or rest.get("environment_rating") or "")
            service = str(rest.get("Service Rating") or rest.get("service_rating") or "")
            lat = rest.get("latitude")
            lon = rest.get("longitude")
            cid = f"restaurant:{slugify(city)}:{slugify(name)}"
            text = (
                f"restaurant {name} in {city}, cuisine {tag}, avg meal price {price}, "
                f"rating {rating}, product {product}, environment {env}, service {service}"
            )
            candidates.append(
                Candidate(
                    candidate_id=cid,
                    entity_type="restaurant",
                    city=city,
                    name=name,
                    price=price,
                    latitude=safe_float(lat, default=0.0) if lat is not None else None,
                    longitude=safe_float(lon, default=0.0) if lon is not None else None,
                    text=text,
                    tags=[t.strip() for t in tag.split(";") if t.strip()],
                    meta={"rating": rating},
                )
            )

        return candidates

    def _build_city_transport(self) -> dict[str, list[dict[str, Any]]]:
        by_city: dict[str, list[dict[str, Any]]] = {}

        for row in self._load_csv("Flight_Schedule.csv"):
            dep = str(row.get("Departure City") or "").strip()
            arr = str(row.get("Arrival City") or "").strip()
            rec = {
                "mode": "flight",
                "from": dep,
                "to": arr,
                "number": row.get("Flight Number"),
                "departure": row.get("Departure Time"),
                "arrival": row.get("Arrival Time"),
                "price": safe_float(row.get("Price")),
                "distance_km": safe_float(row.get("Distance (km)")),
                "on_time": safe_float(row.get("On-Time Performance")),
            }
            by_city.setdefault(dep, []).append(rec)

        # Train schedule is station-based. We approximate city by first token in station name.
        train_rows = self._load_csv("Train_Schedule.csv")
        trains: dict[str, list[dict[str, str]]] = {}
        for row in train_rows:
            trains.setdefault(str(row.get("Train_Number") or ""), []).append(row)

        for train_no, stations in trains.items():
            if len(stations) < 2:
                continue
            ordered = sorted(stations, key=lambda x: safe_int(x.get("Station_Number")))
            first = ordered[0]
            last = ordered[-1]
            dep_station = str(first.get("Station_Name") or "")
            arr_station = str(last.get("Station_Name") or "")
            dep_city = dep_station.split()[0]
            arr_city = arr_station.split()[0]
            rec = {
                "mode": "train",
                "from": dep_city,
                "to": arr_city,
                "number": train_no,
                "departure": first.get("Departure_Time"),
                "arrival": last.get("Arrival_Time"),
                "price": safe_float(last.get("Second_Class_Price")),
                "distance_km": None,
            }
            by_city.setdefault(dep_city, []).append(rec)

        return by_city


def match_candidate_by_name(candidates: list[Candidate], name: str) -> Candidate | None:
    target = normalize_text(name)
    if not target:
        return None

    by_exact = {normalize_text(c.name): c for c in candidates}
    if target in by_exact:
        return by_exact[target]

    for c in candidates:
        n = normalize_text(c.name)
        raw = normalize_text(c.meta.get("name_raw") if isinstance(c.meta, dict) else "")
        if target == raw:
            return c
        if target in n or n in target:
            return c
    return None
