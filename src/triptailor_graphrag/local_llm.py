from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import LocalLLMConfig
from .types import Candidate, PlanResult, QuerySpec
from .utils import clamp, extract_json_payload


@dataclass
class JudgeScore:
    score: float
    reason: str


class LocalLLM:
    def __init__(self, config: LocalLLMConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled and bool(self.config.model_path)

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        if not self.enabled:
            raise RuntimeError("Local LLM is not enabled")

        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise RuntimeError("Local LLM failed to initialize")

        prompt = self._build_prompt(system_prompt, user_prompt)
        encoded = tokenizer(prompt, return_tensors="pt")

        model_device = getattr(model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if self.config.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.config.temperature
            generate_kwargs["top_p"] = self.config.top_p
        else:
            generate_kwargs["do_sample"] = False

        outputs = model.generate(**encoded, **generate_kwargs)
        prompt_length = encoded["input_ids"].shape[-1]
        generated = outputs[0][prompt_length:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> dict[str, Any] | None:
        text = self.generate(system_prompt, user_prompt, max_new_tokens=max_new_tokens)
        payload = extract_json_payload(text)
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local LLM dependencies are missing. Run: python3 -m pip install -r requirements-local-llm.txt"
            ) from exc

        tokenizer_path = self.config.tokenizer_path or self.config.model_path
        load_kwargs = {"trust_remote_code": self.config.trust_remote_code}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **load_kwargs)

        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        dtype = self._resolve_dtype(torch)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id

        self._tokenizer = tokenizer
        self._model = model

    def _resolve_dtype(self, torch_module: Any) -> Any | None:
        dtype_name = (self.config.torch_dtype or "auto").lower()
        if dtype_name == "auto":
            return None
        mapping = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        return mapping.get(dtype_name)

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        if len(user_prompt) > self.config.max_input_chars:
            user_prompt = user_prompt[: self.config.max_input_chars]

        tokenizer = self._tokenizer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        return (
            "System:\n"
            + system_prompt
            + "\n\nUser:\n"
            + user_prompt
            + "\n\nAssistant:\n"
        )


class LocalPersonalizationJudge:
    def __init__(self, llm: LocalLLM | None) -> None:
        self.llm = llm

    def score(self, query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> JudgeScore | None:
        if self.llm is None or not self.llm.enabled:
            return None

        system_prompt = (
            "You are an objective travel-planning judge. "
            "Score how well the plan satisfies the user request. "
            "Return only JSON with keys score and reason. "
            "score must be a number between 0 and 1."
        )
        user_prompt = self._build_prompt(query, plan, candidate_map)
        result = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.llm.config.judge_max_new_tokens,
        )
        if not result:
            return None

        try:
            score = clamp(float(result.get("score", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            return None
        reason = str(result.get("reason", "")).strip()
        return JudgeScore(score=score, reason=reason)

    def _build_prompt(self, query: QuerySpec, plan: PlanResult, candidate_map: dict[str, Candidate]) -> str:
        hotel_name = plan.hotel[0].get("name") if plan.hotel else "None"
        hotel_id = plan.hotel[0].get("candidate_id") if plan.hotel else None
        hotel_category = ""
        if hotel_id and hotel_id in candidate_map:
            hotel_category = str(candidate_map[hotel_id].meta.get("category", ""))

        day_lines: list[str] = []
        for day_key, acts in plan.itinerary.items():
            items = []
            for act in acts:
                candidate = candidate_map.get(act.get("candidate_id"))
                tags = candidate.tags[:3] if candidate else []
                items.append(
                    {
                        "time": act.get("time"),
                        "action": act.get("action"),
                        "candidate_id": act.get("candidate_id"),
                        "location": act.get("location"),
                        "price": act.get("price"),
                        "tags": tags,
                    }
                )
            day_lines.append(f"{day_key}: {json.dumps(items, ensure_ascii=False)}")

        request_info = {
            "query": query.query_text,
            "budget": query.budget,
            "meal_price_range": query.meal_price_range,
            "hotel_category_pref": query.hotel_category_pref,
            "intensity_pref": query.intensity_pref,
            "interest_tags": query.interest_tags,
        }
        plan_info = {
            "hotel": {"name": hotel_name, "category": hotel_category},
            "transportation": plan.transportation,
            "days": day_lines,
        }
        return (
            "Request:\n"
            + json.dumps(request_info, ensure_ascii=False, indent=2)
            + "\n\nPlan:\n"
            + json.dumps(plan_info, ensure_ascii=False, indent=2)
            + "\n\nScore the plan on constraint satisfaction, interest coverage, and itinerary intensity fit."
        )
