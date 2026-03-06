from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol

from .config import LocalLLMConfig


class LocalLLMError(RuntimeError):
    pass


class LocalLLMClient(Protocol):
    def generate(self, prompt: str, system_prompt: str | None = None) -> str: ...


@dataclass
class OllamaLocalLLM:
    config: LocalLLMConfig

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if not self.config.model:
            raise LocalLLMError("Ollama model is not configured.")
        full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        try:
            proc = subprocess.run(
                ["ollama", "run", self.config.model],
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise LocalLLMError("`ollama` command was not found on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise LocalLLMError(f"Ollama generation timed out after {self.config.timeout_seconds}s.") from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise LocalLLMError(f"Ollama generation failed: {stderr or 'unknown error'}")
        return (proc.stdout or "").strip()


@dataclass
class TransformersLocalLLM:
    config: LocalLLMConfig

    def __post_init__(self) -> None:
        if not self.config.model:
            raise LocalLLMError("Transformers model is not configured.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LocalLLMError(
                "Transformers backend requires `transformers` and `torch`. "
                "Install requirements-local-llm.txt first."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = ""
            if system_prompt:
                rendered += f"System:\n{system_prompt}\n\n"
            rendered += f"User:\n{prompt}\n\nAssistant:\n"

        inputs = self._tokenizer(rendered, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_local_llm_client(config: LocalLLMConfig) -> LocalLLMClient | None:
    if not config.backend or not config.model:
        return None
    backend = config.backend.lower()
    if backend == "ollama":
        return OllamaLocalLLM(config)
    if backend == "transformers":
        return TransformersLocalLLM(config)
    raise LocalLLMError(f"Unsupported local LLM backend: {config.backend}")
