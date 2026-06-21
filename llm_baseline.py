"""
llm_baseline.py
================
Baseline Large Language Model (LLM) system: direct question answering
using Llama 3 8B Instruct with NO retrieval component. This serves as the
control arm in the RAG vs. LLM comparative study, mirroring the "standard
LLM" baseline described in Section III-B-2 and Table II of the source
paper.

Supports three interchangeable inference backends, selected via
config.MODEL_CONFIG.backend:
  - "ollama":        calls a local Ollama server (recommended for
                      reproducing this project on a laptop/workstation).
  - "hf_inference":   calls a HuggingFace Inference Endpoint / API.
  - "local":          loads the model in-process via transformers
                      (requires a GPU with sufficient VRAM for 8B params).

Usage:
    from llm_baseline import BaselineLLM

    llm = BaselineLLM()
    response = llm.answer("What are the symptoms of type 2 diabetes?")
    print(response.answer, response.latency_seconds, response.token_count)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

import config
import prompts
from logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Container for a single baseline-LLM generation result."""

    question: str
    answer: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMBackendError(RuntimeError):
    """Raised when the configured inference backend fails or is unreachable."""


def _estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token heuristic).

    Used as a fallback when the backend does not report exact token usage,
    so that latency/efficiency comparisons in evaluation.py always have a
    token-usage figure to plot.
    """
    return max(1, len(text) // 4)


class BaselineLLM:
    """Direct-answer LLM system with no retrieval augmentation.

    This class is intentionally backend-agnostic: it builds the same
    system/user prompt regardless of where the model actually runs, which
    keeps the experimental comparison against rag_pipeline.RAGPipeline
    fair (identical prompting style, only the presence/absence of
    retrieved context differs).
    """

    def __init__(self, model_config: Optional[config.ModelConfig] = None) -> None:
        self.cfg = model_config or config.MODEL_CONFIG
        logger.info(
            "Initialized BaselineLLM with backend=%s model_id=%s",
            self.cfg.backend,
            self.cfg.model_id if self.cfg.backend != "ollama" else self.cfg.ollama_model_name,
        )
        self._local_pipeline = None  # lazy-loaded only for backend == "local"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> LLMResponse:
        """Generate a direct answer to `question` with no retrieval.

        Args:
            question: The healthcare question posed by the user.

        Returns:
            LLMResponse with the generated text, latency, and token usage.

        Raises:
            LLMBackendError: If the configured backend fails after the
                generation attempt (network error, missing model, etc.).
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        system_prompt = prompts.LLM_SYSTEM_PROMPT
        user_prompt = prompts.LLM_USER_TEMPLATE.format(question=question.strip())

        start = time.perf_counter()
        try:
            answer_text = self._dispatch(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - we re-raise as a typed error
            logger.exception("Baseline LLM generation failed")
            raise LLMBackendError(str(exc)) from exc
        latency = time.perf_counter() - start

        prompt_tokens = _estimate_tokens(system_prompt + user_prompt)
        completion_tokens = _estimate_tokens(answer_text)

        logger.info(
            "Baseline LLM answered in %.2fs (%d completion tokens)",
            latency,
            completion_tokens,
        )

        return LLMResponse(
            question=question,
            answer=answer_text.strip(),
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ------------------------------------------------------------------ #
    # Backend dispatch
    # ------------------------------------------------------------------ #
    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        if self.cfg.backend == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        if self.cfg.backend == "hf_inference":
            return self._call_hf_inference(system_prompt, user_prompt)
        if self.cfg.backend == "local":
            return self._call_local(system_prompt, user_prompt)
        raise LLMBackendError(f"Unknown backend: {self.cfg.backend!r}")

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call a local Ollama server running `ollama pull llama3:8b-instruct`."""
        url = f"{self.cfg.ollama_base_url}/api/chat"
        payload = {
            "model": self.cfg.ollama_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": self.cfg.max_new_tokens,
                "repeat_penalty": self.cfg.repetition_penalty,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMBackendError(
                f"Could not reach Ollama at {url}. Is `ollama serve` running and "
                f"have you run `ollama pull {self.cfg.ollama_model_name}`? "
                f"Original error: {exc}"
            ) from exc
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _call_hf_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Call a HuggingFace Inference Endpoint / Serverless API."""
        if not self.cfg.hf_token:
            raise LLMBackendError(
                "HF_TOKEN environment variable is required for the hf_inference backend."
            )
        url = f"https://api-inference.huggingface.co/models/{self.cfg.model_id}"
        headers = {"Authorization": f"Bearer {self.cfg.hf_token}"}
        chat_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        payload = {
            "inputs": chat_text,
            "parameters": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "max_new_tokens": self.cfg.max_new_tokens,
                "repetition_penalty": self.cfg.repetition_penalty,
                "return_full_text": False,
            },
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMBackendError(f"HF Inference API call failed: {exc}") from exc
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        raise LLMBackendError(f"Unexpected HF Inference API response shape: {data}")

    def _call_local(self, system_prompt: str, user_prompt: str) -> str:
        """Run inference in-process via transformers. Requires `transformers`,
        `torch`, and sufficient GPU VRAM (~16GB+ recommended for 8B in fp16)."""
        if self._local_pipeline is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise LLMBackendError(
                    "backend='local' requires `pip install transformers torch accelerate`."
                ) from exc
            logger.info("Loading local model %s ... this may take a while.", self.cfg.model_id)
            self._local_pipeline = pipeline(
                "text-generation",
                model=self.cfg.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        outputs = self._local_pipeline(
            messages,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            do_sample=self.cfg.temperature > 0,
        )
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            # chat-formatted output: last message is the assistant turn
            return generated[-1]["content"]
        return str(generated)


if __name__ == "__main__":
    # Quick manual smoke test: `python llm_baseline.py`
    sample_question = "What are the early warning signs of sepsis?"
    baseline = BaselineLLM()
    try:
        result = baseline.answer(sample_question)
        print(f"Q: {sample_question}\n\nA: {result.answer}\n")
        print(f"Latency: {result.latency_seconds:.2f}s | Tokens: {result.total_tokens}")
    except LLMBackendError as e:
        print(f"[Backend unavailable in this environment] {e}")
