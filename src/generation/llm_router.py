"""LLM Router — OpenRouter with multi-model fallback chain."""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MODEL_CHAIN = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]


class AllProvidersFailedError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"All models failed: {errors}")


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE,
        )
    return _client


def _is_retryable(exc):
    return isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError))


def call_llm(system_prompt, user_prompt, temperature=0.1,
             max_tokens=1500, primary=None):
    client = _get_client()
    errors = {}
    attempts = []

    for model in MODEL_CHAIN:
        attempts.append(model)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            if text and text.strip():
                provider_name = model.split("/")[-1].replace(":free", "")
                return {
                    "text": text,
                    "provider_used": provider_name,
                    "attempts": attempts,
                }
        except Exception as e:
            errors[model] = str(e)[:100]
            if _is_retryable(e):
                time.sleep(1)
            continue

    raise AllProvidersFailedError(errors)
