"""
LLM Router with Provider Fallback Chain
=======================================
Routes generation requests through a fallback chain:
  1. Groq (Llama 3.3 70B) - primary, fast, high-quality
  2. Gemini (gemini-2.0-flash) - secondary, generous free tier

If both fail, raises AllProvidersFailedError so callers can show
graceful degradation messages.

Design notes:
  - Provider-specific errors are normalized to a common interface.
  - Lazy client init (no network calls at import time).
  - Returns provider_used so callers can log which model answered.
"""

import os
import time

from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from groq import APIConnectionError as GroqConnectionError
from groq import APITimeoutError as GroqTimeoutError

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

load_dotenv()


# --- Configuration ---

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1500
REQUEST_TIMEOUT = 60

PER_PROVIDER_RETRIES = 1
RETRY_BACKOFF_SECONDS = 2


# --- Custom exception ---

class AllProvidersFailedError(Exception):
    """Raised when all providers in the fallback chain fail."""
    def __init__(self, errors: dict):
        self.errors = errors
        msg = "All LLM providers failed:\n" + "\n".join(
            f"  - {provider}: {err}" for provider, err in errors.items()
        )
        super().__init__(msg)


# --- Lazy singletons ---

_groq_client = None
_gemini_configured = False


def _get_groq_client():
    """Lazy-initialize the Groq client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")
        _groq_client = Groq(api_key=api_key, timeout=REQUEST_TIMEOUT)
    return _groq_client


def _ensure_gemini_configured():
    """Lazy-configure the Gemini SDK (uses module-level config)."""
    global _gemini_configured
    if not _gemini_configured:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        genai.configure(api_key=api_key)
        _gemini_configured = True


# --- Provider-specific call wrappers ---

def _call_groq(system_prompt, user_prompt, temperature, max_tokens, json_mode=False):
    """Call Groq. Returns response text. Raises on failure."""
    client = _get_groq_client()

    kwargs = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_gemini(system_prompt, user_prompt, temperature, max_tokens, json_mode=False):
    """Call Gemini. Returns response text. Raises on failure."""
    _ensure_gemini_configured()

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    # Gemini doesn't have a separate "system" role; prepend it
    combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=generation_config,
    )
    response = model.generate_content(combined_prompt)
    return response.text


# --- Error classification ---

def _is_retryable(exc):
    """Return True if this error is worth retrying the same provider."""
    return isinstance(exc, (GroqTimeoutError, GroqConnectionError))


def _is_failover_trigger(exc):
    """Return True if this error means we should try the next provider."""
    if isinstance(exc, GroqRateLimitError):
        return True
    if isinstance(exc, google_exceptions.ResourceExhausted):
        return True
    if isinstance(exc, google_exceptions.ServiceUnavailable):
        return True
    if isinstance(exc, GroqConnectionError):
        return True
    return False


# --- Public API: the router ---

def call_llm(
    system_prompt,
    user_prompt,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    json_mode=False,
    primary="groq",
):
    """
    Call an LLM through the fallback chain.

    Returns:
        {
            "text": <response string>,
            "provider_used": "groq" | "gemini",
            "attempts": <list of provider names tried>,
        }

    Raises:
        AllProvidersFailedError if every provider fails.
    """
    if primary == "groq":
        chain = [("groq", _call_groq), ("gemini", _call_gemini)]
    elif primary == "gemini":
        chain = [("gemini", _call_gemini), ("groq", _call_groq)]
    else:
        raise ValueError(f"Unknown primary: {primary}")

    errors = {}
    attempts = []

    for provider_name, provider_fn in chain:
        attempts.append(provider_name)

        last_err = None
        for retry_idx in range(PER_PROVIDER_RETRIES + 1):
            try:
                text = provider_fn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

                if not text or not text.strip():
                    raise RuntimeError(f"{provider_name} returned empty response")

                return {
                    "text": text,
                    "provider_used": provider_name,
                    "attempts": attempts,
                }

            except Exception as e:
                last_err = e

                if _is_retryable(e) and retry_idx < PER_PROVIDER_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * (2 ** retry_idx))
                    continue

                break

        errors[provider_name] = str(last_err)

        if last_err is not None and not _is_failover_trigger(last_err):
            raise AllProvidersFailedError(errors)

    raise AllProvidersFailedError(errors)


# --- Smoke test CLI (DO NOT RUN TODAY - both providers throttled) ---

def main():
    """Test the fallback chain. Skipped today; run tomorrow when quotas reset."""
    print("=" * 60)
    print("LLM Router Smoke Test")
    print("=" * 60)

    print("\nCalling router with primary=groq...")
    try:
        result = call_llm(
            system_prompt="You are a helpful assistant. Reply concisely.",
            user_prompt="What is 2+2? Respond with just the number.",
            temperature=0.0,
            max_tokens=20,
        )
        print(f"  Provider used: {result['provider_used']}")
        print(f"  Attempts: {result['attempts']}")
        print(f"  Response: {result['text'].strip()}")
    except AllProvidersFailedError as e:
        print(f"  All providers failed: {e}")


if __name__ == "__main__":
    main()