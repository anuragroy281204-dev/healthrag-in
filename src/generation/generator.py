"""
Groq LLM Generator
==================
Wraps the Groq API for the generation layer of HealthRAG-IN.
Takes a user question and retrieved chunks, returns a structured
medical answer with citations.

Loads the GROQ_API_KEY from .env automatically.

Usage:
    from src.generation.generator import GroqGenerator
    gen = GroqGenerator()
    answer = gen.generate(question, retrieved_chunks)

Run as CLI for a smoke test:
    python -m src.generation.generator
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.generation.prompts import (
    SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_user_message,
    is_refusal,
    is_emergency_response,
)

# Load environment variables from .env at the project root
load_dotenv()


# --- Configuration ---

# Groq model. Llama 3.3 70B is free, fast, and high-quality.
# Alternatives if rate-limited: "llama-3.1-8b-instant" (smaller, faster)
MODEL_NAME = "llama-3.3-70b-versatile"

# Temperature controls randomness. Low = deterministic (what we want for medical RAG).
TEMPERATURE = 0.1

# Max tokens in the response. ~1500 tokens = ~1100 words, plenty for our format.
MAX_TOKENS = 1500

# Request timeout in seconds.
TIMEOUT = 60


# --- Generator class ---

class GroqGenerator:
    """
    Encapsulates the Groq API client and the call/response logic.
    Loads the API key from .env at init time.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found.\n"
                "  Make sure you have a .env file at the project root with:\n"
                "    GROQ_API_KEY=gsk_your_key_here"
            )

        self.client = Groq(api_key=api_key, timeout=TIMEOUT)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = PROMPT_VERSION

    def generate(
        self,
        question: str,
        retrieved_chunks: list,
    ) -> dict:
        """
        Send the question + retrieved chunks to Groq and return a
        structured response.

        Returns a dict with:
          - text: the raw response from the LLM
          - is_refusal: True if the model refused (out-of-scope)
          - is_emergency: True if the model issued an emergency warning
          - cited_source_numbers: set of source numbers cited in the response
          - prompt_version: version tag for evaluation tracking
          - model_name: which Groq model was used
        """
        user_message = build_user_message(question, retrieved_chunks)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response_text = response.choices[0].message.content

        # Extract which source numbers were cited
        cited_numbers = extract_citation_numbers(response_text)

        return {
            "text": response_text,
            "is_refusal": is_refusal(response_text),
            "is_emergency": is_emergency_response(response_text),
            "cited_source_numbers": cited_numbers,
            "prompt_version": self.prompt_version,
            "model_name": self.model_name,
        }


# --- Citation parsing ---

def extract_citation_numbers(text: str) -> set:
    """
    Find all citation markers in the response text.

    Matches patterns like [1], [1, 3], [1,3], [12], etc.
    Returns a set of integer source numbers that the model cited.

    This is used by the validation layer to check that:
      1. The model actually cited sources (not just claimed to)
      2. The cited numbers map to real retrieved chunks
    """
    cited = set()
    # Pattern: [ N (, N)* ] - matches [1], [2,3], [1, 4, 7], etc.
    matches = re.findall(r"\[([\d,\s]+)\]", text)
    for match in matches:
        for num_str in match.split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                cited.add(int(num_str))
    return cited


# --- Smoke test ---

def main():
    """
    Smoke test: confirms the Groq client can authenticate and respond.
    Sends a tiny dummy request - not the full RAG pipeline.
    """
    print("=" * 60)
    print("Groq Generator Smoke Test")
    print("=" * 60)

    print("\n[1/2] Initializing Groq client...")
    gen = GroqGenerator()
    print(f"  -> Model: {gen.model_name}")
    print(f"  -> Temperature: {gen.temperature}")
    print(f"  -> Prompt version: {gen.prompt_version}")

    print("\n[2/2] Sending tiny test request...")
    fake_chunks = [{
        "text": "HbA1c reflects average plasma glucose over 8-12 weeks. Diagnostic threshold is 6.5%.",
        "metadata": {
            "source": "WHO",
            "parent_title": "Diabetes Fact Sheet",
            "parent_url": "https://example.org/diabetes",
        },
    }]

    result = gen.generate("What is HbA1c?", fake_chunks)

    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(result["text"])
    print("\n" + "=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"  Is refusal: {result['is_refusal']}")
    print(f"  Is emergency: {result['is_emergency']}")
    print(f"  Cited source numbers: {result['cited_source_numbers']}")
    print(f"  Prompt version: {result['prompt_version']}")
    print(f"  Model: {result['model_name']}")


if __name__ == "__main__":
    main()