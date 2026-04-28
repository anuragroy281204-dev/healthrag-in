"""
Medical RAG Generator
=====================
Wraps the LLM router for the generation layer of HealthRAG-IN.
Takes a user question and retrieved chunks, returns a structured
medical answer with citations.

Now uses the LLM router (Groq primary, Gemini fallback) instead of
calling Groq directly. The rest of the interface is unchanged.

Run as CLI for a smoke test:
    python -m src.generation.generator
"""

import re

from src.generation.prompts import (
    SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_user_message,
    is_refusal,
    is_emergency_response,
)
from src.generation.llm_router import call_llm, AllProvidersFailedError


# --- Configuration ---

TEMPERATURE = 0.1
MAX_TOKENS = 1500


# --- Generator class ---

class GroqGenerator:
    """
    The generation layer. Class name kept as 'GroqGenerator' for
    backward compatibility with existing imports - but now uses
    the LLM router under the hood.
    """

    def __init__(self, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = PROMPT_VERSION

    def generate(self, question, retrieved_chunks):
        """
        Send question + retrieved chunks through the router.

        Returns dict with:
          - text: the LLM's response
          - is_refusal: True if the response is a refusal
          - is_emergency: True if it issued an emergency warning
          - cited_source_numbers: which sources were cited
          - prompt_version: prompt template version
          - model_name: which provider answered (groq or gemini)
          - attempts: list of providers tried
        """
        user_message = build_user_message(question, retrieved_chunks)

        try:
            result = call_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                primary="groq",
            )
        except AllProvidersFailedError as e:
            # Total failure - surface a graceful response
            return {
                "text": (
                    "I'm temporarily unable to generate an answer because all "
                    "language model providers are currently unavailable. "
                    "Please try again in a few minutes. The retrieved sources "
                    "below may still be useful."
                ),
                "is_refusal": True,
                "is_emergency": False,
                "cited_source_numbers": set(),
                "prompt_version": self.prompt_version,
                "model_name": "none",
                "attempts": list(e.errors.keys()),
                "error": str(e),
            }

        response_text = result["text"]
        cited_numbers = extract_citation_numbers(response_text)

        return {
            "text": response_text,
            "is_refusal": is_refusal(response_text),
            "is_emergency": is_emergency_response(response_text),
            "cited_source_numbers": cited_numbers,
            "prompt_version": self.prompt_version,
            "model_name": result["provider_used"],
            "attempts": result["attempts"],
        }


# --- Citation parsing ---

def extract_citation_numbers(text):
    """Find all citation markers like [1], [2,3], [1, 4, 7]."""
    cited = set()
    matches = re.findall(r"\[([\d,\s]+)\]", text)
    for match in matches:
        for num_str in match.split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                cited.add(int(num_str))
    return cited


# --- Smoke test (DO NOT RUN TODAY - both providers throttled) ---

def main():
    print("=" * 60)
    print("Generator Smoke Test (with router)")
    print("=" * 60)

    print("\n[1/2] Initializing generator...")
    gen = GroqGenerator()
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
    print(f"  Provider used: {result['model_name']}")
    print(f"  Attempts: {result['attempts']}")
    print(f"  Is refusal: {result['is_refusal']}")
    print(f"  Is emergency: {result['is_emergency']}")
    print(f"  Cited source numbers: {result['cited_source_numbers']}")
    print(f"  Prompt version: {result['prompt_version']}")


if __name__ == "__main__":
    main()