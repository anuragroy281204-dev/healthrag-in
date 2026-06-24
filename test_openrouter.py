import os
from dotenv import load_dotenv
load_dotenv()

# Check the key is loading
key = os.getenv("OPENROUTER_API_KEY")
print("Key loaded:", "YES" if key else "NO — MISSING FROM .env")
if key:
    print("Key starts with:", key[:12])

# Now test the actual router
from src.generation.llm_router import call_llm, AllProvidersFailedError

try:
    result = call_llm(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say the word working.",
        max_tokens=20
    )
    print("SUCCESS:", result["text"])
    print("Provider:", result["provider_used"])
except AllProvidersFailedError as e:
    print("ALL FAILED. Errors per model:")
    for model, err in e.errors.items():
        print(f"  {model}: {err}")