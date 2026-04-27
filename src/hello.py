"""
Smoke test: confirms our environment is set up correctly.
Run this once to make sure Python and our libraries work.
"""

import requests
from dotenv import load_dotenv
import os

print("✅ Python is running.")
print("✅ requests library imported successfully.")
print("✅ python-dotenv imported successfully.")

# Test that requests actually works by hitting a free public API
response = requests.get("https://api.github.com")
if response.status_code == 200:
    print("✅ Internet connection works (got a response from GitHub API).")
else:
    print("❌ Couldn't reach GitHub API.")

print("\n🎉 Setup is complete. You're ready for Step 2.")