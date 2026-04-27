"""
WHO Fact Sheet Scraper
======================
Downloads diabetes-related fact sheets from the WHO website
and saves them as clean JSON records in data/raw/who/.

Run from the project root:
    python -m src.ingestion.who_scraper
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# --- Configuration ---

WHO_INDEX_URL = "https://www.who.int/news-room/fact-sheets"
OUTPUT_DIR = Path("data/raw/who")

RELEVANT_KEYWORDS = [
    "diabetes",
    "obesity",
    "hypertension",
    "cardiovascular",
    "blood pressure",
    "blood sugar",
    "metabolic",
]

HEADERS = {
    "User-Agent": "HealthRAG-IN-Student-Project/1.0 (educational use; contact: royanurag281204@gmail.com)"
}

DELAY_BETWEEN_REQUESTS = 2  # seconds


# --- Helper functions ---

def get_page(url):
    """Fetch a URL and return a parsed BeautifulSoup object."""
    print(f"  -> Fetching {url}")
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def extract_relevant_links(index_soup):
    """Extract fact sheet links matching our keywords."""
    matches = []
    seen_urls = set()

    for link in index_soup.find_all("a", href=True):
        href = link["href"]
        title = link.get_text(strip=True)

        if "/fact-sheets/detail/" not in href:
            continue

        if href.startswith("/"):
            href = "https://www.who.int" + href

        if href in seen_urls:
            continue
        seen_urls.add(href)

        title_lower = title.lower()
        if any(kw in title_lower for kw in RELEVANT_KEYWORDS):
            matches.append({"title": title, "url": href})

    return matches


def extract_fact_sheet_content(soup):
    """Extract the main article text from a WHO fact sheet page."""
    main = soup.find("main") or soup.find("article")
    if not main:
        return ""

    for junk in main.find_all(["script", "style", "nav", "footer", "aside"]):
        junk.decompose()

    text = main.get_text(separator="\n", strip=True)
    return text


def save_record(record, output_dir):
    """Save one fact sheet as its own JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = record["url"].rstrip("/").split("/")[-1]
    filepath = output_dir / f"{slug}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved: {filepath.name}")


# --- Main routine ---

def main():
    print("=" * 60)
    print("WHO Fact Sheet Scraper")
    print("=" * 60)

    print("\n[1/3] Fetching WHO fact sheet index...")
    index_soup = get_page(WHO_INDEX_URL)
    relevant = extract_relevant_links(index_soup)
    print(f"\n  Found {len(relevant)} relevant fact sheets:")
    for sheet in relevant:
        print(f"    - {sheet['title']}")

    if not relevant:
        print("\n[!] No matching fact sheets found.")
        print("    Try opening the URL in a browser to verify it loads.")
        return

    print(f"\n[2/3] Downloading fact sheets (with {DELAY_BETWEEN_REQUESTS}s delay)...")
    records = []
    for sheet in relevant:
        try:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            soup = get_page(sheet["url"])
            content = extract_fact_sheet_content(soup)

            if len(content) < 200:
                print(f"  [!] Skipping (too short): {sheet['title']}")
                continue

            record = {
                "source": "WHO",
                "title": sheet["title"],
                "url": sheet["url"],
                "content": content,
                "content_length": len(content),
            }
            records.append(record)
            save_record(record, OUTPUT_DIR)

        except requests.HTTPError as e:
            print(f"  [X] HTTP error for {sheet['title']}: {e}")
        except Exception as e:
            print(f"  [X] Unexpected error for {sheet['title']}: {e}")

    print(f"\n[3/3] Done.")
    print(f"  Total downloaded: {len(records)}")
    print(f"  Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()