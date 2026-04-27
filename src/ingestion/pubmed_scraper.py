"""
PubMed Abstract Scraper
=======================
Downloads abstracts of diabetes-related research papers from PubMed
using the free NCBI E-utilities API. Saves them as JSON in
data/raw/pubmed/.

Run from the project root:
    python -m src.ingestion.pubmed_scraper
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# --- Configuration ---

# NCBI E-utilities base URL (free, no API key needed for low-volume use)
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

OUTPUT_DIR = Path("data/raw/pubmed")

# The PubMed search query.
# Translation: papers about diabetes (in title or abstract),
# in English, from the last 5 years, that have an abstract.
SEARCH_QUERY = (
    '("diabetes mellitus"[Title/Abstract] OR "type 2 diabetes"[Title/Abstract] '
    'OR "type 1 diabetes"[Title/Abstract] OR "gestational diabetes"[Title/Abstract]) '
    'AND English[Language] '
    'AND ("2020"[Date - Publication] : "3000"[Date - Publication]) '
    'AND hasabstract[All Fields]'
)

# How many papers to fetch (start small, scale up later)
MAX_RESULTS = 500

# How many papers to fetch per API call (NCBI allows up to 200; we'll use 100)
BATCH_SIZE = 100

# NCBI asks for a delay between requests
DELAY_BETWEEN_REQUESTS = 1  # seconds

# Identify ourselves to NCBI (recommended in their guidelines)
TOOL_NAME = "HealthRAG-IN-Student-Project"
EMAIL = "royanurag281204@gmail.com"


# --- Helper functions ---

def search_pubmed(query, max_results):
    """
    Step 1 of PubMed: search by query, get back a list of paper IDs (PMIDs).
    PubMed calls this 'esearch'.
    """
    print(f"  -> Searching PubMed for up to {max_results} papers...")
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "tool": TOOL_NAME,
        "email": EMAIL,
    }
    response = requests.get(ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    pmids = data.get("esearchresult", {}).get("idlist", [])
    total_count = data.get("esearchresult", {}).get("count", "0")
    print(f"  -> PubMed has {total_count} matching papers; fetching {len(pmids)}.")
    return pmids


def fetch_abstracts(pmid_batch):
    """
    Step 2 of PubMed: given a list of paper IDs, download the
    full metadata + abstracts as XML. PubMed calls this 'efetch'.
    """
    params = {
        "db": "pubmed",
        "id": ",".join(pmid_batch),
        "retmode": "xml",
        "rettype": "abstract",
        "tool": TOOL_NAME,
        "email": EMAIL,
    }
    response = requests.get(EFETCH_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.text  # XML as a string


def parse_pubmed_xml(xml_text):
    """
    Parse the XML response from PubMed and extract structured records.
    Each <PubmedArticle> in the XML becomes one record.
    """
    records = []
    root = ET.fromstring(xml_text)

    for article in root.findall(".//PubmedArticle"):
        try:
            # PubMed ID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                continue

            # Title
            title_elem = article.find(".//ArticleTitle")
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

            # Abstract (can have multiple sections like Background/Methods/Results)
            abstract_parts = []
            for abst in article.findall(".//AbstractText"):
                label = abst.get("Label")
                text = "".join(abst.itertext()).strip()
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = "\n\n".join(abstract_parts)

            # Skip papers with no abstract
            if not abstract or len(abstract) < 100:
                continue

            # Journal name
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Publication year
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""

            # Authors (just first 3 to keep records compact)
            authors = []
            for author in article.findall(".//Author")[:3]:
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")

            record = {
                "source": "PubMed",
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "content": f"{title}\n\n{abstract}",
                "abstract": abstract,
                "content_length": len(abstract),
            }
            records.append(record)
        except Exception as e:
            print(f"  [!] Error parsing one article: {e}")
            continue

    return records


def save_records(records, output_dir):
    """Save each record as its own JSON file, named by PMID."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for record in records:
        filepath = output_dir / f"pmid_{record['pmid']}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        saved += 1
    return saved


# --- Main routine ---

def main():
    print("=" * 60)
    print("PubMed Abstract Scraper")
    print("=" * 60)

    # Step 1: search PubMed for matching paper IDs
    print(f"\n[1/3] Searching PubMed...")
    pmids = search_pubmed(SEARCH_QUERY, MAX_RESULTS)

    if not pmids:
        print("\n[!] No papers found. Check the search query or your internet.")
        return

    # Step 2: fetch papers in batches
    print(f"\n[2/3] Fetching abstracts in batches of {BATCH_SIZE}...")
    all_records = []
    total_batches = (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = pmids[start:end]
        print(f"  -> Batch {batch_num + 1}/{total_batches}: fetching {len(batch)} papers...")

        try:
            xml_text = fetch_abstracts(batch)
            records = parse_pubmed_xml(xml_text)
            all_records.extend(records)
            print(f"     [OK] Parsed {len(records)} valid records from this batch.")
        except Exception as e:
            print(f"     [X] Batch failed: {e}")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Step 3: save everything to disk
    print(f"\n[3/3] Saving records to disk...")
    saved_count = save_records(all_records, OUTPUT_DIR)
    print(f"\n  Total saved: {saved_count} papers")
    print(f"  Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()