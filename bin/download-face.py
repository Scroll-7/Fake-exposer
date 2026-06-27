#!/usr/bin/env python3
"""Download a celebrity face image from Wikimedia Commons for face recognition.

Usage: python bin/download-face.py "Person Name"
Example: python bin/download-face.py "Leonardo DiCaprio"
"""

import json, os, sys, urllib.request, urllib.parse, urllib.error

KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def wiki_api(params):
    params["format"] = "json"
    params["origin"] = "*"
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "FakeNewsDetector/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def download_face(name):
    # Step 1: Search for files matching the name
    for search_term in [name, f"{name} portrait", f"{name} face"]:
        search = wiki_api({
            "action": "query",
            "list": "search",
            "srsearch": search_term,
            "srnamespace": "6",  # File namespace
            "srlimit": 10,
        })
        pages = search.get("query", {}).get("search", [])
        if pages:
            break
    if not pages:
        print(f"No Wikimedia results for '{name}'", file=sys.stderr)
        return False

    for page in pages:
        title = page.get("title", "")
        if not title.startswith("File:"):
            continue
        # Skip non-image files
        ext = title.lower().rsplit(".", 1)[-1] if "." in title else ""
        if ext not in ("jpg", "jpeg", "png"):
            continue

        # Get the image URL
        info = wiki_api({
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 500,
        })
        pages_data = info.get("query", {}).get("pages", {})
        for pid, pdata in pages_data.items():
            if pid == "-1":
                continue
            urls = pdata.get("imageinfo", [])
            if urls:
                image_url = urls[0].get("url") or urls[0].get("thumburl")
                if not image_url:
                    continue
                filename = name.lower().replace(" ", "_") + ".jpg"
                filepath = os.path.join(KNOWN_FACES_DIR, filename)

                print(f"Downloading {image_url}...")
                req = urllib.request.Request(image_url, headers={"User-Agent": "FakeNewsDetector/1.0"})
                with urllib.request.urlopen(req, timeout=30) as img_resp:
                    with open(filepath, "wb") as f:
                        f.write(img_resp.read())
                print(f"Saved to {filepath}")
                return True
    print(f"Could not find an image for '{name}'", file=sys.stderr)
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    name = " ".join(sys.argv[1:])
    if not download_face(name):
        sys.exit(1)
