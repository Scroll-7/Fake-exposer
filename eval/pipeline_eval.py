"""Run the live analysis pipeline against a labelled manifest and score it.

This measures the *whole* detector (LLM + heuristics + local models), which is
the thing that actually decides verdicts — not just the text classifier.

Cost note
---------
`/api/analyze/*` uses Groq / Gemini, so running this consumes (free-tier) API
calls and is subject to the server's rate limits (60/hr text, 15/hr image).
Use the pure-local `eval/image_eval.py` for unlimited, zero-cost image scoring.

Manifest format (JSONL, one object per line):
    {"id": "t1", "type": "text", "input": "full article text...",
     "label": 1, "source": "seed", "notes": ""}
    {"id": "u1", "type": "url",  "input": "https://...", "label": 0}
    {"id": "i1", "type": "image","input": "images/fake1.jpg", "label": 1}
`label`: 1 = fake/AI/manipulated, 0 = real/authentic.
Image `input` is a path relative to --data-dir.

Usage:
    python -m eval.pipeline_eval --manifest eval/data/sample_manifest.jsonl
    python -m eval.pipeline_eval --manifest my.jsonl --base-url http://127.0.0.1:3001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from eval.metrics import classification_report_dict, format_report


def post_json(base: str, path: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post_multipart(base: str, path: str, filepath: str, timeout: int) -> dict:
    boundary = "----evalboundary7MA4YWxkTrZu0gW"
    with open(filepath, "rb") as f:
        content = f.read()
    fname = os.path.basename(filepath)
    body = (
        f"--{boundary}\r\n".encode()
        + f'Content-Disposition: form-data; name="image"; filename="{fname}"\r\n'.encode()
        + b"Content-Type: application/octet-stream\r\n\r\n"
        + content
        + b"\r\n"
        + f"--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        base + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def load_manifest(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"manifest line {i}: invalid JSON ({e})")
            if obj.get("type") not in ("text", "url", "image"):
                raise SystemExit(f"manifest line {i}: bad type {obj.get('type')!r}")
            if "label" not in obj or "input" not in obj:
                raise SystemExit(f"manifest line {i}: needs 'input' and 'label'")
            items.append(obj)
    return items


def run_item(base: str, item: dict, data_dir: str, timeout: int) -> dict:
    """Return {'prob': fake_probability, 'raw': response, 'error': None|str}."""
    typ = item["type"]
    try:
        if typ == "text":
            raw = post_json(base, "/api/analyze/text",
                            {"text": item["input"]}, timeout)
        elif typ == "url":
            raw = post_json(base, "/api/analyze/url",
                            {"url": item["input"]}, timeout)
        else:  # image
            path = item["input"]
            if not os.path.isabs(path):
                path = os.path.join(data_dir, path)
            if not os.path.exists(path):
                return {"prob": None, "raw": None, "error": f"image not found: {path}"}
            raw = post_multipart(base, "/api/analyze/image", path, timeout)
    except urllib.error.HTTPError as e:
        return {"prob": None, "raw": None, "error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001 - surface any transport error
        return {"prob": None, "raw": None, "error": str(e)}

    score = raw.get("credibility_score")
    if score is None:
        return {"prob": None, "raw": raw, "error": "no credibility_score in response"}
    # credibility 0..100 (higher = more credible) -> P(fake) = 1 - score/100
    return {"prob": 1.0 - float(score) / 100.0, "raw": raw, "error": None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default=os.path.join("eval", "data"))
    parser.add_argument("--base-url", default="http://127.0.0.1:3001")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not os.path.exists(args.manifest):
        print(f"[!] Manifest not found: {args.manifest}")
        return 1

    items = load_manifest(args.manifest)
    if not items:
        print("[!] Manifest is empty.")
        return 1

    per_type: dict[str, list] = {}
    errors = []
    print(f"\n  Running {len(items)} items against {args.base_url} ...\n")

    for item in items:
        res = run_item(args.base_url, item, args.data_dir, args.timeout)
        tag = f"{item.get('id', '?')} [{item['type']}]"
        if res["error"]:
            print(f"  - {tag:24} ERROR: {res['error']}")
            errors.append((item, res["error"]))
            continue
        prob = res["prob"]
        pred = int(prob >= args.threshold)
        truth = int(item["label"])
        mark = "ok " if pred == truth else "miss"
        print(f"  - {tag:24} p(fake)={prob:.2f} pred={pred} truth={truth}  {mark}")
        per_type.setdefault(item["type"], []).append((truth, prob))

    if errors:
        print(f"\n  {len(errors)} item(s) errored (rate limit or transport). "
              "Re-run later or reduce the set.")

    if not any(per_type.values()):
        print("\n[!] No scorable items. Is the server running?")
        return 1

    print()
    for typ, rows in sorted(per_type.items()):
        y_true = [r[0] for r in rows]
        y_prob = [r[1] for r in rows]
        if len(set(y_true)) < 2:
            print(f"  [{typ}] only one class present ({len(rows)} items) — "
                  "metrics need both real and fake examples.")
            continue
        pred = [int(p >= args.threshold) for p in y_prob]
        report = classification_report_dict(y_true, pred, y_prob)
        print(format_report(f"[{typ}] pipeline", report))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
