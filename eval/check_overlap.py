"""Recompute realistic image eval metrics after deduplication.

Estimates overlap between realistic eval set (eval/data/aihuman/real and ai, n=2000 total) and
the probe training pool (archive (1)/train.csv) by content hash of filenames.
This uses the same CSV layout as make_split (label 0 = human).
"""
from __future__ import annotations
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, 'eval', 'data', 'aihuman')
TRAIN_CSV = os.path.join(ROOT, 'archive (1)', 'train.csv')

def load_filenames(folder):
    out = set()
    for name in os.listdir(folder):
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            out.add(os.path.splitext(name)[0])
    return out

def main():
    real_f = load_filenames(os.path.join(EVAL_DIR, 'real'))
    ai_f = load_filenames(os.path.join(EVAL_DIR, 'ai'))
    eval_stems = real_f | ai_f
    print(f"eval stems: {len(real_f)} real + {len(ai_f)} ai = {len(eval_stems)}")
    if not os.path.exists(TRAIN_CSV):
        print('train CSV not found:', TRAIN_CSV)
        return 0
    seen = set()
    overlaps = 0
    with open(TRAIN_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rel = (row.get('file_name') or '').strip()
            if not rel:
                continue
            stem = os.path.splitext(os.path.basename(rel))[0]
            if stem in eval_stems and stem not in seen:
                seen.add(stem)
                overlaps += 1
    print(f'overlaps with train pool: {overlaps} stems ({overlaps/len(eval_stems)*100:.2f}%)')
    return 0

if __name__ == '__main__':
    sys.exit(main())
