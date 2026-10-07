#!/usr/bin/env python3
"""Pin official GSM8K, deduplicate questions, and keep test isolated."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


def question_hash(question):
    normalized = ' '.join(unicodedata.normalize('NFKC', question).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    if (args.output / 'manifest.json').exists():
        raise RuntimeError('existing data manifest; do not silently regenerate')
    if args.source_dir:
        revision = json.loads((args.source_dir / 'info.json').read_text())['sha']
        dataset = {split: pd.read_parquet(args.source_dir / f'{split}.parquet').to_dict('records') for split in ('train', 'test')}
    else:
        revision = HfApi().dataset_info('openai/gsm8k').sha
        dataset = load_dataset('openai/gsm8k', 'main', revision=revision)
    if len(dataset['train']) != 7473 or len(dataset['test']) != 1319:
        raise ValueError('unexpected official GSM8K sizes')
    seen = set()
    duplicates = []
    def convert(split):
        rows = []
        for i, item in enumerate(dataset[split]):
            digest = question_hash(item['question'])
            if digest in seen:
                duplicates.append({'split': split, 'index': i, 'hash': digest})
                continue
            seen.add(digest)
            answer = re.search(r'####\s*([-+\d.,]+)\s*$', item['answer'])
            if not answer:
                raise ValueError(f'no final numeric reference: {split}:{i}')
            rows.append(dict(data_source='openai/gsm8k',
                prompt=[dict(role='user', content=item['question'] + '\nSolve the problem. Give the final numeric answer after ####.')],
                ability='math', reward_model=dict(style='rule', ground_truth=answer.group(1).replace(',', '')),
                extra_info=dict(split=split, index=i, id=f'gsm8k:{split}:{i}', question_hash=digest)))
        return rows
    # Preserve official test first; remove any overlap from train before splitting.
    test = convert('test')
    train = convert('train')
    random.Random(args.seed).shuffle(train)
    splits = {'probe': train[:500], 'dev': train[500:1473], 'train': train[1473:], 'test': test}
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in splits.items():
        for row in rows:
            row['extra_info']['split'] = name
        path = args.output / f'{name}.parquet'
        pd.DataFrame(rows).to_parquet(path, index=False)
        files[name] = dict(count=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    # Small genuine train/dev slices for the bounded integration smoke.
    for name in ('train', 'dev'):
        pd.DataFrame(splits[name][:40 if name == 'train' else 4]).to_parquet(args.output / f'smoke_{name}.parquet', index=False)
    manifest = dict(dataset='openai/gsm8k', config='main', revision=revision, seed=args.seed,
        files=files, source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.source_dir.glob('*.parquet')} if args.source_dir else {}, removed_duplicates=duplicates, prompt_suffix='Solve the problem. Give the final numeric answer after ####.',
        test_usage='reserved; never passed to smoke or periodic validation')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
