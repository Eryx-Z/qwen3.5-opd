"""CPU-only tokenizer alignment and smoke prompt-budget preflight."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

root = Path('/home/eryx/qwen3.5-opd')
paths = ['/home/eryx/models/Qwen3.5-2B', '/home/eryx/models/Qwen3.5-35B-A3B-FP8']
s, t = [AutoTokenizer.from_pretrained(path, local_files_only=True) for path in paths]
assert s.get_vocab() == t.get_vocab(), 'token string to ID mismatch'
assert s.all_special_tokens == t.all_special_tokens
assert s.all_special_ids == t.all_special_ids
lengths = []
for split in ('smoke_train', 'smoke_dev'):
    for row in pd.read_parquet(root / f'data/gsm8k/{split}.parquet').to_dict('records'):
        ids = s.apply_chat_template(list(row['prompt']), tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        lengths.append(len(ids))
        assert len(ids) <= 512, 'smoke prompt would be filtered'
metadata = {path: {name: hashlib.sha256((Path(path) / name).read_bytes()).hexdigest()
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json')} for path in paths}
result = dict(mapping_equal=True, special_tokens_equal=True, prompt_count=len(lengths),
              max_prompt_tokens=max(lengths), thinking=False, metadata_sha256=metadata)
(root / 'baselines/gsm8k_k1/tokenizer_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
