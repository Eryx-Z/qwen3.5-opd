"""Conservative final-answer verifier. Never executes model text."""
from fractions import Fraction
import re

NUM = r'[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s*/\s*[-+]?\d+)?'


def normalize(text):
    text = text.strip().replace('−', '-').replace(',', '').replace('$', '')
    text = re.sub(r'\\(?:d?frac)\{([-+]?\d+)\}\{([-+]?\d+)\}', r'\1/\2', text)
    text = text.replace(' ', '')
    return Fraction(text)


def extract(text):
    # Do not interpret unrelated numbers in reasoning as the final answer.
    matches = list(re.finditer(r'####\s*([^\n]+)', text))
    if matches:
        tail = matches[-1].group(1).strip().replace('$', '')
        match = re.fullmatch(f'({NUM})\\.?', tail)
        return normalize(match.group(1)) if match else None
    boxes = re.findall(r'\\boxed\{(\\(?:d?frac)\{[-+]?\d+\}\{[-+]?\d+\}|[^{}]+)\}', text)
    if boxes:
        try:
            return normalize(boxes[-1])
        except (ValueError, ZeroDivisionError):
            return None
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != 'openai/gsm8k':
        raise ValueError(f'unexpected data source: {data_source}')
    expected = normalize(ground_truth)  # bad reference is an infrastructure/data error
    try:
        actual = extract(solution_str)
    except (ValueError, ZeroDivisionError):
        actual = None
    return float(actual is not None and actual == expected)
