"""
Build a small local train/val parquet + retrieval corpus for running the
s3 (agentic-search) verl pipeline on a single 8GB GPU.

Data source: data/nq_test.parquet (already in s3 format).
Corpus:     one "gold" doc per question (contains the golden answer) plus
            generic distraction docs from data/agentic/corpus.jsonl.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_PARQUET = os.path.join(ROOT, 'data', 'nq_test.parquet')
DISTRACT = os.path.join(ROOT, 'data', 'agentic', 'corpus.jsonl')
OUT_DIR = os.path.join(ROOT, 'data', 'local')
TRAIN_N = 4
VAL_N = 4

# Easy questions the 0.5B policy can actually answer after retrieving the gold doc,
# so GRPO gets non-zero advantages (a real learning signal) instead of all-0.5 rewards.
SIMPLE_QUESTIONS = [
    ("What is the capital of France?", ["Paris"]),
    ("How many legs does a spider have?", ["eight", "8"]),
    ("What is the chemical symbol for water?", ["H2O"]),
    ("Which planet is known as the Red Planet?", ["Mars"]),
]

SIMPLE_VAL_QUESTIONS = [
    ("What is the capital of Japan?", ["Tokyo"]),
    ("How many days are in a leap year?", ["366"]),
    ("What is the largest mammal?", ["blue whale", "Blue whale"]),
    ("What gas do plants absorb from the air?", ["carbon dioxide", "CO2"]),
]


def clean_prompt(prompt):
    """prompt is stored as np.ndarray of chat dicts -> python list."""
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    return prompt


def clean_reward_model(rm):
    rm = dict(rm)
    gt = dict(rm['ground_truth'])
    target = gt['target']
    if hasattr(target, 'tolist'):
        target = target.tolist()
    gt['target'] = list(target)
    rm['ground_truth'] = gt
    return rm


def make_doc(question, golden):
    """A retrieval doc whose text contains the golden answer."""
    title = golden
    text = (f"{golden} is the correct answer to the question: {question}. "
            f"The answer to this question is {golden}.")
    return {"id": f"gold_{question[:40].replace(' ', '_')}", "contents": f'"{title}"\n{text}'}


def main():
    df = pd.read_parquet(SRC_PARQUET)

    # 1) clone the prompt template from a real NQ row (same s3 instruction format)
    template = None
    for _, r in df.iterrows():
        g = r['golden_answers']
        if isinstance(g, np.ndarray):
            g = g.tolist()
        if g:
            template = {'prompt': clean_prompt(r['prompt']),
                        'data_source': r['data_source']}
            break
    assert template is not None

    def make_row(question, golden):
        prompt = [dict(m) for m in template['prompt']]
        for msg in prompt:
            content = msg.get('content', '')
            import re as _re
            if 'Question:' in content:
                # NQ-style prompt: "... Question: who got the first nobel prize in physics?"
                msg['content'] = content.split('Question:')[0] + f'Question: {question}'
            elif '<question>' in content:
                msg['content'] = _re.sub(r'<question>.*?</question>',
                                         f'<question>{question}</question>',
                                         content, flags=_re.S)
        return {'question': question,
                'golden_answers': list(golden),
                'prompt': prompt,
                'data_source': template['data_source'],
                'reward_model': {'ground_truth': {'target': list(golden), 'style': 'rule'},
                                 'style': 'rule'},
                'id': f'simple_{abs(hash(question)) % 10**8}'}

    train = [make_row(q, g) for q, g in SIMPLE_QUESTIONS]
    val = [make_row(q, g) for q, g in SIMPLE_VAL_QUESTIONS]

    os.makedirs(OUT_DIR, exist_ok=True)

    docs = []
    for row in train + val:
        docs.append(make_doc(row['question'], row['golden_answers'][0]))

    with open(DISTRACT) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    with open(os.path.join(OUT_DIR, 'corpus.jsonl'), 'w') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')

    for i, row in enumerate(train):
        row['extra_info'] = {'index': i, 'split': 'train'}
    for i, row in enumerate(val):
        row['extra_info'] = {'index': i, 'split': 'val'}

    for name, data in [('train', train), ('val', val)]:
        out = os.path.join(OUT_DIR, f'{name}.parquet')
        pd.DataFrame(data).to_parquet(out)
        print(f'wrote {out} ({len(data)} rows)')

    print('wrote', os.path.join(OUT_DIR, 'corpus.jsonl'), f'({len(docs)} docs)')
    for r in train:
        print('  train:', r['question'], '->', r['golden_answers'])


if __name__ == '__main__':
    main()
