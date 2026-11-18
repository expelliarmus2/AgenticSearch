"""
Local rule-based reward for s3-style agentic search RL training.

Why this file exists
--------------------
The original `rag_2.py` rewards answers by calling an external "generator LLM"
(`generator_llms.local_inst.generate_answer*`, which POSTs to an OpenAI-style
server) and imports `pyserini` (a Java-backed retrieval lib). Neither is
available on the local training box. This module provides a pure-Python,
offline reward that keeps the s3 signal structure:

  score = utility_score + generation_score + format_bonus

  * generation_score: 1.0 if the final `<answer>` matches a golden answer (EM)
  * utility_score:    generation_score - zero_shot_baseline(0.0 without a
                      generator LLM, so utility == generation)
  * format_bonus:     0.1 for issuing a `<query>` search, +0.4 for closing
                      with `<answer>` — teaches the agent the search protocol

The function signature mirrors `rag_2.compute_score_rag` so that
`verl/trainer/main_ppo.py` only needs to swap the import.
"""
import re
import string

__all__ = ['compute_score_local', 'output_sequence']


def normalize_answer(s: str) -> str:
    """Lower text, strip articles/punctuation, collapse whitespace (SQuAD-style)."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_score(pred: str, golds) -> float:
    """Exact-match (normalized) between a prediction and a list of gold answers."""
    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return 0.0
    for g in golds:
        g_norm = normalize_answer(str(g))
        if g_norm and pred_norm == g_norm:
            return 1.0
    return 0.0


def f1_score(pred: str, golds, threshold: float = 0.5) -> float:
    """Token-overlap F1; returns 1.0 if max F1 >= threshold, else 0.0.

    Also accepts substring containment (e.g. golden "Paris" inside a verbose
    prediction) — a tiny golden token set otherwise makes F1 unreachable.
    """
    pred_norm = normalize_answer(pred)
    pred_tokens = pred_norm.split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for g in golds:
        g_norm = normalize_answer(str(g))
        gold_tokens = g_norm.split()
        if not gold_tokens:
            continue
        if len(g_norm) >= 2 and g_norm in pred_norm:
            return 1.0
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(gold_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return 1.0 if best >= threshold else 0.0


def _policy_region(solution_str: str) -> str:
    """Slice out the part written by the policy.

    `solution_str` = prompt + generated response, and the prompt embeds few-shot
    examples (<answer> Beijing </answer>). The prompt's question section ends
    with "Question: ...", so everything after the last "Question:" contains the
    question text plus the model's actual trajectory.
    """
    marker = 'Question:'
    idx = solution_str.rfind(marker)
    if idx == -1:
        return solution_str
    return solution_str[idx + len(marker):]


def extract_answer(solution_str: str) -> str:
    """Final answer between <answer> ... </answer> (policy region only)."""
    region = _policy_region(solution_str)
    matches = re.findall(r'<answer>(.*?)</answer>', region, re.DOTALL)
    return matches[-1].strip() if matches else ''


def extract_information(solution_str: str):
    """All retrieved passages wrapped in <information> ... </information>."""
    return re.findall(r'<information>(.*?)</information>', _policy_region(solution_str), re.DOTALL)


def has_search(solution_str: str) -> bool:
    """Whether the agent issued at least one search query."""
    return bool(re.search(r'<query>.*?</query>', _policy_region(solution_str), re.DOTALL))


def has_answer_tag(solution_str: str) -> bool:
    """Whether the policy (not the prompt example) closed with <answer>."""
    return '<answer>' in _policy_region(solution_str)


def _to_golden_list(ground_truth) -> list:
    target = ground_truth.get('target')
    if target is None:
        return []
    if hasattr(target, 'tolist'):
        return [str(g) for g in target.tolist()]
    return [str(g) for g in target]


def compute_score_local(solution_str,
                        ground_truth,
                        data_source='nq',
                        zeroshot_answers=None,
                        use_utility_score=True,
                        use_generation_score=True):
    """
    Pure-local reward. Compatible signature with rag_2.compute_score_rag.
    Returns (score, answer_zeroshot, answer_zeroshot_score).

    NOTE: zeroshot_answers is intentionally ignored — without a generator LLM
    there is no zero-shot baseline, so we assume baseline score = 0.
    """
    question = str(ground_truth.get('question', ''))
    golden = _to_golden_list(ground_truth)

    answer = extract_answer(solution_str)
    info_blocks = extract_information(solution_str)
    searched = has_search(solution_str)

    em = em_score(answer, golden)
    if em == 0.0:
        em = f1_score(answer, golden)

    generation_score = 1.0 if em else 0.0

    # no generator LLM -> zero-shot baseline is 0
    answer_zeroshot = ''
    answer_zeroshot_score = 0.0
    utility_score = generation_score - answer_zeroshot_score

    # small protocol bonus so the agent gets a learning signal even before it
    # can answer correctly
    format_bonus = 0.0
    if searched:
        format_bonus += 0.1
    if has_answer_tag(solution_str) or em:
        format_bonus += 0.4

    if use_utility_score and use_generation_score:
        score = utility_score + generation_score
    elif use_generation_score:
        score = generation_score
    elif use_utility_score:
        score = utility_score
    else:
        score = 0.0

    score = score + format_bonus
    return score, answer_zeroshot, answer_zeroshot_score


def output_sequence(solution_str, ground_truth):
    """Compatibility helper for RewardManager's val_only path (no scoring)."""
    question = str(ground_truth.get('question', ''))
    golden = _to_golden_list(ground_truth)
    blocks = extract_information(solution_str)
    context_with_info = '\n\n'.join(
        f'Doc {i + 1}\n{b}' for i, b in enumerate(blocks))
    return question, golden, context_with_info, solution_str
