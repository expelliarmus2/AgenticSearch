"""s3-style agentic search driven by an OpenAI-compatible chat API (DashScope).

No verl / vLLM / local model needed: the s3 search-copilot loop
(<query> -> retrieve -> <information> -> <important_info> -> <search_complete> ...)
is driven by plain chat-completions calls, and retrieval reuses the repo's
retrieval server (s3/search/retrieval_server.py, FastAPI + BM25/faiss).

Usage:
    python scripts/s3_agentic_api.py --questions "Who built the Eiffel Tower?" \
        --retrieval_url http://127.0.0.1:3000/retrieve --max_turns 3 --topk 8
"""
import argparse
import json
import os
import re
import sys
import time

import httpx

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-8635104661ef4703aca16a2d612aa0ed")

# s3 search-copilot system prompt (verbatim from scripts/data_process/test_s3.py make_prefix, bm25 variant)
SYSTEM_PROMPT = """You are a search copilot for the generation model. Based on a user's query and initial searched results, you will first determine if the searched results are enough to produce an answer.
If the searched results are enough, you will use <search_complete>True</search_complete> to indicate that you have gathered enough information for the generation model to produce an answer.
If the searched results are not enough, you will go through a loop of <query> -> <information> -> <important_info> -> <search_complete> -> <query> (if not complete) ..., to help the generation model to generate a better answer with more relevant information searched.
You should show the search query between <query> and </query> in JSON format.
Based on the search query, we will return the top searched results between <information> and </information>. You need to put the doc ids of the important documents (up to 3 documents, within the current information window) between <important_info> and </important_info> (e.g., <important_info>[1, 4]</important_info>).
A search query MUST be followed by a <search_complete> tag if the search is not complete.
After reviewing the information, you must decide whether to continue searching with a new query or indicate that the search is complete. If you need more information, use <search_complete>False</search_complete> to indicate you want to continue searching with a better query. Otherwise, use <search_complete>True</search_complete> to terminate the search.
During the process, you can add reasoning process within <think></think> tag whenever you want. Note: Only the important information would be used for the generation model to produce an answer.
Note: The search query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately.
For a question and initial searched results:
<question>
[user's question]
</question>
<information>
[initial searched results]
</information>

If the initial searched results are enough to produce an answer, you should output:
<search_complete>
True
</search_complete>

If the initial searched results are not enough to produce an answer, you should output:
<query>
{
    "query": "[search query]"
} 
</query>
<information>
[top searched results based on the above search query]
</information>
<important_info>
[doc ids]
</important_info>
<search_complete>
False
</search_complete>
<query>
{
    "query": "[search query]"
}
</query>
...... (can be several turns until <search_complete> is True)

<search_complete>
True
</search_complete>

Now, start the loop with the following question and initial searched results:"""


def call_llm(messages, model="qwen3.8-max", max_tokens=1500, temperature=0.6, enable_thinking=False):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    # Trust env proxies (WSL codex proxy) if set; falls back to direct otherwise.
    with httpx.Client(timeout=600, trust_env=True) as client:
        for attempt in range(4):
            try:
                resp = client.post(DASHSCOPE_URL, headers=headers, json=payload)
            except httpx.HTTPError as e:
                if attempt == 3:
                    raise
                print(f"[llm] connection error ({e}); retry {attempt + 1}/3...")
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code < 500:
                break
            print(f"[llm] DashScope returned {resp.status_code}; retry {attempt + 1}/3...")
            time.sleep(2 * (attempt + 1))
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def retrieve(query, url, topk, doc_max_chars=2000):
    resp = httpx.post(url, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=120)
    resp.raise_for_status()
    results = resp.json()["result"][0]
    blocks = []
    for idx, item in enumerate(results, 1):
        doc = item["document"]
        content = doc.get("contents") or ""
        title = content.split("\n")[0].strip().strip('"') if content else doc.get("title", "")
        text = "\n".join(content.split("\n")[1:]).strip() if content else doc.get("text", "")
        if doc_max_chars and len(text) > doc_max_chars:
            text = text[:doc_max_chars]
        blocks.append(f"Doc {idx} (Title: {title})\n{text}")
    return "\n".join(blocks)


def parse_response(text):
    """Extract s3 tags from the model output (mirrors s3 postprocess_predictions)."""
    result = {"query": None, "search_complete": None, "important_ids": None}
    m = re.search(r"<query>(.*?)</query>", text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        try:
            result["query"] = json.loads(raw).get("query", "")
        except Exception:
            result["query"] = raw
    m = re.search(r"<search_complete>(.*?)</search_complete>", text, re.DOTALL)
    if m:
        result["search_complete"] = m.group(1).strip().lower() in ("true", "yes", "1", "y")
    m = re.search(r"<important_info>(.*?)</important_info>", text, re.DOTALL)
    if m:
        try:
            result["important_ids"] = json.loads(m.group(1).strip())
        except Exception:
            result["important_ids"] = [int(x) for x in re.findall(r"\d+", m.group(1))]
    return result


def run_agentic_search(question, retrieval_url, model, max_turns=3, topk=8, initial_docs=3, doc_max_chars=2000):
    initial = retrieve(question, retrieval_url, topk, doc_max_chars)
    if initial_docs:
        parts = initial.split("Doc ")
        header, docs = parts[0], parts[1:]
        initial = header + "".join(f"Doc {d}" for d in docs[:initial_docs]).strip()

    history = [{"role": "user", "content": SYSTEM_PROMPT}]
    user_content = f"""<question>
{question}
</question>
<information>
{initial}
</information>"""
    history.append({"role": "user", "content": user_content})

    turns, queries, infos = [], [], [initial]
    done = False
    for step in range(max_turns):
        print(f"  -- turn {step + 1}: calling model...")
        out = call_llm(history, model=model, enable_thinking=False)
        history.append({"role": "assistant", "content": out})
        parsed = parse_response(out)
        turns.append(out)

        if parsed["search_complete"] is True:
            done = True
            print("  -- <search_complete>True</search_complete> -> stopping search")
            break

        if parsed["query"]:
            print(f"  -- search query: {parsed['query']!r}")
            q = parsed["query"]
            queries.append(q)
            info = retrieve(q, retrieval_url, topk, doc_max_chars)
            infos.append(info)
            feedback = f"\n\n<information>{info}</information>\n\n"
            if parsed["important_ids"]:
                feedback += f"\n<important_info>{parsed['important_ids']}</important_info>\n"
            history.append({"role": "user", "content": feedback})
        else:
            # invalid action -> s3-style push toward a query
            feedback = "\n\nThe information is not enough to answer the question. Let me dive deeper by generating a brand new search query between <query> and </query> tags in JSON format:\n<query>\n"
            history.append({"role": "user", "content": feedback})
    if not done:
        print(f"  -- reached max_turns={max_turns} without <search_complete>True")

    # Generator stage: answer from the collected context (s3 separates search copilot & generator)
    all_context = "\n\n".join(infos)
    gen_prompt = (
        f"Based on the following context, answer the question concisely.\n\n"
        f"Question: {question}\n\nContext:\n{all_context}\n\nAnswer:"
    )
    answer = call_llm([{"role": "user", "content": gen_prompt}], model=model, max_tokens=400, temperature=0.2, enable_thinking=False)

    return {
        "question": question,
        "search_turns": turns,
        "queries": queries,
        "infos": infos,
        "final_answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", nargs="+", help="questions passed on the command line")
    parser.add_argument("--questions_file", default=None, help="jsonl file with {query, answer, query_id, ...} per line")
    parser.add_argument("--retrieval_url", default="http://127.0.0.1:3000/retrieve")
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--output", default="data/agentic/s3_api_results.json")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions of --questions_file")
    parser.add_argument("--doc_max_chars", type=int, default=2000, help="truncate each retrieved doc to this many chars")
    args = parser.parse_args()

    questions = []
    if args.questions_file:
        with open(args.questions_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        if args.limit is not None:
            questions = questions[: args.limit]
        # dict form: use the query text as the key, keep answer for reference
        questions = [{"question": q["query"], "answer": q.get("answer")} for q in questions]
    else:
        questions = [{"question": q, "answer": None} for q in (args.questions or [])]
    if not questions:
        parser.error("provide --questions or --questions_file")

    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    for item in questions:
        q = item["question"]
        if q in results:
            print(f"\n>>> (skip, already done) {q[:60]}")
            continue
        print(f"\n>>> {q}")
        t0 = time.time()
        r = run_agentic_search(q, args.retrieval_url, args.model, args.max_turns, args.topk, doc_max_chars=args.doc_max_chars)
        r["golden_answer"] = item["answer"]
        results[q] = r
        print(f"    answer: {r['final_answer'][:200]} (elapsed {round(time.time() - t0, 1)}s)")
        with open(args.output, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
