"""Full-corpus BM25 recall evaluation for BrowseComp-Plus (830 queries, 100,195 docs).

For each query, run BM25 (pyserini) top-k retrieval and check whether the
query's gold_docs / evidence_docs are retrieved. Results are appended line-by-line
to a jsonl so the run is resumable (already-processed query_ids are skipped).

Usage:
    python scripts/evaluation/recall_bm25_browsecomp.py \
        --questions_file dataset/browsecomp-plus/questions.jsonl \
        --index_path dataset/browsecomp-plus/indexes/bm25 \
        --output dataset/browsecomp-plus/bm25_recall_results.jsonl \
        --summary dataset/browsecomp-plus/bm25_recall_summary.json \
        [--topk 100] [--limit N]
"""
import argparse
import json
import os
import time

from pyserini.search.lucene import LuceneSearcher

K_LIST = [1, 5, 10, 20, 50, 100]


def ndcg_at_k(ranked_hit_flags, k, n_relevant):
    """ranked_hit_flags: list of bool (doc relevant or not), best-ranked first."""
    if n_relevant == 0:
        return 0.0
    dcg, idcg = 0.0, 0.0
    for i in range(min(k, len(ranked_hit_flags))):
        rel = 1.0 if ranked_hit_flags[i] else 0.0
        dcg += rel / (i + 2)  # log2(i+2) using i+2 in denominator approx
    for i in range(min(k, n_relevant)):
        idcg += 1.0 / (i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(hits, relevant, k):
    hit_relevant = sum(1 for d in hits[:k] if d in relevant)
    return hit_relevant / len(relevant) if relevant else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_file", default="dataset/browsecomp-plus/questions.jsonl")
    ap.add_argument("--index_path", default="dataset/browsecomp-plus/indexes/bm25")
    ap.add_argument("--output", default="dataset/browsecomp-plus/bm25_recall_results.jsonl")
    ap.add_argument("--summary", default="dataset/browsecomp-plus/bm25_recall_summary.json")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="only process first N queries (for smoke test)")
    args = ap.parse_args()

    searcher = LuceneSearcher(args.index_path)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # already processed query_ids (resume support)
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["query_id"])
    print(f"resuming: {len(done)} already processed", flush=True)

    questions = []
    with open(args.questions_file) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit:
        questions = questions[: args.limit]
    print(f"total queries to evaluate: {len(questions)}", flush=True)

    fout = open(args.output, "a", encoding="utf-8")
    t0 = time.time()
    for i, q in enumerate(questions):
        qid = q["query_id"]
        if qid in done:
            continue
        gold = {str(d["docid"]) for d in q.get("gold_docs", [])}
        evidence = {str(d["docid"]) for d in q.get("evidence_docs", [])}

        hits = searcher.search(q["query"], args.topk)
        hit_docids = [h.docid for h in hits]
        scores = {h.docid: h.score for h in hits}

        gold_hits = {d: round(scores[d], 4) for d in hit_docids if d in gold}
        evidence_hits = {d: round(scores[d], 4) for d in hit_docids if d in evidence}

        record = {
            "query_id": qid,
            "query": q["query"],
            "gold_docs": sorted(gold),
            "evidence_docs": sorted(evidence),
            "hit_docids": hit_docids,
            "gold_hits": gold_hits,
            "evidence_hits": evidence_hits,
            "recall_gold": {str(k): round(recall_at_k(hit_docids, gold, k), 4) for k in K_LIST},
            "recall_evidence": {str(k): round(recall_at_k(hit_docids, evidence, k), 4) for k in K_LIST},
            "ndcg10_gold": round(ndcg_at_k([d in gold for d in hit_docids], 10, len(gold)), 4),
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

        if (i + 1) % 20 == 0 or i == len(questions) - 1:
            print(
                f"[{i+1}/{len(questions)}] elapsed {round(time.time()-t0,1)}s | "
                f"recall@10 gold={record['recall_gold']['10']} evidence={record['recall_evidence']['10']}",
                flush=True,
            )
    fout.close()

    # ---- summary over processed rows ----
    processed = []
    with open(args.output) as f:
        for line in f:
            line = line.strip()
            if line:
                processed.append(json.loads(line))
    if not processed:
        print("no processed rows; nothing to summarize")
        return

    n = len(processed)
    agg = {"n_queries": n}
    for key in ("recall_gold", "recall_evidence"):
        agg[key] = {str(k): round(sum(r[key][str(k)] for r in processed) / n, 4) for k in K_LIST}
    agg["ndcg10_gold_mean"] = round(sum(r["ndcg10_gold"] for r in processed) / n, 4)
    agg["recall10_gold_queries_with_hit"] = sum(1 for r in processed if r["recall_gold"]["10"] > 0)
    agg["recall10_evidence_queries_with_hit"] = sum(1 for r in processed if r["recall_evidence"]["10"] > 0)
    with open(args.summary, "w") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print("\n===== BM25 Recall Summary ({} queries) =====".format(n))
    for k in K_LIST:
        print(f"  Recall@{k:>3}  gold={agg['recall_gold'][str(k)]:.4f}  evidence={agg['recall_evidence'][str(k)]:.4f}")
    print(f"  nDCG@10 (gold relevance) = {agg['ndcg10_gold_mean']:.4f}")
    print(f"  queries with >=1 gold hit @10: {agg['recall10_gold_queries_with_hit']}/{n}")
    print(f"  queries with >=1 evidence hit @10: {agg['recall10_evidence_queries_with_hit']}/{n}")
    print(f"saved summary -> {args.summary}")


if __name__ == "__main__":
    main()
