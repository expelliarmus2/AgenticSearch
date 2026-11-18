"""
Build a tiny faiss FlatIP index from data/local/corpus.jsonl using a local
e5 encoder, for the offline retriever server.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import faiss

from s3.search.retrieval_server import Encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='data/local/corpus.jsonl')
    ap.add_argument('--model', default='models/e5-small-v2')
    ap.add_argument('--out', default='data/local/e5_Flat.index')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    encoder = Encoder(model_name='e5', model_path=args.model,
                      pooling_method='mean', max_length=256, use_fp16=True)

    docs = []
    with open(args.corpus) as f:
        for line in f:
            docs.append(json.loads(line))

    contents = [d['contents'] for d in docs]
    embs = []
    for i in range(0, len(contents), args.batch):
        batch = contents[i:i + args.batch]
        # e5 passages are prefixed by "passage: " inside Encoder.encode
        e = encoder.encode(batch, is_query=False)
        embs.append(e)
        print(f'encoded {min(i + args.batch, len(contents))}/{len(contents)}', flush=True)

    emb = np.vstack(embs).astype('float32')
    print('embedding shape:', emb.shape)

    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    faiss.write_index(index, args.out)
    print('index written to', args.out, 'with', index.ntotal, 'docs')


if __name__ == '__main__':
    main()
