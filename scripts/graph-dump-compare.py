#!/usr/bin/env python3
"""Compare two llama-graph-dump files tensor by tensor.

Tensors are matched by name and occurrence. By default the script lists the first tensors, in the order of the
first file, whose relative RMS difference exceeds --threshold, then a per-layer table of the residual tensor
(--residual) and of top-k expert-set changes (--routing, --topk), with the k-th/(k+1)-th router probability gap
(--probs) of the first file, and of the attention selection (--selection): the share of selected key blocks both files
pick. A routing change explains a large difference from that layer on. Last, the per-token KL divergence between the
--logits of the two files and how often their top-1 tokens agree.

With --write-routing FILE, the top-k experts of the first file's --routing tensors are written as a
llama-graph-dump --dump-routing file and nothing is compared.
"""

import argparse
import re
import struct
import sys
from collections import defaultdict

import numpy as np

GGML_TYPE_F32, GGML_TYPE_I32 = 0, 26


def load(path):
    with open(path, 'rb') as f:
        if f.read(8) != b'LGDUMP\0\1':
            sys.exit(f'{path}: not a graph dump')
        version, dump_pass, n_tokens = struct.unpack('<III', f.read(12))
        if version != 1:
            sys.exit(f'{path}: unsupported version {version}')
        tokens = np.frombuffer(f.read(4 * n_tokens), dtype='<i4')
        records, seen = [], defaultdict(int)
        while header := f.read(4):
            name_len, = struct.unpack('<I', header)
            name = f.read(name_len).decode()
            typ, op = struct.unpack('<II', f.read(8))
            ne = struct.unpack('<4q', f.read(32))
            n_bytes, = struct.unpack('<Q', f.read(8))
            if typ not in (GGML_TYPE_F32, GGML_TYPE_I32):
                sys.exit(f'{path}: unsupported type {typ} for {name}')
            data = np.frombuffer(f.read(n_bytes), dtype='<f4' if typ == GGML_TYPE_F32 else '<i4')
            records.append(dict(key=(name, seen[name]), name=name, type=typ, op=op, ne=ne, data=data))
            seen[name] += 1
    return tokens, records


def rel_rms(a, b):
    # masks hold infinities: those positions must match exactly, the rest is compared numerically
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(finite_a, finite_b) or not np.array_equal(a[~finite_a], b[~finite_b], equal_nan=True):
        return float('inf')
    a, b = a[finite_a].astype(np.float64), b[finite_b]
    if a.size == 0:
        return 0.0
    scale = np.sqrt(np.mean(a ** 2))
    if scale == 0:
        return 0.0 if np.array_equal(a, b) else float('inf')
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def by_layer(records, name):
    pattern = re.compile(re.escape(name) + r'-(\d+)')
    out = {}
    for r in records:
        m = pattern.fullmatch(r['name'])
        if m and int(m[1]) not in out:
            out[int(m[1])] = r
    return out


def write_routing(records, name, k, n_tokens, path):
    layers = by_layer(records, name)
    if not layers or sorted(layers) != list(range(len(layers))):
        sys.exit(f'{name}: expected layers 0..n-1, found {sorted(layers)}')
    with open(path, 'w') as f:
        f.write(f'{len(layers)} {n_tokens} {k}\n')
        for il in range(len(layers)):
            n_expert, n_tok = layers[il]['ne'][:2]
            if n_tok != n_tokens:
                sys.exit(f'{name}-{il}: {n_tok} tokens, expected {n_tokens}')
            ids = layers[il]['data'].reshape(n_tok, n_expert)[:, :k]
            f.write('\n'.join(' '.join(map(str, row)) for row in ids) + '\n')
    print(f'wrote the top-{k} routing of {len(layers)} layers and {n_tokens} tokens to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a')
    parser.add_argument('b', nargs='?')
    parser.add_argument('--write-routing', metavar='FILE', help='write the top-k routing of the first file and exit')
    parser.add_argument('--threshold', type=float, default=1e-4, help='relative RMS listed as a difference')
    parser.add_argument('--limit', type=int, default=20, help='differences to list (0: none)')
    parser.add_argument('--include', help='only list tensors whose name matches this regex')
    parser.add_argument('--unnamed', action='store_true',
                        help='also match auto-named node_<n> tensors, whose numbers shift when the graph changes')
    parser.add_argument('--residual', default='l_last', help='per-layer residual tensor name')
    parser.add_argument('--routing', default='ffn_moe_argsort', help='per-layer expert order tensor name (I32, [n_expert, n_tokens])')
    parser.add_argument('--probs', default='ffn_moe_probs', help='per-layer router probabilities in the first file')
    parser.add_argument('--topk', type=int, default=10, help='experts used per token')
    parser.add_argument('--selection', default='indexer_top_k',
                        help='per-layer selected key blocks (I32, [n_selected, n_tokens], unordered, negative ids ignored)')
    parser.add_argument('--logits', default='result_output', help='output logits tensor name')
    args = parser.parse_args()

    tokens_a, a = load(args.a)
    if args.write_routing:
        write_routing(a, args.routing, args.topk, len(tokens_a), args.write_routing)
        return
    if args.b is None:
        parser.error('two dumps are needed for a comparison')
    tokens_b, b = load(args.b)
    if not np.array_equal(tokens_a, tokens_b):
        print('warning: the dumps decode different tokens')
    unnamed = re.compile(r'node_\d+')
    index_b = {r['key']: r for r in b}
    candidates = [r for r in a if args.unnamed or not unnamed.fullmatch(r['name'])]
    matched = [(r, index_b[r['key']]) for r in candidates if r['key'] in index_b and index_b[r['key']]['data'].shape == r['data'].shape]
    skipped = len(a) - len(candidates)
    print(f'{len(a)} and {len(b)} tensors, {len(matched)} matched by name and occurrence' +
          (f' ({skipped} auto-named tensors skipped, see --unnamed)' if skipped else ''))

    include = re.compile(args.include) if args.include else None
    listed = differing = identical = 0
    for ra, rb in matched:
        if np.array_equal(ra['data'], rb['data']):
            identical += 1
            continue
        if ra['type'] == GGML_TYPE_I32:
            differing += 1
            text = f'{np.sum(ra["data"] != rb["data"])}/{ra["data"].size} values differ'
        else:
            err = rel_rms(ra['data'], rb['data'])
            if err <= args.threshold:
                continue
            differing += 1
            text = f'rel_rms {err:.3e}'
        if listed < args.limit and (include is None or include.search(ra['name'])):
            print(f'  {ra["name"]:44s} #{ra["key"][1]:<3d} ne={list(ra["ne"][:3])} {text}')
            listed += 1
    print(f'{identical} identical, {differing} above threshold {args.threshold:g} (I32: any difference)')

    res_a, res_b = by_layer(a, args.residual), by_layer(b, args.residual)
    route_a, route_b = by_layer(a, args.routing), by_layer(b, args.routing)
    probs_a = by_layer(a, args.probs)
    sel_a, sel_b = by_layer(a, args.selection), by_layer(b, args.selection)
    k = args.topk
    for il in sorted(set(res_a) & set(res_b)):
        line = f'layer {il:3d} {args.residual} rel_rms {rel_rms(res_a[il]["data"], res_b[il]["data"]):.2e}'
        if il in route_a and il in route_b:
            # either file may hold the full expert order or only its top k
            n_tok = route_a[il]['ne'][1]
            ia = route_a[il]['data'].reshape(n_tok, route_a[il]['ne'][0])
            ib = route_b[il]['data'].reshape(route_b[il]['ne'][1], route_b[il]['ne'][0])
            changed = sum(set(ia[t, :k]) != set(ib[t, :k]) for t in range(n_tok))
            line += f'  top-{k} set differs {changed}/{n_tok}'
            if il in probs_a and probs_a[il]['data'].size % n_tok == 0:
                p = -np.sort(-probs_a[il]['data'].reshape(n_tok, -1), axis=1)
                line += f'  min gap {np.min(p[:, k - 1] - p[:, k]):.1e}'
        if il in sel_a and il in sel_b:
            n_tok = sel_a[il]['ne'][1]
            sa = sel_a[il]['data'].reshape(n_tok, sel_a[il]['ne'][0])
            sb = sel_b[il]['data'].reshape(sel_b[il]['ne'][1], sel_b[il]['ne'][0])
            common = total = rows = 0
            for t in range(n_tok):
                ta, tb = set(sa[t][sa[t] >= 0]), set(sb[t][sb[t] >= 0])
                common += len(ta & tb)
                total += max(len(ta), len(tb))
                rows += ta != tb
            line += f'  {args.selection} shared {common / max(total, 1):.4f}, differs {rows}/{n_tok}'
        print(line)

    log_a = next((r for r in a if r['name'] == args.logits), None)
    log_b = next((r for r in b if r['name'] == args.logits), None)
    if log_a is not None and log_b is not None and log_a['ne'] == log_b['ne']:
        def log_softmax(r):
            x = r['data'].reshape(r['ne'][1], r['ne'][0]).astype(np.float64)
            x = x - x.max(axis=1, keepdims=True)
            return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
        la, lb = log_softmax(log_a), log_softmax(log_b)
        kl = (np.exp(la) * (la - lb)).sum(axis=1)
        top = la.argmax(axis=1) == lb.argmax(axis=1)
        print(f'{args.logits}: KL(a||b) per token ' + ' '.join(f'{v:.1e}' for v in kl) +
              f'  mean {kl.mean():.2e}  max {kl.max():.2e}  top-1 same {top.sum()}/{len(top)}')


if __name__ == '__main__':
    main()
