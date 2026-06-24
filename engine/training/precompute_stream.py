"""Streaming precompute: parse FENs line-by-line and save HalfKP feature indices."""
import argparse
import torch
import numpy as np
from train import parse_fen, extract_features, RED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='features.pt')
    args = parser.parse_args()

    own_list = []
    opp_list = []
    scores_list = []
    total = 0

    print(f"Processing {args.data}...", flush=True)
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            fen = ' '.join(parts[:-1])
            try:
                score = float(parts[-1])
            except ValueError:
                continue

            board = parse_fen(fen)
            own_f, opp_f = extract_features(board, RED)
            if own_f.numel() == 0:
                continue
            own_list.append(own_f)
            opp_list.append(opp_f)
            scores_list.append(score)
            total += 1

            if total % 200000 == 0:
                print(f"  {total:,} positions processed ({len(own_list):,} valid)", flush=True)

    print(f"Total: {total:,} lines, {len(scores_list):,} valid positions", flush=True)
    print(f"Saving to {args.output}...", flush=True)

    torch.save({
        'own_list': own_list,
        'opp_list': opp_list,
        'scores': torch.tensor(scores_list, dtype=torch.float32),
    }, args.output)

    print(f"Done. Saved {len(scores_list):,} samples to {args.output}", flush=True)

if __name__ == '__main__':
    main()
