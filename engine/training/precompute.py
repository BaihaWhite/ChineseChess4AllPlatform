"""Pre-compute NNUE feature indices from text data to avoid CPU bottleneck during training."""

import argparse
import torch
import numpy as np
from train import parse_fen, extract_features, load_text_data, RED, PositionDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='features.pt')
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    raw = load_text_data(args.data)
    print(f"  {len(raw)} positions")

    # Augment: both Red and Black perspectives
    all_data = [(fen, score) for fen, score in raw] + [(fen, -score) for fen, score in raw]
    print(f"  {len(all_data)} samples (with perspective augmentation)")

    own_list = []
    opp_list = []
    scores_list = []

    for i, (fen, score) in enumerate(all_data):
        board = parse_fen(fen)
        own_f, opp_f = extract_features(board, RED)
        if own_f.numel() == 0:
            continue
        own_list.append(own_f)
        opp_list.append(opp_f)
        scores_list.append(score)

        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{len(all_data)} processed")

    print(f"Saving {len(scores_list)} samples to {args.output}...")
    torch.save({
        'own_list': own_list,
        'opp_list': opp_list,
        'scores': torch.tensor(scores_list, dtype=torch.float32),
    }, args.output)
    print("Done.")

if __name__ == '__main__':
    main()
