"""Streaming precompute: save features in flat format to save memory.
Uses numpy int32 arrays per chunk, then concatenates via torch at the end.
"""
import argparse, os
import torch
import numpy as np
from train import parse_fen, extract_features, RED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='features_flat.pt')
    args = parser.parse_args()

    own_chunks = []
    opp_chunks = []
    own_lens = []      # per-sample feature counts
    opp_lens = []
    scores_list = []
    results_list = []
    chunk_own = []
    chunk_opp = []
    chunk_scores = []
    chunk_results = []
    CHUNK_SIZE = 100000
    total = 0
    valid = 0
    has_results = None  # auto-detect from first valid line

    print(f"Processing {args.data}...", flush=True)
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            # Format: <fen_board> <side> <score> [game_result]
            # FEN is first 2 tokens: board_rows side_to_move
            fen = ' '.join(tokens[:2])
            try:
                score = float(tokens[2])
                game_result = float(tokens[3]) if len(tokens) >= 4 else 0.0
            except ValueError:
                continue
            total += 1

            board = parse_fen(fen)
            own_f, opp_f = extract_features(board, RED)
            if own_f.numel() == 0:
                continue

            own_arr = own_f.numpy().astype(np.int32)
            opp_arr = opp_f.numpy().astype(np.int32)
            chunk_own.append(own_arr)
            chunk_opp.append(opp_arr)
            chunk_scores.append(score)
            chunk_results.append(game_result)
            valid += 1

            if len(chunk_own) >= CHUNK_SIZE:
                _flush_chunk(chunk_own, chunk_opp, chunk_scores, chunk_results,
                             own_chunks, opp_chunks, own_lens, opp_lens, scores_list, results_list)
                chunk_own.clear()
                chunk_opp.clear()
                chunk_scores.clear()
                chunk_results.clear()
                print(f"  {valid:,} valid / {total:,} total", flush=True)

    # Final chunk
    if chunk_own:
        _flush_chunk(chunk_own, chunk_opp, chunk_scores, chunk_results,
                     own_chunks, opp_chunks, own_lens, opp_lens, scores_list, results_list)

    print(f"Total: {total:,} lines, {valid:,} valid positions", flush=True)

    # Build offsets from per-sample lengths
    print("Building offsets...", flush=True)
    own_offs = [0]
    opp_offs = [0]
    for l in own_lens:
        own_offs.append(own_offs[-1] + l)
    for l in opp_lens:
        opp_offs.append(opp_offs[-1] + l)

    # Concatenate numpy arrays and convert to tensor
    print("Merging chunks...", flush=True)
    own_flat = torch.from_numpy(np.concatenate(own_chunks)).long()
    opp_flat = torch.from_numpy(np.concatenate(opp_chunks)).long()
    del own_chunks, opp_chunks

    scores_t = torch.tensor(scores_list, dtype=torch.float32)
    results_t = torch.tensor(results_list, dtype=torch.float32)
    own_off_t = torch.tensor(own_offs, dtype=torch.int64)
    opp_off_t = torch.tensor(opp_offs, dtype=torch.int64)

    print(f"  own_flat: {list(own_flat.shape)} ({own_flat.numel():,} indices)", flush=True)
    print(f"  opp_flat: {list(opp_flat.shape)} ({opp_flat.numel():,} indices)", flush=True)

    print(f"Saving to {args.output}...", flush=True)
    torch.save({
        'own_flat': own_flat,
        'own_offsets': own_off_t,
        'opp_flat': opp_flat,
        'opp_offsets': opp_off_t,
        'scores': scores_t,
        'game_results': results_t,
        'num_samples': valid,
    }, args.output)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done. Saved {valid:,} samples to {args.output} ({size_mb:.0f} MB)", flush=True)


def _flush_chunk(chunk_own, chunk_opp, chunk_scores, chunk_results,
                 own_chunks, opp_chunks, own_lens, opp_lens, scores_list, results_list):
    """Concatenate chunk samples and store per-sample feature lengths."""
    own_lens.extend(a.shape[0] for a in chunk_own)
    opp_lens.extend(a.shape[0] for a in chunk_opp)
    own_flat_chunk = np.concatenate(chunk_own)
    opp_flat_chunk = np.concatenate(chunk_opp)
    own_chunks.append(own_flat_chunk)
    opp_chunks.append(opp_flat_chunk)
    scores_list.extend(chunk_scores)
    results_list.extend(chunk_results)


if __name__ == '__main__':
    main()
