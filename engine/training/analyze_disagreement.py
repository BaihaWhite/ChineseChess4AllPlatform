#!/usr/bin/env python3
"""Analyze NNUE vs HCE label disagreement across all 11.25M positions.

Finds positions where NNUE predictions differ most from HCE labels.
These are the most valuable positions to re-label with deeper search.

Usage:
  python analyze_disagreement.py
"""

import time, sys
import torch
import torch.nn as nn

# Import training modules
from train_nnue import NnueModel, load_checkpoint, FEAT_TOTAL, HL1, HL2, QA, QA2, FEAT_PER_PSP, LONG

CHECKPOINT = "checkpoint_rl_iter2.pth"
FEATURES = "data/all_features_flat.pt"
BATCH_SIZE = 16384
TOP_K = 200_000  # Number of top disagreements to save


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {CHECKPOINT}")
    model = NnueModel().to(device)
    meta = load_checkpoint(model, CHECKPOINT, device)
    model.eval()
    print(f"  Model loaded, {sum(p.numel() for p in model.parameters()):,} params")

    # Load features (on CPU — 4.4GB is too big for GPU alongside model)
    print(f"\nLoading features: {FEATURES}")
    t0 = time.time()
    data = torch.load(FEATURES, map_location='cpu', weights_only=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    own_flat = data['own_flat']      # [N_total_own] int64, on CPU
    opp_flat = data['opp_flat']      # [N_total_opp] int64, on CPU
    own_off = data['own_offsets']    # [N+1] int64
    opp_off = data['opp_offsets']    # [N+1] int64
    scores = data['scores']          # [N] float32, HCE labels
    n = len(scores)

    print(f"  Samples: {n:,}")
    print(f"  Own flat: {own_flat.numel():,} indices ({own_flat.numel()*8/1024/1024:.0f} MB)")
    print(f"  Opp flat: {opp_flat.numel():,} indices ({opp_flat.numel()*8/1024/1024:.0f} MB)")

    # Move always-needed tensors to GPU
    own_off_gpu = own_off.to(device)
    opp_off_gpu = opp_off.to(device)
    own_flat_gpu = own_flat.to(device)
    opp_flat_gpu = opp_flat.to(device)
    print(f"  Moved feature indices to GPU ({own_flat.numel()*8/1024/1024*2:.0f} MB)")

    # Inference in batches
    print(f"\nRunning NNUE inference on {n:,} positions (batch={BATCH_SIZE})...")
    all_preds = []
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            idx = torch.arange(start, end, device=device, dtype=LONG)

            score_preds, value_preds, _ = model(own_flat_gpu, own_off_gpu, opp_flat_gpu, opp_off_gpu, idx)
            all_preds.append(score_preds.cpu())

            if start % (BATCH_SIZE * 100) == 0:
                done = end / n * 100
                elapsed = time.time() - t0
                rate = end / elapsed if elapsed > 0 else 0
                eta = (n - end) / rate if rate > 0 else 0
                print(f"  {done:.0f}% ({end:,}/{n:,})  {rate:,.0f} pos/s  ETA: {eta:.0f}s", flush=True)

    preds = torch.cat(all_preds)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s ({n/dt:,.0f} pos/s)", flush=True)

    # Compute disagreement
    print(f"\nComputing disagreements...")
    diff = (preds - scores).abs()

    # Statistics
    print(f"\nDisagreement distribution:")
    percentiles = [50, 75, 90, 95, 99, 99.9, 99.99]
    for p in percentiles:
        val = torch.quantile(diff, p / 100).item()
        print(f"  P{p:5.1f}: {val:8.1f}")

    print(f"  Mean:  {diff.mean().item():8.1f}")
    print(f"  Max:   {diff.max().item():8.1f}")
    print(f"  Min:   {diff.min().item():8.1f}")

    # Top disagreements
    print(f"\nTop disagreements (for HCE re-labeling):")
    top_vals, top_idx = torch.topk(diff, min(20, n))
    for i in range(20):
        pos = top_idx[i].item()
        print(f"  #{i+1}: line={pos:,}  HCE={scores[pos]:.0f}  NNUE={preds[pos]:.0f}  diff={top_vals[i]:.0f}")

    # Save top K indices for re-labeling
    topk = min(TOP_K, n)
    _, top_indices = torch.topk(diff, topk)

    output_path = "data/disagreement_top_indices.pt"
    torch.save(top_indices, output_path)
    print(f"\nSaved top {topk:,} disagreement indices to {output_path}")

    # Also save the predictions for later analysis
    preds_path = "data/nnue_predictions_all.pt"
    torch.save({'preds': preds, 'scores': scores, 'diff': diff}, preds_path)
    print(f"Saved all predictions to {preds_path} ({preds.numel()*4/1024/1024:.0f} MB)")

    # Distribution histogram
    print(f"\nDisagreement buckets:")
    buckets = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 99999)]
    for lo, hi in buckets:
        count = ((diff >= lo) & (diff < hi)).sum().item()
        pct = count / n * 100
        bar = '█' * int(pct * 5)
        print(f"  [{lo:5d} - {hi:5d}): {count:>10,} ({pct:5.2f}%) {bar}")


if __name__ == '__main__':
    main()
