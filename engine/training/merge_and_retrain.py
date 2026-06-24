#!/usr/bin/env python3
"""Merge NNUE-rescored labels into existing features and retrain.

1. Load existing all_features_flat.pt
2. Replace scores for the top-K disputed positions with new NNUE depth-8 labels
3. Retrain from checkpoint
4. Export

Usage:
  python merge_and_retrain.py
"""

import time, sys, os
import torch
import torch.nn as nn

from train_nnue import NnueModel, load_checkpoint, export, FEAT_TOTAL, HL1, HL2, QA, QA2, LONG

# Config
CHECKPOINT = "checkpoint_rl_iter2.pth"
FEATURES = "data/all_features_flat.pt"
RESCORED_TXT = "data/rescored_50k_d8.txt"
INDICES_PT = "data/disagreement_top_indices.pt"
OUTPUT_BIN = "nnue_improved.bin"
OUTPUT_PTH = "checkpoint_improved.pth"

EPOCHS = 30
LR = 0.0001
BATCH_SIZE = 16384


def load_rescored_scores(txt_path, n_expected):
    """Load new scores from rescored text file. Returns tensor indexed by original line number."""
    new_scores = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                try:
                    score = float(parts[1])
                    # We need original line index — stored in rescored order (1:1 with indices)
                    new_scores[len(new_scores)] = score
                except ValueError:
                    continue
    print(f"  Loaded {len(new_scores)} new scores from {txt_path}")
    return new_scores


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load indices of disputed positions
    indices = torch.load(INDICES_PT, weights_only=True)
    n_replace = min(50000, len(indices))
    indices = indices[:n_replace]
    print(f"Replacing scores for top {n_replace:,} disputed positions")

    # Load rescored scores
    rescored_map = load_rescored_scores(RESCORED_TXT, n_replace)

    # Load existing features
    print(f"\nLoading {FEATURES}...")
    t0 = time.time()
    data = torch.load(FEATURES, map_location='cpu', weights_only=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    scores = data['scores'].clone()  # [N]
    n = len(scores)
    print(f"  Original scores: {n:,} positions")

    # Replace scores
    replaced = 0
    score_deltas = []
    for i, orig_idx in enumerate(indices.tolist()):
        if orig_idx < n and i in rescored_map:
            old_score = scores[orig_idx].item()
            new_score = rescored_map[i]
            scores[orig_idx] = new_score
            replaced += 1
            score_deltas.append(abs(new_score - old_score))
            if i < 5:
                print(f"  [{i}] line={orig_idx}: {old_score:.0f} -> {new_score:.0f}")

    print(f"  Replaced {replaced:,} scores")
    if score_deltas:
        import statistics
        deltas = sorted(score_deltas)
        print(f"  Score delta: median={deltas[len(deltas)//2]:.0f}  mean={sum(deltas)/len(deltas):.0f}  max={max(deltas):.0f}")

    # Move data to device
    own_flat = data['own_flat'].to(device)
    opp_flat = data['opp_flat'].to(device)
    own_off = data['own_offsets'].to(device)
    opp_off = data['opp_offsets'].to(device)
    scores = scores.to(device)
    game_results = data.get('game_results', torch.zeros(n)).to(device)

    n_train = int(n * 0.9)
    val_idx = torch.arange(n_train, n, device=device, dtype=LONG)

    # Build model
    print(f"\nBuilding model...")
    model = NnueModel().to(device)
    if os.path.exists(CHECKPOINT):
        meta = load_checkpoint(model, CHECKPOINT, device)
        print(f"Loaded checkpoint from {CHECKPOINT}")
    else:
        print(f"No checkpoint, training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params ({n_params*4/1024/1024:.0f} MB fp32)")

    # Freeze L1 for fine-tuning
    model.l1_weight.requires_grad_(False)
    model.l1_bias.requires_grad_(False)
    print("  L1 frozen (fine-tuning mode)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_corr = -1.0
    print(f"\nTraining {EPOCHS} epochs (lr={LR}, batch={BATCH_SIZE})...")
    print()

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Train one epoch
        model.train()
        perm = torch.randperm(n_train, device='cpu')
        total_loss = total_mse_s = total_mse_v = total_reg = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            idx = perm[start:end].to(device)

            score_preds, value_preds, act_reg = model(own_flat, own_off, opp_flat, opp_off, idx)
            mse_s = nn.functional.mse_loss(score_preds, scores[idx])
            mse_v = nn.functional.mse_loss(value_preds, game_results[idx])
            loss = mse_s + 1.0 * mse_v + 0.01 * act_reg

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            nb = idx.numel()
            total_loss += loss.item() * nb
            total_mse_s += mse_s.item() * nb
            total_mse_v += mse_v.item() * nb
            total_reg += act_reg.item() * nb

        dt = time.time() - t0

        # Validate
        model.eval()
        with torch.no_grad():
            preds_all = []
            for start in range(0, val_idx.numel(), BATCH_SIZE):
                end = min(start + BATCH_SIZE, val_idx.numel())
                idx = val_idx[start:end].to(device)
                sp, _, _ = model(own_flat, own_off, opp_flat, opp_off, idx)
                preds_all.append(sp.cpu())
            p = torch.cat(preds_all)
            t = scores[val_idx].cpu()
            val_loss = nn.functional.mse_loss(p, t).item()
            corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()

        # Log
        nf = float(n_train)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"loss={total_loss/nf:7.2f}  mse_s={total_mse_s/nf:7.2f}  "
              f"mse_v={total_mse_v/nf:.4f}  reg={total_reg/nf:.4f}  "
              f"val={val_loss:7.2f}  corr={corr:+.4f}  "
              f"lr={lr:.1e}  dt={dt:.1f}s", flush=True)

        if corr > best_corr:
            best_corr = corr
            print(f"  [new best corr: {corr:.4f}]", flush=True)

    print(f"\nBest corr: {best_corr:.4f}")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': EPOCHS,
        'best_corr': best_corr,
        'lr': LR,
    }, OUTPUT_PTH)
    print(f"Saved checkpoint to {OUTPUT_PTH}")

    # Export
    n_pieces_avg = len(own_flat) / n
    export(model, OUTPUT_BIN, n_pieces_avg)


if __name__ == '__main__':
    main()
