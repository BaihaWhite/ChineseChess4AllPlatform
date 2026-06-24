"""Train nonlinear HalfKP NNUE with custom CUDA L1 kernel + AMP.

Architecture (matches Rust nnue.rs):
  L1: HalfKP sparse → 256 (custom CUDA kernel, fp16 weights)
  CReLU
  L2: 512 → 32 (PyTorch Linear, fp16)
  CReLU
  Output: 32 → 1

Usage:
  python train_nnue.py --features data/all_features_flat.pt --epochs 50 --output nnue_trained.bin
"""

import argparse, struct, time, math, os
import torch
import torch.nn as nn

from halfkp_ops import halfkp_l1, FEAT_TOTAL, HL1

# ---------------------------------------------------------------------------
# Constants (match nnue.rs)
# ---------------------------------------------------------------------------
KING_SQ = 9
PC_TYPE = 14
BOARD_SQ = 90
FEAT_PER_PSP = KING_SQ * PC_TYPE * BOARD_SQ  # 11340
HL2 = 32
QA = 255
QA2 = 255
L1_SCALE = 64
L2_SCALE = 64

LONG = torch.long


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NnueModel(nn.Module):
    def __init__(self):
        super().__init__()
        # L1: fp32 for optimizer/GradScaler compatibility; CUDA kernel reads as fp16
        self.l1_weight = nn.Parameter(torch.zeros(FEAT_TOTAL * HL1, dtype=torch.float32))
        self.l1_bias = nn.Parameter(torch.zeros(HL1, dtype=torch.float32))
        # L2: 512 → 32 (own+opp concatenated)
        self.l2 = nn.Linear(HL1 * 2, HL2)
        # Output: 32 → 1 (score prediction)
        self.output = nn.Linear(HL2, 1)
        # Value head: 32 → 1 (game outcome prediction, +1 Red win, -1 Black win, 0 draw)
        self.value_head = nn.Linear(HL2, 1)
        self._init_weights()

    def _init_weights(self):
        # L1: each position has ~25 features (12.4 pieces/side × 2 perspectives).
        # Accumulator std = σ_weight × √25 ≈ σ_weight × 5.
        # σ=2.0 → raw accum range ~127.5 ± 20 (2σ), covering [87, 168] before clamp.
        # This gives meaningful initial variance between positions without saturation.
        nn.init.normal_(self.l1_weight, std=2.0)
        nn.init.constant_(self.l1_bias, QA / 2.0)
        # L2: small weights to map [0,QA] range to [0,QA2]; bias near center
        nn.init.normal_(self.l2.weight, std=0.01)
        nn.init.constant_(self.l2.bias, QA2 / 2.0)
        nn.init.normal_(self.output.weight, std=0.5)
        nn.init.zeros_(self.output.bias)
        # L2 output is in [0, QA2]=[0,255], centered ~127 with std ~16.
        # To keep pre-tanh activations ~N(0,1) for meaningful gradients:
        # σ ≈ 1/sqrt(32 * E[l2_out²]) ≈ 1/sqrt(32*16535) ≈ 0.0014
        nn.init.normal_(self.value_head.weight, std=0.001)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, own_flat, own_off, opp_flat, opp_off, idx):
        """Forward pass for a batch.

        Args:
            own_flat: int64 [N_own_total] global own-feature indices
            own_off:  int64 [num_samples+1] cumulative offsets
            opp_flat: int64 [N_opp_total] global opponent-feature indices
            opp_off:  int64 [num_samples+1] cumulative offsets
            idx:      int64 [bs] sample indices within this batch
        Returns:
            score_preds: fp32 [bs] predicted scores
            value_preds: fp32 [bs] predicted game outcomes (tanh, range [-1,+1])
        """
        bs = idx.numel()

        # Build per-sample start/length tensors (zero-copy slices)
        own_starts = own_off[idx]                    # [bs]
        own_lens = own_off[idx + 1] - own_starts     # [bs]
        opp_starts = opp_off[idx]                    # [bs]
        opp_lens = opp_off[idx + 1] - opp_starts     # [bs]

        # L1 forward via custom CUDA kernel (fp16 weights → fp32 output)
        l1_own = halfkp_l1(own_flat, own_starts, own_lens, self.l1_weight, self.l1_bias, offset=0)
        l1_opp = halfkp_l1(opp_flat, opp_starts, opp_lens, self.l1_weight, self.l1_bias,
                           offset=FEAT_PER_PSP)

        # Activation regularization on PRE-CLAMP values.
        # This is critical: once clamped to [0, QA], saturated neurons have zero
        # gradient and cannot recover. By regularizing raw accumulator values,
        # we create a penalty gradient that pulls weights back from saturation.
        # Target: raw accum ~QA/2, with penalty growing quadratically beyond.
        act_reg = ((l1_own - QA / 2) ** 2).mean() + ((l1_opp - QA / 2) ** 2).mean()

        # CReLU clamp: match int8 inference activation range
        l1_own = torch.clamp(l1_own, 0, QA)
        l1_opp = torch.clamp(l1_opp, 0, QA)

        # L2 + Output (fp16 matmul via AMP autocast)
        with torch.amp.autocast('cuda'):
            l2_in = torch.cat([l1_own, l1_opp], dim=1)  # [bs, 512]
            l2_out = torch.clamp(self.l2(l2_in), 0, QA2)  # [bs, 32]
            score_preds = self.output(l2_out).squeeze(-1)   # [bs]
            value_preds = torch.tanh(self.value_head(l2_out)).squeeze(-1)  # [bs] in [-1,+1]

        return score_preds.float(), value_preds.float(), act_reg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, own_flat, own_off, opp_flat, opp_off, scores, game_results,
                n_train, optimizer, scaler, batch_size, device, epoch, epochs,
                act_reg_weight=0.01, lambda_value=1.0):
    model.train()
    perm = torch.randperm(n_train, device='cpu')
    total_loss = 0.0
    total_mse_score = 0.0
    total_mse_value = 0.0
    total_reg = 0.0

    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        idx = perm[start:end].to(device)

        score_preds, value_preds, act_reg = model(own_flat, own_off, opp_flat, opp_off, idx)
        score_targets = scores[idx]
        value_targets = game_results[idx]

        mse_score = nn.functional.mse_loss(score_preds, score_targets)
        mse_value = nn.functional.mse_loss(value_preds, value_targets)
        loss = mse_score + lambda_value * mse_value + act_reg_weight * act_reg

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        n_batch = idx.numel()
        total_loss += loss.item() * n_batch
        total_mse_score += mse_score.item() * n_batch
        total_mse_value += mse_value.item() * n_batch
        total_reg += act_reg.item() * n_batch

    n = float(n_train)
    return total_loss / n, total_mse_score / n, total_mse_value / n, total_reg / n


@torch.no_grad()
def validate(model, own_flat, own_off, opp_flat, opp_off, scores, game_results, val_idx, device):
    model.eval()
    bs_val = min(4096, val_idx.numel())
    preds_all = []
    vals_all = []

    for start in range(0, val_idx.numel(), bs_val):
        end = min(start + bs_val, val_idx.numel())
        idx = val_idx[start:end].to(device)
        score_preds, value_preds, _ = model(own_flat, own_off, opp_flat, opp_off, idx)
        preds_all.append(score_preds)
        vals_all.append(value_preds)
    p = torch.cat(preds_all)
    v = torch.cat(vals_all)
    t = scores[val_idx]
    g = game_results[val_idx]
    loss = nn.functional.mse_loss(p, t).item()
    corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
    val_loss = nn.functional.mse_loss(v, g).item()
    val_corr = torch.corrcoef(torch.stack([v, g]))[0, 1].item()
    return loss, corr, p.mean().item(), p.std().item(), val_loss, val_corr


# ---------------------------------------------------------------------------
# Export (matches Rust nnue.rs binary format exactly)
# ---------------------------------------------------------------------------

def export(model, path, n_pieces_avg):
    """Quantize and export to Rust NNUE binary format."""
    cpu = torch.device('cpu')

    # L1 weights: flat [FEAT_TOTAL * HL1] fp32 → quantize with L1_SCALE
    l1_w = (model.l1_weight.data.cpu() * L1_SCALE).round().clamp(-32768, 32767).to(torch.int16)
    l1_b = (model.l1_bias.data.cpu() * L1_SCALE).round().clamp(-32768, 32767).to(torch.int16)
    l1_w_np = l1_w.numpy()
    l1_b_np = l1_b.numpy()

    # L2 weights: PyTorch stores Linear(512,32) as [32, 512] (out × in).
    # Rust accesses as weights[in_idx * HL2 + out_idx] = [512, 32] layout.
    # Transpose to match Rust's access pattern.
    l2_w = (model.l2.weight.data.float().cpu().T * L2_SCALE).round().clamp(-32768, 32767).to(torch.int16)
    l2_b = (model.l2.bias.data.float().cpu() * L2_SCALE).round().clamp(-32768, 32767).to(torch.int16)
    l2_w_np = l2_w.numpy()
    l2_b_np = l2_b.numpy()

    # Output weights: [32] → quantize with L2_SCALE
    out_w = (model.output.weight.data.float().cpu().squeeze(0) * L2_SCALE).round().clamp(-32768, 32767).to(torch.int16)
    out_b = int(round((model.output.bias.data.float().cpu().item() * L2_SCALE)))

    l2_w_np = l2_w_np.astype('int16')
    l2_b_np = l2_b_np.astype('int16')
    out_w_np = out_w.numpy().astype('int16')

    with open(path, 'wb') as f:
        f.write(b'NNUE')
        f.write(struct.pack('<I', FEAT_TOTAL))
        f.write(struct.pack('<I', HL1))
        f.write(struct.pack('<I', HL2))
        for arr in [l1_w_np, l1_b_np, l2_w_np, l2_b_np, out_w_np]:
            f.write(struct.pack('<I', arr.size))
            f.write(arr.tobytes())
        f.write(struct.pack('<i', out_b))

    # Stats
    l1_w_f32 = model.l1_weight.data
    l2_w_f32 = model.l2.weight.data.float()
    print(f"\nExported to {path}")
    print(f"  L1 weight: [{FEAT_TOTAL}×{HL1}] fp32_range=[{l1_w_f32.min():.4f}, {l1_w_f32.max():.4f}]"
          f"  i16_range=[{l1_w_np.min()}, {l1_w_np.max()}]")
    print(f"  L2 weight: [512×32] fp32_range=[{l2_w_f32.min():.4f}, {l2_w_f32.max():.4f}]"
          f"  i16_range=[{l2_w_np.min()}, {l2_w_np.max()}]")
    print(f"  Output bias: {out_b}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_checkpoint(model, path, device):
    """Load model state_dict from a .pth checkpoint.
    Returns metadata dict (may be empty for raw state_dict files).
    Backward compatible: old checkpoints without value_head get fresh value_head init."""
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # Extract state_dict regardless of wrapper format
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('l1_') or k.startswith('l2.') or k.startswith('output.') for k in ckpt.keys()):
        state_dict = ckpt
    else:
        state_dict = ckpt

    # Backward compat: old checkpoints may be missing value_head keys
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [checkpoint compat] Missing keys (using fresh init): {missing}", flush=True)
    if unexpected:
        print(f"  [checkpoint compat] Unexpected keys (ignored): {unexpected}", flush=True)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt
    return {}


def main():
    parser = argparse.ArgumentParser(description='Train nonlinear HalfKP NNUE')
    parser.add_argument('--features', required=True, help='Precomputed flat features .pt file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--output', default='nnue_trained.bin')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Epochs where only L2+output are trained (L1 frozen)')
    parser.add_argument('--checkpoint', default=None,
                        help='Load pre-trained .pth checkpoint for fine-tuning')
    parser.add_argument('--save-checkpoint', default=None,
                        help='Save PyTorch checkpoint after training (default: <output>.pth)')
    parser.add_argument('--freeze-l1', action='store_true',
                        help='Keep L1 frozen during fine-tuning')
    parser.add_argument('--lambda-value', type=float, default=1.0,
                        help='Weight for value-head MSE loss (default: 1.0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}", flush=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading {args.features}...", flush=True)
    t0 = time.time()
    data = torch.load(args.features, map_location='cpu', weights_only=True)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    own_flat = data['own_flat'].to(device)
    opp_flat = data['opp_flat'].to(device)
    own_off = data['own_offsets'].to(device)
    opp_off = data['opp_offsets'].to(device)
    scores = data['scores']
    n = len(scores)

    print(f"  Samples: {n:,}", flush=True)
    unique, counts = torch.unique(scores, return_counts=True)
    for val, cnt in zip(unique.tolist(), counts.tolist()):
        print(f"    score={val:4.0f}: {cnt:>12,} ({100*cnt/n:.1f}%)", flush=True)

    scores = scores.to(device)

    # Load game_results with backward compatibility
    if 'game_results' in data:
        game_results = data['game_results'].to(device)
        print(f"  Loaded game_results ({len(game_results)} samples)", flush=True)
    else:
        game_results = torch.zeros(n, device=device, dtype=torch.float32)
        print(f"  No game_results in data, using zeros (value head will be inactive)", flush=True)

    # Train/val split
    n_train = int(n * 0.9)
    print(f"  Train: {n_train:,}  Val: {n - n_train:,}", flush=True)

    n_pieces_avg = len(own_flat) / n
    print(f"  Avg pieces/pos: {n_pieces_avg:.1f}", flush=True)

    val_idx = torch.arange(n_train, n, device=device, dtype=LONG)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = NnueModel().to(device)
    if args.checkpoint:
        meta = load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint from {args.checkpoint}", flush=True)
        if meta.get('epoch'):
            print(f"  Previously trained for {meta['epoch']} epochs, best_corr={meta.get('best_corr', 'N/A')}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,} ({n_params*4/1024/1024:.0f} MB fp32, "
          f"{n_params*2/1024/1024:.0f} MB fp16)", flush=True)

    # ------------------------------------------------------------------
    # Progressive unfreezing / fine-tuning
    # ------------------------------------------------------------------
    if args.checkpoint and args.freeze_l1:
        # Fine-tuning mode: keep L1 frozen, only train L2+output
        model.l1_weight.requires_grad_(False)
        model.l1_bias.requires_grad_(False)
        args.warmup_epochs = 0
        print("  [Fine-tuning: L1 frozen, training L2+output only]", flush=True)
    else:
        # From scratch or full retrain: progressive unfreezing
        model.l1_weight.requires_grad_(False)
        model.l1_bias.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_corr = -1.0
    t_train = 0.0

    for epoch in range(args.epochs):
        # Unfreeze L1 after warmup (skipped in fine-tuning mode)
        if not args.freeze_l1 and epoch == args.warmup_epochs:
            model.l1_weight.requires_grad_(True)
            model.l1_bias.requires_grad_(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            scaler = torch.amp.GradScaler('cuda')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs)
            print("  [unfroze L1]", flush=True)

        t0 = time.time()
        avg_loss, avg_mse_score, avg_mse_value, avg_reg = train_epoch(
            model, own_flat, own_off, opp_flat, opp_off, scores, game_results,
            n_train, optimizer, scaler, args.batch_size, device,
            epoch, args.epochs, lambda_value=args.lambda_value)
        dt = time.time() - t0
        t_train += dt

        # Validate
        val_loss, corr, pred_mean, pred_std, val_loss_v, val_corr_v = validate(
            model, own_flat, own_off, opp_flat, opp_off, scores, game_results, val_idx, device)

        # Learning rate info
        last_lr = optimizer.param_groups[0]['lr']

        # Activation health check
        l1_w_norm = model.l1_weight.data.norm().item()

        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={avg_loss:7.2f}  mse_s={avg_mse_score:7.2f}  mse_v={avg_mse_value:6.4f}  reg={avg_reg:.4f}  "
              f"val_s={val_loss:7.2f}  corr_s={corr:+.4f}  "
              f"val_v={val_loss_v:.4f}  corr_v={val_corr_v:+.4f}  "
              f"pred(μ={pred_mean:+.1f} σ={pred_std:.1f})  "
              f"|L1|={l1_w_norm:.1f}  lr={last_lr:.1e}  dt={dt:.1f}s", flush=True)

        if corr > best_corr:
            best_corr = corr
            print(f"  [new best corr: {corr:.4f}]", flush=True)

    print(f"\nTotal train time: {t_train/60:.1f} min  Best corr: {best_corr:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Save PyTorch checkpoint
    # ------------------------------------------------------------------
    ckpt_path = args.save_checkpoint or (args.output.rsplit('.', 1)[0] + '.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'best_corr': best_corr,
        'lr': args.lr,
        'warmup_epochs': args.warmup_epochs,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}", flush=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    export(model, args.output, n_pieces_avg)


if __name__ == '__main__':
    main()
