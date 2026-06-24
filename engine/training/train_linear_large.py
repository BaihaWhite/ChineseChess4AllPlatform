"""Train linear HalfKP model — fully vectorized on GPU with repeat_interleave."""
import argparse, struct
import torch

FEAT_TOTAL = 22680
FEAT_PER_PSP = 11340
HL1 = 256
HL2 = 32
L1_SCALE = 64
L2_SCALE = 64

# Shorthand for dtype used everywhere for indices
LONG = torch.long


def train_and_export(features_path, output_path, epochs, lr, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    data = torch.load(features_path, map_location='cpu', weights_only=True)
    own_flat = data['own_flat'].to(device)
    opp_flat = data['opp_flat'].to(device)
    own_off = data['own_offsets'].to(device)
    opp_off = data['opp_offsets'].to(device)
    scores = data['scores']
    n = len(scores)

    print(f"Samples: {n:,}", flush=True)
    unique, counts = torch.unique(scores, return_counts=True)
    for val, cnt in zip(unique.tolist(), counts.tolist()):
        print(f"  score={val:4.0f}: {cnt:>12,} ({100*cnt/n:.1f}%)", flush=True)

    n_train = int(n * 0.9)
    n_val = n - n_train
    scores_gpu = scores.to(device)
    print(f"Train: {n_train:,}  Val: {n_val:,}", flush=True)

    n_pieces_avg = len(own_flat) / n
    print(f"Avg pieces/pos: {n_pieces_avg:.1f}", flush=True)

    w = torch.zeros(FEAT_TOTAL, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    loss_fn = torch.nn.MSELoss()

    val_idx = torch.arange(n_train, n, device=device, dtype=LONG)
    best_corr = -1.0

    for epoch in range(epochs):
        perm = torch.randperm(n_train, device='cpu')
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            bs = end - start
            idx = perm[start:end].to(device)

            # ---- Vectorized batch construction on GPU ----
            own_lens = own_off[idx + 1] - own_off[idx]          # [bs]
            total_own = own_lens.sum().item()
            own_batch = torch.repeat_interleave(
                torch.arange(bs, device=device, dtype=LONG), own_lens)
            own_starts = own_off[idx]                            # [bs]
            cs = torch.cat([
                torch.zeros(1, device=device, dtype=LONG),
                torch.cumsum(own_lens, dim=0)[:-1]
            ])
            pos = (torch.arange(total_own, device=device, dtype=LONG)
                   - torch.repeat_interleave(cs, own_lens))
            own_feat_idx = own_flat[
                torch.repeat_interleave(own_starts, own_lens) + pos]

            # Opponent features (offset by FEAT_PER_PSP)
            opp_lens = opp_off[idx + 1] - opp_off[idx]
            total_opp = opp_lens.sum().item()
            opp_batch = torch.repeat_interleave(
                torch.arange(bs, device=device, dtype=LONG), opp_lens)
            opp_starts = opp_off[idx]
            cs_opp = torch.cat([
                torch.zeros(1, device=device, dtype=LONG),
                torch.cumsum(opp_lens, dim=0)[:-1]
            ])
            pos_opp = (torch.arange(total_opp, device=device, dtype=LONG)
                       - torch.repeat_interleave(cs_opp, opp_lens))
            opp_feat_idx = opp_flat[
                torch.repeat_interleave(opp_starts, opp_lens) + pos_opp
            ] + FEAT_PER_PSP

            # ---- Forward ----
            preds = torch.zeros(bs, device=device)
            preds.index_add_(0, own_batch, w[own_feat_idx])
            preds.index_add_(0, opp_batch, w[opp_feat_idx])
            preds += b

            targets = scores_gpu[idx]
            loss = loss_fn(preds, targets)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w, b], 10.0)
            opt.step()

            total_loss += loss.item() * bs

        avg_loss = total_loss / n_train

        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                bs2 = n_val
                own_lens_v = own_off[val_idx + 1] - own_off[val_idx]
                total_ov = own_lens_v.sum().item()
                ob_v = torch.repeat_interleave(
                    torch.arange(bs2, device=device, dtype=LONG), own_lens_v)
                own_starts_v = own_off[val_idx]
                cs_v = torch.cat([
                    torch.zeros(1, device=device, dtype=LONG),
                    torch.cumsum(own_lens_v, dim=0)[:-1]
                ])
                pis_v = (torch.arange(total_ov, device=device, dtype=LONG)
                         - torch.repeat_interleave(cs_v, own_lens_v))
                ofi_v = own_flat[
                    torch.repeat_interleave(own_starts_v, own_lens_v) + pis_v]

                opp_lens_v = opp_off[val_idx + 1] - opp_off[val_idx]
                total_pv = opp_lens_v.sum().item()
                pb_v = torch.repeat_interleave(
                    torch.arange(bs2, device=device, dtype=LONG), opp_lens_v)
                opp_starts_v = opp_off[val_idx]
                cs_opp_v = torch.cat([
                    torch.zeros(1, device=device, dtype=LONG),
                    torch.cumsum(opp_lens_v, dim=0)[:-1]
                ])
                pis_opp_v = (torch.arange(total_pv, device=device, dtype=LONG)
                             - torch.repeat_interleave(cs_opp_v, opp_lens_v))
                pfi_v = opp_flat[
                    torch.repeat_interleave(opp_starts_v, opp_lens_v) + pis_opp_v
                ] + FEAT_PER_PSP

                p = torch.zeros(bs2, device=device)
                p.index_add_(0, ob_v, w[ofi_v])
                p.index_add_(0, pb_v, w[pfi_v])
                p += b
                t = scores_gpu[val_idx]
                corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
                val_loss = loss_fn(p, t).item()
                pred_std = p.std().item()

            print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.2f}  "
                  f"val_loss={val_loss:.2f}  corr={corr:.4f}  "
                  f"pred_std={pred_std:.1f}  ‖w‖={w.norm():.1f}  b={b.item():.2f}",
                  flush=True)

            if corr > best_corr:
                best_corr = corr
                print(f"  [new best corr: {corr:.4f}]", flush=True)

    # ---- Export NNUE weights ----
    w_cpu = w.detach().cpu()
    w_min = w_cpu.min().item()
    shift = abs(w_min) if w_min < 0 else 0
    w_pos = w_cpu + shift

    l1_w_np = (w_pos / HL1 * L1_SCALE).clamp(-32768, 32767).round()
    l1_w = torch.zeros(FEAT_TOTAL, HL1, dtype=torch.float32)
    for h in range(HL1):
        l1_w[:, h] = l1_w_np
    l1_b = torch.zeros(HL1)

    l2_w = torch.zeros(HL1 * 2, HL2, dtype=torch.float32)
    for j in range(HL2):
        l2_w[j, j] = 1.0
        l2_w[HL1 + j, j] = 1.0
    l2_b = torch.zeros(HL2)

    out_w = torch.ones(HL2)
    # Correct shift compensation: L2 output ≈ n_pieces * shift / HL1
    # Score contribution from shift: HL2 * n_pieces * shift / HL1
    shift_contrib = HL2 * n_pieces_avg * shift / HL1
    out_b = torch.tensor(b.item() - shift_contrib)

    l1_q = (l1_w * L1_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    l1b_q = (l1_b * L1_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    l2_q = (l2_w * L2_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    l2b_q = (l2_b * L2_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    out_q = (out_w * L2_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    outb_q = int((out_b * L2_SCALE).round().item())

    with open(output_path, 'wb') as f:
        f.write(b'NNUE')
        f.write(struct.pack('<I', FEAT_TOTAL))
        f.write(struct.pack('<I', HL1))
        f.write(struct.pack('<I', HL2))
        for arr in [l1_q, l1b_q, l2_q, l2b_q, out_q]:
            f.write(struct.pack('<I', arr.size))
            f.write(arr.tobytes())
        f.write(struct.pack('<i', outb_q))

    print(f"\nExported to {output_path}")
    print(f"Best corr={best_corr:.4f}  shift={shift:.2f}  shift_contrib={shift_contrib:.2f}  out_bias={out_b.item():.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--output', default='/tmp/nnue_trained.bin')
    args = p.parse_args()
    train_and_export(args.features, args.output, args.epochs, args.lr, args.batch_size)
