"""Train a linear model on HalfKP features using vectorized index_add operations."""
import argparse, struct
import torch

FEAT_TOTAL = 22680
FEAT_PER_PSP = 11340
HL1 = 256
HL2 = 32
L1_SCALE = 64
L2_SCALE = 64


def train_and_export(features_path, output_path, epochs, lr, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    data = torch.load(features_path, map_location='cpu', weights_only=True)
    own_list = data['own_list']
    opp_list = data['opp_list']
    scores = data['scores']
    n = len(scores)
    print(f"Samples: {n}, score range: [{scores.min():.1f}, {scores.max():.1f}]", flush=True)

    # Pack features into flat arrays for vectorized access
    print("Packing features for GPU...", flush=True)
    own_offsets = [0]
    opp_offsets = [0]
    own_flat_list = []
    opp_flat_list = []
    for i in range(n):
        own_flat_list.append(own_list[i])
        opp_flat_list.append(opp_list[i])
        own_offsets.append(own_offsets[-1] + len(own_list[i]))
        opp_offsets.append(opp_offsets[-1] + len(opp_list[i]))

    own_flat = torch.cat(own_flat_list).to(device)
    opp_flat = torch.cat(opp_flat_list).to(device)
    own_off = torch.tensor(own_offsets, dtype=torch.long, device=device)
    opp_off = torch.tensor(opp_offsets, dtype=torch.long, device=device)
    scores_gpu = scores.to(device)
    print(f"  own features: {len(own_flat)}, opp features: {len(opp_flat)}", flush=True)

    # Linear model
    w = torch.zeros(FEAT_TOTAL, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Pre-build batch index tensors for each possible batch
    # For speed, build on-the-fly

    for epoch in range(epochs):
        perm = torch.randperm(n, device='cpu')
        total_loss = 0.0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bs = end - start
            idx = perm[start:end].tolist()

            # Build batch index for own features
            own_lens = [own_off[i+1] - own_off[i] for i in idx]
            total_own = sum(own_lens)
            own_batch = torch.empty(total_own, dtype=torch.long, device=device)
            own_feat_idx = torch.empty(total_own, dtype=torch.long, device=device)
            pos = 0
            for bi, i in enumerate(idx):
                o0, o1 = own_off[i].item(), own_off[i+1].item()
                length = o1 - o0
                own_batch[pos:pos+length] = bi
                own_feat_idx[pos:pos+length] = own_flat[o0:o1]
                pos += length

            # Same for opponent features (offset by FEAT_PER_PSP)
            opp_lens = [opp_off[i+1] - opp_off[i] for i in idx]
            total_opp = sum(opp_lens)
            opp_batch = torch.empty(total_opp, dtype=torch.long, device=device)
            opp_feat_idx = torch.empty(total_opp, dtype=torch.long, device=device)
            pos = 0
            for bi, i in enumerate(idx):
                o0, o1 = opp_off[i].item(), opp_off[i+1].item()
                length = o1 - o0
                opp_batch[pos:pos+length] = bi
                opp_feat_idx[pos:pos+length] = opp_flat[o0:o1] + FEAT_PER_PSP
                pos += length

            # Vectorized forward: sum feature weights per sample
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                si = torch.randperm(n)[:500]
                bs2 = 500
                # Quick validation using same vec approach but small batch
                idx_v = si.tolist()
                own_lens_v = [own_off[i+1] - own_off[i] for i in idx_v]
                total_o = sum(own_lens_v)
                ob = torch.empty(total_o, dtype=torch.long, device=device)
                ofi = torch.empty(total_o, dtype=torch.long, device=device)
                pos = 0
                for bi, i in enumerate(idx_v):
                    o0, o1 = own_off[i].item(), own_off[i+1].item()
                    ob[pos:pos+o1-o0] = bi
                    ofi[pos:pos+o1-o0] = own_flat[o0:o1]
                    pos += o1-o0
                opp_lens_v = [opp_off[i+1] - opp_off[i] for i in idx_v]
                total_p = sum(opp_lens_v)
                pb = torch.empty(total_p, dtype=torch.long, device=device)
                pfi = torch.empty(total_p, dtype=torch.long, device=device)
                pos = 0
                for bi, i in enumerate(idx_v):
                    o0, o1 = opp_off[i].item(), opp_off[i+1].item()
                    pb[pos:pos+o1-o0] = bi
                    pfi[pos:pos+o1-o0] = opp_flat[o0:o1] + FEAT_PER_PSP
                    pos += o1-o0
                p = torch.zeros(bs2, device=device)
                p.index_add_(0, ob, w[ofi])
                p.index_add_(0, pb, w[pfi])
                p += b
                t = scores_gpu[idx_v]
                corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
            print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/n:.2f}  corr={corr:.4f}  "
                  f"‖w‖={w.norm():.1f}  b={b.item():.2f}", flush=True)

    # ---- Export ----
    w_cpu = w.detach().cpu()
    w_min = w_cpu.min().item()
    shift = abs(w_min) if w_min < 0 else 0
    w_pos = w_cpu + shift

    l1_w = torch.zeros(FEAT_TOTAL, HL1, dtype=torch.float32)
    for h in range(HL1):
        l1_w[:, h] = w_pos / HL1
    l1_b = torch.zeros(HL1)

    l2_w = torch.zeros(HL1 * 2, HL2, dtype=torch.float32)
    for j in range(HL2):
        l2_w[j, j] = 1.0
        l2_w[HL1 + j, j] = 1.0
    l2_b = torch.zeros(HL2)

    out_w = torch.ones(HL2)
    n_pieces_avg = sum(len(o) for o in own_list) / n
    total_shift = 2.0 * HL2 * n_pieces_avg * shift
    out_b = torch.tensor(b.item() - total_shift)

    l1_q = (l1_w * L1_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    l1b_q = (l1_b * L1_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    l2_q = (l2_w * L2_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    l2b_q = (l2_b * L2_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    out_q = (out_w * L2_SCALE).clamp(-32768, 32767).round().to(torch.int16).numpy()
    outb_q = int((out_b * L2_SCALE).clamp(-32768, 32767).round().item())

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
    print(f"shift={shift:.2f} avg_pieces={n_pieces_avg:.1f} total_shift={total_shift:.2f} out_bias={out_b.item():.2f}")

    # Quick validation
    print("Validation:")
    for idx in [0, 100, 500, 1000]:
        own_f = own_list[idx]
        opp_f = opp_list[idx]
        t = scores[idx].item()
        lin = (w[own_f].sum() + w[opp_f + FEAT_PER_PSP].sum() + b).item()

        o_sum = l1_w[own_f].sum(0) + l1_b
        p_sum = l1_w[opp_f + FEAT_PER_PSP].sum(0) + l1_b
        l2_in = torch.cat([o_sum.clamp(0, 255), p_sum.clamp(0, 255)])
        l2_out = (l2_in @ l2_w + l2_b).clamp(0, 255)
        nnue = (l2_out @ out_w + out_b).item()
        print(f"  {idx}: tgt={t:7.1f}  lin={lin:7.1f}  nnue≈{nnue:7.1f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--output', default='/tmp/nnue_linear.bin')
    args = p.parse_args()
    train_and_export(args.features, args.output, args.epochs, args.lr, args.batch_size)
