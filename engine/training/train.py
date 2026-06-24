"""
NNUE training for Chinese Chess engine.

Architecture: HalfKP (king-relative piece-square features)
  - Input: 2 × 9 × 14 × 90 = 22680 binary features
  - L1: 22680 → 256 (CReLU)
  - L2: 512 → 32 (CReLU) [concatenated own+opp perspectives]
  - Output: 32 → 1

Data format: one sample per line in text format:
  <fen> <score_in_centipawns>

Or binary format (.bin):
  [u8; 90] board bytes + i32 score (little-endian), repeated

Usage:
  python train.py --data positions.txt --epochs 100 --output nnue_weights.bin
"""

import argparse
import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import math

# ---------------------------------------------------------------------------
# Constants (must match Rust nnue.rs)
# ---------------------------------------------------------------------------
KING_SQ = 9       # palace positions per side
PC_TYPE = 14      # 7 piece types × 2 colors (friendly/enemy)
BOARD_SQ = 90     # total board squares
FEAT_PER_PSP = KING_SQ * PC_TYPE * BOARD_SQ  # 11340
FEAT_TOTAL = FEAT_PER_PSP * 2                 # 22680

HL1 = 256
HL2 = 32

QA = 255
QA2 = 255
L1_SCALE = 64
L2_SCALE = 64

# Piece encoding (matches Rust types.rs)
EMPTY, KING, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, PAWN = range(8)
RED, BLACK = 1, 2

# Piece type to index (0-6)
PT_MAP = {KING: 0, ADVISOR: 1, ELEPHANT: 2, HORSE: 3, CHARIOT: 4, CANNON: 5, PAWN: 6}

# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def parse_fen(fen: str) -> np.ndarray:
    """Parse FEN into board array [10][9] of (piece_type, side)."""
    board = np.zeros((10, 9, 2), dtype=np.int32)
    # Strip turn indicator (' w' or ' b') if present
    fen = fen.strip()
    if fen.endswith(' w') or fen.endswith(' b'):
        fen = fen[:-2]
    rows = fen.split('/')
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                side = RED if ch.isupper() else BLACK
                pt = {'K': KING, 'A': ADVISOR, 'B': ELEPHANT, 'E': ELEPHANT,
                      'H': HORSE, 'N': HORSE, 'R': CHARIOT, 'C': CANNON, 'P': PAWN,
                      'k': KING, 'a': ADVISOR, 'b': ELEPHANT, 'e': ELEPHANT,
                      'h': HORSE, 'n': HORSE, 'r': CHARIOT, 'c': CANNON, 'p': PAWN}[ch]
                board[r, c] = [pt, side]
                c += 1
    return board


def mirror_sq(r: int, c: int) -> int:
    """Mirror row for the side-to-move (row flip)."""
    return (9 - r) * 9 + c


def orient_sq(r: int, c: int, stm: int) -> int:
    """Orient square for side-to-move. stm: RED=1, BLACK=2."""
    if stm == RED:
        return (9 - r) * 9 + c
    else:
        return r * 9 + c


def king_idx(r: int, c: int, stm: int) -> int:
    """King square index 0..8 from raw palace position.

    After mirroring, own king lands in rows 7-9, opponent king in rows 0-2.
    Both must map to index range 0..8.
    """
    or_r = 9 - r if stm == RED else r
    if or_r <= 2:
        return or_r * 3 + max(0, c - 3)
    else:
        return (or_r - 7) * 3 + max(0, c - 3)


def piece_color_idx(pt: int, side: int, stm: int) -> int:
    """Piece index 0..13, color-relative to stm."""
    base = 0 if side == stm else 7
    return base + PT_MAP[pt]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(board: np.ndarray, stm: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract HalfKP feature indices for both perspectives.
    Returns (own_features, opp_features) as index tensors.
    """
    # Find kings
    red_king = black_king = None
    for r in range(10):
        for c in range(9):
            pt, side = board[r, c]
            if pt == KING:
                if side == RED:
                    red_king = (r, c)
                else:
                    black_king = (r, c)

    if red_king is None or black_king is None:
        return torch.tensor([]), torch.tensor([])

    rk_r, rk_c = red_king
    bk_r, bk_c = black_king

    rk_ki = king_idx(rk_r, rk_c, stm)  # own king index (if stm==RED)
    bk_ki = king_idx(bk_r, bk_c, stm)  # opponent king index

    own_feats = []
    opp_feats = []

    for r in range(10):
        for c in range(9):
            pt, side = board[r, c]
            if pt == 0:
                continue
            pidx = piece_color_idx(pt, side, stm)
            osq = orient_sq(r, c, stm)

            own_feats.append(rk_ki * PC_TYPE * BOARD_SQ + pidx * BOARD_SQ + osq)
            opp_feats.append(bk_ki * PC_TYPE * BOARD_SQ + pidx * BOARD_SQ + osq)

    return torch.tensor(own_feats, dtype=torch.long), torch.tensor(opp_feats, dtype=torch.long)


# ---------------------------------------------------------------------------
# NNUE Model (PyTorch)
# ---------------------------------------------------------------------------

class NnueModel(nn.Module):
    def __init__(self):
        super().__init__()
        # L1: sparse binary features → 256
        self.l1 = nn.Embedding(FEAT_TOTAL, HL1, padding_idx=None)
        # Actually we use two separate perspectives, each with FEAT_PER_PSP features
        # But we can share the embedding: feat_idx < FEAT_PER_PSP = own perspective,
        # feat_idx >= FEAT_PER_PSP = opponent perspective (offset by FEAT_PER_PSP)
        self.l1_weight = nn.Parameter(torch.zeros(FEAT_TOTAL, HL1))
        self.l1_bias = nn.Parameter(torch.zeros(HL1))

        # L2: 2*HL1 → HL2
        self.l2_weight = nn.Parameter(torch.zeros(HL1 * 2, HL2))
        self.l2_bias = nn.Parameter(torch.zeros(HL2))

        # Output: HL2 → 1
        self.out_weight = nn.Parameter(torch.zeros(HL2))
        self.out_bias = nn.Parameter(torch.zeros(1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.l1_weight, std=0.5)
        nn.init.zeros_(self.l1_bias)
        # Positive mean on both L2 and output so gradient doesn't cancel
        nn.init.normal_(self.l2_weight, mean=0.3, std=0.5)
        nn.init.zeros_(self.l2_bias)
        nn.init.normal_(self.out_weight, mean=0.3, std=0.1)
        nn.init.zeros_(self.out_bias)

    def forward(self, own_list, opp_list):
        """Vectorized forward pass using index_add for batched feature accumulation.

        own_list, opp_list: lists of 1D long tensors (variable-length features per sample).
        """
        batch = len(own_list)
        device = self.l1_weight.device

        # Flatten all features and build batch-index tensors
        own_flat = torch.cat(own_list)
        opp_flat = torch.cat(opp_list)

        own_batch = torch.empty(own_flat.size(0), dtype=torch.long, device=device)
        opp_batch = torch.empty(opp_flat.size(0), dtype=torch.long, device=device)
        pos = 0
        for i, f in enumerate(own_list):
            n = f.size(0)
            own_batch[pos:pos + n] = i
            pos += n
        pos = 0
        for i, f in enumerate(opp_list):
            n = f.size(0)
            opp_batch[pos:pos + n] = i
            pos += n

        # L1: sum feature weights via index_add (fully vectorized on GPU)
        own_sum = torch.zeros(batch, HL1, device=device)
        own_sum.index_add_(0, own_batch, self.l1_weight[own_flat])
        own_sum = own_sum + self.l1_bias

        opp_sum = torch.zeros(batch, HL1, device=device)
        opp_sum.index_add_(0, opp_batch, self.l1_weight[opp_flat + FEAT_PER_PSP])
        opp_sum = opp_sum + self.l1_bias

        # CReLU
        own_out = torch.clamp(own_sum, 0, QA)
        opp_out = torch.clamp(opp_sum, 0, QA)

        # Concatenate → L2 → Output
        l2_in = torch.cat([own_out, opp_out], dim=1)
        l2_out = torch.clamp(l2_in @ self.l2_weight + self.l2_bias, 0, QA2)
        return (l2_out @ self.out_weight + self.out_bias).squeeze(-1)

    def forward_single(self, own_feats: torch.Tensor, opp_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single position (non-batched)."""
        return self.forward([own_feats], [opp_feats])[0]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PositionDataset(Dataset):
    """Dataset of (fen, score) pairs. Parses FEN on-the-fly (slow)."""

    def __init__(self, data: List[Tuple[str, float]], stm: int = RED):
        self.data = data
        self.stm = stm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, score = self.data[idx]
        board = parse_fen(fen)
        own_f, opp_f = extract_features(board, self.stm)
        return own_f, opp_f, torch.tensor(score, dtype=torch.float32)


class PrecomputedDataset(Dataset):
    """Dataset of pre-computed feature indices (fast)."""

    def __init__(self, path: str):
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.own_list = data['own_list']
        self.opp_list = data['opp_list']
        self.scores = data['scores']

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.own_list[idx], self.opp_list[idx], self.scores[idx]


def collate_fn(batch):
    """Custom collate: features are variable-length index lists."""
    own_list, opp_list, scores = zip(*batch)
    return list(own_list), list(opp_list), torch.stack(scores)


def load_text_data(path: str) -> List[Tuple[str, float]]:
    """Load data from text file: <fen> <score> per line."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                fen = ' '.join(parts[:-1])
                try:
                    score = float(parts[-1])
                except ValueError:
                    continue
                data.append((fen, score))
    return data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, dataloader, optimizer, device, epochs):
    model.train()
    loss_fn = nn.MSELoss()

    # Progressive unfreezing: L1 frozen epochs 0..warmup, L2 frozen 0..warmup/2
    warmup = max(1, epochs // 10)
    model.l1_weight.requires_grad_(False)
    model.l2_weight.requires_grad_(False)

    for epoch in range(epochs):
        if epoch == warmup // 2:
            model.l2_weight.requires_grad_(True)
            model.l2_bias.requires_grad_(True)
            print("  [unfroze L2]", flush=True)
        if epoch == warmup:
            model.l1_weight.requires_grad_(True)
            model.l1_bias.requires_grad_(True)
            print("  [unfroze L1]", flush=True)

        total_loss = 0.0
        n = 0
        for own_list, opp_list, scores in dataloader:
            scores = scores.to(device, non_blocking=True)
            own_gpu = [f.to(device, non_blocking=True) for f in own_list]
            opp_gpu = [f.to(device, non_blocking=True) for f in opp_list]

            preds = model.forward(own_gpu, opp_gpu)
            loss = loss_fn(preds, scores)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(scores)
            n += len(scores)

        avg_loss = total_loss / max(n, 1)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  pred_mean={preds.mean().item():.1f}  pred_std={preds.std().item():.1f}", flush=True)


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------

def quantize_and_export(model: NnueModel, path: str):
    """Quantize weights to i16 and export in Rust NNUE binary format."""
    w = {}

    # L1 weights: [FEAT_TOTAL, HL1] → quantized i16
    l1_q = torch.round(model.l1_weight.data.cpu() * L1_SCALE).clamp(-32768, 32767).to(torch.int16)
    w['l1_weights'] = l1_q.numpy()

    # L1 bias
    l1b_q = torch.round(model.l1_bias.data.cpu() * L1_SCALE).clamp(-32768, 32767).to(torch.int16)
    w['l1_bias'] = l1b_q.numpy()

    # L2 weights: [HL1*2, HL2]
    l2_q = torch.round(model.l2_weight.data.cpu() * L2_SCALE).clamp(-32768, 32767).to(torch.int16)
    w['l2_weights'] = l2_q.numpy()

    # L2 bias
    l2b_q = torch.round(model.l2_bias.data.cpu() * L2_SCALE).clamp(-32768, 32767).to(torch.int16)
    w['l2_bias'] = l2b_q.numpy()

    # Output weights
    out_q = torch.round(model.out_weight.data.cpu() * L2_SCALE).clamp(-32768, 32767).to(torch.int16)
    w['out_weights'] = out_q.numpy()

    outb_q = int(torch.round(model.out_bias.data * L2_SCALE).clamp(-32768, 32767).item())
    w['out_bias'] = outb_q

    with open(path, 'wb') as f:
        # Header: magic + dimensions
        f.write(b'NNUE')
        f.write(struct.pack('<I', FEAT_TOTAL))
        f.write(struct.pack('<I', HL1))
        f.write(struct.pack('<I', HL2))

        # Write each array
        for name in ['l1_weights', 'l1_bias', 'l2_weights', 'l2_bias', 'out_weights']:
            arr = w[name]
            f.write(struct.pack('<I', arr.size))
            f.write(arr.tobytes())

        f.write(struct.pack('<i', w['out_bias']))

    print(f"Exported weights to {path}")
    print(f"  L1 weights: {w['l1_weights'].shape} {w['l1_weights'].dtype}")
    print(f"  L2 weights: {w['l2_weights'].shape} {w['l2_weights'].dtype}")
    print(f"  Output weights: {w['out_weights'].shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train NNUE for Chinese Chess')
    parser.add_argument('--data', type=str, required=True, help='Training data (text, .bin, or precomputed .pt)')
    parser.add_argument('--features', type=str, default=None, help='Precomputed features .pt file (bypasses FEN parsing)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='nnue_weights.bin', help='Output weights file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    features_path = args.features or args.data
    if features_path.endswith('.pt'):
        print(f"Loading precomputed features from {features_path}...")
        dataset = PrecomputedDataset(features_path)
        print(f"  Loaded {len(dataset)} samples")
    else:
        print(f"Loading data from {args.data}...")
        raw_data = load_text_data(args.data)
        print(f"  Loaded {len(raw_data)} positions")
        if len(raw_data) == 0:
            print("ERROR: No training data found!")
            return
        # Create datasets for both perspectives
        all_data = [(fen, score) for fen, score in raw_data] + [(fen, -score) for fen, score in raw_data]
        dataset = PositionDataset(all_data)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, pin_memory=True, num_workers=0)

    # Create model
    model = NnueModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print(f"Training for {args.epochs} epochs...")
    train(model, dataloader, optimizer, device, args.epochs)

    # Export
    quantize_and_export(model, args.output)


if __name__ == '__main__':
    main()
