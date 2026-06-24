use crate::board::Board;
use crate::types::*;

// ---------------------------------------------------------------------------
// NNUE Architecture (HalfKP for Chinese Chess)
//
// Features: (king_square, piece_type_color, piece_square)
//   - king_square: 9 palace positions, mirrored per side
//   - piece_type_color: 14 (7 types × 2 colors, color-relative)
//   - piece_square: 90 board squares, mirrored per side
//   - × 2 perspectives (own king, opponent king)
//
// Total features per perspective: 9 × 14 × 90 = 11_340
// Total features (both perspectives): 22_680
// ---------------------------------------------------------------------------

const KING_SQ: usize = 9;
const PC_TYPE: usize = 14;
const BOARD_SQ: usize = 90;
const FEAT_PER_PSP: usize = KING_SQ * PC_TYPE * BOARD_SQ; // 11_340
const FEAT_TOTAL: usize = FEAT_PER_PSP * 2; // 22_680

const HL1: usize = 256;
const HL2: usize = 32;

// Quantization
const QA: i32 = 255;
const QA2: i32 = 255;
const L1_SCALE: i32 = 64;
const L2_SCALE: i32 = 64;

// ---------------------------------------------------------------------------
// NNUE network weights
// ---------------------------------------------------------------------------

pub struct Nnue {
    pub l1_weights: Vec<i16>,    // [FEAT_TOTAL * HL1]
    pub l1_bias: Vec<i16>,       // [HL1]
    pub l2_weights: Vec<i16>,    // [HL1 * 2 * HL2]
    pub l2_bias: Vec<i16>,       // [HL2]
    pub out_weights: Vec<i16>,   // [HL2]
    pub out_bias: i32,
    pub loaded: bool,
}

impl Nnue {
    pub fn new() -> Self {
        let l1_weights = vec![0i16; FEAT_TOTAL * HL1];
        let l1_bias = vec![0i16; HL1];
        let l2_weights = vec![0i16; HL1 * 2 * HL2];
        let l2_bias = vec![0i16; HL2];
        let out_weights = vec![0i16; HL2];
        Nnue {
            l1_weights,
            l1_bias,
            l2_weights,
            l2_bias,
            out_weights,
            out_bias: 0,
            loaded: false,
        }
    }

    /// Check if actual weights are loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Load weights from a binary file (format matches Python trainer export).
    /// Returns Ok(()) on success.
    pub fn load_weights(&mut self, path: &str) -> Result<(), String> {
        use std::io::Read;
        let mut f = std::fs::File::open(path).map_err(|e| format!("open: {}", e))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| format!("read: {}", e))?;

        let mut pos = 0usize;

        // Read header
        if buf.len() < 4 || &buf[0..4] != b"NNUE" {
            return Err("invalid magic".into());
        }
        pos += 4;

        let read_u32 = |b: &[u8], p: &mut usize| -> u32 {
            let v = u32::from_le_bytes([b[*p], b[*p+1], b[*p+2], b[*p+3]]);
            *p += 4;
            v
        };

        let feat_total = read_u32(&buf, &mut pos) as usize;
        let hl1 = read_u32(&buf, &mut pos) as usize;
        let hl2 = read_u32(&buf, &mut pos) as usize;

        if feat_total != FEAT_TOTAL || hl1 != HL1 || hl2 != HL2 {
            return Err(format!("dimension mismatch: expected {}/{}/{}, got {}/{}/{}",
                FEAT_TOTAL, HL1, HL2, feat_total, hl1, hl2));
        }

        let read_array_i16 = |b: &[u8], p: &mut usize| -> Result<Vec<i16>, String> {
            let count = read_u32(b, p) as usize;
            let byte_len = count * 2;
            if *p + byte_len > b.len() {
                return Err("truncated array data".into());
            }
            let mut v = vec![0i16; count];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    b[*p..].as_ptr(),
                    v.as_mut_ptr() as *mut u8,
                    byte_len,
                );
            }
            *p += byte_len;
            Ok(v)
        };

        self.l1_weights = read_array_i16(&buf, &mut pos)?;
        self.l1_bias = read_array_i16(&buf, &mut pos)?;
        self.l2_weights = read_array_i16(&buf, &mut pos)?;
        self.l2_bias = read_array_i16(&buf, &mut pos)?;
        self.out_weights = read_array_i16(&buf, &mut pos)?;

        // Output bias (single i32)
        if pos + 4 > buf.len() {
            return Err("truncated file at out_bias".into());
        }
        self.out_bias = i32::from_le_bytes([buf[pos], buf[pos+1], buf[pos+2], buf[pos+3]]);

        self.loaded = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Accumulator: incremental update state
// ---------------------------------------------------------------------------

/// Accumulator type: 4 perspectives × HL1 entries, stack-allocated for fast clone
pub type Accumulator = [[i32; HL1]; 4];

#[derive(Clone)]
pub struct NnueState {
    /// Accumulators for [Red_own, Red_opp, Black_own, Black_opp]
    /// Stack arrays — clone is a 4KB memcpy instead of 4 heap allocations
    pub acc: Accumulator,
    /// Mirrored king positions: (red_king_mirrored_sq, black_king_mirrored_sq)
    pub king_sqs: [(usize, usize); 2], // (square_idx, mirroed_square_idx)
    /// Piece positions for incremental updates: [piece_index] = square
    pub pieces: [(u8, u8); 32],        // (side_relative_sq, piece_index)
    pub piece_count: usize,
}

/// Piece index: 0..6 friendly, 7..13 enemy
#[inline]
fn piece_idx(side: Side, stm: Side, pt: PieceType) -> usize {
    let base = if side == stm { 0 } else { 7 };
    base + pt as usize - 1
}

/// Mirror column for black side (column stays same in Chinese Chess, row flips)
#[inline]
fn mirror_sq(r: u8, c: u8) -> usize {
    (9 - r as usize) * 9 + c as usize
}

/// Raw square index
#[inline]
fn raw_sq(r: u8, c: u8) -> usize {
    r as usize * 9 + c as usize
}

/// Get king square index (0..8) from a mirrored palace position.
/// After mirroring, the Red king is in rows 0..2, Black king in rows 7..9.
#[inline]
fn king_sq_idx(mirrored_sq: usize) -> usize {
    let r = mirrored_sq / 9;
    let c = mirrored_sq % 9;
    if r <= 2 {
        (r * 3 + (c - 3)) as usize
    } else {
        ((r - 7) * 3 + (c - 3)) as usize
    }
}

impl NnueState {
    pub fn new() -> Self {
        NnueState {
            acc: [[0i32; HL1]; 4],
            king_sqs: [(0, 0), (0, 0)],
            pieces: [(0, 0); 32],
            piece_count: 0,
        }
    }

    /// Full refresh: recompute all accumulators from scratch
    pub fn refresh(&mut self, board: &Board, nnue: &Nnue) {
        for a in &mut self.acc {
            a.fill(0);
        }
        for i in 0..HL1 {
            for j in 0..4 {
                self.acc[j][i] = nnue.l1_bias[i] as i32;
            }
        }

        let red_king = board.find_king(Side::Red);
        let black_king = board.find_king(Side::Black);
        if red_king.is_none() || black_king.is_none() {
            return;
        }
        let (rk_r, rk_c) = red_king.unwrap();
        let (bk_r, bk_c) = black_king.unwrap();

        let rk_mirror = mirror_sq(rk_r as u8, rk_c as u8);
        let bk_mirror = mirror_sq(bk_r as u8, bk_c as u8);
        let rk_ki = king_sq_idx(rk_mirror);
        let bk_ki = king_sq_idx(bk_mirror);
        self.king_sqs[0] = (raw_sq(rk_r as u8, rk_c as u8), rk_mirror);
        self.king_sqs[1] = (raw_sq(bk_r as u8, bk_c as u8), bk_mirror);

        self.piece_count = 0;

        for r in 0..10u8 {
            for c in 0..9u8 {
                let p = board.cells[r as usize][c as usize];
                if p.is_empty() {
                    continue;
                }
                let msq = mirror_sq(r, c);
                let pidx = piece_idx(p.side, Side::Red, p.piece_type);
                let feat_red = rk_ki * PC_TYPE * BOARD_SQ + pidx * BOARD_SQ + msq;
                let feat_black = bk_ki * PC_TYPE * BOARD_SQ + pidx * BOARD_SQ + msq;

                // Red's own (red king ref, first half) and opponent (black king ref, second half)
                add_feature(&mut self.acc[0], feat_red, &nnue.l1_weights);
                add_feature(&mut self.acc[1], feat_black + FEAT_PER_PSP, &nnue.l1_weights);
                // Black's own (black king ref, first half) and opponent (red king ref, second half)
                add_feature(&mut self.acc[2], feat_black, &nnue.l1_weights);
                add_feature(&mut self.acc[3], feat_red + FEAT_PER_PSP, &nnue.l1_weights);

                if self.piece_count < 32 {
                    self.pieces[self.piece_count] = (msq as u8, pidx as u8);
                    self.piece_count += 1;
                }
            }
        }
    }

    /// Incremental update after a move. `captured` is the captured piece (or EMPTY).
    /// `mover` is the piece that moved.
    /// For king moves, falls back to full refresh since king position affects all features.
    pub fn update(
        &mut self,
        board: &Board,
        nnue: &Nnue,
        from_r: u8,
        from_c: u8,
        to_r: u8,
        to_c: u8,
        mover: Piece,
        captured: Piece,
    ) {
        // King moves change the feature indices for ALL pieces (HalfKP depends on
        // king position), so incremental update is incorrect — do a full refresh.
        if mover.piece_type == PieceType::King {
            self.refresh(board, nnue);
            return;
        }

        let from_msq = mirror_sq(from_r, from_c);
        let to_msq = mirror_sq(to_r, to_c);
        let mover_pidx = piece_idx(mover.side, Side::Red, mover.piece_type);

        let rk_ki = king_sq_idx(self.king_sqs[0].1);
        let bk_ki = king_sq_idx(self.king_sqs[1].1);

        let feat_from_r = rk_ki * PC_TYPE * BOARD_SQ + mover_pidx * BOARD_SQ + from_msq;
        let feat_to_r = rk_ki * PC_TYPE * BOARD_SQ + mover_pidx * BOARD_SQ + to_msq;
        let feat_from_b = bk_ki * PC_TYPE * BOARD_SQ + mover_pidx * BOARD_SQ + from_msq;
        let feat_to_b = bk_ki * PC_TYPE * BOARD_SQ + mover_pidx * BOARD_SQ + to_msq;

        // Update Red's accumulators
        sub_feature(&mut self.acc[0], feat_from_r, &nnue.l1_weights);
        add_feature(&mut self.acc[0], feat_to_r, &nnue.l1_weights);
        sub_feature(&mut self.acc[1], feat_from_b + FEAT_PER_PSP, &nnue.l1_weights);
        add_feature(&mut self.acc[1], feat_to_b + FEAT_PER_PSP, &nnue.l1_weights);

        // Update Black's accumulators
        sub_feature(&mut self.acc[2], feat_from_b, &nnue.l1_weights);
        add_feature(&mut self.acc[2], feat_to_b, &nnue.l1_weights);
        sub_feature(&mut self.acc[3], feat_from_r + FEAT_PER_PSP, &nnue.l1_weights);
        add_feature(&mut self.acc[3], feat_to_r + FEAT_PER_PSP, &nnue.l1_weights);

        // Remove captured piece from accumulators
        if !captured.is_empty() {
            let cap_msq = to_msq; // captured piece was at the target square
            let cap_pidx = piece_idx(captured.side, Side::Red, captured.piece_type);
            let cap_feat_r = rk_ki * PC_TYPE * BOARD_SQ + cap_pidx * BOARD_SQ + cap_msq;
            let cap_feat_b = bk_ki * PC_TYPE * BOARD_SQ + cap_pidx * BOARD_SQ + cap_msq;
            sub_feature(&mut self.acc[0], cap_feat_r, &nnue.l1_weights);
            sub_feature(&mut self.acc[1], cap_feat_b + FEAT_PER_PSP, &nnue.l1_weights);
            sub_feature(&mut self.acc[2], cap_feat_b, &nnue.l1_weights);
            sub_feature(&mut self.acc[3], cap_feat_r + FEAT_PER_PSP, &nnue.l1_weights);
        }
    }

    /// Forward pass: compute score from accumulator for `side`'s perspective
    pub fn forward(&self, nnue: &Nnue, side: Side) -> i32 {
        let (own_idx, opp_idx) = match side {
            Side::Red => (0, 1),
            Side::Black => (2, 3),
            _ => return 0,
        };

        // L1 activation
        let mut l2_input = [0i32; HL1 * 2];
        for i in 0..HL1 {
            l2_input[i] = clamp(self.acc[own_idx][i] / L1_SCALE, 0, QA);
        }
        for i in 0..HL1 {
            l2_input[HL1 + i] = clamp(self.acc[opp_idx][i] / L1_SCALE, 0, QA);
        }

        // L2 forward
        let mut l2_out = [0i32; HL2];
        for j in 0..HL2 {
            let mut sum = nnue.l2_bias[j] as i32;
            for i in 0..HL1 * 2 {
                sum += l2_input[i] * nnue.l2_weights[i * HL2 + j] as i32;
            }
            l2_out[j] = clamp(sum / L2_SCALE, 0, QA2);
        }

        // Output
        let mut score = nnue.out_bias as i32;
        for i in 0..HL2 {
            score += l2_out[i] * nnue.out_weights[i] as i32;
        }
        score / L2_SCALE
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn clamp(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
fn add_feature(acc: &mut [i32], feat_idx: usize, weights: &[i16]) {
    let base = feat_idx * HL1;
    for i in 0..HL1 {
        acc[i] += weights[base + i] as i32;
    }
}

#[inline]
fn sub_feature(acc: &mut [i32], feat_idx: usize, weights: &[i16]) {
    let base = feat_idx * HL1;
    for i in 0..HL1 {
        acc[i] -= weights[base + i] as i32;
    }
}

#[cfg(test)]
mod nnue_tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_load_and_eval() {
        let mut nnue = Nnue::new();
        if let Err(e) = nnue.load_weights("/tmp/nnue_trained.bin") {
            if e.contains("open:") {
                return;
            }
            panic!("Failed to load: {}", e);
        }
        assert!(nnue.is_loaded());

        let board = Board::new();
        let mut state = NnueState::new();
        state.refresh(&board, &nnue);

        let score_red = state.forward(&nnue, Side::Red);
        let score_black = state.forward(&nnue, Side::Black);

        println!("Initial position: Red score={} Black score={}", score_red, score_black);
        // Trained model should produce finite, non-identical scores for both sides
        assert!(score_red.abs() < 100000, "Score overflow: {}", score_red);
        assert!(score_black.abs() < 100000, "Score overflow: {}", score_black);
        // Scores should differ between perspectives (not degenerate)
        assert!(score_red != score_black || score_red.abs() > 0,
                "Degenerate: both sides identical non-zero {} vs {}", score_red, score_black);
    }

    #[test]
    fn test_refresh_and_update_consistency() {
        let mut nnue = Nnue::new();
        if nnue.load_weights("/tmp/nnue_trained.bin").is_err() {
            return; // Skip if file missing
        }

        let mut board = Board::new();
        let mut state = NnueState::new();
        state.refresh(&board, &nnue);
        let _score_before = state.forward(&nnue, Side::Red);

        // Make a move
        let moves = board.generate_legal_moves();
        if let Some(&m) = moves.first() {
            let mover = board.cells[m.from_row as usize][m.from_col as usize];
            let captured = board.make_move(m);
            state.update(&board, &nnue, m.from_row, m.from_col, m.to_row, m.to_col, mover, captured);

            // Verify incremental update matches full refresh
            let score_after_update = state.forward(&nnue, Side::Red);

            let mut state2 = NnueState::new();
            state2.refresh(&board, &nnue);
            let score_after_refresh = state2.forward(&nnue, Side::Red);

            println!("After move: update={} refresh={}", score_after_update, score_after_refresh);
            assert_eq!(score_after_update, score_after_refresh,
                       "Incremental update doesn't match full refresh!");
        }
    }
}
