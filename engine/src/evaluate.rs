use crate::board::Board;
use crate::nnue::{Nnue, NnueState};
use crate::types::*;

// ---------------------------------------------------------------------------
// Tapered material values (midgame, endgame)
// ---------------------------------------------------------------------------
pub const MG_BASE: [i32; 8] = [0, 10000, 130, 120, 285, 620, 295, 30];
pub const EG_BASE: [i32; 8] = [0, 10000, 150, 130, 310, 620, 200, 90];

// For backward compat (search.rs move ordering / qsearch delta)
pub const BASE_VALUES: [i32; 8] = MG_BASE;

// Game-phase weights per piece (non-pawn, non-king)
const PHASE_WEIGHT: [i32; 8] = [0, 0, 1, 1, 2, 3, 2, 0];

const TOTAL_PHASE: i32 = 2 * (2 + 2 + 3 + 2); // 18  per side?  R=3 H=2 C=2 E=1 A=1 = 9 * 2 = 18

// ---------------------------------------------------------------------------
// PST tables — midgame (kept from original, with adjustments)
// ---------------------------------------------------------------------------
const MG_PAWN: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 4, 0, 8, 0, 4, 0, 2],
    [0, 0, 0, 6, 0, 6, 0, 0, 0],
    [0, 0, 2, 6, 10, 6, 2, 0, 0],
    [0, 0, 0, 2, 4, 2, 0, 0, 0],
];

const EG_PAWN: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 6, 12, 18, 12, 6, 4, 4],
    [4, 6, 8, 16, 24, 16, 8, 6, 4],
    [4, 6, 10, 20, 30, 20, 10, 6, 4],
    [2, 4, 8, 16, 24, 16, 8, 4, 2],
];

const MG_ADVISOR: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 6, 0, 0, 0],
    [0, 0, 0, 0, 10, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 6, 0, 0, 0],
];

const EG_ADVISOR: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 10, 0, 10, 0, 4, 0],
    [0, 0, 0, 0, 14, 0, 0, 0, 0],
    [0, 4, 0, 10, 0, 10, 0, 4, 0],
];

const MG_ELEPHANT: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 8, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 4, 0, 0],
];

const EG_ELEPHANT: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 6, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 2, 0, 0],
];

const MG_HORSE: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 8, 14, 16, 16, 16, 14, 8, 2],
    [4, 12, 18, 22, 22, 22, 18, 12, 4],
    [4, 14, 20, 26, 26, 26, 20, 14, 4],
    [4, 14, 20, 26, 26, 26, 20, 14, 4],
    [4, 12, 18, 22, 22, 22, 18, 12, 4],
    [2, 8, 14, 16, 16, 16, 14, 8, 2],
    [0, 6, 10, 14, 14, 14, 10, 6, 0],
    [0, 2, 4, 8, 8, 8, 4, 2, 0],
    [0, 0, 2, 4, 4, 4, 2, 0, 0],
];

const EG_HORSE: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 10, 16, 18, 18, 18, 16, 10, 4],
    [6, 14, 20, 24, 24, 24, 20, 14, 6],
    [6, 16, 22, 28, 28, 28, 22, 16, 6],
    [6, 16, 22, 28, 28, 28, 22, 16, 6],
    [6, 14, 20, 24, 24, 24, 20, 14, 6],
    [4, 10, 16, 18, 18, 18, 16, 10, 4],
    [2, 8, 12, 16, 16, 16, 12, 8, 2],
    [0, 4, 6, 10, 10, 10, 6, 4, 0],
    [0, 2, 4, 6, 6, 6, 4, 2, 0],
];

const MG_CHARIOT: [[i32; 9]; 10] = [
    [6, 8, 8, 12, 14, 12, 8, 8, 6],
    [6, 10, 12, 16, 18, 16, 12, 10, 6],
    [4, 8, 10, 14, 16, 14, 10, 8, 4],
    [4, 6, 8, 12, 14, 12, 8, 6, 4],
    [2, 4, 6, 10, 12, 10, 6, 4, 2],
    [0, 2, 4, 8, 10, 8, 4, 2, 0],
    [0, 0, 2, 6, 8, 6, 2, 0, 0],
    [0, 0, 0, 4, 6, 4, 0, 0, 0],
    [0, 0, 0, 2, 4, 2, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
];

const EG_CHARIOT: [[i32; 9]; 10] = [
    [6, 10, 10, 14, 16, 14, 10, 10, 6],
    [6, 12, 14, 18, 20, 18, 14, 12, 6],
    [4, 10, 12, 16, 18, 16, 12, 10, 4],
    [4, 8, 10, 14, 16, 14, 10, 8, 4],
    [2, 6, 8, 12, 14, 12, 8, 6, 2],
    [0, 4, 6, 10, 12, 10, 6, 4, 0],
    [0, 2, 4, 8, 10, 8, 4, 2, 0],
    [0, 0, 2, 6, 8, 6, 2, 0, 0],
    [0, 0, 0, 4, 6, 4, 0, 0, 0],
    [0, 0, 0, 2, 4, 2, 0, 0, 0],
];

const MG_CANNON: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 4, 4, 8, 8, 8, 4, 4, 2],
    [2, 4, 6, 8, 10, 8, 6, 4, 2],
    [0, 2, 4, 6, 8, 6, 4, 2, 0],
    [0, 0, 2, 4, 6, 4, 2, 0, 0],
    [0, 0, 0, 2, 4, 2, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, -2, 0, 4, 0, -2, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
];

const EG_CANNON: [[i32; 9]; 10] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 2, 4, 4, 4, 2, 2, 0],
    [0, 2, 4, 6, 6, 6, 4, 2, 0],
    [0, 0, 2, 4, 4, 4, 2, 0, 0],
    [0, 0, 0, 2, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
];

// ---------------------------------------------------------------------------
// Mobility weights per piece type (midgame, endgame). Applied per reachable
// pseudo-legal square.
// ---------------------------------------------------------------------------
const MOBILITY_MG: [i32; 8] = [0, 0, 2, 2, 4, 3, 2, 1];
const MOBILITY_EG: [i32; 8] = [0, 0, 4, 4, 5, 4, 3, 2];

// ---------------------------------------------------------------------------
// King safety
// ---------------------------------------------------------------------------

// Per attacker-weight near opponent palace.
const ATTACK_WEIGHT: [i32; 8] = [0, 0, 2, 2, 4, 5, 4, 1];
const KING_SAFETY_MAX: i32 = 120;

// Missing advisor / elephant penalties in midgame
const MISSING_ADVISOR_MG: i32 = 30;
const MISSING_ELEPHANT_MG: i32 = 25;

// ---------------------------------------------------------------------------
// Rook on open / half-open file
// ---------------------------------------------------------------------------
const ROOK_OPEN_FILE_MG: i32 = 18;
const ROOK_OPEN_FILE_EG: i32 = 12;
const ROOK_HALF_OPEN_FILE_MG: i32 = 10;
const ROOK_HALF_OPEN_FILE_EG: i32 = 7;

// ---------------------------------------------------------------------------
// Cannon mount
// ---------------------------------------------------------------------------
const CANNON_MOUNT_MG: i32 = 10;
const CANNON_MOUNT_EG: i32 = 4;

// ---------------------------------------------------------------------------
// Passed (crossed-river) pawn bonus, stacked on PST
// ---------------------------------------------------------------------------
const PASSED_PAWN_MG: [i32; 10] = [0, 0, 0, 0, 0, 5, 12, 20, 30, 40];
const PASSED_PAWN_EG: [i32; 10] = [0, 0, 0, 0, 0, 15, 30, 50, 70, 90];

// ---------------------------------------------------------------------------
// Elephant-eye blocking penalty (horse-leg already handled by mobility)
// ---------------------------------------------------------------------------
const ELEPHANT_EYE_PENALTY_MG: i32 = 6;
const ELEPHANT_EYE_PENALTY_EG: i32 = 8;

// ---------------------------------------------------------------------------
// Piece coordination
// ---------------------------------------------------------------------------
const ROOK_CANNON_SAME_FILE_MG: i32 = 8;
const DUAL_ROOK_SAME_FILE_MG: i32 = 12;

// ---------------------------------------------------------------------------
// Forward-declared helpers
// ---------------------------------------------------------------------------
fn flip_r(r: usize) -> usize {
    9 - r
}

fn is_red_side(side: Side) -> bool {
    matches!(side, Side::Red)
}

// ---------------------------------------------------------------------------
// Main entry points
// ---------------------------------------------------------------------------

/// Evaluate the board from Red's perspective (positive = Red advantage).
pub fn evaluate(board: &Board) -> i32 {
    let mut mg = [0i32; 2]; // midgame  [Red, Black]
    let mut eg = [0i32; 2]; // endgame  [Red, Black]
    let mut phase = 0i32;

    // ---- piece loop ----
    for r in 0..10usize {
        for c in 0..9usize {
            let p = board.cells[r][c];
            if p.is_empty() {
                continue;
            }
            let idx = side_idx(p.side);
            let pt = p.piece_type as usize;
            let rr = if is_red_side(p.side) { r } else { flip_r(r) };

            phase += PHASE_WEIGHT[pt];

            // material
            mg[idx] += MG_BASE[pt];
            eg[idx] += EG_BASE[pt];

            // PST
            mg[idx] += pst_mg(pt, rr, c);
            eg[idx] += pst_eg(pt, rr, c);

            // mobility
            let mob = piece_mobility(board, r, c, p);
            mg[idx] += mob * MOBILITY_MG[pt];
            eg[idx] += mob * MOBILITY_EG[pt];

            // per-piece special eval
            match p.piece_type {
                PieceType::King => {
                    // king safety is handled in a separate pass
                }
                PieceType::Elephant => {
                    let blocked = count_blocked_diagonals(board, r, c, p.piece_type);
                    mg[idx] -= blocked as i32 * ELEPHANT_EYE_PENALTY_MG;
                    eg[idx] -= blocked as i32 * ELEPHANT_EYE_PENALTY_EG;
                }
                PieceType::Advisor => {
                    // Advisor has no horse-leg or elephant-eye blocking penalty
                }
                PieceType::Chariot => {
                    // rook open / half-open file
                    mg[idx] += rook_file_bonus(board, r, c, p.side, true);
                    eg[idx] += rook_file_bonus(board, r, c, p.side, false);
                }
                PieceType::Cannon => {
                    // cannon mount
                    mg[idx] += cannon_mount_bonus(board, r, c, p.side, true);
                    eg[idx] += cannon_mount_bonus(board, r, c, p.side, false);
                }
                PieceType::Pawn => {
                    // passed / advanced pawn
                    let pawn_r = if is_red_side(p.side) { r } else { 9 - r };
                    mg[idx] += PASSED_PAWN_MG[pawn_r];
                    eg[idx] += PASSED_PAWN_EG[pawn_r];
                }
                _ => {}
            }
        }
    }

    // ---- king safety (midgame, separate pass) ----
    for &side in &[Side::Red, Side::Black] {
        let idx = side_idx(side);
        mg[idx] -= king_safety_penalty(board, side);
        mg[idx] -= missing_guard_penalty(board, side);
        // endgame king centrality bonus
        eg[idx] += king_centrality_eg(board, side);
    }

    // ---- piece coordination (midgame) ----
    for &side in &[Side::Red, Side::Black] {
        let idx = side_idx(side);
        mg[idx] += coordination_bonus(board, side);
    }

    // ---- tapered interpolation ----
    let phase = phase.min(TOTAL_PHASE);
    let mg_red = mg[0];
    let eg_red = eg[0];
    let mg_black = mg[1];
    let eg_black = eg[1];

    let red = (mg_red * phase + eg_red * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    let black = (mg_black * phase + eg_black * (TOTAL_PHASE - phase)) / TOTAL_PHASE;

    red - black
}

/// Evaluate from the perspective of `side`.
pub fn eval_for_side(board: &Board, side: Side) -> i32 {
    let raw = evaluate(board);
    if side == Side::Red {
        raw
    } else {
        -raw
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn side_idx(side: Side) -> usize {
    match side {
        Side::Red => 0,
        Side::Black => 1,
        _ => 0,
    }
}

#[inline]
fn pst_mg(pt: usize, rr: usize, c: usize) -> i32 {
    match pt {
        2 => MG_ADVISOR[rr][c],
        3 => MG_ELEPHANT[rr][c],
        4 => MG_HORSE[rr][c],
        5 => MG_CHARIOT[rr][c],
        6 => MG_CANNON[rr][c],
        7 => MG_PAWN[rr][c],
        _ => 0,
    }
}

#[inline]
fn pst_eg(pt: usize, rr: usize, c: usize) -> i32 {
    match pt {
        2 => EG_ADVISOR[rr][c],
        3 => EG_ELEPHANT[rr][c],
        4 => EG_HORSE[rr][c],
        5 => EG_CHARIOT[rr][c],
        6 => EG_CANNON[rr][c],
        7 => EG_PAWN[rr][c],
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Mobility (approximate – count reachable squares without generating full
// move list)
// ---------------------------------------------------------------------------

fn piece_mobility(board: &Board, r: usize, c: usize, p: Piece) -> i32 {
    match p.piece_type {
        PieceType::Chariot => chariot_mobility(board, r, c, p.side),
        PieceType::Cannon => cannon_mobility(board, r, c, p.side),
        PieceType::Horse => horse_mobility(board, r, c, p.side),
        PieceType::Elephant => elephant_mobility(board, r, c, p.side),
        PieceType::Advisor => advisor_mobility(board, r, c, p.side),
        PieceType::Pawn => pawn_mobility(board, r, c, p.side),
        _ => 0,
    }
}

fn chariot_mobility(board: &Board, r: usize, c: usize, side: Side) -> i32 {
    let mut cnt = 0i32;
    let dirs: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for (dr, dc) in dirs {
        let (mut cr, mut cc) = (r as i8 + dr, c as i8 + dc);
        while Board::in_board(cr, cc) {
            let piece = board.cells[cr as usize][cc as usize];
            if piece.is_empty() {
                cnt += 1;
            } else {
                if piece.side != side {
                    cnt += 1; // capture square
                }
                break;
            }
            cr += dr;
            cc += dc;
        }
    }
    cnt
}

fn cannon_mobility(board: &Board, r: usize, c: usize, side: Side) -> i32 {
    let mut cnt = 0i32;
    let dirs: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for (dr, dc) in dirs {
        let (mut cr, mut cc) = (r as i8 + dr, c as i8 + dc);
        // slide until a mount piece
        while Board::in_board(cr, cc) {
            if board.cells[cr as usize][cc as usize].is_empty() {
                cnt += 1; // can move here (no capture)
                cr += dr;
                cc += dc;
            } else {
                // found a mount — continue past it looking for a target
                cr += dr;
                cc += dc;
                while Board::in_board(cr, cc) {
                    let tp = board.cells[cr as usize][cc as usize];
                    if !tp.is_empty() {
                        if tp.side != side {
                            cnt += 1; // can capture enemy piece
                        }
                        break;
                    }
                    cr += dr;
                    cc += dc;
                }
                break;
            }
        }
    }
    cnt
}

fn horse_mobility(board: &Board, r: usize, c: usize, side: Side) -> i32 {
    let legs: [(i8, i8, i8, i8); 8] = [
        (-1, 0, -2, -1),
        (-1, 0, -2, 1),
        (1, 0, 2, -1),
        (1, 0, 2, 1),
        (0, -1, -1, -2),
        (0, -1, 1, -2),
        (0, 1, -1, 2),
        (0, 1, 1, 2),
    ];
    let mut cnt = 0i32;
    let ri = r as i8;
    let ci = c as i8;
    let friendly = side;
    for (lr, lc, tr, tc) in legs {
        let leg_r = ri + lr;
        let leg_c = ci + lc;
        let to_r = ri + tr;
        let to_c = ci + tc;
        if Board::in_board(leg_r, leg_c)
            && Board::in_board(to_r, to_c)
            && board.cells[leg_r as usize][leg_c as usize].is_empty()
        {
            let t = board.cells[to_r as usize][to_c as usize];
            if t.is_empty() || t.side != friendly {
                cnt += 1;
            }
        }
    }
    cnt
}

fn elephant_mobility(board: &Board, r: usize, c: usize, side: Side) -> i32 {
    let eyes: [(i8, i8, i8, i8); 4] = [
        (-1, -1, -2, -2),
        (-1, 1, -2, 2),
        (1, -1, 2, -2),
        (1, 1, 2, 2),
    ];
    let mut cnt = 0i32;
    let ri = r as i8;
    let ci = c as i8;
    let friendly = side;
    for (er, ec, tr, tc) in eyes {
        let eye_r = ri + er;
        let eye_c = ci + ec;
        let to_r = ri + tr;
        let to_c = ci + tc;
        if Board::in_board(eye_r, eye_c)
            && Board::in_board(to_r, to_c)
            && board.cells[eye_r as usize][eye_c as usize].is_empty()
            && Board::in_own_half(to_r, side)
        {
            let t = board.cells[to_r as usize][to_c as usize];
            if t.is_empty() || t.side != friendly {
                cnt += 1;
            }
        }
    }
    cnt
}

fn advisor_mobility(board: &Board, r: usize, c: usize, side: Side) -> i32 {
    let mut cnt = 0i32;
    let ri = r as i8;
    let ci = c as i8;
    let friendly = side;
    for dr in [-1i8, 1] {
        for dc in [-1i8, 1] {
            let tr = ri + dr;
            let tc = ci + dc;
            if Board::in_board(tr, tc) && Board::in_palace(tr, tc, side) {
                let t = board.cells[tr as usize][tc as usize];
                if t.is_empty() || t.side != friendly {
                    cnt += 1;
                }
            }
        }
    }
    cnt
}

fn pawn_mobility(board: &Board, r: usize, c: usize, side: Side) -> i32 {
    let ri = r as i8;
    let ci = c as i8;
    let friendly = side;
    let mut cnt = 0i32;

    let fwd: i8 = if is_red_side(side) { -1 } else { 1 };
    let tr = ri + fwd;
    if Board::in_board(tr, ci) {
        let t = board.cells[tr as usize][ci as usize];
        if t.is_empty() || t.side != friendly {
            cnt += 1;
        }
    }
    if !Board::in_own_half(ri, side) {
        for dc in [-1i8, 1] {
            let tc = ci + dc;
            if Board::in_board(ri, tc) {
                let t = board.cells[ri as usize][tc as usize];
                if t.is_empty() || t.side != friendly {
                    cnt += 1;
                }
            }
        }
    }
    cnt
}

// ---------------------------------------------------------------------------
// Blocked diagonals (horse leg / elephant eye)
// ---------------------------------------------------------------------------

fn count_blocked_diagonals(board: &Board, r: usize, c: usize, pt: PieceType) -> u8 {
    match pt {
        PieceType::Horse => {
            let legs: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            let ri = r as i8;
            let ci = c as i8;
            let mut blocked = 0u8;
            for (lr, lc) in legs {
                let lr2 = ri + lr;
                let lc2 = ci + lc;
                if Board::in_board(lr2, lc2) && !board.cells[lr2 as usize][lc2 as usize].is_empty() {
                    blocked += 1;
                }
            }
            blocked
        }
        PieceType::Elephant => {
            let eyes: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
            let ri = r as i8;
            let ci = c as i8;
            let mut blocked = 0u8;
            for (er, ec) in eyes {
                let er2 = ri + er;
                let ec2 = ci + ec;
                if Board::in_board(er2, ec2) && !board.cells[er2 as usize][ec2 as usize].is_empty() {
                    blocked += 1;
                }
            }
            blocked
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Rook open / half-open file
// ---------------------------------------------------------------------------

fn rook_file_bonus(board: &Board, r: usize, c: usize, side: Side, mg: bool) -> i32 {
    let opp = side.opponent();
    let (mut own_pawns, mut opp_pawns) = (0u8, 0u8);
    let mut bonus = 0i32;

    for row in 0..10usize {
        let p = board.cells[row][c];
        if p.piece_type == PieceType::Pawn {
            if p.side == side {
                own_pawns += 1;
            } else if p.side == opp {
                opp_pawns += 1;
            }
        }
    }

    if own_pawns == 0 && opp_pawns == 0 {
        bonus += if mg {
            ROOK_OPEN_FILE_MG
        } else {
            ROOK_OPEN_FILE_EG
        };
    } else if own_pawns == 0 && opp_pawns > 0 {
        bonus += if mg {
            ROOK_HALF_OPEN_FILE_MG
        } else {
            ROOK_HALF_OPEN_FILE_EG
        };
    }

    // Extra if the rook is in the opponent's half of the board
    let in_opp_half = if is_red_side(side) { r <= 4 } else { r >= 5 };
    if in_opp_half && bonus > 0 {
        bonus += bonus / 2;
    }

    bonus
}

// ---------------------------------------------------------------------------
// Cannon mount — is there at least one piece behind the cannon?
// ---------------------------------------------------------------------------

fn cannon_mount_bonus(board: &Board, r: usize, c: usize, _side: Side, mg: bool) -> i32 {
    let dirs: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let mut bonus = 0i32;
    let base = if mg { CANNON_MOUNT_MG } else { CANNON_MOUNT_EG };

    for (dr, dc) in dirs {
        let (mut cr, mut cc) = (r as i8 + dr, c as i8 + dc);
        let mut found_mount = false;
        while Board::in_board(cr, cc) {
            if !board.cells[cr as usize][cc as usize].is_empty() {
                found_mount = true;
                break;
            }
            cr += dr;
            cc += dc;
        }
        if found_mount {
            // check if there's a target beyond the mount
            cr += dr;
            cc += dc;
            while Board::in_board(cr, cc) {
                let tp = board.cells[cr as usize][cc as usize];
                if !tp.is_empty() {
                    bonus += base;
                    break;
                }
                cr += dr;
                cc += dc;
            }
        }
    }
    bonus
}

// ---------------------------------------------------------------------------
// King safety (midgame)
// ---------------------------------------------------------------------------

fn king_safety_penalty(board: &Board, side: Side) -> i32 {
    if let Some((kr, kc)) = board.find_king(side) {
        let opp = side.opponent();
        let mut attack_sum = 0i32;

        for r in 0..10i8 {
            for c in 0..9i8 {
                let p = board.cells[r as usize][c as usize];
                if p.side != opp || p.is_empty() {
                    continue;
                }
                // Manhattan distance from the piece to the king
                let dist = (r - kr).abs() + (c - kc).abs();
                if dist <= 4 {
                    let w = ATTACK_WEIGHT[p.piece_type as usize];
                    attack_sum += w * (5 - dist) as i32;
                }
            }
        }

        attack_sum.min(KING_SAFETY_MAX)
    } else {
        0
    }
}

fn missing_guard_penalty(board: &Board, side: Side) -> i32 {
    let mut penalty = 0i32;
    let (mut a_count, mut e_count) = (0u8, 0u8);

    for r in 0..10usize {
        for c in 0..9usize {
            let p = board.cells[r][c];
            if p.side != side {
                continue;
            }
            match p.piece_type {
                PieceType::Advisor => a_count += 1,
                PieceType::Elephant => e_count += 1,
                _ => {}
            }
        }
    }

    if a_count < 2 {
        penalty += MISSING_ADVISOR_MG * (2 - a_count as i32);
    }
    if e_count < 2 {
        penalty += MISSING_ELEPHANT_MG * (2 - e_count as i32);
    }

    penalty
}

fn king_centrality_eg(board: &Board, side: Side) -> i32 {
    if let Some((kr, kc)) = board.find_king(side) {
        let center_dist_from_edge: i32 = if is_red_side(side) {
            let dr = (kr - 8).abs() as i32;
            let dc = (kc - 4).abs() as i32;
            dr + dc
        } else {
            let dr = (kr - 1).abs() as i32;
            let dc = (kc - 4).abs() as i32;
            dr + dc
        };
        (2 - center_dist_from_edge) * 4
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Piece coordination (midgame)
// ---------------------------------------------------------------------------

fn coordination_bonus(board: &Board, side: Side) -> i32 {
    let mut bonus = 0i32;

    // Rook + Cannon on same file
    for file in 0..9usize {
        let (mut rook, mut cannon) = (false, false);
        for row in 0..10usize {
            let p = board.cells[row][file];
            if p.side != side {
                continue;
            }
            match p.piece_type {
                PieceType::Chariot => rook = true,
                PieceType::Cannon => cannon = true,
                _ => {}
            }
        }
        if rook && cannon {
            bonus += ROOK_CANNON_SAME_FILE_MG;
        }
    }

    // Dual rooks on same file
    for file in 0..9usize {
        let mut rook_count = 0u8;
        for row in 0..10usize {
            let p = board.cells[row][file];
            if p.side == side && p.piece_type == PieceType::Chariot {
                rook_count += 1;
            }
        }
        if rook_count >= 2 {
            bonus += DUAL_ROOK_SAME_FILE_MG;
        }
    }

    bonus
}

// ---------------------------------------------------------------------------
// NNUE evaluation (fallback to handcrafted if NNUE not loaded)
// ---------------------------------------------------------------------------

/// Evaluate using NNUE if loaded, otherwise fall back to handcrafted.
pub fn nnue_evaluate(board: &Board, nnue: &Nnue, state: &NnueState) -> i32 {
    if nnue.is_loaded() {
        state.forward(nnue, Side::Red)
    } else {
        evaluate(board)
    }
}

/// Evaluate from side's perspective using NNUE if loaded.
/// NNUE is always evaluated from Red's perspective (matching training data)
/// and negated for Black. This avoids a mismatch: the model was trained with
/// Red-centric piece encoding (friendly=Red pieces 0-6, enemy=Black 7-13),
/// so calling forward(Black) would swap roles relative to training.
pub fn nnue_eval_for_side(board: &Board, side: Side, nnue: &Nnue, state: &NnueState) -> i32 {
    if nnue.is_loaded() {
        let score = state.forward(nnue, Side::Red);
        if side == Side::Black { -score } else { score }
    } else {
        eval_for_side(board, side)
    }
}
