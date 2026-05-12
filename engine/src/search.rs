use crate::board::Board;
use crate::evaluate::eval_for_side;
use crate::tt::TranspositionTable;
use crate::types::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const INF: i32 = 999999;
const MATE_SCORE: i32 = 90000;
const WINDOW: i32 = 30;

struct SearchState {
    killers: [[Option<Move>; 2]; 128],
    history: [[i32; 90]; 90],
    counter_moves: [[Option<Move>; 90]; 90],
    nodes: u64,
    root_best: Option<Move>,
    search_hashes: [u64; 256],
    game_hash_counts: std::collections::HashMap<u64, u8>,
}

struct SearchResult {
    best_move: Option<Move>,
    best_score: i32,
    completed_depth: i32,
    nodes: u64,
}

pub struct SearchEngine {
    tt: TranspositionTable,
    cancelled: AtomicBool,
    time_limit_ms: u64,
    start_time: Option<Instant>,
}

impl SearchEngine {
    pub fn new() -> Self {
        SearchEngine {
            tt: TranspositionTable::new(2),
            cancelled: AtomicBool::new(false),
            time_limit_ms: 0,
            start_time: None,
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        depth: i32,
        time_ms: u64,
        game_history: &[u64],
    ) -> Option<Move> {
        self.cancelled.store(false, Ordering::Relaxed);
        self.time_limit_ms = time_ms;
        self.start_time = Some(Instant::now());

        let mut state = SearchState {
            killers: [[None; 2]; 128],
            history: [[0; 90]; 90],
            counter_moves: [[None; 90]; 90],
            nodes: 0,
            root_best: None,
            search_hashes: [0u64; 256],
            game_hash_counts: std::collections::HashMap::new(),
        };

        for &h in game_history {
            *state.game_hash_counts.entry(h).or_insert(0) += 1;
        }

        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            return None;
        }
        if moves.len() == 1 {
            return Some(moves[0]);
        }

        let result = self.iterative_deepening(board, depth, &mut state);
        result.best_move.or_else(|| moves.first().copied())
    }

    fn iterative_deepening(
        &mut self,
        board: &mut Board,
        max_depth: i32,
        state: &mut SearchState,
    ) -> SearchResult {
        let mut best_score = -INF;
        let mut last_completed_depth = 0;
        let mut aspiration_alpha = -INF;
        let mut aspiration_beta = INF;

        for d in 1..=max_depth {
            if self.aborted() {
                break;
            }

            let score = if d >= 4 {
                let mut s = self.negamax(
                    board, d, aspiration_alpha, aspiration_beta, 0, true, -1, -1, state,
                );
                if !self.aborted() && (s <= aspiration_alpha || s >= aspiration_beta) {
                    s = self.negamax(board, d, -INF, INF, 0, true, -1, -1, state);
                    aspiration_alpha = -INF;
                    aspiration_beta = INF;
                }
                if !self.aborted() && s > aspiration_alpha && s < aspiration_beta {
                    aspiration_alpha = s - WINDOW;
                    aspiration_beta = s + WINDOW;
                }
                s
            } else {
                let s = self.negamax(board, d, -INF, INF, 0, true, -1, -1, state);
                if d == 3 && !self.aborted() {
                    aspiration_alpha = s - WINDOW;
                    aspiration_beta = s + WINDOW;
                }
                s
            };

            if self.aborted() {
                break;
            }

            if state.root_best.is_some() {
                best_score = score;
                last_completed_depth = d;
            }

            let elapsed = self.elapsed_ms();
            if elapsed > self.time_limit_ms / 2 && d >= 3 {
                break;
            }

            if score.abs() > MATE_SCORE - 100 {
                break;
            }
        }

        SearchResult {
            best_move: state.root_best,
            best_score,
            completed_depth: last_completed_depth,
            nodes: state.nodes,
        }
    }

    fn negamax(
        &mut self,
        board: &mut Board,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        do_null: bool,
        prev_from: i32,
        prev_to: i32,
        state: &mut SearchState,
    ) -> i32 {
        if self.aborted() {
            return 0;
        }
        state.nodes += 1;

        let hash = board.zobrist_hash;
        state.search_hashes[ply] = hash;

        let is_root = ply == 0;
        let in_check = board.is_in_check(board.current_turn);

        if ply > 0 {
            for i in 0..ply {
                if state.search_hashes[i] == hash {
                    if in_check {
                        return MATE_SCORE - ply as i32;
                    }
                    return 0;
                }
            }
            if let Some(&cnt) = state.game_hash_counts.get(&hash) {
                if cnt >= 2 {
                    if in_check {
                        return MATE_SCORE - ply as i32;
                    }
                    return 0;
                }
            }
        }

        if in_check && depth <= 0 && ply < 60 {
            return self.negamax(board, 1, alpha, beta, ply, false, prev_from, prev_to, state);
        }

        if depth <= 0 && !in_check {
            return self.qsearch(board, alpha, beta, ply, state);
        }

        let effective_depth = depth.max(0);

        let tt_entry = self.tt.probe(hash);
        let mut tt_best: Option<Move> = None;
        if let Some(entry) = tt_entry {
            tt_best = entry.best_move;
            if i32::from(entry.depth) >= effective_depth && !is_root {
                match entry.flag {
                    TTFlag::Exact => return entry.score,
                    TTFlag::LowerBound => {
                        if entry.score >= beta {
                            return entry.score;
                        }
                    }
                    TTFlag::UpperBound => {
                        if entry.score <= alpha {
                            return entry.score;
                        }
                    }
                }
            }
        }

        if do_null && effective_depth >= 3 && !in_check && !is_root && ply < 50 {
            let saved_turn = board.current_turn;
            let saved_hash = board.zobrist_hash;
            board.current_turn = board.current_turn.opponent();
            board.zobrist_hash = saved_hash ^ crate::zobrist::SIDE_TO_MOVE_KEY;
            let r = if effective_depth > 6 { 4 } else { 3 };
            let null_value = -self.negamax(
                board,
                effective_depth - 1 - r,
                -beta,
                -beta + 1,
                ply + 1,
                false,
                -1,
                -1,
                state,
            );
            board.current_turn = saved_turn;
            board.zobrist_hash = saved_hash;
            if self.aborted() {
                return 0;
            }
            if null_value >= beta {
                return beta;
            }
        }

        let mut moves = board.generate_legal_moves();
        if moves.is_empty() {
            return -MATE_SCORE + ply as i32;
        }

        self.order_moves(&mut moves, board, effective_depth, &tt_best, prev_from, prev_to, state);

        let mut best_move: Option<Move> = None;
        let mut best_score = -INF;
        let mut flag = TTFlag::UpperBound;
        let mut a = alpha;
        let mut move_count = 0i32;

        let static_eval = eval_for_side(board, board.current_turn);

        for m in moves {
            if self.aborted() {
                break;
            }

            let captured = board.make_move(m);
            let gives_check = board.is_in_check(board.current_turn);
            let is_capture = !captured.is_empty();
            move_count += 1;

            let cur_from = m.from_row as i32 * 9 + m.from_col as i32;
            let cur_to = m.to_row as i32 * 9 + m.to_col as i32;

            let new_depth = effective_depth - 1;

            let mut reduction = 0i32;
            if effective_depth >= 3 && move_count > 3 && !is_capture && !gives_check && !in_check {
                let lmr_idx = (move_count - 1).min(64) as usize;
                let lmr_depth = effective_depth.min(64) as usize;
                reduction = LMR_TABLE[lmr_depth][lmr_idx];
                if Some(m) == state.killers[effective_depth as usize & 127][0]
                    || Some(m) == state.killers[effective_depth as usize & 127][1]
                {
                    reduction = reduction.saturating_sub(1).max(0);
                }
                reduction = reduction.min(new_depth - 1);
                if reduction < 1 {
                    reduction = 0;
                }
            }

            if effective_depth <= 3 && !is_capture && !gives_check && !in_check && reduction == 0 {
                let futility_margin = 200 * effective_depth;
                if static_eval + futility_margin <= a {
                    board.unmake_move(m, captured);
                    continue;
                }
            }

            let value = if move_count == 1 {
                -self.negamax(board, new_depth, -beta, -a, ply + 1, true, cur_from, cur_to, state)
            } else {
                let reduced_depth = (new_depth - reduction).max(0);
                let mut v = -self.negamax(
                    board, reduced_depth, -a - 1, -a, ply + 1, true, cur_from, cur_to, state,
                );
                if reduction > 0 && v > a {
                    v = -self.negamax(
                        board, new_depth, -a - 1, -a, ply + 1, true, cur_from, cur_to, state,
                    );
                }
                if v > a && v < beta {
                    -self.negamax(
                        board, new_depth, -beta, -a, ply + 1, true, cur_from, cur_to, state,
                    )
                } else {
                    v
                }
            };

            board.unmake_move(m, captured);

            if self.aborted() {
                break;
            }

            if value > best_score {
                best_score = value;
                best_move = Some(m);
                if value > a {
                    a = value;
                    flag = TTFlag::Exact;
                }
            }
            if a >= beta {
                if !is_capture && (effective_depth as usize) < 128 {
                    let d = effective_depth as usize & 127;
                    if Some(m) != state.killers[d][0] {
                        state.killers[d][1] = state.killers[d][0];
                        state.killers[d][0] = Some(m);
                    }
                    let fi = m.from_row as usize * 9 + m.from_col as usize;
                    let ti = m.to_row as usize * 9 + m.to_col as usize;
                    state.history[fi][ti] += effective_depth * effective_depth;
                    if prev_from >= 0 && prev_to >= 0 {
                        state.counter_moves[prev_from as usize][prev_to as usize] = Some(m);
                    }
                }
                flag = TTFlag::LowerBound;
                break;
            }
        }

        if best_score <= alpha {
            flag = TTFlag::UpperBound;
        }
        if !self.aborted() {
            self.tt.store(hash, effective_depth as i16, best_score, flag, best_move);
            if is_root {
                state.root_best = best_move;
            }
        }

        best_score
    }

    fn qsearch(
        &mut self,
        board: &mut Board,
        alpha: i32,
        beta: i32,
        ply: usize,
        state: &mut SearchState,
    ) -> i32 {
        if self.aborted() {
            return 0;
        }
        state.nodes += 1;

        let stand_pat = eval_for_side(board, board.current_turn);
        if stand_pat >= beta {
            return beta;
        }
        let mut a = alpha.max(stand_pat);

        let mut moves = board.generate_legal_moves();
        self.order_moves_captures(&mut moves, board);

        for m in moves {
            let captured = board.cells[m.to_row as usize][m.to_col as usize];
            if captured.is_empty() {
                continue;
            }

            let value = BASE_VALUES[captured.piece_type as usize] * 10
                - BASE_VALUES[board.cells[m.from_row as usize][m.from_col as usize].piece_type as usize];
            if value + stand_pat + 200 < a {
                continue;
            }

            let cap = board.make_move(m);
            let score = -self.qsearch(board, -beta, -a, ply + 1, state);
            board.unmake_move(m, cap);

            if self.aborted() {
                return 0;
            }
            if score > a {
                a = score;
                if a >= beta {
                    return beta;
                }
            }
        }

        a
    }

    fn order_moves(
        &self,
        moves: &mut Vec<Move>,
        board: &Board,
        depth: i32,
        tt_best: &Option<Move>,
        prev_from: i32,
        prev_to: i32,
        state: &SearchState,
    ) {
        let depth_idx = (depth as usize) & 127;
        moves.sort_by(|a, b| {
            let sa = self.move_score(*a, board, depth_idx, tt_best, prev_from, prev_to, state);
            let sb = self.move_score(*b, board, depth_idx, tt_best, prev_from, prev_to, state);
            sb.cmp(&sa)
        });
    }

    fn move_score(
        &self,
        m: Move,
        board: &Board,
        depth_idx: usize,
        tt_best: &Option<Move>,
        prev_from: i32,
        prev_to: i32,
        state: &SearchState,
    ) -> i32 {
        if Some(m) == *tt_best {
            return 50_000_000;
        }

        let victim = board.cells[m.to_row as usize][m.to_col as usize];
        let attacker = board.cells[m.from_row as usize][m.from_col as usize];

        if !victim.is_empty() {
            return 40_000_000 + BASE_VALUES[victim.piece_type as usize] * 10
                - BASE_VALUES[attacker.piece_type as usize];
        }

        if Some(m) == state.killers[depth_idx][0] {
            return 30_000_000;
        }
        if Some(m) == state.killers[depth_idx][1] {
            return 29_000_000;
        }

        if prev_from >= 0 {
            if let Some(cm) = state.counter_moves[prev_from as usize][prev_to as usize] {
                if m == cm {
                    return 28_000_000;
                }
            }
        }

        let fi = m.from_row as usize * 9 + m.from_col as usize;
        let ti = m.to_row as usize * 9 + m.to_col as usize;
        state.history[fi][ti]
    }

    fn order_moves_captures(&self, moves: &mut Vec<Move>, board: &Board) {
        moves.sort_by(|a, b| {
            let va = board.cells[a.to_row as usize][a.to_col as usize];
            let vb = board.cells[b.to_row as usize][b.to_col as usize];
            let sa = if va.is_empty() {
                0
            } else {
                BASE_VALUES[va.piece_type as usize] * 10
                    - BASE_VALUES[board.cells[a.from_row as usize][a.from_col as usize].piece_type as usize]
            };
            let sb = if vb.is_empty() {
                0
            } else {
                BASE_VALUES[vb.piece_type as usize] * 10
                    - BASE_VALUES[board.cells[b.from_row as usize][b.from_col as usize].piece_type as usize]
            };
            sb.cmp(&sa)
        });
    }

    fn aborted(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed) || self.time_up()
    }

    fn time_up(&self) -> bool {
        if self.time_limit_ms == 0 {
            return false;
        }
        self.elapsed_ms() >= self.time_limit_ms
    }

    fn elapsed_ms(&self) -> u64 {
        match self.start_time {
            Some(t) => t.elapsed().as_millis() as u64,
            None => 0,
        }
    }
}

const BASE_VALUES: [i32; 8] = [0, 10000, 120, 120, 270, 600, 285, 30];

const LMR_TABLE: [[i32; 65]; 65] = {
    let mut table = [[0i32; 65]; 65];
    let mut d = 1;
    while d < 65 {
        let mut m = 1;
        while m < 65 {
            let log_d = approx_ln(d);
            let log_m = approx_ln(m);
            table[d][m] = (log_d * log_m / 200) as i32;
            m += 1;
        }
        d += 1;
    }
    table
};

const fn approx_ln(x: usize) -> i32 {
    if x <= 1 { return 0; }
    let mut result = 0;
    let mut val = x;
    while val > 1 {
        val /= 2;
        result += 69;
    }
    result
}
