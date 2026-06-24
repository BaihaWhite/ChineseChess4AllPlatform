use crate::board::Board;
use crate::evaluate::nnue_eval_for_side;
use crate::nnue::{Nnue, NnueState};
use crate::tt::TranspositionTable;
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const INF: i32 = 999999;
const MATE_SCORE: i32 = 90000;
const WINDOW: i32 = 40;

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

/// Per-thread search worker. Holds mutable per-thread state and shared references.
struct Searcher<'a> {
    tt: &'a TranspositionTable,
    nnue: &'a Nnue,
    nnue_state: NnueState,
    cancelled: &'a AtomicBool,
    time_limit_ms: u64,
    start_time: Instant,
}

impl<'a> Searcher<'a> {
    fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    fn time_up(&self) -> bool {
        if self.time_limit_ms == 0 {
            return false;
        }
        self.elapsed_ms() >= self.time_limit_ms
    }

    fn aborted(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed) || self.time_up()
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
                    // Don't return draw if we're giving check — keep searching for mate
                    if !board.is_in_check(board.current_turn.opponent()) {
                        return 0;
                    }
                }
            }
            if let Some(&cnt) = state.game_hash_counts.get(&hash) {
                if cnt >= 2 {
                    if in_check {
                        return MATE_SCORE - ply as i32;
                    }
                    // Don't return draw if we're giving check — keep searching for mate
                    if !board.is_in_check(board.current_turn.opponent()) {
                        return 0;
                    }
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

        // IID: when TT misses at a PV node, do a shallow search for move ordering
        if tt_best.is_none() && effective_depth >= 3 && !is_root && !in_check {
            self.negamax(
                board, effective_depth - 2, -beta, -alpha, ply + 1, false, -1, -1, state,
            );
            if self.aborted() {
                return 0;
            }
            if let Some(entry) = self.tt.probe(hash) {
                tt_best = entry.best_move;
            }
        }

        if do_null && effective_depth >= 3 && !in_check && !is_root && ply < 50 {
            let saved_turn = board.current_turn;
            let saved_hash = board.zobrist_hash;
            board.current_turn = board.current_turn.opponent();
            board.zobrist_hash = saved_hash ^ crate::zobrist::SIDE_TO_MOVE_KEY;
            let r = 3 + effective_depth / 6;
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

        let static_eval = nnue_eval_for_side(board, board.current_turn, self.nnue, &self.nnue_state);

        // Razoring: at low depth, skip search if eval is hopeless
        if !in_check && effective_depth == 1 && static_eval + 350 <= alpha {
            return self.qsearch(board, alpha, beta, ply, state);
        }

        self.order_moves(&mut moves, board, effective_depth, &tt_best, prev_from, prev_to, state);

        // Late Move Pruning: at low depth, skip quiet moves after a limit
        let lmp_limit = if !in_check && effective_depth <= 3 {
            [0, 4, 8, 16][effective_depth as usize]
        } else {
            moves.len()
        };

        let mut best_move: Option<Move> = None;
        let mut best_score = -INF;
        let mut flag = TTFlag::UpperBound;
        let mut a = alpha;
        let mut move_count = 0i32;
        let mut quiets_searched = 0usize;

        for m in moves {
            if self.aborted() {
                break;
            }

            let mover = board.cells[m.from_row as usize][m.from_col as usize];
            let saved_nnue = self.nnue_state.clone();
            let captured = board.make_move(m);
            self.nnue_state.update(board, self.nnue, m.from_row, m.from_col, m.to_row, m.to_col, mover, captured);
            let gives_check = board.is_in_check(board.current_turn);
            let is_capture = !captured.is_empty();
            move_count += 1;

            let is_quiet = !is_capture && !gives_check && !in_check;
            if is_quiet {
                quiets_searched += 1;
                if quiets_searched > lmp_limit {
                    self.nnue_state = saved_nnue;
                    board.unmake_move(m, captured);
                    continue;
                }
            }

            let cur_from = m.from_row as i32 * 9 + m.from_col as i32;
            let cur_to = m.to_row as i32 * 9 + m.to_col as i32;

            let mut new_depth = effective_depth - 1;
            if gives_check {
                new_depth += 1;
            }

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
                    self.nnue_state = saved_nnue;
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

            self.nnue_state = saved_nnue;
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

        let stand_pat = nnue_eval_for_side(board, board.current_turn, self.nnue, &self.nnue_state);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat + 650 < alpha {
            return alpha; // delta pruning: even best capture can't reach alpha
        }
        let mut a = alpha.max(stand_pat);

        let mut moves = board.generate_legal_captures();
        self.order_moves_captures(&mut moves, board);

        for m in moves {
            let see_val = see_capture(board, m);
            if see_val < -100 {
                continue;
            }
            if see_val + stand_pat + 150 < a {
                continue;
            }

            let mover = board.cells[m.from_row as usize][m.from_col as usize];
            let saved_nnue = self.nnue_state.clone();
            let cap = board.make_move(m);
            self.nnue_state.update(board, self.nnue, m.from_row, m.from_col, m.to_row, m.to_col, mover, cap);
            let score = -self.qsearch(board, -beta, -a, ply + 1, state);
            self.nnue_state = saved_nnue;
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
        moves.sort_unstable_by(|a, b| {
            let sa = move_score(*a, board, depth_idx, tt_best, prev_from, prev_to, state);
            let sb = move_score(*b, board, depth_idx, tt_best, prev_from, prev_to, state);
            sb.cmp(&sa)
        });
    }

    fn order_moves_captures(&self, moves: &mut Vec<Move>, board: &Board) {
        moves.sort_unstable_by(|a, b| {
            let sa = see_capture(board, *a);
            let sb = see_capture(board, *b);
            sb.cmp(&sa)
        });
    }
}

fn move_score(
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

    if !victim.is_empty() {
        let see_val = see_capture(board, m);
        if see_val >= 0 {
            return 40_000_000 + see_val;
        } else {
            return 35_000_000 + see_val;
        }
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

// ---------------------------------------------------------------------------
// SearchEngine — public API
// ---------------------------------------------------------------------------

pub struct SearchEngine {
    pub tt: Arc<TranspositionTable>,
    pub nnue: Arc<Nnue>,
    pub nnue_state: NnueState,
    pub last_search_score: i32,
    pub last_completed_depth: i32,
    pub last_nodes: u64,
    cancelled: Arc<AtomicBool>,
    time_limit_ms: u64,
    start_time: Option<Instant>,
    num_threads: usize,
}

impl SearchEngine {
    pub fn new() -> Self {
        let mut nnue = Nnue::new();
        // Search order: cwd, exe directory, /tmp
        let load_result = nnue.load_weights("nnue_trained.bin")
            .or_else(|_| {
                // Try the executable's directory (e.g. engine/target/release/)
                if let Ok(exe) = std::env::current_exe() {
                    if let Some(dir) = exe.parent() {
                        let path = dir.join("nnue_trained.bin");
                        return nnue.load_weights(path.to_str().unwrap_or(""));
                    }
                }
                Err("no exe path".into())
            })
            .or_else(|_| nnue.load_weights("/tmp/nnue_trained.bin"))
            .or_else(|_| nnue.load_weights("/tmp/chinese-chess-natives/nnue_trained.bin"))
            .or_else(|_| {
                let tmp = std::env::temp_dir().join("chinese-chess-natives").join("nnue_trained.bin");
                nnue.load_weights(tmp.to_str().unwrap_or(""))
            });
        let msg = match &load_result {
            Ok(()) => format!("NNUE loaded OK"),
            Err(e) => format!("NNUE FAILED: {}. Using HCE fallback.", e),
        };
        let _ = std::fs::write("/tmp/nnue_load_status.txt", &msg);
        eprintln!("[NNUE] {}", msg);

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        SearchEngine {
            tt: Arc::new(TranspositionTable::new(64)),
            nnue: Arc::new(nnue),
            nnue_state: NnueState::new(),
            last_search_score: 0,
            last_completed_depth: 0,
            last_nodes: 0,
            cancelled: Arc::new(AtomicBool::new(false)),
            time_limit_ms: 0,
            start_time: None,
            num_threads,
        }
    }

    pub fn new_without_nnue() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        SearchEngine {
            tt: Arc::new(TranspositionTable::new(64)),
            nnue: Arc::new(Nnue::new()),
            nnue_state: NnueState::new(),
            last_search_score: 0,
            last_completed_depth: 0,
            last_nodes: 0,
            cancelled: Arc::new(AtomicBool::new(false)),
            time_limit_ms: 0,
            start_time: None,
            num_threads,
        }
    }

    pub fn new_with_nnue_path(path: &str) -> Self {
        let mut nnue = Nnue::new();
        match nnue.load_weights(path) {
            Ok(()) => eprintln!("[NNUE] Weights loaded from {}", path),
            Err(e) => eprintln!("[NNUE] WARNING: Failed to load from {}: {}. Using HCE fallback.", path, e),
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        SearchEngine {
            tt: Arc::new(TranspositionTable::new(64)),
            nnue: Arc::new(nnue),
            nnue_state: NnueState::new(),
            last_search_score: 0,
            last_completed_depth: 0,
            last_nodes: 0,
            cancelled: Arc::new(AtomicBool::new(false)),
            time_limit_ms: 0,
            start_time: None,
            num_threads,
        }
    }

    pub fn thread_count(&self) -> usize {
        self.num_threads
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Reload NNUE weights from a file path. Used by JNI to load from Android assets.
    pub fn reload_nnue(&mut self, path: &str) {
        let mut nnue = Nnue::new();
        match nnue.load_weights(path) {
            Ok(()) => {
                eprintln!("[NNUE] Loaded from {}", path);
                self.nnue = Arc::new(nnue);
                self.nnue_state = NnueState::new();
            }
            Err(e) => {
                eprintln!("[NNUE] Failed to load from {}: {}", path, e);
            }
        }
    }

    /// Single-threaded search (kept for backward compatibility with gen_data etc.)
    pub fn search(
        &mut self,
        board: &mut Board,
        depth: i32,
        time_ms: u64,
        game_history: &[u64],
    ) -> Option<Move> {
        self.search_with_threads(board, depth, time_ms, game_history, 1)
    }

    /// Lazy SMP search with specified number of threads.
    /// Threads share the TT and NNUE weights. Each thread has its own board,
    /// NnueState, and SearchState, and perturbs move ordering at the root.
    pub fn search_with_threads(
        &mut self,
        board: &mut Board,
        depth: i32,
        time_ms: u64,
        game_history: &[u64],
        num_threads: usize,
    ) -> Option<Move> {
        let num_threads = num_threads.max(1);
        self.cancelled.store(false, Ordering::Relaxed);
        self.time_limit_ms = time_ms;
        let t0 = Instant::now();
        self.start_time = Some(t0);

        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            return None;
        }
        eprintln!("[FEN] {}", board.to_fen());
        if moves.len() == 1 {
            self.last_search_score = 0;
            self.last_completed_depth = 1;
            self.last_nodes = 1;
            return Some(moves[0]);
        }

        if num_threads == 1 {
            // Single-threaded path — use directly for efficiency
            self.nnue_state.refresh(board, &self.nnue);

            let mut game_hash_counts = std::collections::HashMap::new();
            for &h in game_history {
                *game_hash_counts.entry(h).or_insert(0) += 1;
            }
            let mut state = SearchState {
                killers: [[None; 2]; 128],
                history: [[0; 90]; 90],
                counter_moves: [[None; 90]; 90],
                nodes: 0,
                root_best: None,
                search_hashes: [0u64; 256],
                game_hash_counts,
            };

            let mut searcher = Searcher {
                tt: self.tt.as_ref(),
                nnue: self.nnue.as_ref(),
                nnue_state: self.nnue_state.clone(),
                cancelled: self.cancelled.as_ref(),
                time_limit_ms: time_ms,
                start_time: t0,
            };
            let result = searcher.iterative_deepening(board, depth, &mut state);
            self.nnue_state = searcher.nnue_state;
            self.last_search_score = result.best_score;
            self.last_completed_depth = result.completed_depth;
            self.last_nodes = result.nodes;
            result.best_move.or_else(|| moves.first().copied())
        } else {
            // Lazy SMP: spawn N threads sharing TT and NNUE
            let best_move = Arc::new(std::sync::Mutex::new(None::<Move>));
            let best_score = Arc::new(AtomicI32::new(-INF));
            let best_depth = Arc::new(AtomicI32::new(0));
            let total_nodes = Arc::new(AtomicU64::new(0));
            let initial_board = board.clone();

            std::thread::scope(|scope| {
                let mut handles = Vec::new();

                for _thread_id in 0..num_threads {
                    let tt = Arc::clone(&self.tt);
                    let nnue = Arc::clone(&self.nnue);
                    let cancelled = Arc::clone(&self.cancelled);
                    let best_move = Arc::clone(&best_move);
                    let best_score = Arc::clone(&best_score);
                    let best_depth = Arc::clone(&best_depth);
                    let total_nodes = Arc::clone(&total_nodes);
                    let gm_history: Vec<u64> = game_history.to_vec();
                    let root_moves = moves.clone();
                    let mut thread_board = initial_board.clone();

                    let handle = scope.spawn(move || {
                        let mut nnue_state = NnueState::new();
                        nnue_state.refresh(&thread_board, &nnue);

                        let mut game_hash_counts = std::collections::HashMap::new();
                        for &h in &gm_history {
                            *game_hash_counts.entry(h).or_insert(0) += 1;
                        }
                        let mut state = SearchState {
                            killers: [[None; 2]; 128],
                            history: [[0; 90]; 90],
                            counter_moves: [[None; 90]; 90],
                            nodes: 0,
                            root_best: None,
                            search_hashes: [0u64; 256],
                            game_hash_counts,
                        };

                        let mut searcher = Searcher {
                            tt: tt.as_ref(),
                            nnue: nnue.as_ref(),
                            nnue_state,
                            cancelled: cancelled.as_ref(),
                            time_limit_ms: time_ms,
                            start_time: t0,
                        };

                        let result = searcher.iterative_deepening(&mut thread_board, depth, &mut state);

                        total_nodes.fetch_add(result.nodes, Ordering::Relaxed);

                        // Atomically update best result using CAS on depth
                        // to prevent a shallower result overwriting a deeper one.
                        loop {
                            let cur_depth = best_depth.load(Ordering::Relaxed);
                            if result.completed_depth < cur_depth {
                                break; // another thread has a deeper result
                            }
                            let cur_score = best_score.load(Ordering::Relaxed);
                            if result.completed_depth == cur_depth
                                && result.best_score <= cur_score
                            {
                                break; // same depth, not a better score
                            }
                            // Try to claim the depth slot
                            let desired = if result.completed_depth > cur_depth {
                                result.completed_depth
                            } else {
                                cur_depth // same depth — CAS to serialize
                            };
                            if best_depth
                                .compare_exchange_weak(
                                    cur_depth,
                                    desired,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                                .is_err()
                            {
                                continue; // CAS failed, retry
                            }
                            // Won the race — update score and move
                            best_score.store(result.best_score, Ordering::Relaxed);
                            if let Some(m) = result.best_move {
                                if root_moves.contains(&m) {
                                    *best_move.lock().unwrap() = Some(m);
                                }
                            }
                            break;
                        }
                    });

                    handles.push(handle);
                }

                for h in handles {
                    let _ = h.join();
                }
            });

            self.last_search_score = best_score.load(Ordering::Relaxed);
            self.last_completed_depth = best_depth.load(Ordering::Relaxed);
            self.last_nodes = total_nodes.load(Ordering::Relaxed);

            let result = *best_move.lock().unwrap();
            result.or_else(|| moves.first().copied())
        }
    }

}

// ---------------------------------------------------------------------------
// SEE (Static Exchange Evaluation) — free functions
// ---------------------------------------------------------------------------

const SEE_VAL: [i32; 8] = [0, 10000, 120, 120, 270, 600, 285, 30];

pub fn see(board: &Board, tr: usize, tc: usize, side: Side) -> i32 {
    let victim = board.cells[tr][tc];
    if victim.is_empty() {
        return 0;
    }
    let mut used = [[false; 9]; 10];

    let mut att_vals = [0i32; 32];
    let mut n = 0usize;
    att_vals[n] = SEE_VAL[victim.piece_type as usize];
    n += 1;

    let mut cur = side;
    loop {
        let (ar, ac, _) = find_lva(board, tr, tc, cur, &used);
        if ar < 0 {
            break;
        }
        used[ar as usize][ac as usize] = true;
        att_vals[n] = SEE_VAL[board.cells[ar as usize][ac as usize].piece_type as usize];
        n += 1;
        cur = cur.opponent();

        if n >= 31 {
            break;
        }
    }

    let mut gain = 0i32;
    for i in (0..n).rev() {
        gain = (att_vals[i] - gain).max(0);
    }
    gain
}

fn find_lva(board: &Board, tr: usize, tc: usize, side: Side, used: &[[bool; 9]; 10]) -> (i8, i8, i32) {
    let tr_i = tr as i8;
    let tc_i = tc as i8;
    let mut best_val = i32::MAX;
    let mut best_r: i8 = -1;
    let mut best_c: i8 = -1;

    for r in 0..10i8 {
        for c in 0..9i8 {
            if used[r as usize][c as usize] {
                continue;
            }
            let p = board.cells[r as usize][c as usize];
            if p.side != side || p.is_empty() {
                continue;
            }
            if board.is_valid_move(r, c, tr_i, tc_i) {
                let val = SEE_VAL[p.piece_type as usize];
                if val < best_val {
                    best_val = val;
                    best_r = r;
                    best_c = c;
                }
            }
        }
    }
    (best_r, best_c, best_val)
}

pub fn see_capture(board: &Board, m: Move) -> i32 {
    let victim = board.cells[m.to_row as usize][m.to_col as usize];
    if victim.is_empty() {
        return 0;
    }
    let attacker = board.cells[m.from_row as usize][m.from_col as usize];
    // Use a stack copy of cells instead of cloning the whole Board (avoids Vec heap allocs)
    let mut cells = board.cells;
    cells[m.to_row as usize][m.to_col as usize] = attacker;
    cells[m.from_row as usize][m.from_col as usize] = Piece::EMPTY;
    let net = SEE_VAL[victim.piece_type as usize]
        - see_cells(&cells, m.to_row as usize, m.to_col as usize, attacker.side.opponent());
    net
}

/// SEE using raw cells array — avoids Board clone.
fn see_cells(cells: &[[Piece; 9]; 10], tr: usize, tc: usize, side: Side) -> i32 {
    let victim = cells[tr][tc];
    if victim.is_empty() {
        return 0;
    }
    let mut used = [[false; 9]; 10];

    let mut att_vals = [0i32; 32];
    let mut n = 0usize;
    att_vals[n] = SEE_VAL[victim.piece_type as usize];
    n += 1;

    let mut cur = side;
    loop {
        let (ar, ac, _) = find_lva_cells(cells, tr, tc, cur, &used);
        if ar < 0 {
            break;
        }
        used[ar as usize][ac as usize] = true;
        att_vals[n] = SEE_VAL[cells[ar as usize][ac as usize].piece_type as usize];
        n += 1;
        cur = cur.opponent();

        if n >= 31 {
            break;
        }
    }

    let mut gain = 0i32;
    for i in (0..n).rev() {
        gain = (att_vals[i] - gain).max(0);
    }
    gain
}

fn find_lva_cells(
    cells: &[[Piece; 9]; 10],
    tr: usize,
    tc: usize,
    side: Side,
    used: &[[bool; 9]; 10],
) -> (i8, i8, i32) {
    let tr_i = tr as i8;
    let tc_i = tc as i8;
    let mut best_val = i32::MAX;
    let mut best_r: i8 = -1;
    let mut best_c: i8 = -1;

    for r in 0..10i8 {
        for c in 0..9i8 {
            if used[r as usize][c as usize] {
                continue;
            }
            let p = cells[r as usize][c as usize];
            if p.side != side || p.is_empty() {
                continue;
            }
            if is_valid_move_cells(cells, r, c, tr_i, tc_i, side) {
                let val = SEE_VAL[p.piece_type as usize];
                if val < best_val {
                    best_val = val;
                    best_r = r;
                    best_c = c;
                }
            }
        }
    }
    (best_r, best_c, best_val)
}

/// Simplified move validation using only the cells array.
/// Mirrors `Board::is_valid_move` but without borrowing Board.
fn is_valid_move_cells(
    cells: &[[Piece; 9]; 10],
    fx: i8,
    fy: i8,
    tx: i8,
    ty: i8,
    side: Side,
) -> bool {
    if !Board::in_board(tx, ty) {
        return false;
    }
    if fx == tx && fy == ty {
        return false;
    }
    let mover = cells[fx as usize][fy as usize];
    if mover.side != side {
        return false;
    }
    let target = cells[tx as usize][ty as usize];
    if target.side == side {
        return false;
    }

    let dx = tx - fx;
    let dy = ty - fy;
    let adx = dx.abs();
    let ady = dy.abs();

    match mover.piece_type {
        PieceType::King => {
            ((adx == 1 && ady == 0) || (adx == 0 && ady == 1))
                && Board::in_palace(tx, ty, mover.side)
        }
        PieceType::Advisor => {
            adx == 1 && ady == 1 && Board::in_palace(tx, ty, mover.side)
        }
        PieceType::Elephant => {
            adx == 2
                && ady == 2
                && Board::in_own_half(tx, mover.side)
                && cells[(fx + dx / 2) as usize][(fy + dy / 2) as usize].is_empty()
        }
        PieceType::Horse => {
            (adx == 2 && ady == 1 || adx == 1 && ady == 2)
                && if adx == 2 {
                    cells[(fx + dx / 2) as usize][fy as usize].is_empty()
                } else {
                    cells[fx as usize][(fy + dy / 2) as usize].is_empty()
                }
        }
        PieceType::Chariot => {
            (dx == 0 || dy == 0) && count_between_cells(cells, fx, fy, tx, ty) == 0
        }
        PieceType::Cannon => {
            (dx == 0 || dy == 0)
                && if target.is_empty() {
                    count_between_cells(cells, fx, fy, tx, ty) == 0
                } else {
                    count_between_cells(cells, fx, fy, tx, ty) == 1
                }
        }
        PieceType::Pawn => match mover.side {
            Side::Red => {
                if Board::in_own_half(fx, mover.side) {
                    dx == -1 && dy == 0
                } else {
                    dx != 1 && (adx + ady == 1) && (dx == -1 || (dx == 0 && ady == 1))
                }
            }
            Side::Black => {
                if Board::in_own_half(fx, mover.side) {
                    dx == 1 && dy == 0
                } else {
                    dx != -1 && (adx + ady == 1) && (dx == 1 || (dx == 0 && ady == 1))
                }
            }
            _ => false,
        },
        PieceType::Empty => false,
    }
}

fn count_between_cells(cells: &[[Piece; 9]; 10], fx: i8, fy: i8, tx: i8, ty: i8) -> i8 {
    let mut cnt = 0i8;
    let dx = (tx - fx).signum();
    let dy = (ty - fy).signum();
    let mut cx = fx + dx;
    let mut cy = fy + dy;
    while cx != tx || cy != ty {
        if !cells[cx as usize][cy as usize].is_empty() {
            cnt += 1;
        }
        cx += dx;
        cy += dy;
    }
    cnt
}

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
