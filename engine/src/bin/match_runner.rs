/// Depth-ladder self-play match: find NNUE's equivalent HCE depth.
///
/// NNUE engine uses a fixed search depth. HCE engine (no NNUE weights) varies
/// depth across a range. For each (nnue_depth, hce_depth) pair we play N games
/// (half as Red, half as Black) and compute Elo difference.
/// The HCE depth where NNUE win rate ≈ 50% is the "equivalent depth".
///
/// Usage: cargo run --release --bin match_runner -- <nnue_depth> <hce_min> <hce_max> <games>
/// Example: cargo run --release --bin match_runner -- 2 1 6 100

use chess_engine::board::Board;
use chess_engine::search::SearchEngine;
use chess_engine::types::*;
use std::env;

const MAX_MOVES: u32 = 200;
const TIME_PER_MOVE_MS: u64 = 0; // unlimited — depth-limited search

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: match_runner <nnue_depth> <hce_min> <hce_max> <games> [nnue_path]\n\
             Example: match_runner 2 1 6 100 /tmp/nnue_trained.bin"
        );
        return;
    }

    let nnue_depth: i32 = args[1].parse().expect("invalid nnue_depth");
    let hce_min: i32 = args[2].parse().expect("invalid hce_min");
    let hce_max: i32 = args[3].parse().expect("invalid hce_max");
    let games: usize = args[4].parse().expect("invalid games");
    let nnue_path: Option<&str> = if args.len() > 5 { Some(&args[5]) } else { None };

    println!(
        "NNUE depth={} vs HCE depth ladder [{}-{}] ({} games each)\n",
        nnue_depth, hce_min, hce_max, games
    );
    if let Some(p) = nnue_path {
        println!("  Using NNUE weights: {}\n", p);
    }

    let mut results: Vec<(i32, f64, f64)> = Vec::new(); // (hce_depth, win_rate, elo)

    for hce_depth in hce_min..=hce_max {
        let (nnue_score, total) = run_match(nnue_depth, hce_depth, games, nnue_path);
        let win_rate = nnue_score / (total as f64 * 2.0); // normalize: each game = 2pts max
        let elo = elo_from_winrate(win_rate);
        results.push((hce_depth, win_rate, elo));
    }

    // Print summary table
    println!(
        "{:─<60}",
        format!(" NNUE depth={} vs HCE depth ladder ({} games each) ", nnue_depth, games)
    );
    for &(hce_depth, wr, elo) in &results {
        let marker = if (wr - 0.5).abs() < 0.03 { " ← equiv?" } else { "" };
        println!(
            "HCE d={}:  {:.0}%  [Elo {:+4.0}]{marker}",
            hce_depth,
            wr * 100.0,
            elo,
        );
    }
    println!("{:─<60}", "");

    // Find equivalent depth
    let mut best_hce = hce_min;
    let mut best_dist = f64::MAX;
    for &(hce_depth, wr, _) in &results {
        let dist = (wr - 0.5).abs();
        if dist < best_dist {
            best_dist = dist;
            best_hce = hce_depth;
        }
    }

    if best_dist < 0.03 {
        println!(
            "Equivalent depth: NNUE(d={}) ≈ HCE(d={})",
            nnue_depth, best_hce
        );
    } else {
        // Find interval where 50% falls
        let mut lower = None;
        let mut upper = None;
        for i in 0..results.len() {
            if results[i].1 > 0.5 {
                lower = Some(results[i].0);
            }
            if results[i].1 < 0.5 && upper.is_none() {
                upper = Some(results[i].0);
            }
        }
        if let (Some(lo), Some(hi)) = (lower, upper) {
            println!(
                "Equivalent depth: NNUE(d={}) between HCE(d={}) and HCE(d={})",
                nnue_depth, lo, hi
            );
        } else if results.iter().all(|r| r.1 > 0.5) {
            let max = results.last().unwrap().0;
            println!(
                "NNUE(d={}) stronger than HCE(d={}) — try higher hce_max",
                nnue_depth, max
            );
        } else {
            let min = results.first().unwrap().0;
            println!(
                "NNUE(d={}) weaker than HCE(d={}) — try lower hce_min",
                nnue_depth, min
            );
        }
    }
}

/// Run 2 × games matches (NNUE as Red/Black alternately), return (nnue_total_score, total_games).
/// Each win = 2 pts, draw = 1 pt, loss = 0 pt. So score ∈ [0, 2×total_games].
fn run_match(nnue_depth: i32, hce_depth: i32, games: usize, nnue_path: Option<&str>) -> (f64, usize) {
    let mut nnue_total = 0f64;
    let total = games * 2;

    for game_idx in 0..total {
        let nnue_is_red = game_idx % 2 == 0;

        let result = play_game(nnue_depth, hce_depth, nnue_is_red, nnue_path);

        match result {
            GameResult::NnueWin => nnue_total += 2.0,
            GameResult::Draw => nnue_total += 1.0,
            GameResult::NnueLoss => {}
        }

        // Progress indicator
        if (game_idx + 1) % 20 == 0 || game_idx == total - 1 {
            let wr = nnue_total / ((game_idx + 1) as f64 * 2.0) * 100.0;
            println!(
                "  HCE d={}: game {}/{}  NNUE wr={:.1}%",
                hce_depth,
                game_idx + 1,
                total,
                wr,
            );
        }
    }

    (nnue_total, total)
}

enum GameResult {
    NnueWin,
    Draw,
    NnueLoss,
}

fn play_game(nnue_depth: i32, hce_depth: i32, nnue_is_red: bool, nnue_path: Option<&str>) -> GameResult {
    let mut board = Board::new();
    let mut nnue_engine = match nnue_path {
        Some(p) => SearchEngine::new_with_nnue_path(p),
        None => SearchEngine::new(),
    };
    let mut hce_engine = SearchEngine::new_without_nnue();

    let nnue_side = if nnue_is_red { Side::Red } else { Side::Black };

    for move_num in 0..MAX_MOVES {
        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            // Side to move has no legal moves → checkmated or stalemated
            return if board.current_turn == nnue_side {
                GameResult::NnueLoss
            } else {
                GameResult::NnueWin
            };
        }

        if move_num >= 120 && moves.len() == 1 {
            // likely a trivial endgame, draw
            break;
        }

        // Choose engine based on current turn
        let m = if board.current_turn == nnue_side {
            nnue_engine.search(&mut board, nnue_depth, TIME_PER_MOVE_MS, &[])
        } else {
            hce_engine.search(&mut board, hce_depth, TIME_PER_MOVE_MS, &[])
        };

        let m = match m {
            Some(mv) => mv,
            None => {
                // search failed — pick first legal move
                moves[0]
            }
        };

        board.make_move(m);

        // Repetition draw
        if board.position_history.iter().filter(|&&h| h == board.zobrist_hash).count() >= 2 {
            return GameResult::Draw;
        }
    }

    // Move limit reached → draw
    GameResult::Draw
}

/// Elo difference from win rate: ΔElo = -400 × log₁₀(1/W - 1)
fn elo_from_winrate(wr: f64) -> f64 {
    let w = wr.clamp(0.001, 0.999);
    -400.0 * (1.0 / w - 1.0).log10()
}
