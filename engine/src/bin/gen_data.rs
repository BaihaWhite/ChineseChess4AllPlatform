/// Self-play data generator for NNUE training with RL game outcomes.
/// Plays complete self-play games, records (FEN, search_score, game_result) triples.
/// Games are played in parallel using rayon.
///
/// Asymmetric mode: weak_depth controls the weaker side's search depth.
/// Half the games give Black the full depth (Red weak), half give Red the full depth (Black weak).
/// This produces decisive games for value head training.
///
/// Usage: cargo run --release --bin gen_data -- <output.txt> <num_games> <depth> [nnue_path] [weak_depth]

use chess_engine::board::Board;
use chess_engine::search::SearchEngine;
use chess_engine::types::*;
use rayon::prelude::*;
use std::env;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_POSITIONS_PER_GAME: u64 = 600;
const MAX_NO_PROGRESS: u32 = 120;

#[derive(Clone, Copy, Debug, PartialEq)]
enum GameOutcome {
    RedWin,
    BlackWin,
    Draw,
}

impl GameOutcome {
    fn to_result_str(self) -> &'static str {
        match self {
            GameOutcome::RedWin => "+1.0",
            GameOutcome::BlackWin => "-1.0",
            GameOutcome::Draw => "0.0",
        }
    }
}

fn board_to_fen(board: &Board) -> String {
    let pt_char = |pt: PieceType| -> char {
        match pt {
            PieceType::King => 'k',
            PieceType::Advisor => 'a',
            PieceType::Elephant => 'b',
            PieceType::Horse => 'h',
            PieceType::Chariot => 'r',
            PieceType::Cannon => 'c',
            PieceType::Pawn => 'p',
            _ => '?',
        }
    };

    let mut parts: Vec<String> = Vec::new();
    for r in 0..10 {
        let mut row = String::new();
        let mut empty = 0u8;
        for c in 0..9 {
            let p = board.cells[r][c];
            if p.is_empty() {
                empty += 1;
            } else {
                if empty > 0 {
                    row.push_str(&empty.to_string());
                    empty = 0;
                }
                let ch = pt_char(p.piece_type);
                if p.side == Side::Red {
                    row.push(ch.to_ascii_uppercase());
                } else {
                    row.push(ch);
                }
            }
        }
        if empty > 0 {
            row.push_str(&empty.to_string());
        }
        parts.push(row);
    }

    let turn_char = if board.current_turn == Side::Red {
        'w'
    } else {
        'b'
    };
    format!("{} {}", parts.join("/"), turn_char)
}

fn play_one_game(
    nnue_path: Option<String>,
    strong_depth: i32,
    weak_depth: i32,
    time_per_move_ms: u64,
    game_idx: usize,
    total_games: usize,
    counter: &AtomicU64,
) -> (Vec<String>, GameOutcome) {
    let mut board = Board::new();
    let mut engine = match &nnue_path {
        Some(p) => SearchEngine::new_with_nnue_path(p),
        None => SearchEngine::new(),
    };
    let mut lines: Vec<String> = Vec::new();
    let mut positions_in_game = 0u64;
    let mut no_progress = 0u32;
    let outcome: GameOutcome;

    // Asymmetric depth: half the games give Red the full depth (Black weak),
    // half give Black the full depth (Red weak). This produces winning AND losing examples.
    let (red_depth, black_depth) = if game_idx % 2 == 0 {
        (weak_depth, strong_depth) // Black stronger
    } else {
        (strong_depth, weak_depth) // Red stronger
    };

    loop {
        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            let in_check = board.is_in_check(board.current_turn);
            outcome = if in_check {
                // Checkmate: side to move loses
                match board.current_turn {
                    Side::Red => GameOutcome::BlackWin,
                    Side::Black => GameOutcome::RedWin,
                    _ => GameOutcome::Draw,
                }
            } else {
                // Stalemate
                GameOutcome::Draw
            };
            break;
        }

        // Pick search depth based on side to move
        let cur_depth = if board.current_turn == Side::Red { red_depth } else { black_depth };

        // Record position with search score and game result (to be filled)
        let fen = board_to_fen(&board);
        let best_move = engine.search(&mut board, cur_depth, time_per_move_ms, &[]);
        let raw_score = engine.last_search_score.clamp(-2000, 2000);
        let training_score = if board.current_turn == Side::Black {
            -raw_score
        } else {
            raw_score
        };

        // Pick a move: use search result if available, else first legal
        let m = best_move.unwrap_or(moves[0]);
        if !moves.contains(&m) {
            let _ = board.make_move(moves[0]);
            // Track progress
            let captured = board.cells[moves[0].to_row as usize][moves[0].to_col as usize];
            let mover = board.cells[moves[0].from_row as usize][moves[0].from_col as usize];
            let is_reset = !captured.is_empty() || mover.piece_type == PieceType::Pawn;
            if is_reset {
                no_progress = 0;
            } else {
                no_progress += 1;
            }
            lines.push(format!("{} {} {}", fen, training_score, "?"));
            positions_in_game += 1;
            counter.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let mover = board.cells[m.from_row as usize][m.from_col as usize];
        let captured = board.make_move(m);

        // Reset no_progress on capture or pawn advance
        if !captured.is_empty() || mover.piece_type == PieceType::Pawn {
            no_progress = 0;
        } else {
            no_progress += 1;
        }

        lines.push(format!("{} {} {}", fen, training_score, "?"));
        positions_in_game += 1;
        counter.fetch_add(1, Ordering::Relaxed);

        // Check repetition
        let hash_count = board
            .position_history
            .iter()
            .filter(|&&h| h == board.zobrist_hash)
            .count();
        if hash_count >= 2 {
            outcome = GameOutcome::Draw;
            break;
        }

        // Move limit
        if positions_in_game >= MAX_POSITIONS_PER_GAME {
            outcome = GameOutcome::Draw;
            break;
        }

        // No-progress limit
        if no_progress >= MAX_NO_PROGRESS {
            outcome = GameOutcome::Draw;
            break;
        }
    }

    // Backfill game result in all recorded lines
    let result_str = outcome.to_result_str();
    for line in &mut lines {
        // Replace trailing "?" with the actual result
        if line.ends_with(" ?") {
            let space_pos = line.rfind(' ').unwrap();
            line.replace_range(space_pos + 1.., result_str);
        }
    }

    if game_idx % 10 == 0 || game_idx == total_games - 1 {
        eprintln!(
            "  ~Game {}/{} done ({} positions total)",
            game_idx + 1,
            total_games,
            counter.load(Ordering::Relaxed),
        );
    }

    (lines, outcome)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: gen_data <output.txt> <num_games> <depth> [nnue_path] [weak_depth]");
        return;
    }

    let output_path = args[1].clone();
    let num_games: usize = args[2].parse().unwrap_or(10);
    let strong_depth: i32 = args[3].parse().unwrap_or(5);
    let nnue_path: Option<String> = if args.len() > 4 {
        Some(args[4].clone())
    } else {
        None
    };
    let weak_depth: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    let weak_depth = if weak_depth > 0 { weak_depth } else { strong_depth }; // 0 = symmetric
    let time_per_move_ms: u64 = 500;

    if weak_depth < strong_depth {
        println!(
            "Asymmetric self-play: strong_depth={} weak_depth={} (alternating sides)",
            strong_depth, weak_depth
        );
    }
    println!(
        "Generating {} games (parallel, max {} moves/game)...",
        num_games, MAX_POSITIONS_PER_GAME
    );
    if let Some(ref p) = nnue_path {
        println!("  Using NNUE weights: {}", p);
    }

    let counter = AtomicU64::new(0);

    let all_results: Vec<(Vec<String>, GameOutcome)> = (0..num_games)
        .into_par_iter()
        .map(|game_idx| {
            play_one_game(
                nnue_path.clone(),
                strong_depth,
                weak_depth,
                time_per_move_ms,
                game_idx,
                num_games,
                &counter,
            )
        })
        .collect();

    let total_positions = counter.load(Ordering::Relaxed);
    println!("Writing {} positions to {}...", total_positions, output_path);

    let mut out = std::fs::File::create(&output_path).expect("create output file");
    let mut red_wins = 0u64;
    let mut black_wins = 0u64;
    let mut draws = 0u64;

    for (game_lines, outcome) in &all_results {
        for line in game_lines {
            writeln!(out, "{}", line).ok();
        }
        match outcome {
            GameOutcome::RedWin => red_wins += 1,
            GameOutcome::BlackWin => black_wins += 1,
            GameOutcome::Draw => draws += 1,
        }
    }

    println!(
        "Done. {} positions written to {}\nGames: {} Red wins, {} Black wins, {} draws",
        total_positions, output_path, red_wins, black_wins, draws
    );
}
