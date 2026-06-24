/// Re-score FEN positions with NNUE-accelerated depth-N search.
/// Reads FEN lines, runs NNUE search, outputs FEN+new_score.
///
/// Usage: cargo run --release --bin rescore -- <input.txt> <output.txt> [depth=8] [threads=6] [nnue_bin]

use chess_engine::board::Board;
use chess_engine::search::SearchEngine;
use chess_engine::types::*;
use rayon::prelude::*;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

fn board_from_fen(fen: &str) -> Board {
    let mut board = Board::empty();
    let fen_board = fen.split_whitespace().next().unwrap_or(fen);
    let rows: Vec<&str> = fen_board.split('/').collect();
    if rows.len() != 10 {
        return board;
    }

    let pt_from_char = |ch: char| -> (PieceType, Side) {
        let side = if ch.is_uppercase() { Side::Red } else { Side::Black };
        let pt = match ch.to_ascii_lowercase() {
            'k' => PieceType::King, 'a' => PieceType::Advisor, 'b' | 'e' => PieceType::Elephant,
            'h' | 'n' => PieceType::Horse, 'r' => PieceType::Chariot, 'c' => PieceType::Cannon,
            'p' => PieceType::Pawn, _ => PieceType::Empty,
        };
        (pt, side)
    };

    for r in 0..10 {
        let mut c = 0usize;
        for ch in rows[r].chars() {
            if ch.is_ascii_digit() {
                c += ch.to_digit(10).unwrap() as usize;
            } else {
                let (pt, side) = pt_from_char(ch);
                board.cells[r][c] = Piece::new(pt, side);
                c += 1;
            }
        }
    }

    board.current_turn = Side::Red;
    if let Some(turn_part) = fen.split_whitespace().nth(1) {
        if turn_part.starts_with('b') {
            board.current_turn = Side::Black;
        }
    }

    board.zobrist_hash = board.compute_zobrist();
    board
}

fn score_position(fen_line: &str, depth: i32, time_ms: u64, nnue_path: &Option<String>) -> Option<String> {
    let line = fen_line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // FEN is the first 2 space-separated tokens (board_rows side_to_move)
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let fen = if tokens.len() >= 2 {
        format!("{} {}", tokens[0], tokens[1])
    } else {
        line.to_string()
    };

    let mut board = board_from_fen(&fen);
    if board.cells.iter().all(|row| row.iter().all(|p| p.is_empty())) {
        return None;
    }

    let mut engine = match nnue_path {
        Some(p) => SearchEngine::new_with_nnue_path(p),
        None => SearchEngine::new(),
    };
    engine.search(&mut board, depth, time_ms, &[]);
    let raw_score = engine.last_search_score.clamp(-2000, 2000);

    // Convert to Red's perspective (same as gen_data.rs training format)
    let training_score = if board.current_turn == Side::Black {
        -raw_score
    } else {
        raw_score
    };

    Some(format!("{} {}", fen, training_score))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rescore <input.txt> <output.txt> [depth=12] [threads=6]");
        return;
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let depth: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let threads: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    let nnue_path: Option<String> = args.get(5).cloned();
    let time_per_move_ms: u64 = 5000;

    println!(
        "Rescoring {} at depth={} with {} threads (time_limit={}ms)...",
        input_path, depth, threads, time_per_move_ms
    );

    let input = std::fs::File::open(input_path).expect("open input");
    let lines: Vec<String> = BufReader::new(input)
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .collect();

    let total = lines.len();
    println!("  {} positions to score", total);

    let counter = AtomicU64::new(0);
    let output = Mutex::new(
        std::fs::File::create(output_path).expect("create output"),
    );
    let valid = AtomicU64::new(0);

    // Configure rayon thread pool
    if let Ok(n) = env::var("RAYON_NUM_THREADS") {
        println!("  RAYON_NUM_THREADS={}", n);
    } else {
        env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    lines
        .par_iter()
        .filter_map(|line| {
            let result = score_position(line, depth, time_per_move_ms, &nnue_path);
            let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 5000 == 0 || count == total as u64 {
                let pct = 100.0 * count as f64 / total as f64;
                eprintln!("  {}/{} ({:.1}%)", count, total, pct);
            }
            result
        })
        .for_each(|scored_line| {
            let mut out = output.lock().unwrap();
            writeln!(out, "{}", scored_line).ok();
            valid.fetch_add(1, Ordering::Relaxed);
        });

    let v = valid.load(Ordering::Relaxed);
    println!(
        "Done. {}/{} positions written to {}",
        v, total, output_path
    );
}
