use std::time::Instant;
use chess_engine::board::Board;
use chess_engine::search::SearchEngine;
use chess_engine::types::*;

fn board_from_fen(fen: &str) -> Board {
    let mut board = Board::empty();
    let rows: Vec<&str> = fen.split('/').collect();
    for r in 0..10 {
        let mut c = 0usize;
        for ch in rows[r].chars() {
            if ch.is_ascii_digit() { c += ch.to_digit(10).unwrap() as usize; }
            else {
                let side = if ch.is_uppercase() { Side::Red } else { Side::Black };
                let pt = match ch.to_ascii_lowercase() {
                    'k' => PieceType::King, 'a' => PieceType::Advisor, 'b'|'e' => PieceType::Elephant,
                    'h'|'n' => PieceType::Horse, 'r' => PieceType::Chariot, 'c' => PieceType::Cannon,
                    'p' => PieceType::Pawn, _ => PieceType::Empty,
                };
                board.cells[r][c] = Piece::new(pt, side); c += 1;
            }
        }
    }
    board.current_turn = Side::Red;
    board.zobrist_hash = board.compute_zobrist();
    board
}

fn main() {
    let mut engine = SearchEngine::new();
    
    let positions = [
        ("Initial", "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR"),
        ("Midgame 1", "rheakaehr/9/1c2c4/p1p1p3p/4h4/2P6/P3P1P1P/1C2C4/9/RHEAKAEHR"),
        ("Midgame 2", "r1eakaehr/9/1c2c4/p1p3p1p/3h5/2P1P4/P4P1PP/1C2C4/9/RHEAKAE1R"),
    ];
    
    for (name, fen) in &positions {
        let board = board_from_fen(fen);
        println!("=== {} ===", name);
        for depth in 1..=12 {
            let mut b = board.clone();
            let start = Instant::now();
            let _ = engine.search(&mut b, depth, 60000, &[]);
            let elapsed = start.elapsed().as_millis();
            println!("  depth {:2}: {:>8}ms  score={:>5}", depth, elapsed, engine.last_search_score);
            if elapsed > 30000 { break; }
        }
    }
}
