/// Parse ICCS PGN files and output (FEN, game_result_score) pairs for NNUE training.
///
/// Usage: cargo run --release --bin parse_pgn -- <input.pgns> <output.txt>
///
/// ICCS coordinate mapping:
///   - Uppercase cols A-I (Red's perspective): A=0, B=1, ..., I=8
///   - Lowercase cols a-i (Black's perspective): a=8, b=7, ..., i=0
///   - Rows 0-9 from Black's side: our_row = 9 - iccs_row

use chess_engine::board::Board;
use chess_engine::types::*;
use std::env;
use std::io::{BufRead, BufReader, Write};

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

/// Parse an ICCS move like "H2-E2" or "b9-c7" into (from_row, from_col, to_row, to_col)
fn parse_iccs_move(s: &str) -> Option<(u8, u8, u8, u8)> {
    let s = s.trim();
    if s.len() < 5 {
        return None;
    }
    let bytes = s.as_bytes();
    let sep_pos = s.find('-')?;

    let from_col = parse_iccs_col(bytes[0] as char)?;
    let from_row = parse_iccs_row(bytes[1] as char)?;
    // Skip '-'
    let to_col = parse_iccs_col(bytes[sep_pos + 1] as char)?;
    let to_row = parse_iccs_row(bytes[sep_pos + 2] as char)?;

    Some((from_row, from_col, to_row, to_col))
}

fn parse_iccs_col(ch: char) -> Option<u8> {
    match ch {
        'A'..='I' => Some(ch as u8 - b'A'),
        'a'..='i' => Some(8 - (ch as u8 - b'a')),
        _ => None,
    }
}

/// ICCS row 0=Black's side → our row 9; ICCS row 9=Red's side → our row 0
fn parse_iccs_row(ch: char) -> Option<u8> {
    ch.to_digit(10).map(|r| 9 - r as u8)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: parse_pgn <input.pgns> <output.txt>");
        return;
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let input = std::fs::File::open(input_path).expect("open input");
    let reader = BufReader::new(input);
    let mut out = std::fs::File::create(output_path).expect("create output");

    let mut board = Board::new();
    let mut fen: Option<String> = None;
    let mut result_score: i32 = 0;
    let mut in_game = false;
    let mut in_moves = false;
    let mut pos_count = 0u64;
    let mut error_count = 0u64;

    for line in reader.lines() {
        let line = line.unwrap_or_default();
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if in_moves {
                // Game ended
                in_game = false;
                in_moves = false;
            }
            continue;
        }

        if trimmed == "[Game \"Chinese Chess\"]" || trimmed == "[Game \"Chinese Chess\"]" {
            in_game = true;
            in_moves = false;
            board = Board::new();
            fen = None;
            result_score = 0;
            continue;
        }

        if trimmed.starts_with('[') && in_game {
            // Parse tag
            if trimmed.starts_with("[FEN \"") {
                let fen_str = &trimmed[6..trimmed.len() - 2];
                fen = Some(fen_str.to_string());
            } else if trimmed.starts_with("[Result \"") {
                let result = &trimmed[9..trimmed.len() - 2];
                result_score = match result {
                    "1-0" => 200,
                    "0-1" => -200,
                    _ => 0, // 1/2-1/2 or *
                };
            } else if trimmed.starts_with("[Format") {
                // Start of moves section follows after tags
                in_moves = false;
            }
            continue;
        }

        // Moves section: lines with move numbers and ICCS moves
        if in_game && !trimmed.starts_with('[') {
            in_moves = true;

            // Parse the starting FEN if provided
            if let Some(ref fen_str) = fen {
                // Load the starting position from FEN
                board = board_from_fen(fen_str);
                fen = None; // Only do this once per game
            }

            // Parse moves from this line
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            for token in tokens {
                // Skip move numbers (e.g., "1.", "10.", "1...")
                if token.ends_with('.') {
                    continue;
                }
                // Skip result at end
                if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" {
                    break;
                }
                // Skip comments
                if token.starts_with('{') || token.starts_with('(') {
                    continue;
                }

                // Record the current position BEFORE making the move
                let stm = board.current_turn;
                let score = if stm == Side::Red {
                    result_score
                } else {
                    -result_score
                };
                let fen_line = board_to_fen(&board);
                writeln!(out, "{} {}", fen_line, score).ok();
                pos_count += 1;

                // Parse and apply the move
                if let Some((fr, fc, tr, tc)) = parse_iccs_move(token) {
                    let moves = board.generate_legal_moves();
                    let matching: Vec<_> = moves
                        .iter()
                        .filter(|m| {
                            m.from_row == fr && m.from_col == fc && m.to_row == tr && m.to_col == tc
                        })
                        .collect();
                    if matching.len() == 1 {
                        board.make_move(*matching[0]);
                    } else {
                        // Try to find by from/to only (some moves have ambiguous piece)
                        let matching2: Vec<_> = moves
                            .iter()
                            .filter(|m| m.from_row == fr && m.from_col == fc && m.to_row == tr && m.to_col == tc)
                            .collect();
                        if matching2.len() == 1 {
                            board.make_move(*matching2[0]);
                        } else {
                            error_count += 1;
                            if error_count <= 5 {
                                eprintln!(
                                    "Warning: move {} at {}:{} not found ({} matches), fen={}",
                                    token, fr, fc, matching2.len(),
                                    board_to_fen(&board)
                                );
                            }
                            break; // Skip rest of this game
                        }
                    }
                } else {
                    error_count += 1;
                    if error_count <= 5 {
                        eprintln!("Warning: could not parse move '{}'", token);
                    }
                    break;
                }
            }
        }
    }

    // Record final position of last game
    if in_game {
        let stm = board.current_turn;
        let score = if stm == Side::Red {
            result_score
        } else {
            -result_score
        };
        let fen_line = board_to_fen(&board);
        writeln!(out, "{} {}", fen_line, score).ok();
        pos_count += 1;
    }

    println!(
        "Done. {} positions written to {} ({} parse errors)",
        pos_count, output_path, error_count
    );
}

/// Minimal FEN parser for standard initial position.
/// For custom starting positions in the PGN, this initializes the board from FEN.
fn board_from_fen(fen: &str) -> Board {
    let mut board = Board::new();

    // If it's the standard initial position FEN, Board::new() already set it up
    let standard = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR";
    if fen.trim() == standard || fen.trim().starts_with(standard) {
        return board;
    }

    // Parse non-standard FEN
    let fen_board = fen.split_whitespace().next().unwrap_or(fen);
    let rows: Vec<&str> = fen_board.split('/').collect();
    if rows.len() != 10 {
        eprintln!("Warning: unexpected FEN row count: {} in '{}'", rows.len(), fen);
        return board;
    }

    let pt_from_char = |ch: char| -> (PieceType, Side) {
        let side = if ch.is_uppercase() { Side::Red } else { Side::Black };
        let pt = match ch.to_ascii_lowercase() {
            'k' => PieceType::King,
            'a' => PieceType::Advisor,
            'b' => PieceType::Elephant,
            'h' => PieceType::Horse,
            'r' => PieceType::Chariot,
            'c' => PieceType::Cannon,
            'p' => PieceType::Pawn,
            _ => PieceType::Empty,
        };
        (pt, side)
    };

    for r in 0..10 {
        let mut c = 0usize;
        for ch in rows[r].chars() {
            if ch.is_ascii_digit() {
                let skip = ch.to_digit(10).unwrap() as usize;
                for i in 0..skip {
                    board.cells[r][c + i] = Piece::EMPTY;
                }
                c += skip;
            } else {
                let (pt, side) = pt_from_char(ch);
                board.cells[r][c] = Piece::new(pt, side);
                c += 1;
            }
        }
    }

    // Set side to move from FEN
    let parts: Vec<&str> = fen.split_whitespace().collect();
    if parts.len() >= 2 {
        board.current_turn = if parts[1] == "b" || parts[1] == "b" {
            Side::Black
        } else {
            Side::Red
        };
    }

    board
}
