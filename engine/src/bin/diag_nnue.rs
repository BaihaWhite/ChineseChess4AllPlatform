use chess_engine::board::Board;
use chess_engine::nnue::{Nnue, NnueState};
use chess_engine::types::*;

fn main() {
    let mut nnue = Nnue::new();
    nnue.load_weights("/tmp/nnue_trained.bin").expect("load");

    let board = Board::new();
    let mut state = NnueState::new();
    state.refresh(&board, &nnue);

    let score_red = state.forward(&nnue, Side::Red);
    let score_black = state.forward(&nnue, Side::Black);
    println!("Initial: Red={} Black={}", score_red, score_black);

    // Red Rook+King vs Black Bare King
    let mut board2 = Board::empty();
    board2.cells[9][0] = Piece::new(PieceType::Chariot, Side::Red);
    board2.cells[0][0] = Piece::new(PieceType::King, Side::Black);
    board2.cells[9][4] = Piece::new(PieceType::King, Side::Red);
    board2.current_turn = Side::Red;
    board2.zobrist_hash = board2.compute_zobrist();
    let mut state2 = NnueState::new();
    state2.refresh(&board2, &nnue);
    let s2 = state2.forward(&nnue, Side::Red);
    println!("Red Rook+King vs Bare King: Red={}", s2);

    // Black Rook+King vs Red Bare King
    let mut board3 = Board::empty();
    board3.cells[9][0] = Piece::new(PieceType::Chariot, Side::Black);
    board3.cells[0][0] = Piece::new(PieceType::King, Side::Red);
    board3.cells[0][4] = Piece::new(PieceType::King, Side::Black);
    board3.current_turn = Side::Red;
    board3.zobrist_hash = board3.compute_zobrist();
    let mut state3 = NnueState::new();
    state3.refresh(&board3, &nnue);
    let s3 = state3.forward(&nnue, Side::Red);
    println!("Black Rook+King vs Bare King (Red view): Red={}", s3);

    println!("\nHCE comparison:");
    println!("  Initial: {}", chess_engine::evaluate::evaluate(&board));
    println!("  Red Rook: {}", chess_engine::evaluate::evaluate(&board2));
    println!("  Black Rook: {}", chess_engine::evaluate::evaluate(&board3));
}
