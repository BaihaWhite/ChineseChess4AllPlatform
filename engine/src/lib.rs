pub mod types;
pub mod zobrist;
pub mod board;
pub mod evaluate;
pub mod tt;
pub mod search;
pub mod ffi;

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::search::SearchEngine;
    use crate::types::*;

    #[test]
    fn test_initial_legal_moves() {
        let mut board = Board::new();
        let moves = board.generate_legal_moves();
        assert!(!moves.is_empty(), "Initial position should have legal moves");
    }

    #[test]
    fn test_search_returns_move() {
        let mut board = Board::new();
        let mut engine = SearchEngine::new();
        let result = engine.search(&mut board, 3, 5000, &[]);
        assert!(result.is_some(), "Search should return a move");
    }

    #[test]
    fn test_zobrist_consistency() {
        let mut board = Board::new();
        let h1 = board.zobrist_hash;
        let moves = board.generate_legal_moves();
        if let Some(&m) = moves.first() {
            let cap = board.make_move(m);
            let h2 = board.zobrist_hash;
            assert_ne!(h1, h2, "Hash should change after move");
            board.unmake_move(m, cap);
            assert_eq!(board.zobrist_hash, h1, "Hash should restore after unmake");
        }
    }
}
