use crate::types::*;

thread_local! {
    static ZOBRIST_TABLE: [[[u64; 9]; 10]; 16] = {
        let mut rng = SplitMix64 { state: 20240101u64 };
        let mut table = [[[0u64; 9]; 10]; 16];
        for pt in 0..8usize {
            for side in 0..2usize {
                for r in 0..10usize {
                    for c in 0..9usize {
                        let idx = pt * 2 + side;
                        table[idx][r][c] = rng.next();
                    }
                }
            }
        }
        table
    };
}

pub const SIDE_TO_MOVE_KEY: u64 = {
    let mut rng = SplitMix64 { state: 20240102u64 };
    rng.next()
};

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

pub fn zobrist_piece(pt: PieceType, side: Side, row: u8, col: u8) -> u64 {
    let pt_idx = pt as usize;
    let side_idx = match side {
        Side::Red => 0usize,
        Side::Black => 1usize,
        _ => return 0,
    };
    ZOBRIST_TABLE.with(|table| table[pt_idx * 2 + side_idx][row as usize][col as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zobrist_deterministic() {
        let h1 = zobrist_piece(PieceType::King, Side::Red, 9, 4);
        let h2 = zobrist_piece(PieceType::King, Side::Red, 9, 4);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }
}
