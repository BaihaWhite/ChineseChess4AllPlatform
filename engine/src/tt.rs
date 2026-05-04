use crate::types::*;

pub struct TranspositionTable {
    hashes: Vec<u64>,
    depths: Vec<i16>,
    scores: Vec<i32>,
    flags: Vec<u8>,
    best_from: Vec<u16>,
    best_to: Vec<u16>,
    mask: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub depth: i16,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Option<Move>,
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let size = (mb * 1024 * 1024 / 32).next_power_of_two();
        let mask = (size - 1) as u64;
        TranspositionTable {
            hashes: vec![0; size],
            depths: vec![-1; size],
            scores: vec![0; size],
            flags: vec![0; size],
            best_from: vec![0xFFFF; size],
            best_to: vec![0xFFFF; size],
            mask,
        }
    }

    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let idx = (hash & self.mask) as usize;
        if self.hashes[idx] == hash && self.depths[idx] >= 0 {
            let best_move = if self.best_from[idx] != 0xFFFF {
                let fv = self.best_from[idx];
                let tv = self.best_to[idx];
                Some(Move::new(
                    (fv >> 8) as u8,
                    (fv & 0xFF) as u8,
                    (tv >> 8) as u8,
                    (tv & 0xFF) as u8,
                ))
            } else {
                None
            };
            let flag = match self.flags[idx] {
                1 => TTFlag::LowerBound,
                2 => TTFlag::UpperBound,
                _ => TTFlag::Exact,
            };
            Some(TTEntry {
                depth: self.depths[idx],
                score: self.scores[idx],
                flag,
                best_move,
            })
        } else {
            None
        }
    }

    pub fn store(&mut self, hash: u64, depth: i16, score: i32, flag: TTFlag, best_move: Option<Move>) {
        let idx = (hash & self.mask) as usize;
        if self.hashes[idx] != hash || depth >= self.depths[idx] {
            self.hashes[idx] = hash;
            self.depths[idx] = depth;
            self.scores[idx] = score;
            self.flags[idx] = match flag {
                TTFlag::LowerBound => 1,
                TTFlag::UpperBound => 2,
                TTFlag::Exact => 0,
            };
            if let Some(m) = best_move {
                self.best_from[idx] = ((m.from_row as u16) << 8) | m.from_col as u16;
                self.best_to[idx] = ((m.to_row as u16) << 8) | m.to_col as u16;
            } else {
                self.best_from[idx] = 0xFFFF;
            }
        }
    }

    pub fn clear(&mut self) {
        self.hashes.fill(0);
        self.depths.fill(-1);
        self.best_from.fill(0xFFFF);
    }
}
