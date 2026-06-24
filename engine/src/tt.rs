use crate::types::*;
use std::sync::atomic::{AtomicI16, AtomicI32, AtomicU16, AtomicU64, AtomicU8, Ordering};

pub struct TranspositionTable {
    hashes: Vec<AtomicU64>,
    depths: Vec<AtomicI16>,
    scores: Vec<AtomicI32>,
    flags: Vec<AtomicU8>,
    best_from: Vec<AtomicU16>,
    best_to: Vec<AtomicU16>,
    mask: u64,
}

// Atomic types are Send+Sync, so the whole struct is automatically Send+Sync.

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
            hashes: (0..size).map(|_| AtomicU64::new(0)).collect(),
            depths: (0..size).map(|_| AtomicI16::new(-1)).collect(),
            scores: (0..size).map(|_| AtomicI32::new(0)).collect(),
            flags: (0..size).map(|_| AtomicU8::new(0)).collect(),
            best_from: (0..size).map(|_| AtomicU16::new(0xFFFF)).collect(),
            best_to: (0..size).map(|_| AtomicU16::new(0xFFFF)).collect(),
            mask,
        }
    }

    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let idx = (hash & self.mask) as usize;
        let stored_hash = self.hashes[idx].load(Ordering::Acquire);
        if stored_hash != hash {
            return None;
        }
        let depth = self.depths[idx].load(Ordering::Relaxed);
        if depth < 0 {
            return None;
        }
        let score = self.scores[idx].load(Ordering::Relaxed);
        let flag_raw = self.flags[idx].load(Ordering::Relaxed);
        let from = self.best_from[idx].load(Ordering::Relaxed);
        let to = self.best_to[idx].load(Ordering::Relaxed);

        let best_move = if from != 0xFFFF {
            Some(Move::new(
                (from >> 8) as u8,
                (from & 0xFF) as u8,
                (to >> 8) as u8,
                (to & 0xFF) as u8,
            ))
        } else {
            None
        };
        let flag = match flag_raw {
            1 => TTFlag::LowerBound,
            2 => TTFlag::UpperBound,
            _ => TTFlag::Exact,
        };
        Some(TTEntry {
            depth,
            score,
            flag,
            best_move,
        })
    }

    pub fn store(&self, hash: u64, depth: i16, score: i32, flag: TTFlag, best_move: Option<Move>) {
        let idx = (hash & self.mask) as usize;

        // Depth-preferred replacement with CAS loop: only overwrite if this is
        // the same position (hash match) or the new entry is deeper.
        loop {
            let cur_hash = self.hashes[idx].load(Ordering::Acquire);
            let cur_depth = self.depths[idx].load(Ordering::Relaxed);
            if cur_hash == hash || depth >= cur_depth {
                // Try to claim the slot via CAS on depth.
                match self.depths[idx].compare_exchange_weak(
                    cur_depth,
                    depth,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        // We won the race — write remaining fields.
                        self.hashes[idx].store(hash, Ordering::Release);
                        self.scores[idx].store(score, Ordering::Relaxed);
                        self.flags[idx].store(
                            match flag {
                                TTFlag::LowerBound => 1,
                                TTFlag::UpperBound => 2,
                                TTFlag::Exact => 0,
                            },
                            Ordering::Relaxed,
                        );
                        if let Some(m) = best_move {
                            self.best_from[idx].store(
                                ((m.from_row as u16) << 8) | m.from_col as u16,
                                Ordering::Relaxed,
                            );
                            self.best_to[idx].store(
                                ((m.to_row as u16) << 8) | m.to_col as u16,
                                Ordering::Relaxed,
                            );
                        } else {
                            self.best_from[idx].store(0xFFFF, Ordering::Relaxed);
                        }
                        return;
                    }
                    Err(_) => {
                        // Another thread modified depth concurrently — retry
                        // the outer check (depth may have increased).
                        continue;
                    }
                }
            } else {
                // Existing entry is deeper for a different position — skip.
                return;
            }
        }
    }

    pub fn clear(&self) {
        for i in 0..self.hashes.len() {
            self.hashes[i].store(0, Ordering::Relaxed);
            self.depths[i].store(-1, Ordering::Relaxed);
            self.best_from[i].store(0xFFFF, Ordering::Relaxed);
        }
    }
}
