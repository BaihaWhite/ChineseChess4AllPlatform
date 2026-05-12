use crate::types::*;
use crate::zobrist;

pub struct Board {
    pub cells: [[Piece; 9]; 10],
    pub current_turn: Side,
    pub zobrist_hash: u64,
    pub position_history: Vec<u64>,
    pub consecutive_checks: [u8; 2],
    checks_history: Vec<[u8; 2]>,
}

impl Board {
    pub fn new() -> Self {
        let mut b = Board {
            cells: [[Piece::EMPTY; 9]; 10],
            current_turn: Side::Red,
            zobrist_hash: 0,
            position_history: Vec::new(),
            consecutive_checks: [0, 0],
            checks_history: Vec::new(),
        };
        b.init();
        b
    }

    pub fn init(&mut self) {
        for r in 0..10 {
            for c in 0..9 {
                self.cells[r][c] = Piece::EMPTY;
            }
        }

        let set = |b: &mut Board, r: usize, c: usize, pt: PieceType, s: Side| {
            b.cells[r][c] = Piece::new(pt, s);
        };

        set(self, 9, 0, PieceType::Chariot, Side::Red);
        set(self, 9, 1, PieceType::Horse, Side::Red);
        set(self, 9, 2, PieceType::Elephant, Side::Red);
        set(self, 9, 3, PieceType::Advisor, Side::Red);
        set(self, 9, 4, PieceType::King, Side::Red);
        set(self, 9, 5, PieceType::Advisor, Side::Red);
        set(self, 9, 6, PieceType::Elephant, Side::Red);
        set(self, 9, 7, PieceType::Horse, Side::Red);
        set(self, 9, 8, PieceType::Chariot, Side::Red);
        set(self, 7, 1, PieceType::Cannon, Side::Red);
        set(self, 7, 7, PieceType::Cannon, Side::Red);
        for c in (0..9).step_by(2) {
            set(self, 6, c, PieceType::Pawn, Side::Red);
        }

        set(self, 0, 0, PieceType::Chariot, Side::Black);
        set(self, 0, 1, PieceType::Horse, Side::Black);
        set(self, 0, 2, PieceType::Elephant, Side::Black);
        set(self, 0, 3, PieceType::Advisor, Side::Black);
        set(self, 0, 4, PieceType::King, Side::Black);
        set(self, 0, 5, PieceType::Advisor, Side::Black);
        set(self, 0, 6, PieceType::Elephant, Side::Black);
        set(self, 0, 7, PieceType::Horse, Side::Black);
        set(self, 0, 8, PieceType::Chariot, Side::Black);
        set(self, 2, 1, PieceType::Cannon, Side::Black);
        set(self, 2, 7, PieceType::Cannon, Side::Black);
        for c in (0..9).step_by(2) {
            set(self, 3, c, PieceType::Pawn, Side::Black);
        }

        self.current_turn = Side::Red;
        self.position_history.clear();
        self.consecutive_checks = [0, 0];
        self.checks_history.clear();
        self.zobrist_hash = self.compute_zobrist();
    }

    #[inline]
    pub fn in_board(r: i8, c: i8) -> bool {
        r >= 0 && r < 10 && c >= 0 && c < 9
    }

    #[inline]
    pub fn in_palace(r: i8, c: i8, side: Side) -> bool {
        if !(3..=5).contains(&c) {
            return false;
        }
        match side {
            Side::Red => (7..=9).contains(&r),
            Side::Black => (0..=2).contains(&r),
            _ => false,
        }
    }

    #[inline]
    pub fn in_own_half(r: i8, side: Side) -> bool {
        match side {
            Side::Red => r >= 5,
            Side::Black => r <= 4,
            _ => false,
        }
    }

    pub fn count_between(&self, fx: i8, fy: i8, tx: i8, ty: i8) -> i8 {
        let mut cnt = 0i8;
        let dx = (tx - fx).signum();
        let dy = (ty - fy).signum();
        let mut cx = fx + dx;
        let mut cy = fy + dy;
        while cx != tx || cy != ty {
            if !self.cells[cx as usize][cy as usize].is_empty() {
                cnt += 1;
            }
            cx += dx;
            cy += dy;
        }
        cnt
    }

    pub fn is_valid_move(&self, fx: i8, fy: i8, tx: i8, ty: i8) -> bool {
        if !Self::in_board(tx, ty) {
            return false;
        }
        if fx == tx && fy == ty {
            return false;
        }
        let mover = self.cells[fx as usize][fy as usize];
        if mover.side != self.current_turn {
            return false;
        }
        let target = self.cells[tx as usize][ty as usize];
        if target.side == self.current_turn {
            return false;
        }

        let dx = tx - fx;
        let dy = ty - fy;
        let adx = dx.abs();
        let ady = dy.abs();

        match mover.piece_type {
            PieceType::King => {
                ((adx == 1 && ady == 0) || (adx == 0 && ady == 1))
                    && Self::in_palace(tx, ty, mover.side)
            }
            PieceType::Advisor => {
                adx == 1 && ady == 1 && Self::in_palace(tx, ty, mover.side)
            }
            PieceType::Elephant => {
                adx == 2
                    && ady == 2
                    && Self::in_own_half(tx, mover.side)
                    && self.cells[(fx + dx / 2) as usize][(fy + dy / 2) as usize].is_empty()
            }
            PieceType::Horse => {
                (adx == 2 && ady == 1 || adx == 1 && ady == 2)
                    && if adx == 2 {
                        self.cells[(fx + dx / 2) as usize][fy as usize].is_empty()
                    } else {
                        self.cells[fx as usize][(fy + dy / 2) as usize].is_empty()
                    }
            }
            PieceType::Chariot => (dx == 0 || dy == 0) && self.count_between(fx, fy, tx, ty) == 0,
            PieceType::Cannon => {
                (dx == 0 || dy == 0)
                    && if target.is_empty() {
                        self.count_between(fx, fy, tx, ty) == 0
                    } else {
                        self.count_between(fx, fy, tx, ty) == 1
                    }
            }
            PieceType::Pawn => match mover.side {
                Side::Red => {
                    if Self::in_own_half(fx, mover.side) {
                        dx == -1 && dy == 0
                    } else {
                        dx != 1 && (adx + ady == 1) && (dx == -1 || (dx == 0 && ady == 1))
                    }
                }
                Side::Black => {
                    if Self::in_own_half(fx, mover.side) {
                        dx == 1 && dy == 0
                    } else {
                        dx != -1 && (adx + ady == 1) && (dx == 1 || (dx == 0 && ady == 1))
                    }
                }
                _ => false,
            },
            PieceType::Empty => false,
        }
    }

    pub fn find_king(&self, side: Side) -> Option<(i8, i8)> {
        for r in 0..10i8 {
            for c in 0..9i8 {
                let p = self.cells[r as usize][c as usize];
                if p.piece_type == PieceType::King && p.side == side {
                    return Some((r, c));
                }
            }
        }
        None
    }

    pub fn kings_are_facing(&self) -> bool {
        let rk = self.find_king(Side::Red);
        let bk = self.find_king(Side::Black);
        match (rk, bk) {
            (Some((rx, ry)), Some((bx, by))) => {
                if ry != by {
                    return false;
                }
                let lo = rx.min(bx) + 1;
                let hi = rx.max(bx);
                for i in lo..hi {
                    if !self.cells[i as usize][ry as usize].is_empty() {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn is_in_check(&mut self, side: Side) -> bool {
        let king = match self.find_king(side) {
            Some(k) => k,
            None => return true,
        };
        let opp = side.opponent();
        let saved = self.current_turn;
        self.current_turn = opp;
        for r in 0..10i8 {
            for c in 0..9i8 {
                if self.cells[r as usize][c as usize].side == opp
                    && self.is_valid_move(r, c, king.0, king.1)
                {
                    self.current_turn = saved;
                    return true;
                }
            }
        }
        self.current_turn = saved;
        false
    }

    pub fn would_be_in_check(&mut self, fx: i8, fy: i8, tx: i8, ty: i8, side: Side) -> bool {
        let captured = self.cells[tx as usize][ty as usize];
        let mover = self.cells[fx as usize][fy as usize];
        self.cells[tx as usize][ty as usize] = mover;
        self.cells[fx as usize][fy as usize] = Piece::EMPTY;
        let check = self.is_in_check(side);
        self.cells[fx as usize][fy as usize] = mover;
        self.cells[tx as usize][ty as usize] = captured;
        check
    }

    pub fn would_kings_face(&self, fx: i8, fy: i8, tx: i8, ty: i8) -> bool {
        let mover = self.cells[fx as usize][fy as usize];
        let mut b = self.clone();
        b.cells[tx as usize][ty as usize] = mover;
        b.cells[fx as usize][fy as usize] = Piece::EMPTY;
        b.kings_are_facing()
    }

    pub fn would_repeat(&self, fx: i8, fy: i8, tx: i8, ty: i8, gives_check: bool) -> bool {
        let mover = self.cells[fx as usize][fy as usize];
        let captured = self.cells[tx as usize][ty as usize];
        let mut new_hash = self.zobrist_hash;
        new_hash ^= zobrist::zobrist_piece(mover.piece_type, mover.side, fx as u8, fy as u8);
        new_hash ^= zobrist::zobrist_piece(mover.piece_type, mover.side, tx as u8, ty as u8);
        if !captured.is_empty() {
            new_hash ^= zobrist::zobrist_piece(captured.piece_type, captured.side, tx as u8, ty as u8);
        }
        new_hash ^= zobrist::SIDE_TO_MOVE_KEY;

        let count = self.position_history.iter().filter(|&&h| h == new_hash).count();
        let threshold: usize = if gives_check { 1 } else { 2 };
        count >= threshold
    }

    fn would_give_check(&mut self, fx: i8, fy: i8, tx: i8, ty: i8) -> bool {
        let mover = self.cells[fx as usize][fy as usize];
        let captured = self.cells[tx as usize][ty as usize];
        let opp = mover.side.opponent();
        self.cells[tx as usize][ty as usize] = mover;
        self.cells[fx as usize][fy as usize] = Piece::EMPTY;
        let result = self.is_in_check(opp);
        self.cells[fx as usize][fy as usize] = mover;
        self.cells[tx as usize][ty as usize] = captured;
        result
    }

    pub fn is_legal_move(&mut self, fx: i8, fy: i8, tx: i8, ty: i8) -> bool {
        if !self.is_valid_move(fx, fy, tx, ty) {
            return false;
        }
        let mover = self.cells[fx as usize][fy as usize];
        if self.would_be_in_check(fx, fy, tx, ty, mover.side) {
            return false;
        }
        if self.would_kings_face(fx, fy, tx, ty) {
            return false;
        }
        let gives_check = self.would_give_check(fx, fy, tx, ty);
        if self.would_repeat(fx, fy, tx, ty, gives_check) {
            return false;
        }
        if gives_check {
            let idx = side_index(mover.side);
            if self.consecutive_checks[idx] >= 2 {
                return false;
            }
        }
        true
    }

    pub fn generate_legal_moves(&mut self) -> Vec<Move> {
        let side = self.current_turn;
        let mut moves = Vec::new();
        for r in 0..10i8 {
            for c in 0..9i8 {
                if self.cells[r as usize][c as usize].side == side {
                    for tr in 0..10i8 {
                        for tc in 0..9i8 {
                            if self.is_legal_move(r, c, tr, tc) {
                                moves.push(Move::new(r as u8, c as u8, tr as u8, tc as u8));
                            }
                        }
                    }
                }
            }
        }
        moves
    }

    pub fn make_move(&mut self, m: Move) -> Piece {
        let fx = m.from_row as usize;
        let fy = m.from_col as usize;
        let tx = m.to_row as usize;
        let ty = m.to_col as usize;

        self.position_history.push(self.zobrist_hash);
        self.checks_history.push(self.consecutive_checks);

        let mover = self.cells[fx][fy];
        let captured = self.cells[tx][ty];

        self.zobrist_hash ^= zobrist::zobrist_piece(mover.piece_type, mover.side, m.from_row, m.from_col);
        self.zobrist_hash ^= zobrist::zobrist_piece(mover.piece_type, mover.side, m.to_row, m.to_col);
        if !captured.is_empty() {
            self.zobrist_hash ^= zobrist::zobrist_piece(captured.piece_type, captured.side, m.to_row, m.to_col);
        }
        self.zobrist_hash ^= zobrist::SIDE_TO_MOVE_KEY;

        self.cells[tx][ty] = mover;
        self.cells[fx][fy] = Piece::EMPTY;
        let mover_side = self.current_turn;
        self.current_turn = self.current_turn.opponent();

        let gives_check = self.is_in_check(self.current_turn);
        let idx = side_index(mover_side);
        if gives_check {
            self.consecutive_checks[idx] += 1;
        } else {
            self.consecutive_checks[idx] = 0;
        }

        captured
    }

    pub fn unmake_move(&mut self, m: Move, captured: Piece) {
        let fx = m.from_row as usize;
        let fy = m.from_col as usize;
        let tx = m.to_row as usize;
        let ty = m.to_col as usize;

        let mover = self.cells[tx][ty];

        self.zobrist_hash ^= zobrist::zobrist_piece(mover.piece_type, mover.side, m.from_row, m.from_col);
        self.zobrist_hash ^= zobrist::zobrist_piece(mover.piece_type, mover.side, m.to_row, m.to_col);
        if !captured.is_empty() {
            self.zobrist_hash ^= zobrist::zobrist_piece(captured.piece_type, captured.side, m.to_row, m.to_col);
        }
        self.zobrist_hash ^= zobrist::SIDE_TO_MOVE_KEY;

        self.cells[fx][fy] = mover;
        self.cells[tx][ty] = captured;
        self.current_turn = self.current_turn.opponent();

        self.position_history.pop();
        self.consecutive_checks = self.checks_history.pop().unwrap_or([0, 0]);
    }

    pub fn compute_zobrist(&self) -> u64 {
        let mut h = 0u64;
        for r in 0..10u8 {
            for c in 0..9u8 {
                let p = self.cells[r as usize][c as usize];
                if !p.is_empty() {
                    h ^= zobrist::zobrist_piece(p.piece_type, p.side, r, c);
                }
            }
        }
        if self.current_turn == Side::Black {
            h ^= zobrist::SIDE_TO_MOVE_KEY;
        }
        h
    }

    pub fn from_bytes(data: &[u8; 90]) -> Self {
        let mut b = Board {
            cells: [[Piece::EMPTY; 9]; 10],
            current_turn: Side::Red,
            zobrist_hash: 0,
            position_history: Vec::new(),
            consecutive_checks: [0, 0],
            checks_history: Vec::new(),
        };
        for r in 0..10 {
            for c in 0..9 {
                let v = data[r * 9 + c];
                b.cells[r][c] = decode_piece(v);
            }
        }
        b.zobrist_hash = b.compute_zobrist();
        b
    }
}

#[inline]
fn side_index(side: Side) -> usize {
    match side {
        Side::Red => 0,
        Side::Black => 1,
        _ => 0,
    }
}

fn decode_piece(v: u8) -> Piece {
    let side = if v & 0x80 != 0 {
        Side::Black
    } else if v & 0x40 != 0 {
        Side::Red
    } else {
        Side::None
    };
    let pt = match v & 0x0F {
        1 => PieceType::King,
        2 => PieceType::Advisor,
        3 => PieceType::Elephant,
        4 => PieceType::Horse,
        5 => PieceType::Chariot,
        6 => PieceType::Cannon,
        7 => PieceType::Pawn,
        _ => PieceType::Empty,
    };
    Piece::new(pt, side)
}

impl Clone for Board {
    fn clone(&self) -> Self {
        Board {
            cells: self.cells,
            current_turn: self.current_turn,
            zobrist_hash: self.zobrist_hash,
            position_history: self.position_history.clone(),
            consecutive_checks: self.consecutive_checks,
            checks_history: self.checks_history.clone(),
        }
    }
}
