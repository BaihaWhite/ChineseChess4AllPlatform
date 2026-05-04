#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceType {
    Empty = 0,
    King = 1,
    Advisor = 2,
    Elephant = 3,
    Horse = 4,
    Chariot = 5,
    Cannon = 6,
    Pawn = 7,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    None = 0,
    Red = 1,
    Black = 2,
}

impl Side {
    pub fn opponent(self) -> Side {
        match self {
            Side::Red => Side::Black,
            Side::Black => Side::Red,
            _ => Side::None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub piece_type: PieceType,
    pub side: Side,
}

impl Piece {
    pub const EMPTY: Piece = Piece {
        piece_type: PieceType::Empty,
        side: Side::None,
    };

    pub fn new(piece_type: PieceType, side: Side) -> Self {
        Piece { piece_type, side }
    }

    pub fn is_empty(self) -> bool {
        self.piece_type == PieceType::Empty
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from_row: u8,
    pub from_col: u8,
    pub to_row: u8,
    pub to_col: u8,
}

impl Move {
    pub fn new(from_row: u8, from_col: u8, to_row: u8, to_col: u8) -> Self {
        Move { from_row, from_col, to_row, to_col }
    }

    pub fn encode(self) -> u32 {
        ((self.from_row as u32) << 24)
            | ((self.from_col as u32) << 16)
            | ((self.to_row as u32) << 8)
            | (self.to_col as u32)
    }

    pub fn decode(v: u32) -> Self {
        Move {
            from_row: ((v >> 24) & 0xFF) as u8,
            from_col: ((v >> 16) & 0xFF) as u8,
            to_row: ((v >> 8) & 0xFF) as u8,
            to_col: (v & 0xFF) as u8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}
