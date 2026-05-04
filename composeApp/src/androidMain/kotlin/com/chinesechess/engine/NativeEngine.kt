package com.chinesechess.engine

object NativeEngine {
    init {
        System.loadLibrary("chess_engine")
    }

    fun search(
        board: Array<Array<Piece>>,
        turn: Int,
        depth: Int,
        timeMs: Int,
        history: LongArray
    ): Int {
        val boardBytes = ByteArray(90)
        for (r in 0..9) {
            for (c in 0..8) {
                val p = board[r][c]
                val v = when (p.type) {
                    PType.KING -> 1
                    PType.ADVISOR -> 2
                    PType.ELEPHANT -> 3
                    PType.HORSE -> 4
                    PType.CHARIOT -> 5
                    PType.CANNON -> 6
                    PType.PAWN -> 7
                    else -> 0
                }
                val side = when (p.side) {
                    PSide.RED -> 0x40
                    PSide.BLACK -> 0x80
                    else -> 0
                }
                boardBytes[r * 9 + c] = (v or side).toByte()
            }
        }
        return chessSearch(boardBytes, turn, depth, timeMs, history, history.size)
    }

    fun cancel() {
        chessCancel()
    }

    private external fun chessSearch(
        board: ByteArray,
        turn: Int,
        depth: Int,
        timeMs: Int,
        history: LongArray,
        historyLen: Int
    ): Int

    private external fun chessCancel()
}
