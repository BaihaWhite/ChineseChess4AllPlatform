package com.chinesechess.engine

import kotlin.random.Random

class ChessEngine {
    val board = Array(10) { Array(9) { Piece() } }
    var currentTurn = PSide.RED
        internal set
    var gameOver = false
        internal set
    var drawGame = false
        private set

    fun setDrawGame(v: Boolean) { drawGame = v }
    var gameMode = GameMode.NONE
    var playerSide = PSide.RED
    var selRow = -1
        private set
    var selCol = -1
        private set
    var validMoves = emptyList<Move>()
        private set
    var userName = ""
    var aiDepth: Int = 3
    var lastAIMove: Move? = null
        internal set

    var consecutiveChecksRed = 0
        private set
    var consecutiveChecksBlack = 0
        private set

    var zobristHash: ULong = 0UL
        internal set

    companion object {
        val zobristTable: Array<Array<Array<ULongArray>>> = run {
            val rng = Random(20240101L)
            Array(8) {
                Array(2) {
                    Array(10) {
                        ULongArray(9) { rng.nextLong().toULong() }
                    }
                }
            }
        }
        val sideToMoveKey: ULong = Random(20240102L).nextLong().toULong()

        fun calculateZobrist(board: Array<Array<Piece>>, currentTurn: PSide): ULong {
            var h = 0UL
            for (r in 0..9) for (c in 0..8) {
                val p = board[r][c]
                if (p.type != PType.EMPTY) {
                    h = h xor zobristTable[p.type.ordinal][p.side.ordinal][r][c]
                }
            }
            if (currentTurn == PSide.BLACK) h = h xor sideToMoveKey
            return h
        }

        fun zobristPiece(type: PType, side: PSide, row: Int, col: Int): ULong =
            zobristTable[type.ordinal][side.ordinal][row][col]
    }

    fun computeZobrist(): ULong = calculateZobrist(board, currentTurn)

    fun recalcZobrist() {
        zobristHash = computeZobrist()
    }

    fun clone(): ChessEngine {
        val c = ChessEngine()
        for (r in 0..9) for (co in 0..8) c.board[r][co] = board[r][co].copy()
        c.currentTurn = currentTurn
        c.gameMode = gameMode
        c.playerSide = playerSide
        c.zobristHash = zobristHash
        c.positionHistory.addAll(positionHistory)
        c.consecutiveChecksRed = consecutiveChecksRed
        c.consecutiveChecksBlack = consecutiveChecksBlack
        return c
    }

    fun getPositionHistory(): List<ULong> = positionHistory.toList()

    fun executeMoveSimple(fx: Int, fy: Int, tx: Int, ty: Int): Piece {
        val mover = board[fx][fy]
        val captured = board[tx][ty]
        board[tx][ty] = mover
        board[fx][fy] = Piece()
        currentTurn = if (currentTurn == PSide.RED) PSide.BLACK else PSide.RED
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, fx, fy)
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, tx, ty)
        if (captured.type != PType.EMPTY)
            zobristHash = zobristHash xor zobristPiece(captured.type, captured.side, tx, ty)
        zobristHash = zobristHash xor sideToMoveKey
        return captured
    }

    fun undoMoveSimple(fx: Int, fy: Int, tx: Int, ty: Int, captured: Piece) {
        val mover = board[tx][ty]
        board[fx][fy] = mover
        board[tx][ty] = captured
        currentTurn = if (currentTurn == PSide.RED) PSide.BLACK else PSide.RED
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, fx, fy)
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, tx, ty)
        if (captured.type != PType.EMPTY)
            zobristHash = zobristHash xor zobristPiece(captured.type, captured.side, tx, ty)
        zobristHash = zobristHash xor sideToMoveKey
    }

    fun copyBoard(): Array<Array<Piece>> = Array(10) { r -> Array(9) { c -> board[r][c].copy() } }

    private val positionHistory = mutableListOf<ULong>()
    private val moveHistory = mutableListOf<MoveRecord>()
    private val checksRedHistory = mutableListOf<Int>()
    private val checksBlackHistory = mutableListOf<Int>()

    fun initBoard() {
        for (i in 0..9) for (j in 0..8) board[i][j] = Piece()

        val setP = { r: Int, c: Int, t: PType, s: PSide -> board[r][c] = Piece(t, s) }

        setP(9, 0, PType.CHARIOT, PSide.RED); setP(9, 1, PType.HORSE, PSide.RED)
        setP(9, 2, PType.ELEPHANT, PSide.RED); setP(9, 3, PType.ADVISOR, PSide.RED)
        setP(9, 4, PType.KING, PSide.RED); setP(9, 5, PType.ADVISOR, PSide.RED)
        setP(9, 6, PType.ELEPHANT, PSide.RED); setP(9, 7, PType.HORSE, PSide.RED)
        setP(9, 8, PType.CHARIOT, PSide.RED)
        setP(7, 1, PType.CANNON, PSide.RED); setP(7, 7, PType.CANNON, PSide.RED)
        for (j in 0..8 step 2) setP(6, j, PType.PAWN, PSide.RED)

        setP(0, 0, PType.CHARIOT, PSide.BLACK); setP(0, 1, PType.HORSE, PSide.BLACK)
        setP(0, 2, PType.ELEPHANT, PSide.BLACK); setP(0, 3, PType.ADVISOR, PSide.BLACK)
        setP(0, 4, PType.KING, PSide.BLACK); setP(0, 5, PType.ADVISOR, PSide.BLACK)
        setP(0, 6, PType.ELEPHANT, PSide.BLACK); setP(0, 7, PType.HORSE, PSide.BLACK)
        setP(0, 8, PType.CHARIOT, PSide.BLACK)
        setP(2, 1, PType.CANNON, PSide.BLACK); setP(2, 7, PType.CANNON, PSide.BLACK)
        for (j in 0..8 step 2) setP(3, j, PType.PAWN, PSide.BLACK)

        currentTurn = PSide.RED
        gameOver = false
        drawGame = false
        selRow = -1
        selCol = -1
        validMoves = emptyList()
        positionHistory.clear()
        moveHistory.clear()
        checksRedHistory.clear()
        checksBlackHistory.clear()
        consecutiveChecksRed = 0
        consecutiveChecksBlack = 0
        zobristHash = computeZobrist()
    }

    fun inBoard(r: Int, c: Int): Boolean = r in 0..9 && c in 0..8

    fun inPalace(r: Int, c: Int, s: PSide): Boolean {
        if (c !in 3..5) return false
        return if (s == PSide.RED) r in 7..9 else r in 0..2
    }

    fun inOwnHalf(r: Int, s: PSide): Boolean = if (s == PSide.RED) r >= 5 else r <= 4

    fun countBetween(fx: Int, fy: Int, tx: Int, ty: Int): Int {
        var cnt = 0
        val dx = when { tx > fx -> 1; tx < fx -> -1; else -> 0 }
        val dy = when { ty > fy -> 1; ty < fy -> -1; else -> 0 }
        var cx = fx + dx
        var cy = fy + dy
        while (cx != tx || cy != ty) {
            if (board[cx][cy].type != PType.EMPTY) cnt++
            cx += dx; cy += dy
        }
        return cnt
    }

    fun isValidMove(fx: Int, fy: Int, tx: Int, ty: Int): Boolean {
        if (!inBoard(tx, ty)) return false
        if (fx == tx && fy == ty) return false
        if (board[fx][fy].side != currentTurn) return false
        if (board[tx][ty].side == currentTurn) return false

        val pc = board[fx][fy]
        val dx = tx - fx
        val dy = ty - fy
        val adx = kotlin.math.abs(dx)
        val ady = kotlin.math.abs(dy)

        return when (pc.type) {
            PType.KING -> {
                (adx == 1 && ady == 0 || adx == 0 && ady == 1) && inPalace(tx, ty, pc.side)
            }
            PType.ADVISOR -> {
                adx == 1 && ady == 1 && inPalace(tx, ty, pc.side)
            }
            PType.ELEPHANT -> {
                adx == 2 && ady == 2 && inOwnHalf(tx, pc.side) &&
                    board[fx + dx / 2][fy + dy / 2].type == PType.EMPTY
            }
            PType.HORSE -> {
                (adx == 2 && ady == 1 || adx == 1 && ady == 2) &&
                    if (adx == 2) board[fx + dx / 2][fy].type == PType.EMPTY
                    else board[fx][fy + dy / 2].type == PType.EMPTY
            }
            PType.CHARIOT -> {
                (dx == 0 || dy == 0) && countBetween(fx, fy, tx, ty) == 0
            }
            PType.CANNON -> {
                (dx == 0 || dy == 0) &&
                    if (board[tx][ty].type == PType.EMPTY) countBetween(fx, fy, tx, ty) == 0
                    else countBetween(fx, fy, tx, ty) == 1
            }
            PType.PAWN -> {
                if (pc.side == PSide.RED) {
                    if (inOwnHalf(fx, pc.side)) dx == -1 && dy == 0
                    else dx != 1 && (adx + ady == 1) && (dx == -1 || dx == 0 && ady == 1)
                } else {
                    if (inOwnHalf(fx, pc.side)) dx == 1 && dy == 0
                    else dx != -1 && (adx + ady == 1) && (dx == 1 || dx == 0 && ady == 1)
                }
            }
            PType.EMPTY -> false
        }
    }

    fun kingsAreFacing(): Boolean {
        var rx = -1; var ry = -1; var bx = -1; var by = -1
        for (i in 0..9) for (j in 0..8) {
            if (board[i][j].type == PType.KING) {
                if (board[i][j].side == PSide.RED) { rx = i; ry = j }
                else { bx = i; by = j }
            }
        }
        if (ry != by) return false
        for (i in minOf(rx, bx) + 1 until maxOf(rx, bx)) {
            if (board[i][ry].type != PType.EMPTY) return false
        }
        return true
    }

    fun isInCheck(s: PSide): Boolean {
        var kx = -1; var ky = -1
        for (i in 0..9) for (j in 0..8) {
            if (board[i][j].type == PType.KING && board[i][j].side == s) { kx = i; ky = j }
        }
        if (kx == -1) return true
        val opp = if (s == PSide.RED) PSide.BLACK else PSide.RED
        val saved = currentTurn
        currentTurn = opp
        for (i in 0..9) for (j in 0..8) {
            if (board[i][j].side == opp && isValidMove(i, j, kx, ky)) {
                currentTurn = saved
                return true
            }
        }
        currentTurn = saved
        return false
    }

    fun wouldBeInCheck(fx: Int, fy: Int, tx: Int, ty: Int, s: PSide): Boolean {
        val cap = board[tx][ty]
        val mov = board[fx][fy]
        board[tx][ty] = mov
        board[fx][fy] = Piece()
        val ck = isInCheck(s)
        board[fx][fy] = mov
        board[tx][ty] = cap
        return ck
    }

    fun wouldKingsFace(fx: Int, fy: Int, tx: Int, ty: Int): Boolean {
        val cap = board[tx][ty]
        val mov = board[fx][fy]
        board[tx][ty] = mov
        board[fx][fy] = Piece()
        val kf = kingsAreFacing()
        board[fx][fy] = mov
        board[tx][ty] = cap
        return kf
    }

    private fun wouldRepeat(fx: Int, fy: Int, tx: Int, ty: Int): Boolean {
        val mover = board[fx][fy]
        val captured = board[tx][ty]
        var newHash = zobristHash
        newHash = newHash xor zobristPiece(mover.type, mover.side, fx, fy)
        newHash = newHash xor zobristPiece(mover.type, mover.side, tx, ty)
        if (captured.type != PType.EMPTY)
            newHash = newHash xor zobristPiece(captured.type, captured.side, tx, ty)
        newHash = newHash xor sideToMoveKey

        var count = 0
        for (h in positionHistory) {
            if (h == newHash) count++
        }
        return count >= 2
    }

    private fun wouldGiveCheck(fx: Int, fy: Int, tx: Int, ty: Int): Boolean {
        val mover = board[fx][fy]
        val captured = board[tx][ty]
        val opp = if (mover.side == PSide.RED) PSide.BLACK else PSide.RED
        board[tx][ty] = mover
        board[fx][fy] = Piece()
        val result = isInCheck(opp)
        board[fx][fy] = mover
        board[tx][ty] = captured
        return result
    }

    fun isLegalMove(fx: Int, fy: Int, tx: Int, ty: Int): Boolean {
        if (!isValidMove(fx, fy, tx, ty)) return false
        if (wouldBeInCheck(fx, fy, tx, ty, board[fx][fy].side)) return false
        if (wouldKingsFace(fx, fy, tx, ty)) return false
        if (wouldRepeat(fx, fy, tx, ty)) return false
        val side = board[fx][fy].side
        if (wouldGiveCheck(fx, fy, tx, ty)) {
            val checks = if (side == PSide.RED) consecutiveChecksRed else consecutiveChecksBlack
            if (checks >= 2) return false
        }
        return true
    }

    fun legalMovesForCell(row: Int, col: Int): List<Move> {
        val saved = currentTurn
        currentTurn = board[row][col].side
        val moves = buildList {
            for (ti in 0..9) for (tj in 0..8) {
                if (isLegalMove(row, col, ti, tj)) add(Move(row, col, ti, tj))
            }
        }
        currentTurn = saved
        return moves
    }

    fun getAllLegalMoves(s: PSide): List<Move> {
        val moves = mutableListOf<Move>()
        val saved = currentTurn
        currentTurn = s
        for (i in 0..9) for (j in 0..8) {
            if (board[i][j].side == s) {
                for (ti in 0..9) for (tj in 0..8) {
                    if (isLegalMove(i, j, ti, tj)) {
                        moves.add(Move(i, j, ti, tj))
                    }
                }
            }
        }
        currentTurn = saved
        return moves
    }

    var onMoveExecuted: ((Move) -> Unit)? = null

    fun executeMove(fx: Int, fy: Int, tx: Int, ty: Int) {
        val moverSide = board[fx][fy].side
        moveHistory.add(MoveRecord(fx, fy, tx, ty, board[tx][ty]))
        positionHistory.add(zobristHash)
        checksRedHistory.add(consecutiveChecksRed)
        checksBlackHistory.add(consecutiveChecksBlack)

        board[tx][ty] = board[fx][fy]
        board[fx][fy] = Piece()
        currentTurn = if (currentTurn == PSide.RED) PSide.BLACK else PSide.RED
        val mover = board[tx][ty]
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, fx, fy)
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, tx, ty)
        val cap = moveHistory.last().captured
        if (cap.type != PType.EMPTY)
            zobristHash = zobristHash xor zobristPiece(cap.type, cap.side, tx, ty)
        zobristHash = zobristHash xor sideToMoveKey

        if (isInCheck(currentTurn)) {
            if (moverSide == PSide.RED) consecutiveChecksRed++ else consecutiveChecksBlack++
        } else {
            if (moverSide == PSide.RED) consecutiveChecksRed = 0 else consecutiveChecksBlack = 0
        }

        if (getAllLegalMoves(currentTurn).isEmpty()) {
            gameOver = true
            drawGame = false
        }

        onMoveExecuted?.invoke(Move(fx, fy, tx, ty))
    }

    fun canUndo(): Boolean = moveHistory.isNotEmpty()

    fun undoMove(): Boolean {
        if (moveHistory.isEmpty()) return false
        val last = moveHistory.removeLast()
        val mover = board[last.toRow][last.toCol]
        board[last.fromRow][last.fromCol] = mover
        board[last.toRow][last.toCol] = last.captured
        currentTurn = if (currentTurn == PSide.RED) PSide.BLACK else PSide.RED
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, last.fromRow, last.fromCol)
        zobristHash = zobristHash xor zobristPiece(mover.type, mover.side, last.toRow, last.toCol)
        if (last.captured.type != PType.EMPTY)
            zobristHash = zobristHash xor zobristPiece(last.captured.type, last.captured.side, last.toRow, last.toCol)
        zobristHash = zobristHash xor sideToMoveKey
        gameOver = false
        drawGame = false
        positionHistory.removeLastOrNull()
        consecutiveChecksRed = checksRedHistory.removeLastOrNull() ?: 0
        consecutiveChecksBlack = checksBlackHistory.removeLastOrNull() ?: 0
        selRow = -1
        selCol = -1
        validMoves = emptyList()
        return true
    }

    fun winnerText(): String {
        if (drawGame) return "平局！"
        if (!gameOver) return ""
        return if (currentTurn == PSide.RED) "黑方胜！" else "红方胜！"
    }

    fun clickCell(row: Int, col: Int): Int {
        if (gameOver) return 0

        if (selRow < 0) {
            if (board[row][col].side == currentTurn) {
                selRow = row
                selCol = col
                validMoves = buildList {
                    for (ti in 0..9) for (tj in 0..8) {
                        if (isLegalMove(row, col, ti, tj)) add(Move(row, col, ti, tj))
                    }
                }
                return 1
            }
            return 0
        }

        for (m in validMoves) {
            if (m.toRow == row && m.toCol == col) {
                executeMove(m.fromRow, m.fromCol, m.toRow, m.toCol)
                selRow = -1
                selCol = -1
                validMoves = emptyList()
                return 2
            }
        }

        if (board[row][col].side == currentTurn) {
            selRow = row
            selCol = col
            validMoves = buildList {
                for (ti in 0..9) for (tj in 0..8) {
                    if (isLegalMove(row, col, ti, tj)) add(Move(row, col, ti, tj))
                }
            }
            return 1
        }

        selRow = -1
        selCol = -1
        validMoves = emptyList()
        return 0
    }

    fun isAiTurn(): Boolean {
        if (gameMode != GameMode.VS_AI || gameOver) return false
        val aiSide = if (playerSide == PSide.RED) PSide.BLACK else PSide.RED
        return currentTurn == aiSide
    }

    fun resetGame() {
        initBoard()
        lastAIMove = null
    }
}
