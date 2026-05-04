package com.chinesechess.engine

enum class PType { EMPTY, KING, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, PAWN }
enum class PSide { RED, BLACK, NONE }
enum class GameMode { NONE, VS_AI, VS_HUMAN, VS_ONLINE }
enum class TTFlag { EXACT, LOWER_BOUND, UPPER_BOUND }

data class Piece(val type: PType = PType.EMPTY, val side: PSide = PSide.NONE) {
    val isEmpty: Boolean get() = type == PType.EMPTY
}

data class Move(val fromRow: Int, val fromCol: Int, val toRow: Int, val toCol: Int)

data class MoveRecord(val fromRow: Int, val fromCol: Int, val toRow: Int, val toCol: Int, val captured: Piece)

class TranspositionTable(size: Int) {
    private val capacity = size
    private val hashes = ULongArray(size)
    private val depths = IntArray(size) { -1 }
    private val scores = IntArray(size)
    private val flags = IntArray(size)
    private val bestFromRow = IntArray(size) { -1 }
    private val bestFromCol = IntArray(size) { -1 }
    private val bestToRow = IntArray(size) { -1 }
    private val bestToCol = IntArray(size) { -1 }
    private val mask = (size - 1).toULong()

    fun probe(hash: ULong): TTEntry? {
        val idx = (hash and mask).toInt()
        if (hashes[idx] == hash && depths[idx] >= 0) {
            val best = if (bestFromRow[idx] >= 0) Move(bestFromRow[idx], bestFromCol[idx], bestToRow[idx], bestToCol[idx]) else null
            return TTEntry(hash, depths[idx], scores[idx], when (flags[idx]) { 1 -> TTFlag.LOWER_BOUND; 2 -> TTFlag.UPPER_BOUND; else -> TTFlag.EXACT }, best)
        }
        return null
    }

    fun store(hash: ULong, depth: Int, score: Int, flag: TTFlag, bestMove: Move?) {
        val idx = (hash and mask).toInt()
        if (hashes[idx] != hash || depth >= depths[idx]) {
            hashes[idx] = hash
            depths[idx] = depth
            scores[idx] = score
            flags[idx] = when (flag) { TTFlag.LOWER_BOUND -> 1; TTFlag.UPPER_BOUND -> 2; else -> 0 }
            if (bestMove != null) {
                bestFromRow[idx] = bestMove.fromRow
                bestFromCol[idx] = bestMove.fromCol
                bestToRow[idx] = bestMove.toRow
                bestToCol[idx] = bestMove.toCol
            } else {
                bestFromRow[idx] = -1
            }
        }
    }

    fun clear() {
        hashes.fill(0UL)
        depths.fill(-1)
        bestFromRow.fill(-1)
    }
}

data class TTEntry(
    val hash: ULong = 0UL,
    val depth: Int = -1,
    val score: Int = 0,
    val flag: TTFlag = TTFlag.EXACT,
    val bestMove: Move? = null
)

object PieceNames {
    val RED = mapOf(
        PType.KING to "帅", PType.ADVISOR to "仕", PType.ELEPHANT to "相",
        PType.HORSE to "馬", PType.CHARIOT to "車", PType.CANNON to "炮", PType.PAWN to "兵"
    )
    val BLACK = mapOf(
        PType.KING to "将", PType.ADVISOR to "士", PType.ELEPHANT to "象",
        PType.HORSE to "馬", PType.CHARIOT to "車", PType.CANNON to "砲", PType.PAWN to "卒"
    )

    fun name(type: PType, side: PSide): String =
        if (side == PSide.RED) RED[type] ?: "" else BLACK[type] ?: ""
}
