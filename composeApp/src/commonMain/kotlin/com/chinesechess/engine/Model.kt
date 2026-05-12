package com.chinesechess.engine

enum class PType { EMPTY, KING, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, PAWN }
enum class PSide { RED, BLACK, NONE }
enum class GameMode { NONE, VS_AI, VS_HUMAN, VS_ONLINE }

data class Piece(val type: PType = PType.EMPTY, val side: PSide = PSide.NONE) {
    val isEmpty: Boolean get() = type == PType.EMPTY
}

data class Move(val fromRow: Int, val fromCol: Int, val toRow: Int, val toCol: Int)

data class MoveRecord(val fromRow: Int, val fromCol: Int, val toRow: Int, val toCol: Int, val captured: Piece)

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
