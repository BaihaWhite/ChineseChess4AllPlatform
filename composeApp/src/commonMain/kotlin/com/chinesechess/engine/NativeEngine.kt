package com.chinesechess.engine

expect object NativeEngine {
    val isAvailable: Boolean
    val threadCount: Int
    val lastSearchDepth: Int
    val lastSearchNodes: Long
    fun search(board: Array<Array<Piece>>, turn: Int, depth: Int, timeMs: Int, history: LongArray, redChecks: Int, blackChecks: Int): Int
    fun cancel()
}
