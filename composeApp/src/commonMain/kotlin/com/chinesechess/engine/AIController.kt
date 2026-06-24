package com.chinesechess.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.random.Random

class AIController(private val engine: ChessEngine, private val openingBook: OpeningBook? = null) {

    @Volatile
    var cancelled = false
        private set

    fun cancel() {
        cancelled = true
        if (NativeEngine.isAvailable) {
            NativeEngine.cancel()
        }
    }

    suspend fun getAIMove(
        searchDepth: Int = 3,
        bookRandomChance: Float = 0f,
        userTimeLimit: Long = 0L,
        randomMoveChance: Float = 0f
    ): Move? = withContext(Dispatchers.Default) {
        cancelled = false

        val aiSide = if (engine.playerSide == PSide.RED) PSide.BLACK else PSide.RED
        val allMoves = engine.getAllLegalMoves(aiSide)
        if (allMoves.isEmpty()) return@withContext null

        if (randomMoveChance > 0f && Random.nextFloat() < randomMoveChance) {
            DebugLog.debug("AI", "Random move")
            return@withContext allMoves.random()
        }

        openingBook?.let { book ->
            val bookMoves = book.probeWeighted(engine.zobristHash)
            if (bookMoves.isNotEmpty()) {
                val validBookMoves = bookMoves.mapNotNull { (move, _) ->
                    if (allMoves.any { it.fromRow == move.fromRow && it.fromCol == move.fromCol && it.toRow == move.toRow && it.toCol == move.toCol }) move
                    else null
                }
                if (validBookMoves.isNotEmpty()) {
                    DebugLog.info("Book", "${validBookMoves.size} moves")
                    if (bookRandomChance > 0f && Random.nextFloat() < bookRandomChance) {
                        DebugLog.debug("Book", "Noise override")
                        return@withContext allMoves.random()
                    }
                    return@withContext validBookMoves.random()
                }
            }
        }

        val maxDepth = searchDepth.coerceAtLeast(1)
        val timeLimit = if (userTimeLimit > 0L) userTimeLimit else when {
            searchDepth <= 3 -> 3000L
            searchDepth <= 6 -> 10000L
            else -> 30000L
        }

        if (!NativeEngine.isAvailable) {
            DebugLog.error("AI", "Native engine unavailable")
            return@withContext allMoves.firstOrNull()
        }

        val turn = if (aiSide == PSide.RED) 1 else 2
        val history = engine.getPositionHistory().map { it.toLong() }.toLongArray()

        val t0 = System.nanoTime()
        val rawResult = NativeEngine.search(engine.board, turn, maxDepth, timeLimit.toInt(), history, engine.consecutiveChecksRed, engine.consecutiveChecksBlack)
        val elapsed = (System.nanoTime() - t0) / 1_000_000
        val actualDepth = NativeEngine.lastSearchDepth
        val nodes = NativeEngine.lastSearchNodes

        val threads = NativeEngine.threadCount
        DebugLog.info("AI", "Depth=$actualDepth Nodes=${nodes / 1000}K ${elapsed}ms ${threads}T")

        if (rawResult != 0) {
            val fromRow = (rawResult shr 24) and 0xFF
            val fromCol = (rawResult shr 16) and 0xFF
            val toRow = (rawResult shr 8) and 0xFF
            val toCol = rawResult and 0xFF
            val matchedMove = allMoves.find {
                it.fromRow == fromRow && it.fromCol == fromCol && it.toRow == toRow && it.toCol == toCol
            }
            if (matchedMove != null) {
                return@withContext matchedMove
            }
        }

        DebugLog.warn("AI", "Engine returned no valid move, picking first")
        allMoves.firstOrNull()
    }
}
