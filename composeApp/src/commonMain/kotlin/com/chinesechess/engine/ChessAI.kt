package com.chinesechess.engine

import com.chinesechess.engine.DebugLog
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.Dispatchers
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.time.TimeSource

class ChessAI(private val engine: ChessEngine, private val openingBook: OpeningBook? = null) {

    private val baseValues = intArrayOf(0, 10000, 180, 220, 400, 900, 450, 100)

    private val redPawnPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        170, 160, 150, 140, 140, 150, 160, 170, 0,
        100, 95, 85, 70, 70, 85, 95, 100, 0,
        50, 45, 35, 20, 20, 35, 45, 50, 0,
        20, 15, 10, 0, 0, 10, 15, 20, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    )

    private val blackPawnPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        20, 15, 10, 0, 0, 10, 15, 20, 0,
        50, 45, 35, 20, 20, 35, 45, 50, 0,
        100, 95, 85, 70, 70, 85, 95, 100, 0,
        170, 160, 150, 140, 140, 150, 160, 170, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    )

    private val knightPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -10, 0, 0, 0, 0, 0, -10, 0,
        0, 0, 10, 15, 15, 15, 10, 0, 0,
        0, 0, 15, 25, 30, 25, 15, 0, 0,
        0, 0, 10, 20, 25, 20, 10, 0, 0,
        0, 0, 0, 10, 10, 10, 0, 0, 0,
        0, -5, -5, 0, 0, 0, -5, -5, 0,
        0, -10, -5, -5, -5, -5, -5, -10, 0,
        0, -20, -15, -10, -10, -10, -15, -20, 0
    )

    private val cannonPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 10, 10, 20, 10, 10, 5, 0,
        0, 5, 5, 15, 25, 15, 5, 5, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -5, 0, 0, 0, 0, 0, -5, 0,
        0, -10, -5, 0, 0, 0, -5, -10, 0,
        0, -5, -10, -10, -10, -10, -10, -5, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    )

    private val chariotPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 5, 5, 5, 5, 5, 5, 5, 0,
        0, -10, 0, 0, 0, 0, 0, -10, 0,
        0, -10, 0, 0, 0, 0, 0, -10, 0,
        0, -10, 0, 0, 0, 0, 0, -10, 0,
        0, -10, 0, 0, 0, 0, 0, -10, 0,
        0, -10, 0, 0, 0, 0, 0, -10, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
    )

    private val elephantPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 10, 0, 0, 0, 0, 0, 10, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, 0, -5,
        0, 0, -5, 0, 0, 0, -5, 0, 0
    )

    private val advisorPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 10, 10, 10, 0, 0, 0,
        0, 0, 0, 0, 20, 0, 0, 0, 0,
        0, 0, 0, 10, 0, 10, 0, 0, 0
    )

    private val kingPST = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 5, 5, 5, 0, 0, 0,
        0, 0, 0, 0, 10, 0, 0, 0, 0,
        0, 0, 0, 5, 0, 5, 0, 0, 0
    )

    private fun pstValue(type: PType, side: PSide, row: Int, col: Int): Int {
        val r = if (side == PSide.RED) row else 9 - row
        val idx = r * 9 + col
        return when (type) {
            PType.PAWN -> if (side == PSide.RED) redPawnPST[idx] else blackPawnPST[idx]
            PType.HORSE -> knightPST[idx]
            PType.CANNON -> cannonPST[idx]
            PType.CHARIOT -> chariotPST[idx]
            PType.ELEPHANT -> elephantPST[idx]
            PType.ADVISOR -> advisorPST[idx]
            PType.KING -> kingPST[idx]
            else -> 0
        }
    }

    private val tt = TranspositionTable(1 shl 16)

    @Volatile
    var cancelled = false
        private set

    private var startMark = TimeSource.Monotonic.markNow()
    private var timeLimit = 0L

    private fun elapsedMs(): Long = startMark.elapsedNow().inWholeMilliseconds

    private val lmrTable = Array(65) { d ->
        IntArray(65) { i ->
            if (d < 3 || i < 3) 0
            else max(1, (1.0 + ln(d.toDouble()) * ln(i.toDouble()) / 2.25).toInt())
        }
    }

    private class SearchState {
        val killers = Array(128) { arrayOfNulls<Move>(2) }
        val history = Array(90) { IntArray(90) }
        val counterMoves = Array(90) { arrayOfNulls<Move>(90) }
        @Volatile var rootBest: Move? = null
        var nodes = 0L
        val searchHashes = ULongArray(128)
        var gameHashCounts: Map<ULong, Int> = emptyMap()
    }

    private data class SearchResult(
        val bestMove: Move?,
        val bestScore: Int,
        val completedDepth: Int,
        val nodes: Long
    )

    fun cancel() {
        cancelled = true
    }

    fun clearHistory() {
        tt.clear()
    }

    private fun evaluate(board: Array<Array<Piece>>): Int {
        var score = 0
        for (row in 0..9) {
            for (col in 0..8) {
                val piece = board[row][col]
                if (piece.type == PType.EMPTY) continue
                val value = baseValues[piece.type.ordinal] + pstValue(piece.type, piece.side, row, col)
                if (piece.side == PSide.RED) score += value else score -= value
            }
        }
        return score
    }

    private fun evalForSide(board: Array<Array<Piece>>, side: PSide): Int {
        val raw = evaluate(board)
        return if (side == PSide.RED) raw else -raw
    }

    private fun mvvLva(victim: Piece, attacker: Piece): Int {
        if (victim.type == PType.EMPTY) return 0
        return baseValues[victim.type.ordinal] * 10 - baseValues[attacker.type.ordinal]
    }

    private fun moveScore(
        move: Move, board: Array<Array<Piece>>, depth: Int, ttBest: Move?,
        prevFrom: Int, prevTo: Int, state: SearchState
    ): Int {
        if (move == ttBest) return 50_000_000

        val victim = board[move.toRow][move.toCol]
        val attacker = board[move.fromRow][move.fromCol]

        if (victim.type != PType.EMPTY) {
            return 40_000_000 + mvvLva(victim, attacker)
        }

        if (depth in 0 until 128) {
            if (move == state.killers[depth][0]) return 30_000_000
            if (move == state.killers[depth][1]) return 29_000_000
        }

        if (prevFrom >= 0) {
            val cm = state.counterMoves[prevFrom][prevTo]
            if (cm != null && cm == move) return 28_000_000
        }

        val fromIdx = move.fromRow * 9 + move.fromCol
        val toIdx = move.toRow * 9 + move.toCol
        return state.history[fromIdx][toIdx]
    }

    private fun orderMoves(
        moves: List<Move>, board: Array<Array<Piece>>, depth: Int, ttBest: Move?,
        prevFrom: Int, prevTo: Int, state: SearchState
    ): List<Move> {
        return moves.sortedByDescending { moveScore(it, board, depth, ttBest, prevFrom, prevTo, state) }
    }

    private fun isTimeUp(): Boolean {
        if (timeLimit == 0L) return false
        return elapsedMs() >= timeLimit
    }

    private fun aborted(): Boolean = cancelled || isTimeUp()

    private fun qsearch(clone: ChessEngine, alpha: Int, beta: Int, ply: Int, state: SearchState): Int {
        if (aborted()) return 0
        state.nodes++

        val standPat = evalForSide(clone.board, clone.currentTurn)

        if (standPat >= beta) return beta
        var a = max(alpha, standPat)

        val captures = clone.getAllLegalMoves(clone.currentTurn)
            .filter { clone.board[it.toRow][it.toCol].type != PType.EMPTY }
            .sortedByDescending { mvvLva(clone.board[it.toRow][it.toCol], clone.board[it.fromRow][it.fromCol]) }

        for (m in captures) {
            if (aborted()) break
            val captured = clone.executeMoveSimple(m.fromRow, m.fromCol, m.toRow, m.toCol)
            val value = -qsearch(clone, -beta, -a, ply + 1, state)
            clone.undoMoveSimple(m.fromRow, m.fromCol, m.toRow, m.toCol, captured)
            if (value > a) a = value
            if (a >= beta) return beta
        }

        return a
    }

    private fun negamax(
        clone: ChessEngine, depth: Int, alpha: Int, beta: Int, ply: Int,
        doNull: Boolean, prevFrom: Int, prevTo: Int, state: SearchState
    ): Int {
        if (aborted()) return 0
        state.nodes++

        val hash = clone.zobristHash
        state.searchHashes[ply] = hash

        if (ply > 0) {
            for (i in 0 until ply) {
                if (state.searchHashes[i] == hash) return 0
            }
            val gameCount = state.gameHashCounts[hash] ?: 0
            if (gameCount >= 2) return 0
        }

        val isRoot = ply == 0
        val inCheck = clone.isInCheck(clone.currentTurn)

        if (inCheck && depth <= 0 && ply < 60) {
            return negamax(clone, 1, alpha, beta, ply, false, prevFrom, prevTo, state)
        }

        if (depth <= 0 && !inCheck) {
            return qsearch(clone, alpha, beta, ply, state)
        }

        val effectiveDepth = max(depth, 0)

        val ttEntry = tt.probe(hash)
        var ttBest: Move? = null
        if (ttEntry != null) {
            ttBest = ttEntry.bestMove
            if (ttEntry.depth >= effectiveDepth && !isRoot) {
                val ttScore = ttEntry.score
                when (ttEntry.flag) {
                    TTFlag.EXACT -> return ttScore
                    TTFlag.LOWER_BOUND -> if (ttScore >= beta) return ttScore
                    TTFlag.UPPER_BOUND -> if (ttScore <= alpha) return ttScore
                }
            }
        }

        if (doNull && effectiveDepth >= 3 && !inCheck && !isRoot && ply < 50) {
            val savedTurn = clone.currentTurn
            val savedHash = clone.zobristHash
            clone.currentTurn = if (savedTurn == PSide.RED) PSide.BLACK else PSide.RED
            clone.zobristHash = savedHash xor ChessEngine.sideToMoveKey
            val r = if (effectiveDepth > 6) 4 else 3
            val nullValue = -negamax(clone, effectiveDepth - 1 - r, -beta, -beta + 1, ply + 1, false, -1, -1, state)
            clone.currentTurn = savedTurn
            clone.zobristHash = savedHash
            if (aborted()) return 0
            if (nullValue >= beta) return beta
        }

        val moves = clone.getAllLegalMoves(clone.currentTurn)
        if (moves.isEmpty()) {
            return -MATE_SCORE + ply
        }

        val orderedMoves = orderMoves(moves, clone.board, effectiveDepth, ttBest, prevFrom, prevTo, state)

        var bestMove: Move? = null
        var bestScore = -INF
        var flag = TTFlag.UPPER_BOUND
        var a = alpha
        var moveCount = 0

        val staticEval = evalForSide(clone.board, clone.currentTurn)

        for (m in orderedMoves) {
            if (aborted()) break

            val captured = clone.executeMoveSimple(m.fromRow, m.fromCol, m.toRow, m.toCol)
            val givesCheck = clone.isInCheck(clone.currentTurn)
            val isCapture = captured.type != PType.EMPTY
            moveCount++

            val curFrom = m.fromRow * 9 + m.fromCol
            val curTo = m.toRow * 9 + m.toCol

            var newDepth = effectiveDepth - 1

            var reduction = 0
            if (effectiveDepth >= 3 && moveCount > 3 && !isCapture && !givesCheck && !inCheck) {
                val lmrIdx = min(moveCount - 1, 64)
                val lmrDepth = min(effectiveDepth, 64)
                reduction = lmrTable[lmrDepth][lmrIdx]
                if (m == state.killers[effectiveDepth][0] || m == state.killers[effectiveDepth][1]) {
                    reduction = max(0, reduction - 1)
                }
                reduction = min(reduction, newDepth - 1)
                if (reduction < 1) reduction = 0
            }

            if (effectiveDepth <= 3 && !isCapture && !givesCheck && !inCheck && reduction == 0) {
                val futilityMargin = 200 * effectiveDepth
                if (staticEval + futilityMargin <= a) {
                    clone.undoMoveSimple(m.fromRow, m.fromCol, m.toRow, m.toCol, captured)
                    continue
                }
            }

            val value: Int = if (moveCount == 1) {
                -negamax(clone, newDepth, -beta, -a, ply + 1, true, curFrom, curTo, state)
            } else {
                val reducedDepth = max(0, newDepth - reduction)
                var v = -negamax(clone, reducedDepth, -a - 1, -a, ply + 1, true, curFrom, curTo, state)

                if (reduction > 0 && v > a) {
                    v = -negamax(clone, newDepth, -a - 1, -a, ply + 1, true, curFrom, curTo, state)
                }

                if (v > a && v < beta) {
                    -negamax(clone, newDepth, -beta, -a, ply + 1, true, curFrom, curTo, state)
                } else {
                    v
                }
            }

            clone.undoMoveSimple(m.fromRow, m.fromCol, m.toRow, m.toCol, captured)

            if (aborted()) break

            if (value > bestScore) {
                bestScore = value
                bestMove = m
                if (value > a) {
                    a = value
                    flag = TTFlag.EXACT
                }
            }
            if (a >= beta) {
                if (!isCapture && effectiveDepth < 128) {
                    if (m != state.killers[effectiveDepth][0]) {
                        state.killers[effectiveDepth][1] = state.killers[effectiveDepth][0]
                        state.killers[effectiveDepth][0] = m
                    }
                    val fromIdx = m.fromRow * 9 + m.fromCol
                    val toIdx = m.toRow * 9 + m.toCol
                    state.history[fromIdx][toIdx] += effectiveDepth * effectiveDepth
                    if (prevFrom >= 0 && prevTo >= 0) {
                        state.counterMoves[prevFrom][prevTo] = m
                    }
                }
                flag = TTFlag.LOWER_BOUND
                break
            }
        }

        if (bestScore <= alpha) flag = TTFlag.UPPER_BOUND
        if (!aborted()) {
            tt.store(hash, effectiveDepth, bestScore, flag, bestMove)
            if (isRoot) {
                state.rootBest = bestMove
            }
        }

        return bestScore
    }

    private fun iterativeDeepening(
        clone: ChessEngine, maxDepth: Int, state: SearchState, workerId: Int
    ): SearchResult {
        var bestScore = -INF
        var lastCompletedDepth = 0
        var aspirationAlpha = -INF
        var aspirationBeta = INF

        for (iterDepth in 1..maxDepth) {
            if (aborted()) break

            val score: Int = if (iterDepth >= 4) {
                var s = negamax(clone, iterDepth, aspirationAlpha, aspirationBeta, 0, true, -1, -1, state)

                if (!aborted() && (s <= aspirationAlpha || s >= aspirationBeta)) {
                    s = negamax(clone, iterDepth, -INF, INF, 0, true, -1, -1, state)
                    aspirationAlpha = -INF
                    aspirationBeta = INF
                }

                if (!aborted() && s > aspirationAlpha && s < aspirationBeta) {
                    aspirationAlpha = s - WINDOW
                    aspirationBeta = s + WINDOW
                }
                s
            } else {
                val s = negamax(clone, iterDepth, -INF, INF, 0, true, -1, -1, state)
                if (iterDepth == 3 && !aborted()) {
                    aspirationAlpha = s - WINDOW
                    aspirationBeta = s + WINDOW
                }
                s
            }

            if (aborted()) break

            val iterBest = state.rootBest
            if (iterBest != null) {
                bestScore = score
                lastCompletedDepth = iterDepth
            }

            val elapsed = elapsedMs()
            if (workerId == 0) {
                DebugLog.log("AI: depth $iterDepth score=$score nodes=${state.nodes} time=${elapsed}ms best=${iterBest?.let { "${it.fromRow},${it.fromCol}->${it.toRow},${it.toCol}" }}")
            }

            if (elapsed > timeLimit / 2 && iterDepth >= 3) {
                if (workerId == 0) DebugLog.log("AI: time budget half used, stopping")
                break
            }

            if (abs(score) > MATE_SCORE - 100) break
        }

        return SearchResult(state.rootBest, bestScore, lastCompletedDepth, state.nodes)
    }

    suspend fun getAIMove(
        searchDepth: Int = 3,
        noise: Int = 0,
        randomPickChance: Float = 0f,
        userTimeLimit: Long = 0L,
        randomMoveChance: Float = 0f,
        threads: Int = 1
    ): Move? {
        cancelled = false
        startMark = TimeSource.Monotonic.markNow()

        val aiSide = if (engine.playerSide == PSide.RED) PSide.BLACK else PSide.RED
        val allMoves = engine.getAllLegalMoves(aiSide)
        if (allMoves.isEmpty()) return null

        if (randomMoveChance > 0f && kotlin.random.Random.nextFloat() < randomMoveChance) {
            DebugLog.log("AI: random move pick ($randomMoveChance)")
            return allMoves.random()
        }

        openingBook?.let { book ->
            val bookMoves = book.probeWeighted(engine.zobristHash)
            if (bookMoves.isNotEmpty()) {
                val validBookMoves = bookMoves.mapNotNull { (move, _) ->
                    if (allMoves.any { it.fromRow == move.fromRow && it.fromCol == move.fromCol && it.toRow == move.toRow && it.toCol == move.toCol }) move
                    else null
                }
                if (validBookMoves.isNotEmpty()) {
                    DebugLog.log("AI: book hit, ${validBookMoves.size} moves")
                    if (randomPickChance > 0f && kotlin.random.Random.nextFloat() < randomPickChance) {
                        DebugLog.log("AI: book noise pick, random")
                        return allMoves.random()
                    }
                    return validBookMoves.random()
                }
            }
        }

        val maxDepth = searchDepth.coerceAtLeast(1)

        timeLimit = if (userTimeLimit > 0L) userTimeLimit else when {
            searchDepth <= 3 -> 3000L
            searchDepth <= 6 -> 10000L
            else -> 30000L
        }

        val performanceCores = detectPerformanceCores()
        val actualThreads = threads.coerceAtMost(performanceCores).coerceAtLeast(1)

        val gameHashCounts = mutableMapOf<ULong, Int>()
        for (h in engine.getPositionHistory()) {
            gameHashCounts[h] = (gameHashCounts[h] ?: 0) + 1
        }

        val result = if (actualThreads <= 1) {
            DebugLog.log("AI: single thread (perfCores=$performanceCores, config=$threads)")
            val state = SearchState()
            state.gameHashCounts = gameHashCounts
            val clone = engine.clone()
            clone.currentTurn = aiSide
            clone.zobristHash = clone.computeZobrist()
            iterativeDeepening(clone, maxDepth, state, 0)
        } else {
            DebugLog.log("AI: starting $actualThreads workers (perfCores=$performanceCores, config=$threads)")
            coroutineScope {
                val deferreds = (0 until actualThreads).map { workerId ->
                    async(Dispatchers.Default) {
                        val state = SearchState()
                        state.gameHashCounts = gameHashCounts
                        val clone = engine.clone()
                        clone.currentTurn = aiSide
                        clone.zobristHash = clone.computeZobrist()
                        iterativeDeepening(clone, maxDepth, state, workerId)
                    }
                }
                val results = deferreds.awaitAll()
                results.filter { it.bestMove != null }.maxWithOrNull(
                    compareByDescending<SearchResult> { it.completedDepth }
                        .thenByDescending { it.bestScore }
                ) ?: results.first()
            }
        }

        var bestMove = result.bestMove
        if (bestMove == null && allMoves.isNotEmpty()) {
            bestMove = allMoves.first()
        }

        if (noise > 0 && bestMove != null && result.bestScore > -MATE_SCORE + 100) {
            val threshold = noise * 2
            val candidates = allMoves.filter { move ->
                if (move == bestMove) true
                else {
                    val clone2 = engine.clone()
                    clone2.currentTurn = aiSide
                    clone2.zobristHash = clone2.computeZobrist()
                    clone2.executeMoveSimple(move.fromRow, move.fromCol, move.toRow, move.toCol)
                    val s = evalForSide(clone2.board, aiSide)
                    result.bestScore - s <= threshold
                }
            }
            if (candidates.size > 1) {
                bestMove = candidates.random()
                DebugLog.log("AI: noise pick from ${candidates.size} candidates within $threshold of ${result.bestScore}")
            }
        }

        if (randomPickChance > 0f && bestMove != null && kotlin.random.Random.nextFloat() < randomPickChance) {
            val others = allMoves.filter { it != bestMove }
            if (others.isNotEmpty()) {
                bestMove = others.random()
            }
        }

        val totalTime = elapsedMs()
        DebugLog.log("AI: chose ${bestMove?.let { "${it.fromRow},${it.fromCol}->${it.toRow},${it.toCol}" }} depth=${result.completedDepth} nodes=${result.nodes} time=${totalTime}ms threads=$actualThreads")

        return bestMove
    }

    companion object {
        private const val INF = 99999
        private const val MATE_SCORE = 99000
        private const val WINDOW = 50
        private val baseValues = intArrayOf(0, 10000, 180, 220, 400, 900, 450, 100)

        fun staticEvaluate(board: Array<Array<Piece>>, maximizing: Boolean): Int {
            var score = 0
            for (row in 0..9) {
                for (col in 0..8) {
                    val piece = board[row][col]
                    if (piece.type == PType.EMPTY) continue
                    var value = baseValues[piece.type.ordinal]
                    if (piece.type == PType.PAWN) {
                        value += when {
                            piece.side == PSide.RED && row >= 3 -> (9 - row) * 40 + 10
                            piece.side == PSide.BLACK && row <= 6 -> row * 40 + 10
                            else -> 0
                        }
                    }
                    if (piece.side == PSide.RED) score += value else score -= value
                }
            }
            return if (maximizing) score else -score
        }
    }
}
