package com.chinesechess.engine

import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.LinkedBlockingQueue
import kotlin.random.Random
import kotlin.system.measureTimeMillis

object OpeningBookGenerator {
    @JvmStatic
    fun main(args: Array<String>) {
        val gameCount = args.getOrNull(0)?.toIntOrNull() ?: 500
        val maxPly = args.getOrNull(1)?.toIntOrNull() ?: 16
        val topK = args.getOrNull(2)?.toIntOrNull() ?: 4
        val outputPath = args.getOrNull(3) ?: "opening_book.txt"
        val computeThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 128)
        val saveInterval = args.getOrNull(4)?.toIntOrNull() ?: 2000

        println("=== Chinese Chess Opening Book Generator ===")
        println("Config: games=$gameCount, maxPly=$maxPly, topK=$topK, compute=$computeThreads, saveInterval=$saveInterval")
        println("Output: $outputPath")

        val mergedBook = OpeningBook()
        val outFile = File(outputPath)
        if (outFile.exists()) {
            mergedBook.deserialize(outFile.readText())
            println("Resuming from existing: ${mergedBook.size} positions")
        }

        val computePool = Executors.newFixedThreadPool(computeThreads)
        val saverThread = Thread {
            while (!Thread.currentThread().isInterrupted) {
                try {
                    val data = saveQueue.take() ?: break
                    outFile.writeText(data)
                } catch (e: InterruptedException) { break }
                catch (_: Exception) {}
            }
        }.apply { isDaemon = true; start() }

        val processBatch = computeThreads * 4

        val elapsed = measureTimeMillis {
            var submitted = 0
            while (submitted < gameCount) {
                val batchEnd = minOf(submitted + processBatch, gameCount)
                val batchCount = batchEnd - submitted

                val batchFutures = ArrayList<Future<OpeningBook>>(batchCount)
                for (i in 0 until batchCount) {
                    val gameIdx = submitted + i
                    batchFutures.add(computePool.submit(Callable {
                        val rng = Random(42L + gameIdx.toLong())
                        val book = OpeningBook()
                        val engine = ChessEngine()
                        engine.initBoard()
                        var ply = 0

                        while (!engine.gameOver && ply < maxPly) {
                            val side = engine.currentTurn
                            val moves = engine.getAllLegalMoves(side)
                            if (moves.isEmpty()) break

                            var chosen: Move
                            if (moves.size == 1) {
                                chosen = moves[0]
                            } else {
                                val maximizing = side == PSide.BLACK
                                val scored = moves.map { move ->
                                    val clone = engine.clone()
                                    clone.currentTurn = side
                                    clone.zobristHash = clone.computeZobrist()
                                    clone.executeMoveSimple(move.fromRow, move.fromCol, move.toRow, move.toCol)
                                    val s = ChessAI.staticEvaluate(clone.board, maximizing)
                                    Pair(move, s)
                                }

                                val sorted = if (maximizing)
                                    scored.sortedByDescending { it.second }
                                else
                                    scored.sortedBy { it.second }

                                val candidates = sorted.take(topK.coerceAtMost(moves.size))
                                val totalWeight = (1..candidates.size).sum()
                                var r = rng.nextInt(totalWeight)
                                chosen = candidates.first().first
                                for ((idx, cand) in candidates.withIndex()) {
                                    r -= (candidates.size - idx)
                                    if (r < 0) { chosen = cand.first; break }
                                }
                            }

                            val hashBefore = engine.zobristHash
                            engine.executeMove(chosen.fromRow, chosen.fromCol, chosen.toRow, chosen.toCol)
                            book.add(hashBefore, chosen)
                            ply++
                        }
                        book
                    }))
                }

                synchronized(mergedBook) {
                    for (f in batchFutures) {
                        mergedBook.merge(f.get())
                    }
                }
                submitted = batchEnd

                if (submitted % saveInterval < processBatch || submitted == gameCount) {
                    val snapshot = synchronized(mergedBook) { mergedBook.serialize() }
                    val size = snapshot.length
                    println("  $submitted/$gameCount (${mergedBook.size} positions, ${size} bytes)")
                    saveQueue.put(snapshot)
                }
            }
            saveQueue.put("")
        }

        computePool.shutdown()
        saverThread.interrupt()
        saverThread.join(5000)

        println("")
        println("Done in ${elapsed / 1000}s!")
        println("Total positions: ${mergedBook.size}")
        println("Total move entries: ${mergedBook.totalMoves}")
        println("Saved $outputPath (${outFile.length()} bytes)")
    }

    private val saveQueue = LinkedBlockingQueue<String?>()
}
