package com.chinesechess.engine

import android.content.Context
import java.io.File

actual object NativeEngine {
    private var loaded = false

    init {
        try {
            System.loadLibrary("chess_engine")
            loaded = true
        } catch (e: UnsatisfiedLinkError) {
            DebugLog.error("Native", "Failed to load: ${e.message}")
        }
    }

    /**
     * Copy NNUE weights from assets to internal storage and load into engine.
     * Call from MainActivity.onCreate after library load.
     */
    fun loadNNUE(context: Context) {
        if (!loaded) return
        try {
            val dest = File(context.filesDir, "nnue_trained.bin")
            if (!dest.exists()) {
                context.assets.open("nnue_trained.bin").use { input ->
                    dest.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                DebugLog.info("NNUE", "Copied to ${dest.absolutePath}")
            }
            val ok = chessLoadNNUE(dest.absolutePath)
            DebugLog.info("NNUE", "Load result: $ok")
        } catch (e: Exception) {
            DebugLog.error("NNUE", "Failed: ${e.message}")
        }
    }

    actual val isAvailable: Boolean get() = loaded

    actual val threadCount: Int get() = 1
    actual val lastSearchDepth: Int get() = 0
    actual val lastSearchNodes: Long get() = 0L

    actual fun search(
        board: Array<Array<Piece>>,
        turn: Int,
        depth: Int,
        timeMs: Int,
        history: LongArray,
        redChecks: Int,
        blackChecks: Int
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
        return chessSearch(boardBytes, turn, depth, timeMs, history, history.size, redChecks, blackChecks)
    }

    actual fun cancel() {
        chessCancel()
    }

    private external fun chessLoadNNUE(path: String): Boolean

    private external fun chessSearch(
        board: ByteArray,
        turn: Int,
        depth: Int,
        timeMs: Int,
        history: LongArray,
        historyLen: Int,
        redChecks: Int,
        blackChecks: Int
    ): Int

    private external fun chessCancel()
}
