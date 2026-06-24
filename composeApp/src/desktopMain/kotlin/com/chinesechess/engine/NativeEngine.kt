package com.chinesechess.engine

import java.io.File

actual object NativeEngine {
    private var loaded = false

    init {
        try {
            loadFromResources()
            loaded = true
            DebugLog.info("Native", "Rust engine loaded")
        } catch (e: UnsatisfiedLinkError) {
            DebugLog.error("Native", "Failed to load: ${e.message}")
        } catch (e: Exception) {
            DebugLog.error("Native", "Failed: ${e.message}")
        }
    }

    private fun loadFromResources() {
        val os = System.getProperty("os.name").lowercase()
        val (dir, libName) = when {
            os.contains("win") -> "windows-x86-64" to "chess_engine.dll"
            os.contains("linux") -> "linux-x86-64" to "libchess_engine.so"
            os.contains("mac") -> "macos-x86-64" to "libchess_engine.dylib"
            else -> throw UnsatisfiedLinkError("Unsupported OS: $os")
        }
        val resourcePath = "/natives/$dir/$libName"
        val input = NativeEngine::class.java.getResourceAsStream(resourcePath)
            ?: throw UnsatisfiedLinkError("Native lib not found in resources: $resourcePath")
        val tmpDir = File(System.getProperty("java.io.tmpdir"), "chinese-chess-natives")
        tmpDir.mkdirs()
        val tmpFile = File(tmpDir, libName)
        if (!tmpFile.exists() || tmpFile.length() == 0L) {
            tmpFile.outputStream().use { input.copyTo(it) }
            tmpFile.deleteOnExit()
        } else {
            input.close()
        }
        System.load(tmpFile.absolutePath)

        // Extract and load NNUE weights from resources
        try {
            val nnueResource = NativeEngine::class.java.getResourceAsStream("/nnue_trained.bin")
            if (nnueResource != null) {
                val nnueFile = File(tmpDir, "nnue_trained.bin")
                if (!nnueFile.exists() || nnueFile.length() == 0L) {
                    nnueFile.outputStream().use { nnueResource.copyTo(it) }
                    nnueFile.deleteOnExit()
                } else {
                    nnueResource.close()
                }
                val ok = chessLoadNNUE(nnueFile.absolutePath)
                DebugLog.info("NNUE", "Loaded: $ok")
            } else {
                DebugLog.warn("NNUE", "Not found in resources, using HCE")
            }
        } catch (e: Throwable) {
            DebugLog.error("NNUE", "JNI failed: ${e.message}")
        }
    }

    actual val isAvailable: Boolean get() = loaded

    actual val threadCount: Int get() = if (loaded) chessGetThreadCount() else 1
    actual val lastSearchDepth: Int get() = if (loaded) chessGetLastDepth() else 0
    actual val lastSearchNodes: Long get() = if (loaded) chessGetLastNodes() else 0L

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

    private external fun chessLoadNNUE(path: String): Boolean
    private external fun chessGetThreadCount(): Int
    private external fun chessGetLastDepth(): Int
    private external fun chessGetLastNodes(): Long
    private external fun chessCancel()
}
