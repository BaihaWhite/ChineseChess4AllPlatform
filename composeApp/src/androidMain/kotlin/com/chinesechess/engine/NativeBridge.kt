package com.chinesechess.engine

actual fun setupNativeEngine(ai: ChessAI) {
    try {
        val nativeEngine = NativeEngine
        ai.useNativeEngine = true
        ai.nativeSearchFn = { board, turn, depth, timeMs, history ->
            val result = nativeEngine.search(board, turn, depth, timeMs, history)
            if (result != 0) result else null
        }
        ai.nativeCancelFn = { nativeEngine.cancel() }
        DebugLog.log("NativeEngine: Rust engine loaded successfully")
    } catch (e: UnsatisfiedLinkError) {
        DebugLog.log("NativeEngine: failed to load: ${e.message}")
    } catch (e: Exception) {
        DebugLog.log("NativeEngine: error: ${e.message}")
    }
}
