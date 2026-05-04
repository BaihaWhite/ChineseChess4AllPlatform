package com.chinesechess.engine

object DebugLog {
    private const val MAX = 200
    private const val TAG = "ChessAI"
    private val buffer = ArrayDeque<String>(MAX)
    private var counter = 0L

    fun log(msg: String) {
        platformLog(TAG, msg)
        synchronized(buffer) {
            counter++
            if (buffer.size >= MAX) buffer.removeFirst()
            buffer.add("#$counter $msg")
        }
    }

    fun getRecent(n: Int): String = synchronized(buffer) {
        buffer.takeLast(n).joinToString("\n")
    }

    fun getAll(): String = synchronized(buffer) {
        buffer.joinToString("\n")
    }
}
