package com.chinesechess.engine

enum class LogLevel { ERROR, WARN, INFO, DEBUG }

object DebugLog {
    private const val MAX = 200
    private var counter = 0L
    private val buffer = ArrayDeque<String>(MAX)

    var minLevel: LogLevel = LogLevel.DEBUG

    fun error(tag: String, msg: String) = log(LogLevel.ERROR, tag, msg)
    fun warn(tag: String, msg: String) = log(LogLevel.WARN, tag, msg)
    fun info(tag: String, msg: String) = log(LogLevel.INFO, tag, msg)
    fun debug(tag: String, msg: String) = log(LogLevel.DEBUG, tag, msg)

    private fun log(level: LogLevel, tag: String, msg: String) {
        if (level.ordinal > minLevel.ordinal) return
        val line = "[${level.name.padEnd(5)}] $tag: $msg"
        platformLog(tag, msg)
        synchronized(buffer) {
            counter++
            if (buffer.size >= MAX) buffer.removeFirst()
            buffer.add("#$counter $line")
        }
    }

    fun getRecent(n: Int): String = synchronized(buffer) {
        buffer.takeLast(n).joinToString("\n")
    }

    fun clear() = synchronized(buffer) { buffer.clear() }
}
