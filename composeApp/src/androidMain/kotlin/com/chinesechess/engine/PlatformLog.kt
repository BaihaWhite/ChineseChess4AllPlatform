package com.chinesechess.engine

actual fun platformLog(tag: String, msg: String) {
    android.util.Log.d(tag, msg)
}

actual fun platformExportLogs(
    logContent: String,
    onSuccess: (String) -> Unit,
    onError: (String) -> Unit
) {
    try {
        val context = com.chinesechess.app.MainActivity.instance
            ?: throw IllegalStateException("No activity context")
        val logDir = java.io.File(context.cacheDir, "logs")
        logDir.mkdirs()
        val logFile = java.io.File(logDir, "chinese-chess-log.txt")
        logFile.writeText(logContent)
        val uri = androidx.core.content.FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            logFile
        )
        val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(android.content.Intent.EXTRA_STREAM, uri)
            putExtra(android.content.Intent.EXTRA_SUBJECT, "中国象棋日志")
            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(android.content.Intent.createChooser(intent, "分享日志"))
        onSuccess("已打开分享")
    } catch (e: Exception) {
        onError(e.message ?: "分享失败")
    }
}
