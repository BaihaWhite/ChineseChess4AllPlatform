package com.chinesechess.engine

actual fun platformLog(tag: String, msg: String) {
    println("[$tag] $msg")
}

actual fun platformExportLogs(
    logContent: String,
    onSuccess: (String) -> Unit,
    onError: (String) -> Unit
) {
    try {
        val chooser = javax.swing.JFileChooser().apply {
            dialogTitle = "保存日志"
            selectedFile = java.io.File("chinese-chess-log.txt")
            fileFilter = javax.swing.filechooser.FileNameExtensionFilter("文本文件 (*.txt)", "txt")
        }
        val result = chooser.showSaveDialog(null)
        if (result == javax.swing.JFileChooser.APPROVE_OPTION) {
            var file = chooser.selectedFile
            if (!file.name.endsWith(".txt", ignoreCase = true)) {
                file = java.io.File(file.absolutePath + ".txt")
            }
            file.writeText(logContent)
            onSuccess("已保存到 ${file.name}")
        }
    } catch (e: Exception) {
        onError(e.message ?: "保存失败")
    }
}
