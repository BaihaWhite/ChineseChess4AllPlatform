package com.chinesechess.engine

expect fun platformLog(tag: String, msg: String)

/**
 * Export logs — Android: system share; Desktop: save file dialog.
 * @param logContent the full log text to export
 * @param onSuccess called when export succeeds (message for user)
 * @param onError called on failure
 */
expect fun platformExportLogs(logContent: String, onSuccess: (String) -> Unit, onError: (String) -> Unit)
