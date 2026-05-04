package com.chinesechess.engine

actual fun platformLog(tag: String, msg: String) {
    android.util.Log.d(tag, msg)
}
