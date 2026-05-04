package com.chinesechess

import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.chinesechess.ui.App

fun main() = application {
    Window(
        onCloseRequest = ::exitApplication,
        title = "中国象棋 Chinese Chess",
        state = rememberWindowState(width = 480.dp, height = 720.dp)
    ) {
        App(onExitApp = ::exitApplication)
    }
}
