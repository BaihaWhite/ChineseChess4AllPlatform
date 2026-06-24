package com.chinesechess.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.chinesechess.engine.NativeEngine
import com.chinesechess.ui.App

class MainActivity : ComponentActivity() {
    companion object {
        var instance: MainActivity? = null
            private set
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        instance = this
        super.onCreate(savedInstanceState)
        NativeEngine.loadNNUE(this)
        enableEdgeToEdge()
        setContent {
            App(onExitApp = { finish() })
        }
    }
}
