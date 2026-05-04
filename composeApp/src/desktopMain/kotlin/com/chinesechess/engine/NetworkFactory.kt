package com.chinesechess.engine

actual fun createNetworkClient(): NetworkClient = JvmNetworkClient()

actual fun detectPerformanceCores(): Int = Runtime.getRuntime().availableProcessors()
