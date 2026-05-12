package com.chinesechess.engine

actual fun createNetworkClient(): NetworkClient = AndroidNetworkClient()
