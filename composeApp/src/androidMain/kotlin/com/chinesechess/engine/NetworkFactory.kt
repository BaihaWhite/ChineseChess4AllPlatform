package com.chinesechess.engine

actual fun createNetworkClient(): NetworkClient = AndroidNetworkClient()

actual fun detectPerformanceCores(): Int {
    val cpuDir = java.io.File("/sys/devices/system/cpu")
    val cpuDirs = cpuDir.listFiles { f -> f.name.matches(Regex("cpu\\d+")) } ?: return Runtime.getRuntime().availableProcessors()
    val totalCores = cpuDirs.size
    val freqs = mutableListOf<Long>()

    for (dir in cpuDirs) {
        val freqFile = java.io.File(dir, "cpufreq/cpuinfo_max_freq")
        if (freqFile.exists()) {
            try {
                val freq = freqFile.readText().trim().toLongOrNull()
                if (freq != null && freq > 0) freqs.add(freq)
            } catch (_: Exception) {}
        }
    }

    if (freqs.size >= 2) {
        val sorted = freqs.sorted()
        val median = sorted[sorted.size / 2]
        val bigCoreCount = freqs.count { it > median }
        val littleCoreCount = freqs.size - bigCoreCount
        DebugLog.log("CPU: $totalCores cores ($bigCoreCount big + $littleCoreCount LITTLE), freqs=$sorted")
    } else {
        DebugLog.log("CPU: $totalCores cores (freq info unavailable)")
    }

    return totalCores
}
