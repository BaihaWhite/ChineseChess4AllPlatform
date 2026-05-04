package com.chinesechess.engine

class OpeningBook {
    private var bookHashes = ULongArray(0)
    private var bookMoves = IntArray(0)
    private var bookCounts = IntArray(0)
    private var bookOffsets = IntArray(0)
    private var bookLens = IntArray(0)
    private var numPositions = 0
    private var _totalMoves = 0
    val totalMoves: Int get() = _totalMoves
    var loaded = false
        private set

    private val tempEntries = mutableMapOf<ULong, MutableList<Pair<Move, Int>>>()

    fun probe(hash: ULong): List<Move> {
        val idx = binarySearch(hash)
        if (idx < 0) return emptyList()
        val off = bookOffsets[idx]
        val len = bookLens[idx]
        val result = mutableListOf<Move>()
        for (i in 0 until len) {
            val mi = (off + i) * 4
            result.add(Move(bookMoves[mi], bookMoves[mi + 1], bookMoves[mi + 2], bookMoves[mi + 3]))
        }
        return result
    }

    fun probeWeighted(hash: ULong): List<Pair<Move, Int>> {
        val idx = binarySearch(hash)
        if (idx < 0) return emptyList()
        val off = bookOffsets[idx]
        val len = bookLens[idx]
        val result = mutableListOf<Pair<Move, Int>>()
        for (i in 0 until len) {
            val mi = (off + i) * 4
            val move = Move(bookMoves[mi], bookMoves[mi + 1], bookMoves[mi + 2], bookMoves[mi + 3])
            result.add(Pair(move, bookCounts[off + i]))
        }
        return result
    }

    fun add(hash: ULong, move: Move) {
        val list = tempEntries.getOrPut(hash) { mutableListOf() }
        val existing = list.indexOfFirst { it.first == move }
        if (existing >= 0) {
            val (m, c) = list[existing]
            list[existing] = Pair(m, c + 1)
        } else {
            list.add(Pair(move, 1))
        }
    }

    fun merge(other: OpeningBook) {
        for ((hash, moves) in other.tempEntries) {
            for ((move, count) in moves) {
                repeat(count) { add(hash, move) }
            }
        }
    }

    fun serialize(): String = buildString {
        if (numPositions > 0) {
            for (i in 0 until numPositions) {
                val hash = bookHashes[i]
                val off = bookOffsets[i]
                val len = bookLens[i]
                if (len == 0) continue
                append(hash.toString(16))
                append(':')
                val parts = (0 until len).map { j ->
                    val mi = (off + j) * 4
                    "${bookMoves[mi]},${bookMoves[mi + 1]},${bookMoves[mi + 2]},${bookMoves[mi + 3]}=${bookCounts[off + j]}"
                }
                append(parts.joinToString(";"))
                append('\n')
            }
        } else {
            for ((hash, moves) in tempEntries) {
                if (moves.isEmpty()) continue
                append(hash.toString(16))
                append(':')
                append(moves.joinToString(";") { (move, count) ->
                    "${move.fromRow},${move.fromCol},${move.toRow},${move.toCol}=$count"
                })
                append('\n')
            }
        }
    }

    fun deserialize(data: String) {
        tempEntries.clear()
        bookHashes = ULongArray(0)
        bookMoves = IntArray(0)
        bookCounts = IntArray(0)
        bookOffsets = IntArray(0)
        bookLens = IntArray(0)
        numPositions = 0
        _totalMoves = 0

        val positions = mutableListOf<ULong>()
        val moveData = mutableListOf<Int>()
        val countData = mutableListOf<Int>()
        val offsetData = mutableListOf<Int>()
        val lenData = mutableListOf<Int>()

        var moveCount = 0

        for (line in data.lines()) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue
            val colonIdx = trimmed.indexOf(':')
            if (colonIdx < 0) continue
            val hash = trimmed.substring(0, colonIdx).toULong(16)
            val movesStr = trimmed.substring(colonIdx + 1)
            var posMoveCount = 0
            for (part in movesStr.split(';')) {
                if (part.isEmpty()) continue
                val eqIdx = part.lastIndexOf('=')
                if (eqIdx < 0) continue
                val coords = part.substring(0, eqIdx).split(',')
                val count = part.substring(eqIdx + 1).toIntOrNull() ?: continue
                if (coords.size != 4) continue
                val fr = coords[0].toIntOrNull() ?: continue
                val fc = coords[1].toIntOrNull() ?: continue
                val tr = coords[2].toIntOrNull() ?: continue
                val tc = coords[3].toIntOrNull() ?: continue
                moveData.add(fr)
                moveData.add(fc)
                moveData.add(tr)
                moveData.add(tc)
                countData.add(count)
                posMoveCount++
            }
            if (posMoveCount > 0) {
                positions.add(hash)
                offsetData.add(moveCount)
                lenData.add(posMoveCount)
                moveCount += posMoveCount
            }
        }

        numPositions = positions.size
        _totalMoves = moveCount

        bookHashes = ULongArray(numPositions) { positions[it] }
        bookMoves = IntArray(moveCount * 4) { moveData[it] }
        bookCounts = IntArray(moveCount) { countData[it] }
        bookOffsets = IntArray(numPositions) { offsetData[it] }
        bookLens = IntArray(numPositions) { lenData[it] }

        loaded = true
    }

    private fun binarySearch(hash: ULong): Int {
        var lo = 0
        var hi = numPositions - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (bookHashes[mid] == hash) return mid
            if (bookHashes[mid] < hash) lo = mid + 1 else hi = mid - 1
        }
        return -1
    }

    val size: Int get() = numPositions

    fun clear() {
        tempEntries.clear()
        bookHashes = ULongArray(0)
        bookMoves = IntArray(0)
        bookCounts = IntArray(0)
        bookOffsets = IntArray(0)
        bookLens = IntArray(0)
        numPositions = 0
        _totalMoves = 0
        loaded = false
    }
}
