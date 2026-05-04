package com.chinesechess.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.chinesechess.engine.*

@Composable
fun ChessBoard(
    engine: ChessEngine,
    flipped: Boolean,
    lastMove: Move?,
    selectedRow: Int,
    selectedCol: Int,
    frozenBoard: Array<Array<Piece>>? = null,
    validMoves: List<Move> = emptyList(),
    onCellClick: (row: Int, col: Int) -> Unit,
    modifier: Modifier = Modifier
) {
    val textMeasurer = rememberTextMeasurer()
    val renderBoard = frozenBoard ?: engine.board

    fun toEngineRow(vr: Int): Int = if (flipped) 9 - vr else vr
    fun toEngineCol(vc: Int): Int = if (flipped) 8 - vc else vc

    BoxWithConstraints(modifier = modifier) {
        val boardWidth = constraints.maxWidth.toFloat()
        if (boardWidth <= 0f) return@BoxWithConstraints

        val cellSize = boardWidth / 10f
        val marginX = cellSize
        val marginY = cellSize * 0.5f
        val pieceR = cellSize * 0.42f
        val boardW = 8 * cellSize
        val boardH = 9 * cellSize

        fun boardCX(col: Int): Float = marginX + col * cellSize
        fun boardCY(row: Int): Float = marginY + row * cellSize

        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(boardWidth) {
                    detectTapGestures { offset ->
                        val cs = boardWidth / 10f
                        val mx = cs
                        val my = cs * 0.5f
                        val col = ((offset.x - mx + cs / 2) / cs).toInt().coerceIn(0, 8)
                        val row = ((offset.y - my + cs / 2) / cs).toInt().coerceIn(0, 9)
                        onCellClick(toEngineRow(row), toEngineCol(col))
                    }
                }
        ) {
            drawRect(ChessColors.BoardBg, topLeft = Offset(marginX - cellSize / 2, marginY - cellSize / 2),
                size = Size(boardW + cellSize, boardH + cellSize))

            for (i in 0..9) {
                drawLine(ChessColors.BoardLine, Offset(marginX, marginY + i * cellSize),
                    Offset(marginX + boardW, marginY + i * cellSize), strokeWidth = 1.5f)
            }
            for (j in 0..8) {
                if (j == 0 || j == 8) {
                    drawLine(ChessColors.BoardLine, Offset(marginX + j * cellSize, marginY),
                        Offset(marginX + j * cellSize, marginY + boardH), strokeWidth = 1.5f)
                } else {
                    drawLine(ChessColors.BoardLine, Offset(marginX + j * cellSize, marginY),
                        Offset(marginX + j * cellSize, marginY + 4 * cellSize), strokeWidth = 1.5f)
                    drawLine(ChessColors.BoardLine, Offset(marginX + j * cellSize, marginY + 5 * cellSize),
                        Offset(marginX + j * cellSize, marginY + boardH), strokeWidth = 1.5f)
                }
            }

            drawLine(ChessColors.BoardLine, Offset(marginX + 3 * cellSize, marginY),
                Offset(marginX + 5 * cellSize, marginY + 2 * cellSize), strokeWidth = 1.5f)
            drawLine(ChessColors.BoardLine, Offset(marginX + 5 * cellSize, marginY),
                Offset(marginX + 3 * cellSize, marginY + 2 * cellSize), strokeWidth = 1.5f)
            drawLine(ChessColors.BoardLine, Offset(marginX + 3 * cellSize, marginY + 7 * cellSize),
                Offset(marginX + 5 * cellSize, marginY + 9 * cellSize), strokeWidth = 1.5f)
            drawLine(ChessColors.BoardLine, Offset(marginX + 5 * cellSize, marginY + 7 * cellSize),
                Offset(marginX + 3 * cellSize, marginY + 9 * cellSize), strokeWidth = 1.5f)

            val fontSizeSp = (pieceR * 1.0f / density).sp

            for (r in 0..9) for (c in 0..8) {
                val p = renderBoard[r][c]
                if (p.isEmpty) continue
                val vc = if (flipped) 8 - c else c
                val vr = if (flipped) 9 - r else r
                val cx = boardCX(vc)
                val cy = boardCY(vr)

                drawCircle(ChessColors.PieceBg, radius = pieceR, center = Offset(cx, cy))
                drawCircle(
                    if (p.side == PSide.RED) ChessColors.RedPiece else ChessColors.BlackPiece,
                    radius = pieceR, center = Offset(cx, cy), style = Stroke(width = 2.5f)
                )

                val name = PieceNames.name(p.type, p.side)
                val textColor = if (p.side == PSide.RED) ChessColors.RedPiece else ChessColors.BlackPiece
                val textLayoutResult = textMeasurer.measure(
                    name,
                    style = TextStyle(
                        color = textColor,
                        fontSize = fontSizeSp,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    ),
                    constraints = androidx.compose.ui.unit.Constraints(
                        minWidth = (pieceR * 2).toInt(),
                        maxWidth = (pieceR * 2).toInt(),
                        minHeight = (pieceR * 2).toInt(),
                        maxHeight = (pieceR * 2).toInt()
                    )
                )
                val textWidth = textLayoutResult.size.width.toFloat()
                val textHeight = textLayoutResult.size.height.toFloat()
                drawText(
                    textLayoutResult,
                    topLeft = Offset(cx - textWidth / 2, cy - textHeight / 2)
                )
            }

            for (m in validMoves) {
                val vc = if (flipped) 8 - m.toCol else m.toCol
                val vr = if (flipped) 9 - m.toRow else m.toRow
                val cx = boardCX(vc)
                val cy = boardCY(vr)
                val targetPiece = renderBoard[m.toRow][m.toCol]
                if (!targetPiece.isEmpty) {
                    drawCircle(ChessColors.ValidCapture, radius = pieceR + 2.dp.toPx(),
                        center = Offset(cx, cy), style = Stroke(width = 2.5f))
                } else {
                    drawCircle(ChessColors.ValidDot, radius = 5.dp.toPx(), center = Offset(cx, cy))
                }
            }

            if (selectedRow >= 0 && selectedCol >= 0) {
                val vc = if (flipped) 8 - selectedCol else selectedCol
                val vr = if (flipped) 9 - selectedRow else selectedRow
                val sx = boardCX(vc)
                val sy = boardCY(vr)
                drawCircle(ChessColors.SelectedGlow, radius = pieceR + 3.dp.toPx(),
                    center = Offset(sx, sy))
                drawCircle(ChessColors.Selected, radius = pieceR + 4.dp.toPx(),
                    center = Offset(sx, sy), style = Stroke(width = 3f))
            }

            if (lastMove != null) {
                val vfc = if (flipped) 8 - lastMove.fromCol else lastMove.fromCol
                val vfr = if (flipped) 9 - lastMove.fromRow else lastMove.fromRow
                val vtc = if (flipped) 8 - lastMove.toCol else lastMove.toCol
                val vtr = if (flipped) 9 - lastMove.toRow else lastMove.toRow
                val fx = boardCX(vfc); val fy = boardCY(vfr)
                val tx = boardCX(vtc); val ty = boardCY(vtr)
                drawLine(ChessColors.TrajectoryGlow, Offset(fx, fy), Offset(tx, ty), strokeWidth = 7f)
                drawLine(ChessColors.Trajectory, Offset(fx, fy), Offset(tx, ty), strokeWidth = 3f)
                val angle = kotlin.math.atan2((ty - fy).toDouble(), (tx - fx).toDouble()).toFloat()
                val ah = cellSize * 0.3f
                for (s in -1..1 step 2) {
                    val ax = tx - ah * kotlin.math.cos(angle - 0.45f * s)
                    val ay = ty - ah * kotlin.math.sin(angle - 0.45f * s)
                    drawLine(ChessColors.Trajectory, Offset(tx, ty), Offset(ax, ay), strokeWidth = 3f)
                }
            }
        }
    }
}
