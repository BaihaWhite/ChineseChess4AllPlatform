package com.chinesechess.ui

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.chinesechess.engine.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.jetbrains.compose.resources.ExperimentalResourceApi
import org.jetbrains.compose.resources.InternalResourceApi
import org.jetbrains.compose.resources.readResourceBytes

enum class Screen { MENU, SELECT, PLAY, LOBBY }

data class Difficulty(val label: String, val depth: Int, val evalNoise: Int, val bookNoiseChance: Float, val timeLimit: Long, val randomMoveChance: Float, val threads: Int = 1)

val DIFFICULTIES = listOf(
    Difficulty("初级", 2, 200, 0.50f, 1000L, 0.15f, 2),
    Difficulty("中级", 4, 50, 0.10f, 3000L, 0f, 4),
    Difficulty("高级", 7, 0, 0f, 10000L, 0f, 8),
    Difficulty("大师", 12, 0, 0f, 60000L, 0f, 16)
)

@Composable
fun App(
    engine: ChessEngine = remember { ChessEngine().also { it.initBoard() } },
    onExitApp: () -> Unit = {}
) {
    var screen by remember { mutableStateOf(Screen.MENU) }
    var lastMove by remember { mutableStateOf<Move?>(null) }
    var aiThinking by remember { mutableStateOf(false) }
    var userName by remember { mutableStateOf(engine.userName) }
    var showNameDialog by remember { mutableStateOf(false) }
    var showInviteDialog by remember { mutableStateOf<InviteInfo?>(null) }
    val scope = rememberCoroutineScope()
    var aiJob by remember { mutableStateOf<Job?>(null) }
    var currentAI by remember { mutableStateOf<AIController?>(null) }
    var aiDepth by remember { mutableIntStateOf(DIFFICULTIES[1].depth) }
    var difficulty by remember { mutableIntStateOf(1) } // 0=初级 1=中级 2=高级 3=大师
    var frozenBoard by remember { mutableStateOf<Array<Array<Piece>>?>(null) }
    var boardVersion by remember { mutableIntStateOf(0) }

    val networkClient = remember { createNetworkClient() }
    var openingBook by remember { mutableStateOf<OpeningBook?>(null) }

    @OptIn(ExperimentalResourceApi::class, InternalResourceApi::class)
    suspend fun loadOpeningBook(): OpeningBook? {
        return withContext(Dispatchers.Default) {
            try {
                DebugLog.info("Book", "Loading...")
                val bytes = readResourceBytes("composeResources/chinese_chess.composeapp.generated.resources/files/opening_book.txt")
                if (bytes.isNotEmpty()) {
                    OpeningBook().apply {
                        deserialize(bytes.decodeToString())
                        DebugLog.info("Book", "Loaded $size entries")
                    }
                } else {
                    DebugLog.warn("Book", "File empty")
                    null
                }
            } catch (e: Exception) {
                DebugLog.error("Book", "Failed: ${e.message}")
                null
            }
        }
    }

    LaunchedEffect(Unit) {
        openingBook = loadOpeningBook()
    }

    fun cancelAI() {
        currentAI?.cancel()
        aiJob?.cancel()
        aiJob = null
        currentAI = null
        frozenBoard = null
        aiThinking = false
    }

    fun startAI(engine: ChessEngine, onMove: (Move?) -> Unit) {
        cancelAI()
        frozenBoard = engine.copyBoard()
        aiThinking = true
        aiJob = scope.launch {
            try {
                delay(300)
                val ai = AIController(engine, openingBook)
                currentAI = ai
                val diff = DIFFICULTIES[difficulty.coerceIn(0, 3)]
                DebugLog.info("AI", "${diff.label} depth=${diff.depth} time=${diff.timeLimit}ms")
                val move = withContext(Dispatchers.Default) {
                    ai.getAIMove(diff.depth, diff.evalNoise, diff.bookNoiseChance, diff.timeLimit, diff.randomMoveChance, diff.threads)
                }
                currentAI = null
                frozenBoard = null
                if (move != null) {
                    engine.executeMove(move.fromRow, move.fromCol, move.toRow, move.toCol)
                    engine.lastAIMove = move
                    DebugLog.debug("AI", "Move ${move.fromRow},${move.fromCol}->${move.toRow},${move.toCol}")
                } else {
                    engine.gameOver = true
                    engine.setDrawGame(false)
                    DebugLog.info("AI", "No moves — loses")
                }
                aiThinking = false
                boardVersion++
                onMove(move)
            } catch (e: Exception) {
                DebugLog.error("AI", "${e.message}")
                currentAI = null
                frozenBoard = null
                aiThinking = false
                onMove(null)
            }
        }
    }

    MaterialTheme(colorScheme = darkColorScheme()) {
        val clipboardManager = LocalClipboardManager.current
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(ChessColors.Background)
                .windowInsetsPadding(WindowInsets.safeDrawing),
            contentAlignment = Alignment.TopCenter
        ) {
            when (screen) {
                Screen.MENU -> MenuScreen(
                    userName = userName,
                    onVsAi = {
                        engine.gameMode = GameMode.VS_AI
                        screen = Screen.SELECT
                    },
                    onVsHuman = {
                        engine.gameMode = GameMode.VS_HUMAN
                        engine.playerSide = PSide.RED
                        engine.resetGame()
                        lastMove = null
                        screen = Screen.PLAY
                    },
                    onOnline = {
                        if (userName.isEmpty()) {
                            showNameDialog = true
                        } else {
                            engine.gameMode = GameMode.VS_ONLINE
                            networkClient.init(userName)
                            screen = Screen.LOBBY
                            scope.launch {
                                while (screen == Screen.LOBBY) {
                                    val invite = networkClient.poll()
                                    if (invite != null) showInviteDialog = invite
                                    delay(2000)
                                }
                            }
                        }
                    },
                    onExit = onExitApp,
                    onNameClick = { showNameDialog = true }
                )
                Screen.SELECT -> SelectScreen(
                    difficulty = difficulty,
                    onDifficultyChange = {
                        difficulty = it
                        aiDepth = DIFFICULTIES[it].depth
                    },
                    onRed = {
                        engine.playerSide = PSide.RED
                        engine.resetGame()
                        lastMove = null
                        screen = Screen.PLAY
                    },
                    onBlack = {
                        engine.playerSide = PSide.BLACK
                        engine.resetGame()
                        lastMove = null
                        screen = Screen.PLAY
                        startAI(engine) { lastMove = it }
                    },
                    onBack = { screen = Screen.MENU }
                )
                Screen.PLAY -> GameScreen(
                    engine = engine,
                    lastMove = lastMove,
                    aiThinking = aiThinking,
                    frozenBoard = frozenBoard,
                    boardVersion = boardVersion,
                    onBoardChanged = { boardVersion++ },
                    networkClient = networkClient,
                    onLastMoveUpdate = { lastMove = it },
                    onAiThinkingUpdate = { aiThinking = it },
                    onStartAI = { startAI(engine) { lastMove = it } },
                    onCancelAI = { cancelAI() },
                    onBack = {
                        cancelAI()
                        if (networkClient.isOnline) networkClient.cleanup()
                        engine.resetGame()
                        lastMove = null
                        screen = Screen.MENU
                    }
                )
                Screen.LOBBY -> LobbyScreen(
                    userName = userName,
                    myIp = networkClient.myIp,
                    discoveredUsers = networkClient.discoveredUsers,
                    onRefresh = { networkClient.refreshUsers() },
                    onBack = {
                        networkClient.cleanup()
                        screen = Screen.MENU
                    }
                )
            }

            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(8.dp)
                    .clip(RoundedCornerShape(6.dp))
                    .background(ChessColors.CardBg.copy(alpha = 0.7f))
                    .clickable {
                        val logs = DebugLog.getRecent(80)
                        clipboardManager.setText(AnnotatedString(logs))
                    }
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Text("📋 日志", fontSize = 11.sp, color = ChessColors.TextSecondary)
            }
        }

        if (showNameDialog) {
            NameDialog(
                initialName = userName,
                onConfirm = {
                    userName = it
                    engine.userName = it
                    showNameDialog = false
                },
                onDismiss = { showNameDialog = false }
            )
        }

        if (showInviteDialog != null) {
            val invite = showInviteDialog!!
            InviteDialog(
                fromName = invite.fromName,
                onRed = {
                    networkClient.acceptInvite(PSide.RED)
                    engine.resetGame()
                    lastMove = null
                    screen = Screen.PLAY
                    showInviteDialog = null
                },
                onBlack = {
                    networkClient.acceptInvite(PSide.BLACK)
                    engine.resetGame()
                    lastMove = null
                    screen = Screen.PLAY
                    showInviteDialog = null
                },
                onReject = { showInviteDialog = null }
            )
        }
    }
}

@Composable
fun MenuScreen(
    userName: String,
    onVsAi: () -> Unit,
    onVsHuman: () -> Unit,
    onOnline: () -> Unit,
    onExit: () -> Unit,
    onNameClick: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(24.dp))
        Text("中国象棋", fontSize = 36.sp, fontWeight = FontWeight.Bold, color = ChessColors.TextPrimary)
        Spacer(Modifier.height(4.dp))
        Text("Chinese Chess", fontSize = 16.sp, color = ChessColors.TextSecondary)
        Spacer(Modifier.height(40.dp))

        val buttons = listOf(
            Triple("人机对战", ChessColors.ButtonBlue, onVsAi),
            Triple("面对面对战", ChessColors.ButtonGreen, onVsHuman),
            Triple("联机大厅", ChessColors.ButtonOrange, onOnline),
            Triple("退出游戏", ChessColors.ButtonGray, onExit)
        )

        for ((text, color, action) in buttons) {
            Box(
                modifier = Modifier
                    .width(280.dp).height(52.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(color)
                    .clickable { action() },
                contentAlignment = Alignment.Center
            ) {
                Text(text, color = androidx.compose.ui.graphics.Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(12.dp))
        }

        Spacer(Modifier.weight(1f))

        Box(
            modifier = Modifier
                .width(220.dp).height(40.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(ChessColors.CardBg)
                .clickable { onNameClick() },
            contentAlignment = Alignment.Center
        ) {
            if (userName.isEmpty()) {
                Text("点击登录", color = ChessColors.TextSecondary, fontSize = 14.sp)
            } else {
                Text(userName, color = ChessColors.TextPrimary, fontSize = 14.sp)
            }
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
fun SelectScreen(
    difficulty: Int,
    onDifficultyChange: (Int) -> Unit,
    onRed: () -> Unit,
    onBlack: () -> Unit,
    onBack: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(24.dp))
        Text("选择先后手", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = ChessColors.TextPrimary)
        Spacer(Modifier.height(32.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
            SideButton("执红先行", "帅", ChessColors.RedPiece, onRed)
            SideButton("执黑后行", "将", ChessColors.BlackPiece, onBlack)
        }

        Spacer(Modifier.height(40.dp))

        val diff = DIFFICULTIES[difficulty.coerceIn(0, 3)]
        Text("AI 难度", fontSize = 18.sp, fontWeight = FontWeight.Medium, color = ChessColors.TextPrimary)
        Spacer(Modifier.height(4.dp))
        Text(diff.label + " · 深度 ${diff.depth} · ${diff.timeLimit / 1000}s · ≤${diff.threads}线程", fontSize = 13.sp, color = ChessColors.TextSecondary)
        Spacer(Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("初级", fontSize = 12.sp, color = if (difficulty == 0) ChessColors.Gold else ChessColors.TextSecondary)
            Slider(
                value = difficulty.toFloat(),
                onValueChange = { onDifficultyChange(it.toInt().coerceIn(0, 3)) },
                valueRange = 0f..3f,
                steps = 2,
                modifier = Modifier.weight(1f),
                colors = SliderDefaults.colors(
                    thumbColor = ChessColors.ButtonBlue,
                    activeTrackColor = ChessColors.ButtonBlue
                )
            )
            Text("大师", fontSize = 12.sp, color = if (difficulty == 3) ChessColors.Gold else ChessColors.TextSecondary)
        }

        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            DIFFICULTIES.forEachIndexed { idx, d ->
                Text(
                    d.label,
                    fontSize = 11.sp,
                    color = if (idx == difficulty) ChessColors.Gold else ChessColors.TextSecondary
                )
            }
        }

        Spacer(Modifier.height(32.dp))

        Box(
            modifier = Modifier
                .width(280.dp).height(44.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(ChessColors.CardBg)
                .clickable { onBack() },
            contentAlignment = Alignment.Center
        ) {
            Text("返回主菜单", color = ChessColors.TextSecondary, fontSize = 16.sp)
        }
    }
}

@Composable
fun RowScope.SideButton(label: String, king: String, color: androidx.compose.ui.graphics.Color, onClick: () -> Unit) {
    Column(
        modifier = Modifier
            .weight(1f)
            .clip(RoundedCornerShape(12.dp))
            .background(color)
            .clickable { onClick() }
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(label, color = androidx.compose.ui.graphics.Color.White, fontSize = 16.sp)
        Spacer(Modifier.height(12.dp))
        Box(
            modifier = Modifier.size(48.dp).clip(RoundedCornerShape(24.dp)).background(ChessColors.PieceBg),
            contentAlignment = Alignment.Center
        ) {
            Text(king, color = color, fontSize = 24.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun GameScreen(
    engine: ChessEngine,
    lastMove: Move?,
    aiThinking: Boolean,
    frozenBoard: Array<Array<Piece>>?,
    boardVersion: Int,
    onBoardChanged: () -> Unit,
    networkClient: NetworkClient,
    onLastMoveUpdate: (Move?) -> Unit,
    onAiThinkingUpdate: (Boolean) -> Unit,
    onStartAI: () -> Unit,
    onCancelAI: () -> Unit,
    onBack: () -> Unit
) {
    val flipped = engine.gameMode == GameMode.VS_AI && engine.playerSide == PSide.BLACK
    var selRow by remember { mutableIntStateOf(-1) }
    var selCol by remember { mutableIntStateOf(-1) }
    var validMoves by remember { mutableStateOf(emptyList<Move>()) }

    fun syncSelection() {
        selRow = engine.selRow
        selCol = engine.selCol
        validMoves = engine.validMoves
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().background(ChessColors.HeaderBg).padding(horizontal = 8.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            val modeLabel = when {
                networkClient.isOnline -> "联机对战"
                engine.gameMode == GameMode.VS_AI -> "人机对战"
                else -> "面对面"
            }
            Text(modeLabel, fontSize = 12.sp, color = ChessColors.TextSecondary, modifier = Modifier.weight(1f), textAlign = TextAlign.Center)
        }

        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            GameButton("重新开始", ChessColors.ButtonBlue) {
                onCancelAI()
                engine.resetGame()
                selRow = -1; selCol = -1; validMoves = emptyList()
                onBoardChanged()
                onLastMoveUpdate(null)
                if (flipped) {
                    onStartAI()
                }
            }
            GameButton("悔  棋", ChessColors.ButtonOrange) {
                if (engine.gameOver || aiThinking || !engine.canUndo()) return@GameButton
                onCancelAI()
                if (engine.gameMode == GameMode.VS_AI) {
                    engine.undoMove()
                    engine.undoMove()
                    DebugLog.debug("Game", "Undo 2")
                } else {
                    engine.undoMove()
                    DebugLog.debug("Game", "Undo 1")
                }
                engine.lastAIMove = null
                selRow = -1; selCol = -1; validMoves = emptyList()
                onBoardChanged()
                onLastMoveUpdate(null)
                if (flipped && engine.isAiTurn()) {
                    onStartAI()
                }
            }
            GameButton("返  回", ChessColors.ButtonRed) { onBack() }
        }

        Row(
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
            horizontalArrangement = Arrangement.Center
        ) {
            val turnText = when {
                engine.gameOver -> engine.winnerText()
                aiThinking -> "思考中..."
                engine.currentTurn == PSide.RED -> "红方走棋"
                else -> "黑方走棋"
            }
            val turnColor = when {
                engine.gameOver -> ChessColors.Gold
                aiThinking -> ChessColors.TextSecondary
                engine.currentTurn == PSide.RED -> ChessColors.RedTurn
                else -> ChessColors.BlackTurn
            }
            Text(turnText, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = turnColor)
        }

        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            key(boardVersion) {
            ChessBoard(
                engine = engine,
                flipped = flipped,
                lastMove = lastMove,
                selectedRow = selRow,
                selectedCol = selCol,
                frozenBoard = frozenBoard,
                validMoves = validMoves,
                onCellClick = { row, col ->
                    if (engine.gameOver || aiThinking) return@ChessBoard

                    val beforeR = engine.selRow
                    val beforeC = engine.selCol
                    val result = engine.clickCell(row, col)
                    syncSelection()

                    if (result == 2) {
                        selRow = -1; selCol = -1; validMoves = emptyList()
                        onBoardChanged()
                        val move = if (beforeR >= 0 && beforeC >= 0) Move(beforeR, beforeC, row, col) else null
                        onLastMoveUpdate(move)
                        if (networkClient.isOnline) {
                            move?.let { networkClient.sendMove(it) }
                        }
                        if (engine.isAiTurn() && !engine.gameOver) {
                            onStartAI()
                        }
                    }
                }
            )
            }

            if (aiThinking) {
                Box(modifier = Modifier.fillMaxSize().background(ChessColors.AIOverlay))
            }

            if (engine.gameOver) {
                Box(
                    modifier = Modifier.fillMaxSize().background(ChessColors.Overlay),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(engine.winnerText(), fontSize = 42.sp, fontWeight = FontWeight.Bold, color = ChessColors.Gold)
                        Spacer(Modifier.height(16.dp))
                        Text("点击重新开始", fontSize = 16.sp, color = ChessColors.TextSecondary)
                    }
                }
            }
        }
    }

    LaunchedEffect(networkClient.isOnline) {
        while (networkClient.isOnline) {
            val move = networkClient.recvMove()
            if (move != null) {
                engine.executeMove(move.fromRow, move.fromCol, move.toRow, move.toCol)
                onLastMoveUpdate(move)
            }
            delay(1500)
        }
    }
}

@Composable
fun GameButton(text: String, color: androidx.compose.ui.graphics.Color, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .height(40.dp)
            .clip(RoundedCornerShape(6.dp))
            .background(color)
            .clickable { onClick() }
            .padding(horizontal = 16.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(text, color = androidx.compose.ui.graphics.Color.White, fontSize = 14.sp)
    }
}

@Composable
fun LobbyScreen(
    userName: String,
    myIp: String,
    discoveredUsers: List<OnlineUser>,
    onRefresh: () -> Unit,
    onBack: () -> Unit
) {
    var users by remember { mutableStateOf(discoveredUsers) }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(24.dp))
        Text("联机大厅", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = ChessColors.ButtonOrange)
        Spacer(Modifier.height(4.dp))
        Text("我: $userName · $myIp", fontSize = 13.sp, color = ChessColors.TextSecondary)

        Spacer(Modifier.height(12.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Box(
                modifier = Modifier
                    .height(36.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(ChessColors.ButtonOrange)
                    .clickable {
                        onRefresh()
                        users = discoveredUsers
                    }
                    .padding(horizontal = 20.dp),
                contentAlignment = Alignment.Center
            ) {
                Text("刷新", color = androidx.compose.ui.graphics.Color.White, fontSize = 14.sp)
            }
            Box(
                modifier = Modifier
                    .height(36.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(ChessColors.CardBg)
                    .clickable { onBack() }
                    .padding(horizontal = 20.dp),
                contentAlignment = Alignment.Center
            ) {
                Text("返回", color = ChessColors.TextSecondary, fontSize = 14.sp)
            }
        }

        Spacer(Modifier.height(12.dp))

        if (users.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .clip(RoundedCornerShape(8.dp))
                    .background(ChessColors.CardBg),
                contentAlignment = Alignment.Center
            ) {
                Text("暂无在线用户\n点击「刷新」扫描局域网", fontSize = 14.sp, color = ChessColors.TextSecondary, textAlign = TextAlign.Center)
            }
        } else {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .clip(RoundedCornerShape(8.dp))
                    .background(ChessColors.CardBg)
                    .padding(8.dp)
            ) {
                Text("在线用户 (${users.size})", fontSize = 13.sp, color = ChessColors.TextSecondary)
                Spacer(Modifier.height(8.dp))
                users.forEach { user ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(6.dp))
                            .background(ChessColors.HeaderBg)
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(user.name, fontSize = 14.sp, color = ChessColors.TextPrimary)
                            Text(user.ip, fontSize = 11.sp, color = ChessColors.TextSecondary)
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                }
            }
        }
    }
}

@Composable
fun NameDialog(initialName: String, onConfirm: (String) -> Unit, onDismiss: () -> Unit) {
    var name by remember { mutableStateOf(initialName) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("输入昵称") },
        text = {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it },
                singleLine = true
            )
        },
        confirmButton = {
            TextButton(onClick = { onConfirm(name) }) { Text("确定") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("取消") }
        }
    )
}

@Composable
fun InviteDialog(fromName: String, onRed: () -> Unit, onBlack: () -> Unit, onReject: () -> Unit) {
    AlertDialog(
        onDismissRequest = onReject,
        title = { Text("对战邀请") },
        text = { Text("$fromName 邀请你对战") },
        confirmButton = {
            Row {
                TextButton(onClick = onRed) { Text("执红") }
                TextButton(onClick = onBlack) { Text("执黑") }
            }
        },
        dismissButton = {
            TextButton(onClick = onReject) { Text("拒绝") }
        }
    )
}
