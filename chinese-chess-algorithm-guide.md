# 中国象棋算法实现指南

> 基于 Pikafish (Stockfish 派生)、ElephantEye、Fairy-Stockfish 等知名开源项目的实战经验总结。

## 目录

1. [核心设计原则](#1-核心设计原则)
2. [数据结构选型](#2-数据结构选型)
3. [走法生成技术](#3-走法生成技术)
4. [局面评估函数](#4-局面评估函数)
5. [搜索算法](#5-搜索算法)
6. [优化策略](#6-优化策略)
7. [测试方法论](#7-测试方法论)
8. [附录：算法演进路线图](#8-附录算法演进路线图)

---

## 1. 核心设计原则

### 1.1 引擎与 UI 解耦

将棋局逻辑与用户界面完全分离，这是所有成熟象棋引擎（Pikafish、Stockfish）的共同架构：

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   UI 层      │────▶│  游戏控制器   │────▶│  引擎核心    │
│ (ChessBoard) │◀────│  (App.kt)    │◀────│ (Engine/AI) │
└─────────────┘     └──────────────┘     └─────────────┘
```

- **引擎核心**：纯逻辑层，无 UI 依赖，可独立测试
- **游戏控制器**：管理回合、胜负判定的胶水层
- **UI 层**：只负责渲染和用户输入转发

### 1.2 不可变数据优先

搜索过程中，AI 需要对棋盘状态进行大量克隆。设计上应确保：

- 棋子数据是可复制（Copy/Clone）的轻量结构
- 搜索中的走子/悔子操作为 O(1)
- 避免在搜索路径上产生 GC 压力

### 1.3 正确性优先于速度

实现顺序严格遵守：
1. 先实现完全正确的走法生成
2. 再实现正确的胜负判定
3. 然后加入搜索算法
4. 最后进行性能优化

---

## 2. 数据结构选型

### 2.1 棋盘表示

三种主流方案对比：

| 方案 | 内存占用 | 走法生成速度 | 实现难度 | 代表项目 |
|------|---------|------------|---------|---------|
| 10×9 二维数组 (Mailbox) | 90 cells | 中等 | ★☆☆☆☆ | 本指南示例、象眼 1.x |
| 128-bit Bitboard | 2×128 bits | 快速 | ★★★★☆ | Pikafish、Fairy-SF |
| 256-bit Bitboard | 2×256 bits | 最快 | ★★★★★ | 部分商业引擎 |

#### 方案 A：二维数组（Mailbox）— 推荐入门

```kotlin
// 10 行 × 9 列的二维数组，每个格子存棋子信息
val board = Array(10) { Array(9) { Piece() } }

data class Piece(
    val type: PType = PType.EMPTY,   // 棋子类型
    val side: PSide = PSide.NONE      // 红方/黑方/空
)

enum class PType { EMPTY, KING, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, PAWN }
enum class PSide { RED, BLACK, NONE }
```

**优点**：直观，调试方便，适合理解搜索算法原理。
**缺点**：走法生成需遍历棋盘，速度较 Bitboard 慢。

#### 方案 B：128-bit Bitboard — 生产级方案（Pikafish 做法）

Pikafish 继承了 Stockfish 的 Bitboard 架构，用两个 64-bit 整数表示一种颜色的所有棋子位置。由于中国象棋有 90 个交叉点，需要扩展为 128-bit bitboard：

```cpp
// C++ 伪代码 (Pikafish 实际结构)
struct Bitboard128 {
    uint64_t lo;  // 低 64 位（行列 0-63 的格子）
    uint64_t hi;  // 高 64 位（行列 64-89 的格子）
};

// 每种棋子类型都有独立的 bitboard
Bitboard128 pawns[2];    // 兵/卒
Bitboard128 knights[2];  // 马
Bitboard128 bishops[2];  // 象
// ... 所有棋子类型
Bitboard128 occupied[2]; // 每方所有棋子
```

Bitboard 通过预计算的攻击表（attack tables）实现走法生成，核心是一次位运算即可得到所有可能目标位。

### 2.2 走法表示

Pikafish 使用 16-bit 整数编码走法，极致紧凑：

```
Bit: 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
     [ 目标格 7bit ] [ 起始格 7bit ] [类型2bit]
```

对于入门实现，直接使用结构体更清晰：

```kotlin
data class Move(
    val fromRow: Int,
    val fromCol: Int,
    val toRow: Int,
    val toCol: Int
)
```

### 2.3 走法记录与悔棋

```kotlin
data class MoveRecord(
    val fromRow: Int,
    val fromCol: Int,
    val toRow: Int,
    val toCol: Int,
    val captured: Piece    // 被吃掉的棋子（悔棋还原用）
)

// 引擎中维护
private val moveHistory = mutableListOf<MoveRecord>()
```

---

## 3. 走法生成技术

### 3.1 合法性判定层次

一个合法的走法必须满足以下所有条件（判定顺序影响性能）：

```
第一层（最快，纯几何判定）
  ├─ 目标格在棋盘内
  ├─ 目标格与起始格不同
  ├─ 目标格上无己方棋子
  └─ 走法符合棋子类型规则
        ├─ 帅/将：九宫内一步直行
        ├─ 仕/士：九宫内对角一步
        ├─ 相/象：田字对角 + 象眼检查 + 不过河
        ├─ 馬：日字 + 蹩脚检查
        ├─ 車：直线移动无遮挡
        ├─ 砲/炮：直线移动，吃子时隔一子，不吃时无遮挡
        └─ 兵/卒：未过河前只进，过河后可左右前

第二层（模拟走子后检查）
  ├─ 不会导致己方被将（走后不能送将）
  ├─ 不会导致将帅对面
  └─ 不会导致长捉违规
```

### 3.2 典型实现：車的走法生成

```kotlin
private fun countBetween(fx: Int, fy: Int, tx: Int, ty: Int): Int {
    var cnt = 0
    val dx = when { tx > fx -> 1; tx < fx -> -1; else -> 0 }
    val dy = when { ty > fy -> 1; ty < fy -> -1; else -> 0 }
    var cx = fx + dx
    var cy = fy + dy
    while (cx != tx || cy != ty) {
        if (board[cx][cy].type != PType.EMPTY) cnt++
        cx += dx; cy += dy
    }
    return cnt
}

// 車：直线无遮挡
PType.CHARIOT -> (dx == 0 || dy == 0) && countBetween(fx, fy, tx, ty) == 0

// 炮：直线，吃子时隔一子(dst 非空且挡一子)，不吃时无遮挡(dst 空且无遮挡)
PType.CANNON -> (dx == 0 || dy == 0) &&
    if (board[tx][ty].type == PType.EMPTY) countBetween(fx, fy, tx, ty) == 0
    else countBetween(fx, fy, tx, ty) == 1
```

### 3.3 全量合法走法收集

```kotlin
fun getAllLegalMoves(side: PSide): List<Move> {
    val moves = mutableListOf<Move>()
    for (fromRow in 0..9) {
        for (fromCol in 0..8) {
            if (board[fromRow][fromCol].side != side) continue
            for (toRow in 0..9) {
                for (toCol in 0..8) {
                    if (isLegalMove(fromRow, fromCol, toRow, toCol)) {
                        moves.add(Move(fromRow, fromCol, toRow, toCol))
                    }
                }
            }
        }
    }
    return moves
}
```

### 3.4 Bitboard 走法生成（进阶）

Pikafish 的做法是预计算攻击表。以马为例：

```cpp
// 预计算：对每个格子，存储马能跳到的所有目标位的 bitboard
Bitboard128 knightAttacks[90];

void initKnightAttacks() {
    for (int sq = 0; sq < 90; sq++) {
        // 马的8个可能方向，过滤蹩脚位
        for (int d = 0; d < 8; d++) {
            int to = sq + knightOffsets[d];
            int leg = sq + knightLegOffsets[d];
            if (isOnBoard(to) && !isOnBoard(leg)) {
                // 此处简化，实际需检查 leg 位不在棋盘边界
                knightAttacks[sq].setBit(to);
            }
        }
    }
}

// 走法生成变为一次查表 + 一次位运算
Bitboard128 targets = knightAttacks[fromSq] & ~occupied[ownSide];
```

---

## 4. 局面评估函数

### 4.1 基础子力价值

参考 Pikafish 和多个学术文献的典型权重（单位为"分"，帅/将为无穷大）：

```kotlin
val baseValues = mapOf(
    PType.KING     to 10000,  // 帅/将：被将死即输，设极大值
    PType.CHARIOT  to 900,    // 車：最强棋子
    PType.CANNON   to 450,    // 炮：中期强，残局弱
    PType.HORSE    to 400,    // 马：灵活，但怕蹩脚
    PType.ELEPHANT to 220,    // 象/相：防守型，不过河
    PType.ADVISOR  to 180,    // 仕/士：近身护卫
    PType.PAWN     to 100,    // 兵/卒：基础值，需加上位置加成
    PType.EMPTY    to 0
)
```

**关键细节**：仕(180) 和 相(220) 不能设为相同值。相能守更多位置（田字范围），价值应略高于仕。

### 4.2 位置加成表（Piece-Square Tables）

兵/卒的过河加成是最基本也最重要的位置评估：

```kotlin
fun pawnBonus(row: Int, side: PSide): Int {
    if (side == PSide.RED) {
        return when (row) {
            6 -> 0     // 初始位置
            5 -> 20    // 刚过河
            4 -> 50    // 深入敌阵
            3 -> 100   // 接近底线
            else -> 170 // 底线兵（威力最大）
        }
    } else {
        return when (row) {
            3 -> 0
            4 -> 20
            5 -> 50
            6 -> 100
            else -> 170
        }
    }
}
```

**进一步优化的方向**：
- 马的位置表：中心位置价值高，边角低
- 炮的位置表：中期有炮架时价值高，残局无炮架时价值降
- 車的位置表：占据要道（肋道、卒林线）加分

### 4.3 评估函数实现

```kotlin
private fun evaluate(board: Array<Array<Piece>>): Int {
    var score = 0
    for (row in 0..9) {
        for (col in 0..8) {
            val piece = board[row][col]
            if (piece.type == PType.EMPTY) continue

            var value = baseValues[piece.type]!!

            // 兵/卒的过河加成
            if (piece.type == PType.PAWN) {
                value += pawnBonus(row, piece.side)
            }

            // 红方加分，黑方减分（从红方视角）
            if (piece.side == PSide.RED) score += value
            else score -= value
        }
    }
    return score  // 正数 = 红优，负数 = 黑优
}
```

### 4.4 NNUE 评估（Pikafish 核心）

Pikafish 的核心优势是 NNUE（Efficiently Updatable Neural Network）评估：

```
传统评估：手工特征 + 手工权重 → 难以调优
NNUE评估：神经网络自动学习权重 → 棋力远超手工评估
```

NNUE 架构简示：

```
输入层 (棋盘特征向量)
  ↓
第1隐藏层 (全连接 + ReLU) × 2 → 拼接
  ↓
第2隐藏层 (全连接 + ReLU)
  ↓
输出层 (单个分数值)
```

关键创新——增量更新：走一步棋时，不需要重新计算整个网络，只需更新受影响的几个神经元，将每步评估从 O(n) 降到 O(1)。

**入门建议**：先用传统评估函数跑通全流程，再考虑引入 NNUE。

---

## 5. 搜索算法

### 5.1 Minimax 基础

```
原理：我方选最大收益，对方选最小收益（对我方最不利）

搜索树示意 (depth=2)：

         [当前局面]
        /    |    \
    走法1  走法2  走法3    ← 我方走 (MAX)
     /|\    /|\    /|\
   ...    ...    ...       ← 对方走 (MIN)
```

### 5.2 Alpha-Beta 剪枝

```kotlin
private fun minimax(
    engine: ChessEngine,
    depth: Int,
    alpha: Int,      // 当前路径已确保的最小值下界
    beta: Int,       // 当前路径已确保的最大值上界
    maximizing: Boolean  // 当前是 MAX 还是 MIN 层
): Int {
    if (depth == 0) return evaluate(engine.board)

    val side = engine.currentTurn  // 重要：用实际轮次而非 hardcode
    val moves = engine.getAllLegalMoves(side)

    if (moves.isEmpty()) {
        // 无子可走 = 被将死 或 困毙
        return if (maximizing) -99999 else 99999
    }

    if (maximizing) {
        var best = -99999
        var a = alpha
        for (move in moves) {
            val captured = engine.executeMoveSimple(move)
            val value = minimax(engine, depth - 1, a, beta, false)
            engine.undoMoveSimple(move, captured)
            if (value > best) best = value
            if (best > a) a = best
            if (a >= beta) break  // Beta 剪枝
        }
        return best
    } else {
        var best = 99999
        var b = beta
        for (move in moves) {
            val captured = engine.executeMoveSimple(move)
            val value = minimax(engine, depth - 1, alpha, b, true)
            engine.undoMoveSimple(move, captured)
            if (value < best) best = value
            if (best < b) b = best
            if (alpha >= b) break  // Alpha 剪枝
        }
        return best
    }
}
```

**剪枝效率**：最优情况下（走法已排好序），搜索节点数从 O(b^d) 降为 O(b^(d/2))，同等时间可搜索约 2 倍深度。

### 5.3 多线程并行搜索（根节点并行化）

根节点的每个候选走法是一条独立搜索路径，天然可并行：

```kotlin
suspend fun getAIMove(searchDepth: Int): Move? = coroutineScope {
    val aiSide = if (engine.playerSide == PSide.RED) PSide.BLACK else PSide.RED
    val moves = engine.getAllLegalMoves(aiSide)
    if (moves.isEmpty()) return@coroutineScope null

    val maximizing = aiSide == PSide.BLACK

    val results = moves.map { move ->
        async {
            // 每个协程使用独立的棋盘副本
            val clone = engine.clone()
            clone.currentTurn = aiSide
            clone.executeMoveSimple(move.fromRow, move.fromCol, move.toRow, move.toCol)
            val value = minimax(clone, searchDepth - 1, -99999, 99999, maximizing)
            Pair(move, value)
        }
    }.mapNotNull { it.await() }

    // 根据 AI 方颜色选最佳走法
    if (aiSide == PSide.RED) results.maxByOrNull { it.second }?.first
    else results.minByOrNull { it.second }?.first
}
```

**重要**：每个线程/协程必须操作独立的棋盘副本，不能共享同一个棋盘状态！

### 5.4 高级搜索技术（Pikafish 路线）

| 技术 | 简要说明 | 提升效果 |
|------|---------|---------|
| **迭代加深** | depth=1 搜到 depth=N，每次加深可利用上次结果排序 | 时间控制、走法排序 |
| **PVS (Principal Variation Search)** | 假设第一个走法最好，其余用零窗口搜索 | 约 10-20% 加速 |
| **空着裁剪 (Null Move Pruning)** | 让对方走两步再评估，若仍 >= beta 则裁剪 | 约 30-50% 加速 |
| **静态搜索 (Quiescence Search)** | 到达叶子节点后继续搜索吃子走法 | 消除"地平线效应" |
| **杀手走法 (Killer Moves)** | 记录同一层引发剪枝的走法，优先尝试 | 提高剪枝效率 |
| **历史启发 (History Heuristic)** | 记录每对 [from][to] 的剪枝次数，优先排序 | 持续优化走法顺序 |

### 5.5 走法排序（Move Ordering）— 最关键的性能优化

Alpha-Beta 的剪枝效率极度依赖走法排序。排序越接近最优，剪枝越多。

排序优先级（从高到低）：

```
1. 置换表中的最佳走法（如果实现了 TT）
2. 吃子走法，按 MVV-LVA 排序
   MVV-LVA = Most Valuable Victim - Least Valuable Attacker
   即：用最小代价吃最大价值的子排最前
3. 杀手走法
4. 历史表分数高的走法
5. 其他安静走法（非吃子走法）
```

MVV-LVA 实现示例：

```kotlin
private fun mvvLvaScore(move: Move, board: Array<Array<Piece>>): Int {
    val victim = board[move.toRow][move.toCol]
    val attacker = board[move.fromRow][move.fromCol]
    if (victim.type == PType.EMPTY) return 0
    // 被吃子价值 × 10 - 攻击子价值：确保吃車优于吃兵
    return baseValues[victim.type.ordinal] * 10 - baseValues[attacker.type.ordinal]
}
```

---

## 6. 优化策略

### 6.1 Zobrist 哈希 + 置换表（Transposition Table）

**问题**：不同走法顺序可能到达相同局面，重复搜索浪费计算。

**Zobrist 哈希原理**：

```kotlin
// 预生成随机数表
// 维度：[棋子类型][颜色][行][列]，共 7×2×10×9 = 1260 个随机数
val zobristTable = Array(7) { Array(2) { Array(10) { Array(9) { randomULong() } } } }

// 哈希更新——走子时只需异或4个数（走子前/后、吃子前/后），O(1) 操作
fun zobristUpdate(hash: ULong, move: Move, piece: Piece, captured: Piece): ULong {
    var h = hash
    h = h xor zobristTable[piece.type][piece.side][move.fromRow][move.fromCol]  // 移除原棋子
    h = h xor zobristTable[piece.type][piece.side][move.toRow][move.toCol]      // 放置到目标
    if (captured.type != PType.EMPTY)
        h = h xor zobristTable[captured.type][captured.side][move.toRow][move.toCol] // 移除被吃子
    return h
}
```

**置换表结构**：

```kotlin
data class TTEntry(
    val hash: ULong,        // 验证用
    val depth: Int,         // 搜索深度
    val score: Int,         // 评估分数
    val flag: TTFlag,       // EXACT / LOWER_BOUND / UPPER_BOUND
    val bestMove: Move?     // 最佳走法（用于走法排序）
)

enum class TTFlag { EXACT, LOWER_BOUND, UPPER_BOUND }

// 对于 2^20 (约百万) 个槽位的置换表
val transpositionTable = Array(1 shl 20) { TTEntry() }
```

**Pikafish 的置换表策略**：
- 使用 4 路组相联缓存
- Always-Replace 策略 + 深度优先保护
- 多线程共享，使用无锁并发

### 6.2 重复局面检测

```kotlin
private val positionHistory = mutableListOf<ULong>()

// 每步走后记录哈希
positionHistory.add(zobristHash)

// 检查是否可能长捉
fun wouldRepeatCheck(fx: Int, fy: Int, tx: Int, ty: Int): Boolean {
    // 模拟走子 → 检查对方是否被将 → 检查此局面是否出现过
    val cap = board[tx][ty]
    board[tx][ty] = board[fx][fy]
    board[fx][fy] = Piece()

    val inCheck = isInCheck(opponent)
    val repeated = inCheck && positionHistory.contains(boardHash())

    // 还原
    board[fx][fy] = board[tx][ty]
    board[tx][ty] = cap
    return repeated
}
```

### 6.3 开局库（Opening Book）

原理：预存常见开局走法，开局时直接查表而不搜索。

```kotlin
// 简单开局库：局面哈希 → 推荐走法列表
val openingBook = mutableMapOf<ULong, List<Move>>()

fun loadOpeningBook(path: String) {
    // 从文件加载预计算的 PGN/EPD 格式开局数据
    // 或从 ElephantEye 的开局库（.obk 格式）导入
}

fun getOpeningMove(hash: ULong): Move? {
    val moves = openingBook[hash] ?: return null
    return moves.random()  // 随机选一个走法增加变化性
}
```

**象眼 (ElephantEye) 的做法**：自带一个经过大量对局统计优化的开局库，开局阶段几乎零搜索时间。

### 6.4 残局库（Endgame Tablebase）

对于棋子极少的残局（如 KPK、KRK 等），可以预计算所有可能局面的必胜/必和判定。

```
运行方式：脱机预生成 → 文件存储 → 运行时 Probe
适用范围：通常 ≤ 4 子（包括帅将）
格式：Syzygy、DTZ50 等标准格式
```

### 6.5 难度调整策略

| 难度 | 搜索深度 | 评估噪声 | 其他限制 |
|------|---------|---------|---------|
| 初级 | 1-2 | ±200 随机分 | 50% 概率不选最佳走法 |
| 中级 | 3-4 | ±100 随机分 | 不限制 |
| 高级 | 5-6 | 无噪声 | 不限制 |
| 大师 | 8+ | 无噪声 | 不用随机开局库 |

**注意**：调整搜索深度是最直接有效的难度控制手段，不需要修改算法本身。

### 6.6 C++ 原生层（可选进阶）

当 Kotlin/Java 层的性能达到瓶颈时，可考虑：

```
Kotlin/Java (UI + 控制器)
        ↓ JNI
C++ (搜索核心 + Bitboard + NNUE)

性能提升：5-10 倍（编译器优化 + 缓存友好的内存布局）
```

这也是 Pikafish 的实际做法：C++ 核心引擎，通过 UCI 协议与各种前端（GUI、Android、Web）通信。

---

## 7. 测试方法论

### 7.1 单元测试：走法正确性

确保每种棋子的每种走法路径都被覆盖：

```kotlin
@Test
fun testKnightMoveLegBlocked() {
    val engine = ChessEngine()
    engine.initBoard()
    // 初始位置：红马在 (9, 7)，目标 (7, 6) 和 (7, 8) 被己方棋子蹩脚？
    // 验证每步跳蹩脚都被正确拦截
}

@Test
fun testCannonCaptureOverOnePiece() {
    // 开局：红炮在 (7, 7)，黑马在 (0, 7)，中间有黑卒在 (3, 7)
    // 炮吃马应该合法（隔一子）
    // 炮吃卒不应该合法（中间无子或隔多子）
}

@Test
fun testGeneralFacingRule() {
    // 将帅对面的局面，走法不应导致将帅见面
}
```

**测试覆盖清单**：
- [ ] 每种棋子类型的合法走法和非法走法（至少 2 正 2 反）
- [ ] 蹩脚马、塞象眼的所有方向
- [ ] 炮的翻山吃子（隔 0 子/隔 1 子/隔 2+ 子）
- [ ] 过河兵/卒的横移
- [ ] 九宫内的仕和帅
- [ ] 将帅对面检查
- [ ] 走子后不被将
- [ ] 长捉/长将禁止
- [ ] 无子可走的困毙

### 7.2 集成测试：对局完整性

```
1. 加载标准开局 → 执行 10 步随机合法走法 → 确认无崩溃
2. 特定残局 FEN → AI 搜索 → 确认能找到杀棋
3. 悔棋 5 次 → 确认棋盘恢复到 5 步前
```

### 7.3 性能基准测试

```kotlin
@Test
fun benchmarkSearchDepth4() {
    val engine = ChessEngine()
    engine.initBoard()
    val ai = ChessAI(engine)

    val start = System.nanoTime()
    val move = runBlocking { ai.getAIMove(searchDepth = 4) }
    val elapsed = (System.nanoTime() - start) / 1_000_000

    assertTrue(move != null, "AI should find a move")
    assertTrue(elapsed < 5000, "Search depth 4 should take < 5 seconds")
}
```

**性能基准参考值**（Mailbox 实现，单线程）：

| 搜索深度 | 局面数（近似） | 预期时间 |
|---------|-------------|---------|
| 2 | ~2,000 | < 0.1s |
| 3 | ~80,000 | < 1s |
| 4 | ~2,500,000 | 3-10s |
| 5 | ~80,000,000 | 30-120s |

### 7.4 自对弈测试（Self-Play）

```kotlin
fun selfPlay(games: Int = 100): List<GameResult> {
    val results = mutableListOf<GameResult>()

    repeat(games) {
        val engine = ChessEngine()
        engine.initBoard()
        engine.gameMode = GameMode.NONE  // 人机对弈模式（AI 走双方）

        while (!engine.gameOver) {
            // AI 为当前方走一子
            val move = /* search with engine.currentTurn */
            engine.executeMove(move.fromRow, move.fromCol, move.toRow, move.toCol)

            // 检测无限循环
            if (moveCount > 200) { engine.drawGame = true; break }
        }
        results.add(parseResult(engine))
    }
    return results
}
```

**自对弈的用途**：
- 验证新版评估函数的胜率是否提升
- 发现引擎的盲点（总是输的某些局面模式）
- NNUE 训练数据的来源

---

## 8. 附录：算法演进路线图

```
第一阶段：基础原型（1-2 周）
├─ 棋盘表示：二维数组
├─ 走法生成：完整合法性检查
├─ 基础 UI：点击走子
├─ 胜负判定
└─ 目标：两个人在同一设备上对弈

第二阶段：人机对弈（1 周）
├─ Minimax + Alpha-Beta
├─ 基础评估函数（子力 + 兵位置表）
├─ 单线程搜索
└─ 目标：depth 3-4 有基本棋力

第三阶段：性能优化（1-2 周）
├─ 走法排序 (MVV-LVA)
├─ 多线程并行搜索
├─ Zobrist 哈希 + 置换表
├─ 静态搜索 (Quiescence Search)
└─ 目标：depth 5-6 流畅运行

第四阶段：棋力提升（2-4 周）
├─ 空着裁剪 (Null Move Pruning)
├─ 迭代加深 + PVS
├─ 杀手走法 + 历史启发
├─ 完善位置评估表
├─ 开局库
└─ 目标：depth 8+ 接近业余高手

第五阶段：专业级（持续迭代）
├─ Bitboard 重构（可选）
├─ NNUE 神经网络评估
├─ 残局库 (Tablebase)
├─ C++ 原生层（可选）
└─ 目标：对标开源强引擎
```

---

## 参考资料

| 资源 | 描述 | 链接 |
|------|------|------|
| Pikafish | 基于 Stockfish 的中国象棋引擎，支持 NNUE | github.com/pikafish/Pikafish |
| ElephantEye (象眼) | 经典中国象棋引擎，文档详尽 | github.com/elephanteye/ElephantEye |
| Fairy-Stockfish | Stockfish 变体，支持多种象棋变体包括中国象棋 | github.com/ianfab/Fairy-Stockfish |
| Chess Programming Wiki | 所有象棋编程技术的百科全书式文档 | chessprogramming.org |

> **本指南所附示例代码基于 Kotlin Multiplatform 实现，但你可以用任何语言（Python、JavaScript、C++、Java 等）遵循相同逻辑实现。核心算法思想是语言无关的。**
