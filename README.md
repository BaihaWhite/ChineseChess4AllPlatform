# 中国象棋全平台版 / Chinese Chess 4 All Platforms

跨平台中国象棋应用，基于 Kotlin Multiplatform + Compose Multiplatform 构建，Rust 引擎驱动 AI 对弈。

## 特性

- 🎮 **人机对弈** — 四级难度（初级/中级/高级/大师）
- 🤖 **Rust AI 引擎** — Negamax + PVS + LMR + Null Move Pruning
- 🧠 **NNUE 神经网络** — HalfKP 架构，量化推理，自动回退 HCE
- 📱 **Android 原生支持** — JNI 集成 Rust 引擎，4 架构支持
- 🖥️ **Desktop 跨平台** — Windows/Linux/macOS，单文件 uber-jar 分发
- 📖 **开局库** — 110 万+ 开局走法，加权随机选择
- 🔄 **循环检测** — 长将/长捉禁止着法检测，允许连将杀法
- 📤 **日志导出** — Android 系统分享 / Desktop 保存文件
- 🎨 **Material 3 主题** — 动态配色

## 快速开始

### Desktop（Windows/Linux）

需要 Java 17+，下载 `chinese-chess.jar` 后运行：

```bash
java -jar chinese-chess.jar
```

### Android

下载 `chinese-chess.apk` 安装。

## 架构

```
┌─────────────────────────────────────────────┐
│           Compose Multiplatform UI          │
│              (App.kt / ChessBoard.kt)       │
├─────────────────────────────────────────────┤
│            Kotlin Engine Layer              │
│  ChessEngine / AIController / OpeningBook   │
├──────────────────┬──────────────────────────┤
│   Android: JNI   │   Desktop: JNI           │
│   NativeEngine   │   NativeEngine           │
├──────────────────┴──────────────────────────┤
│              Rust Engine                    │
│  board / search / evaluate / nnue / tt      │
└─────────────────────────────────────────────┘
```

### Rust 引擎模块

| 模块 | 功能 |
|------|------|
| `types` | 棋子类型、走法编码 |
| `board` | 棋盘表示、走法生成、合法性验证、FEN 输出 |
| `zobrist` | Zobrist 哈希 |
| `tt` | 置换表（Structure of Arrays，64MB） |
| `evaluate` | HCE 评估函数（PST + 物质价值） |
| `nnue` | NNUE 网络（HalfKP，22680→256→32→1） |
| `search` | 搜索引擎（Lazy SMP 多线程） |
| `ffi` | JNI 接口导出 |

## 构建指南

### 前置条件

- JDK 17+
- Android SDK (NDK 27+)
- Rust toolchain (`rustup`)
- Android Rust targets:
  ```bash
  rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android
  rustup target add x86_64-pc-windows-gnu  # Windows 交叉编译
  ```

### 编译 Rust 引擎

```bash
cd engine

# Desktop (host platform)
cargo build --release --lib

# Android (all architectures)
./build_android.sh

# Windows 交叉编译
cargo build --release --target x86_64-pc-windows-gnu --lib
```

### 构建应用

```bash
# Android APK
./gradlew assembleDebug

# Desktop 运行
./gradlew run

# 跨平台 uber-jar (Linux + Windows, 含 NNUE)
./gradlew packageCrossPlatformJar
cp composeApp/build/libs/chinese-chess-crossplatform.jar chinese-chess.jar
```

### NNUE 训练

```bash
cd engine/training
python train_nnue.py       # 训练 NNUE
python gen_data.py         # 生成训练数据
```

## 难度等级

| 等级 | 搜索深度 | 时间限制 | 线程数 | 特点 |
|------|---------|---------|--------|------|
| 初级 | 3 | 3s | 1 | 评估噪声 + 随机走法 |
| 中级 | 6 | 10s | 2 | 轻度噪声 |
| 高级 | 9 | 30s | 4 | 无噪声 |
| 大师 | 12 | 60s | 全核 | 最强搜索 |

## 技术细节

### 搜索算法

- **Lazy SMP** 多线程：共享 TT 和 NNUE 权重
- **Iterative Deepening** + **Aspiration Window**
- **PVS** (Principal Variation Search) + **LMR** (Late Move Reductions)
- **Null Move Pruning** + **Futility Pruning** + **Razoring** + **LMP**
- **SEE** (Static Exchange Evaluation)
- **Killer Moves** + **History Heuristic** + **Counter Moves`

### NNUE

HalfKP 架构，特征：`(king_square, piece_type_color, piece_square) × 2 perspectives`。网络结构：22680→256→32→1，量化推理。权重文件 `nnue_trained.bin`，加载失败时自动回退到 HCE。

### 循环检测

- 基于 Zobrist 哈希检测重复局面
- 将军方：继续搜索找连将杀法
- 被将军方：返回认输分数
- 长将/长捉等禁止着法在 `ChessEngine.isLegalMove()` 中过滤

### FEN 日志

每次 AI 搜索输出 `[FEN] rnbakabnr/9/1c5c1/... w 0000000000000000`，可直接复制到调试工具还原局面。

## License

MIT
