# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

跨平台中国象棋应用，Kotlin Multiplatform + Compose Multiplatform 构建 UI，Rust 引擎驱动 AI 对弈。支持 Android 和 Desktop（Windows/Linux/macOS）。

## Build Commands

### Rust Engine

```bash
cd engine

# Desktop (host platform)
cargo build --release --lib

# Android (all 4 architectures)
./build_android.sh

# Windows cross-compilation (Linux host)
cargo build --release --target x86_64-pc-windows-gnu --lib
```

编译产物：
- Linux: `engine/target/release/libchess_engine.so`
- Windows: `engine/target/x86_64-pc-windows-gnu/release/chess_engine.dll`
- Android: 复制到 `composeApp/src/androidMain/jniLibs/` (4 个架构)

### Gradle (Kotlin/Compose)

```bash
./gradlew assembleDebug                # Android APK
./gradlew run                          # Desktop 运行
./gradlew packageCrossPlatformJar      # 跨平台 uber-jar (Linux+Windows, 含 NNUE)
./gradlew packageDeb                   # Linux 打包
./gradlew packageMsi                   # Windows 打包
./gradlew packageDmg                   # macOS 打包
```

跨平台 jar 产出：`composeApp/build/libs/chinese-chess-crossplatform.jar`，需手动复制到根目录：
```bash
cp composeApp/build/libs/chinese-chess-crossplatform.jar chinese-chess.jar
```

### Rust 测试

```bash
cd engine && cargo test
```

### NNUE 训练

```bash
cd engine/training
python train_nnue.py       # 训练 NNUE
python gen_data.py         # 生成训练数据
```

训练产物：`engine/training/nnue_trained.bin`，复制到 `engine/nnue_trained.bin`。

## Architecture

```
composeApp/src/
├── commonMain/kotlin/com/chinesechess/
│   ├── engine/          # 游戏逻辑层
│   │   ├── ChessEngine.kt    # 棋盘状态、走法生成、合法性验证
│   │   ├── AIController.kt   # AI 调度（开局库 → NativeEngine）
│   │   ├── NativeEngine.kt   # expect 声明，JNI 桥接接口
│   │   ├── OpeningBook.kt    # 开局库（Zobrist 哈希查询）
│   │   ├── Model.kt          # Piece/Move/PSide/PType 数据类
│   │   ├── DebugLog.kt       # 跨平台日志（内存环形缓冲 200 条）
│   │   └── PlatformLog.kt    # expect: platformLog + platformExportLogs
│   └── ui/
│       ├── App.kt            # 主 Composable、屏幕导航、日志导出按钮
│       ├── ChessBoard.kt     # Canvas 棋盘渲染
│       └── Theme.kt          # Material 3 颜色定义
├── androidMain/kotlin/com/chinesechess/
│   ├── engine/
│   │   ├── NativeEngine.kt   # actual: System.loadLibrary + loadNNUE
│   │   └── PlatformLog.kt    # actual: FileProvider + Intent 分享日志
│   └── app/MainActivity.kt   # Activity, 暴露 instance 给 FileProvider
└── desktopMain/kotlin/com/chinesechess/engine/
    ├── NativeEngine.kt       # actual: 资源提取 + System.load
    └── PlatformLog.kt        # actual: JFileChooser 保存日志

engine/src/
├── lib.rs              # 模块导出
├── types.rs            # Piece/Move/Side/PieceType 定义
├── board.rs            # 棋盘表示、走法生成、合法性验证、to_fen()
├── zobrist.rs          # Zobrist 哈希
├── tt.rs               # 置换表（Structure of Arrays，64MB）
├── evaluate.rs         # HCE 评估函数（PST + 物质价值）
├── nnue.rs             # NNUE 网络（HalfKP 架构，22680→256→32→1）
├── search.rs           # Negamax/PVS/LMR/Null Move/Aspiration/SEE
└── ffi.rs              # JNI 导出函数
```

## JNI 接口

`ffi.rs` 导出的 JNI 函数（通过 `#[no_mangle] extern "system"`）：

| 函数 | 说明 |
|------|------|
| `JNI_OnLoad` | 初始化 ENGINE（lazy_static），加载 NNUE |
| `chessSearch` | 搜索最佳走法，返回 `int` 编码 |
| `chessCancel` | 取消搜索 |
| `chessLoadNNUE` | 从指定路径加载 NNUE 权重 |
| `chessGetThreadCount` | 获取线程数 |
| `chessGetLastDepth` | 获取上次搜索深度 |
| `chessGetLastNodes` | 获取上次搜索节点数 |

棋盘通过 90 字节 `ByteArray` 传输（10×9 棋盘，每格 1 字节：低 4 位棋子类型，bit6=红方，bit7=黑方）。走法返回值编码为 `int`：`(fromRow << 24) | (fromCol << 16) | (toRow << 8) | toCol`。错误返回 `-1`。

JNI 版本使用 `JNI_VERSION_1_6`（Android 兼容性）。

## NNUE 加载流程

1. **Kotlin 端**：从 JAR/APK 资源提取 `nnue_trained.bin` 到临时目录
2. **JNI 调用**：`chessLoadNNUE(tmpPath)` 尝试加载
3. **Rust 自动加载**（JNI 失败时的回退）：按顺序搜索：
   - `nnue_trained.bin`（当前工作目录）
   - `<exe_dir>/nnue_trained.bin`
   - `/tmp/nnue_trained.bin`
   - `/tmp/chinese-chess-natives/nnue_trained.bin`
   - `std::env::temp_dir()/chinese-chess-natives/nnue_trained.bin`
4. 所有路径失败 → 回退到 HCE（手工评估函数）

`chessLoadNNUE` 的 `UnsatisfiedLinkError` 必须用 `catch (e: Throwable)` 捕获（`Error` 子类，不是 `Exception`）。

## 搜索引擎

- **Lazy SMP** 多线程：共享 TT 和 NNUE 权重，每线程独立 SearchState
- **Iterative Deepening** + **Aspiration Window**（WINDOW=40）
- **PVS** (Principal Variation Search) + **LMR** (Late Move Reductions)
- **Null Move Pruning** + **Futility Pruning** + **Razoring** + **Late Move Pruning**
- **IID** (Internal Iterative Deepening) — TT 未命中时做浅搜索获取 move ordering
- **SEE** (Static Exchange Evaluation) — qsearch 中过滤负 SEE 吃子
- **Killer Moves** + **History Heuristic** + **Counter Moves`
- **FEN 日志**：每次搜索输出 `[FEN] <position>` 便于诊断

### 循环检测（重复局面）

基于 Zobrist 哈希检测，分两层：
1. **搜索树内重复**（`search_hashes[0..ply]`）：同一搜索路径中局面重复
2. **棋谱历史重复**（`game_hash_counts`）：局面在对局历史中已出现 ≥2 次

处理逻辑：
- 被将军方（`in_check`）：返回 `MATE_SCORE - ply`（认输）
- 将军方（正在给对手将军）：**不返回和棋**，继续搜索找连将杀法
- 非将军局面：返回 `0`（和棋）

长将/长捉等禁止着法在 `ChessEngine.isLegalMove()` 中过滤（Kotlin 端）。

## 日志导出

日志按钮（📤 日志）行为：
- **Android**：通过 FileProvider + Intent.ACTION_SEND 调用系统分享
- **Desktop**：JFileChooser 弹出保存文件对话框，保存为 `.txt`
- 保留全部日志（`DebugLog.getAll()`），不限条数

## 跨平台资源打包

Desktop 的 native lib 和 NNUE 通过 JAR 资源打包：
- `natives/linux-x86-64/libchess_engine.so`
- `natives/windows-x86-64/chess_engine.dll`
- `nnue_trained.bin`

运行时提取到 `java.io.tmpdir/chinese-chess-natives/`。

## Dependencies

- **Rust**: `jni` (0.21), `rayon`, `lazy_static`
- **Kotlin**: Compose Multiplatform, Material 3, Skiko (Desktop: linux/windows/macos x64)
- **Android**: NDK 27+, compileSdk 35, minSdk 26, FileProvider
- **JDK**: 17+

## Rust Binary Tools

`engine/src/bin/` 下的工具用于开发和调试：
- `bench_search` — 搜索性能基准测试
- `gen_data` — 生成 NNUE 训练数据
- `match_runner` — 引擎对弈测试
- `rescore` — 重新评分棋局
- `diag_nnue` — NNUE 诊断
- `parse_pgn` — PGN 解析
