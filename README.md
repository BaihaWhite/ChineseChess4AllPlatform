# 中国象棋全平台版 / Chinese Chess 4 All Platforms

跨平台中国象棋应用，基于 Kotlin Multiplatform + Compose Multiplatform 构建，Rust 引擎驱动 AI 对弈。

## 特性

- 🎮 **人机对弈** — 四级难度（初级/中级/高级/大师）
- 🤖 **Rust AI 引擎** — 高性能搜索引擎，Negamax + PVS + LMR + Null Move Pruning
- 📱 **Android 原生支持** — JNI 集成 Rust 引擎，极低内存占用
- 🖥️ **Desktop 支持** — Windows/Linux/macOS 桌面端
- 📖 **开局库** — 110 万+ 开局走法，加权随机选择
- 🔄 **循环检测** — 长将/长捉/长杀等禁止着法检测
- 🌐 **在线对弈** — 局域网/互联网对战（开发中）
- 🎨 **Material 3 主题** — 动态配色

## 架构

```
┌─────────────────────────────────────────────┐
│           Compose Multiplatform UI          │
│              (App.kt / ChessBoard.kt)       │
├─────────────────────────────────────────────┤
│            Kotlin Engine Layer              │
│  ChessEngine / ChessAI / OpeningBook / TT   │
├──────────────────┬──────────────────────────┤
│   Android: JNI   │   Desktop: Pure Kotlin   │
│   NativeEngine   │   (fallback)             │
├──────────────────┤                          │
│   Rust Engine    │                          │
│   (libchess_     │                          │
│    engine.so)    │                          │
└──────────────────┴──────────────────────────┘
```

### Rust 引擎模块

| 模块 | 功能 |
|------|------|
| `types` | 棋子类型、走法编码 |
| `board` | 棋盘表示、走法生成、合法性验证 |
| `zobrist` | Zobrist 哈希（位置标识） |
| `tt` | 置换表（Structure of Arrays，2MB） |
| `evaluate` | 评估函数（棋子价值 + 位置表） |
| `search` | 搜索引擎（Negamax/PVS/LMR/Null Move/Aspiration） |
| `ffi` | JNI 接口导出 |

## 性能对比

| 指标 | Kotlin 引擎 | Rust 引擎 |
|------|------------|-----------|
| 大师模式搜索时间 | 10-30s+ | 1-4s |
| 内存占用 | 200MB+ (频繁 GC) | ~2MB |
| .so 体积 | N/A | 390-540KB |
| GC 压力 | 严重 | 无 |

## 构建指南

### 前置条件

- JDK 17+
- Android SDK (NDK 27+)
- Rust toolchain (`rustup`)
- Android Rust targets:
  ```bash
  rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android
  ```

### 编译 Rust 引擎

```bash
cd engine

# 编译所有 Android 架构
./build_android.sh

# 或单独编译
cargo build --release --target aarch64-linux-android
```

### 构建 Android APK

```bash
./gradlew assembleDebug
# 或 Release
./gradlew assembleRelease
```

### 构建 Desktop

```bash
./gradlew run          # 运行
./gradlew packageDmg   # macOS
./gradlew packageMsi   # Windows
./gradlew packageDeb   # Linux
```

## 分支说明

| 分支 | 说明 |
|------|------|
| `kotlin` | 纯 Kotlin 实现，Kotlin AI 引擎 |
| `rust` | Rust 后端引擎，JNI 集成（当前分支） |

## 难度等级

| 等级 | 搜索深度 | 时间限制 | 线程数 | 特点 |
|------|---------|---------|--------|------|
| 初级 | 3 | 3s | 1 | 评估噪声 + 随机走法 |
| 中级 | 6 | 10s | 2 | 轻度噪声 |
| 高级 | 9 | 30s | 4 | 无噪声 |
| 大师 | 12 | 60s | 全核 | 最强搜索 |

## 技术细节

### 搜索算法

- **Negamax** 带 Alpha-Beta 剪枝
- **PVS** (Principal Variation Search)
- **LMR** (Late Move Reductions)
- **Null Move Pruning**
- **Aspiration Window**
- **Futility Pruning**
- **Quiescence Search** (静态搜索)
- **Iterative Deepening** (迭代加深)
- **Killer Moves** + **History Heuristic** + **Counter Moves**

### 评估函数

- 棋子基础价值（帅10000/車600/馬270/炮285/相120/仕120/兵30-170）
- 位置评估表（PST）— 每种棋子红/黑方独立位置分
- 红方视角评估，取负为黑方

### 循环检测

- 基于 Zobrist 哈希历史检测重复局面
- 长将/长捉/长杀等禁止着法过滤
- 无合法走法时判负

## License

MIT
