# Rust 引擎代码审查修复清单

## 🔴 严重
- [x] #1 `tt.rs` — UnsafeCell 并发写入 UB，改用原子类型 ✅
- [x] #2 `nnue.rs` — read_array_i16 缺乏边界检查 ✅
- [x] #3 `evaluate.rs` — Advisor/Elephant 分支内死代码 + Advisor 错误惩罚 ✅

## 🟠 重要
- [x] #4 `board.rs` — is_in_check 不应需要 &mut self ✅
- [x] #5 `search.rs` — see_capture 每次 clone Board ✅
- [x] #6 `search.rs` — Lazy SMP 最佳走法选择 race condition ✅
- [x] #7 `board.rs` — generate_legal_moves 每次分配新 Vec ✅
- [x] #8 `ffi.rs` — JNI 返回值 0 无法区分错误和走法 ✅

## 🟡 中等
- [x] #9 `zobrist.rs` — thread_local 改 static ✅
- [x] #10 `nnue.rs` — update() 未处理王移动 ✅
- [x] #11 `board.rs` — from_bytes 不设置 position_history ✅
- [x] #12 `evaluate.rs` — mobility 函数使用 board.current_turn 判断敌友 ✅

## 🔵 建议
- [x] #13 `board.rs` — would_kings_face 不需要 clone 整个 Board ✅
- [x] #14 `search.rs` — aspiration window 初始值太小 ✅
- [x] #15 `Cargo.toml` — 缺少 parse_pgn bin 声明 ✅

## 验证结果
- `cargo test`: 6/6 通过，0 warning
- `cargo build --release`: 成功
