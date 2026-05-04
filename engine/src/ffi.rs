use crate::board::Board;
use crate::search::SearchEngine;
use crate::types::*;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref ENGINE: Mutex<SearchEngine> = Mutex::new(SearchEngine::new());
}

#[no_mangle]
pub extern "system" fn Java_com_chinesechess_engine_NativeEngine_chessSearch(
    mut _env: jni::JNIEnv,
    _class: jni::objects::JClass,
    board_bytes: jni::objects::JByteArray,
    turn: jni::sys::jint,
    depth: jni::sys::jint,
    time_ms: jni::sys::jint,
    history: jni::objects::JLongArray,
    history_len: jni::sys::jint,
) -> jni::sys::jint {
    let bytes: Vec<u8> = match _env.convert_byte_array(board_bytes) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    if bytes.len() != 90 {
        return 0;
    }

    let mut arr = [0u8; 90];
    arr.copy_from_slice(&bytes);

    let mut board = Board::from_bytes(&arr);
    board.current_turn = match turn {
        1 => Side::Red,
        2 => Side::Black,
        _ => Side::Red,
    };

    let game_history = if history_len > 0 {
        let mut buf = vec![0i64; history_len as usize];
        _env.get_long_array_region(history, 0, &mut buf).unwrap_or(());
        buf.iter().map(|&v| v as u64).collect()
    } else {
        Vec::new()
    };

    let mut engine = match ENGINE.lock() {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let result = engine.search(&mut board, depth, time_ms as u64, &game_history);

    match result {
        Some(m) => m.encode() as jni::sys::jint,
        None => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_chinesechess_engine_NativeEngine_chessCancel(
    _env: jni::JNIEnv,
    _class: jni::objects::JClass,
) {
    if let Ok(engine) = ENGINE.lock() {
        engine.cancel();
    }
}
