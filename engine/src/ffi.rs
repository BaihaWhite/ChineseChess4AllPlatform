use crate::board::Board;
use crate::search::SearchEngine;
use crate::types::*;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref ENGINE: Mutex<SearchEngine> = Mutex::new(SearchEngine::new());
}

#[no_mangle]
pub extern "system" fn JNI_OnLoad(
    _vm: *mut std::ffi::c_void,
    _reserved: *mut std::ffi::c_void,
) -> jni::sys::jint {
    // Force engine initialization so NNUE load status is logged
    let _ = &*ENGINE;
    jni::sys::JNI_VERSION_1_6
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
    red_checks: jni::sys::jint,
    black_checks: jni::sys::jint,
) -> jni::sys::jint {
    let bytes: Vec<u8> = match _env.convert_byte_array(board_bytes) {
        Ok(b) => b,
        Err(_) => return -1,
    };
    if bytes.len() != 90 {
        return -1;
    }

    let mut arr = [0u8; 90];
    arr.copy_from_slice(&bytes);

    let mut board = Board::from_bytes(&arr);
    board.current_turn = match turn {
        1 => Side::Red,
        2 => Side::Black,
        _ => Side::Red,
    };
    board.consecutive_checks[0] = red_checks as u8;
    board.consecutive_checks[1] = black_checks as u8;

    let game_history = if history_len > 0 {
        let mut buf = vec![0i64; history_len as usize];
        _env.get_long_array_region(history, 0, &mut buf).unwrap_or(());
        buf.iter().map(|&v| v as u64).collect()
    } else {
        Vec::new()
    };

    let mut engine = match ENGINE.lock() {
        Ok(e) => e,
        Err(_) => return -1,
    };
    let threads = engine.thread_count();
    let result = engine.search_with_threads(&mut board, depth, time_ms as u64, &game_history, threads);

    match result {
        Some(m) => m.encode() as jni::sys::jint,
        None => -1,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_chinesechess_engine_NativeEngine_chessGetThreadCount(
    _env: jni::JNIEnv,
    _class: jni::objects::JClass,
) -> jni::sys::jint {
    match ENGINE.lock() {
        Ok(e) => e.thread_count() as i32,
        Err(_) => 1,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_chinesechess_engine_NativeEngine_chessGetLastDepth(
    _env: jni::JNIEnv,
    _class: jni::objects::JClass,
) -> jni::sys::jint {
    match ENGINE.lock() {
        Ok(e) => e.last_completed_depth,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_chinesechess_engine_NativeEngine_chessGetLastNodes(
    _env: jni::JNIEnv,
    _class: jni::objects::JClass,
) -> jni::sys::jlong {
    match ENGINE.lock() {
        Ok(e) => e.last_nodes as i64,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_chinesechess_engine_NativeEngine_chessLoadNNUE(
    mut _env: jni::JNIEnv,
    _class: jni::objects::JClass,
    path: jni::objects::JString,
) -> jni::sys::jboolean {
    let path_str: String = match _env.get_string(&path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    match ENGINE.lock() {
        Ok(mut engine) => {
            engine.reload_nnue(&path_str);
            if engine.nnue.is_loaded() { 1 } else { 0 }
        }
        Err(_) => 0,
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
