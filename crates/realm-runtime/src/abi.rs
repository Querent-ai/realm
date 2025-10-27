/// C ABI exports for wasm-chord runtime
///
/// Provides stable FFI interface for host languages.
use crate::{context::*, inference::*, set_last_error, take_last_error, ErrorCode};
use std::os::raw::c_char;
use std::slice;
use std::sync::Mutex;

#[cfg(test)]
use std::ffi::CStr;

// Global runtime context (protected by mutex for thread safety)
static RUNTIME: Mutex<Option<RuntimeContext>> = Mutex::new(None);

/// Initialize runtime with JSON config
///
/// Returns 0 on success, error code otherwise.
///
/// # Safety
/// `config_ptr` must be valid for reads of `config_len` bytes, or null.
#[no_mangle]
pub unsafe extern "C" fn wasmchord_init(config_ptr: *const u8, config_len: usize) -> u32 {
    let config_str = if config_ptr.is_null() || config_len == 0 {
        "{}" // Default config
    } else {
        let bytes = slice::from_raw_parts(config_ptr, config_len);
        match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid UTF-8 in config: {}", e));
                return ErrorCode::InvalidArgument as u32;
            }
        }
    };

    let config = match RuntimeConfig::from_json(config_str) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(format!("Config parse error: {}", e));
            return ErrorCode::InvalidArgument as u32;
        }
    };

    let mut runtime = RUNTIME.lock().unwrap();
    *runtime = Some(RuntimeContext::new(config));

    ErrorCode::Ok as u32
}

/// Load model from memory
///
/// Returns model handle (>0) on success, 0 on error.
///
/// # Safety
/// `model_bytes_ptr` must be valid for reads of `model_bytes_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn wasmchord_load_model(
    model_bytes_ptr: *const u8,
    model_bytes_len: usize,
) -> u32 {
    if model_bytes_ptr.is_null() || model_bytes_len == 0 {
        set_last_error("Invalid model data pointer".to_string());
        return 0;
    }

    let bytes = slice::from_raw_parts(model_bytes_ptr, model_bytes_len);

    // Parse GGUF (simplified - just create placeholder)
    use std::io::Cursor;
    use realm_core::formats::gguf::GGUFParser;

    let cursor = Cursor::new(bytes);
    let mut parser = GGUFParser::new(cursor);

    let meta = match parser.parse_header() {
        Ok(m) => m,
        Err(e) => {
            set_last_error(format!("Model parse error: {}", e));
            return 0;
        }
    };

    let model = ModelHandle { name: "loaded_model".to_string(), meta };

    let mut runtime = RUNTIME.lock().unwrap();
    if let Some(ref mut rt) = *runtime {
        rt.register_model(model)
    } else {
        set_last_error("Runtime not initialized".to_string());
        0
    }
}

/// Free model from memory
#[no_mangle]
pub extern "C" fn wasmchord_free_model(model_handle: u32) -> u32 {
    let mut runtime = RUNTIME.lock().unwrap();
    if let Some(ref mut rt) = *runtime {
        if rt.remove_model(model_handle).is_some() {
            ErrorCode::Ok as u32
        } else {
            set_last_error(format!("Invalid model handle: {}", model_handle));
            ErrorCode::InvalidArgument as u32
        }
    } else {
        set_last_error("Runtime not initialized".to_string());
        ErrorCode::GenericFailure as u32
    }
}

/// Perform blocking inference
///
/// Returns stream handle (>0) on success, 0 on error.
///
/// # Safety
/// `prompt_ptr` must be valid for reads of `prompt_len` bytes.
/// `opts_ptr` must be valid for reads or null.
#[no_mangle]
pub unsafe extern "C" fn wasmchord_infer(
    model_handle: u32,
    prompt_ptr: *const u8,
    prompt_len: usize,
    opts_ptr: *const GenOptions,
) -> u32 {
    if prompt_ptr.is_null() || prompt_len == 0 {
        set_last_error("Invalid prompt pointer".to_string());
        return 0;
    }

    let prompt_bytes = slice::from_raw_parts(prompt_ptr, prompt_len);
    let _prompt = match std::str::from_utf8(prompt_bytes) {
        Ok(s) => s.to_string(),
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in prompt: {}", e));
            return 0;
        }
    };

    let options = if opts_ptr.is_null() { GenOptions::default() } else { *opts_ptr };

    // TODO: Tokenize prompt (_prompt) before creating session
    // For now, use empty token vector as placeholder
    let prompt_tokens = Vec::new(); // Will be tokenized in full implementation

    // Create inference session (in real implementation, store in runtime)
    let _session = InferenceSession::new(model_handle, prompt_tokens, options);

    // Placeholder: return dummy stream handle
    // In real implementation, we'd store session and return its ID
    1
}

/// Get next token from stream
///
/// Returns number of bytes written, -1 for end, -2 for error.
///
/// # Safety
/// `buf_ptr` must be valid for writes of `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn wasmchord_next_token(
    stream_handle: u32,
    buf_ptr: *mut u8,
    buf_len: usize,
) -> i32 {
    if stream_handle == 0 {
        set_last_error("Invalid stream handle".to_string());
        return -2;
    }

    if buf_ptr.is_null() || buf_len == 0 {
        set_last_error("Invalid buffer pointer".to_string());
        return -2;
    }

    // Placeholder implementation
    // In real version, we'd look up session and call next_token()
    let token = "placeholder_token";
    let token_bytes = token.as_bytes();

    if token_bytes.len() > buf_len {
        set_last_error("Buffer too small".to_string());
        return -2;
    }

    std::ptr::copy_nonoverlapping(token_bytes.as_ptr(), buf_ptr, token_bytes.len());

    token_bytes.len() as i32
}

/// Close inference stream
#[no_mangle]
pub extern "C" fn wasmchord_close_stream(stream_handle: u32) -> u32 {
    if stream_handle == 0 {
        set_last_error("Invalid stream handle".to_string());
        return ErrorCode::InvalidArgument as u32;
    }

    // Placeholder: in real implementation, clean up session
    ErrorCode::Ok as u32
}

/// Get last error message
///
/// Returns number of bytes written to buffer.
///
/// # Safety
/// `buf_ptr` must be valid for writes of `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn wasmchord_last_error(buf_ptr: *mut c_char, buf_len: usize) -> usize {
    if buf_ptr.is_null() || buf_len == 0 {
        return 0;
    }

    let error_msg = take_last_error().unwrap_or_else(|| "No error".to_string());
    let bytes = error_msg.as_bytes();
    let copy_len = bytes.len().min(buf_len - 1); // Leave room for null terminator

    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf_ptr as *mut u8, copy_len);
    *buf_ptr.add(copy_len) = 0; // Null terminator

    copy_len
}

/// Get runtime version string
#[no_mangle]
pub extern "C" fn wasmchord_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let config = b"{}";
        let result = unsafe { wasmchord_init(config.as_ptr(), config.len()) };
        assert_eq!(result, ErrorCode::Ok as u32);
    }

    #[test]
    fn test_version() {
        let ver_ptr = wasmchord_version();
        assert!(!ver_ptr.is_null());

        let c_str = unsafe { CStr::from_ptr(ver_ptr) };
        let version = c_str.to_str().unwrap();
        assert!(!version.is_empty());
    }
}
