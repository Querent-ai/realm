//! Realm Node.js SDK - Native Addon
//!
//! Provides host-side model storage functions for WASM module via Neon bindings

use neon::prelude::*;
use neon::types::buffer::TypedArray;
use realm_runtime::model_storage::GLOBAL_MODEL_STORAGE;

/// Store model from bytes in HOST storage
///
/// Auto-generates deterministic model ID from content hash.
/// Same model content = same ID (automatic deduplication).
///
/// # Arguments
/// * `gguf_bytes` - GGUF file bytes (Buffer)
///
/// # Returns
/// Model ID (number) - deterministic hash-based ID
fn store_model(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let model_bytes = cx.argument::<JsBuffer>(0)?;

    let bytes = model_bytes.as_slice(&cx);

    // Always use None to auto-generate ID from content hash
    match GLOBAL_MODEL_STORAGE.store_model(bytes, None) {
        Ok(id) => Ok(cx.number(id as f64)),
        Err(e) => cx.throw_error(format!("Failed to store model: {}", e)),
    }
}

/// Get tensor from HOST storage (returns dequantized f32 array as ArrayBuffer)
///
/// # Arguments
/// * `model_id` - Model ID (number)
/// * `tensor_name` - Tensor name string (e.g., "blk.0.attn_q.weight")
///
/// # Returns
/// ArrayBuffer containing dequantized f32 data
fn get_tensor(mut cx: FunctionContext) -> JsResult<JsArrayBuffer> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;
    let tensor_name = cx.argument::<JsString>(1)?.value(&mut cx);

    let tensor = match GLOBAL_MODEL_STORAGE.get_tensor(model_id, &tensor_name) {
        Ok(t) => t,
        Err(e) => return cx.throw_error(format!("Failed to get tensor: {}", e)),
    };

    // Dequantize tensor
    use realm_core::quant::dequantize_tensor;
    let element_count = tensor.element_count() as usize;
    let dequantized = match dequantize_tensor(&tensor.data, tensor.dtype, element_count) {
        Ok(d) => d,
        Err(e) => return cx.throw_error(format!("Failed to dequantize: {}", e)),
    };

    // Convert to bytes (little-endian f32)
    let f32_bytes: Vec<u8> = dequantized.iter().flat_map(|&f| f.to_le_bytes()).collect();

    let mut buffer = cx.array_buffer(f32_bytes.len())?;
    buffer.as_mut_slice(&mut cx).copy_from_slice(&f32_bytes);
    Ok(buffer)
}

/// Get model metadata
///
/// # Arguments
/// * `model_id` - Model ID (number)
///
/// # Returns
/// Object with `tensor_count` and `total_size` properties
fn get_model_info(mut cx: FunctionContext) -> JsResult<JsObject> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;

    let model = match GLOBAL_MODEL_STORAGE.get_model(model_id) {
        Ok(m) => m,
        Err(e) => return cx.throw_error(format!("Failed to get model: {}", e)),
    };

    let result = cx.empty_object();
    let tensor_count = cx.number(model.tensor_count() as f64);
    let total_size = cx.number(model.total_size as f64);

    result.set(&mut cx, "tensor_count", tensor_count)?;
    result.set(&mut cx, "total_size", total_size)?;

    Ok(result)
}

/// Remove model from storage
///
/// # Arguments
/// * `model_id` - Model ID (number)
fn remove_model(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;

    match GLOBAL_MODEL_STORAGE.remove_model(model_id) {
        Ok(()) => Ok(cx.undefined()),
        Err(e) => cx.throw_error(format!("Failed to remove model: {}", e)),
    }
}

/// Embed tokens using HOST-side computation
///
/// # Arguments
/// * `model_id` - Model ID (number)
/// * `token_ids` - Token IDs array (Uint32Array/Buffer)
///
/// # Returns
/// ArrayBuffer containing hidden states (f32 array: token_count * hidden_size)
fn embed_tokens(mut cx: FunctionContext) -> JsResult<JsArrayBuffer> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;
    let token_ids_buffer = cx.argument::<JsBuffer>(1)?;

    // Read token IDs from buffer
    let token_ids_bytes = token_ids_buffer.as_slice(&cx);
    let token_count = token_ids_bytes.len() / 4; // u32 = 4 bytes
    let mut token_ids = Vec::with_capacity(token_count);

    for chunk in token_ids_bytes.chunks_exact(4) {
        let token_id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        token_ids.push(token_id);
    }

    // Get model and config
    let model = match GLOBAL_MODEL_STORAGE.get_model(model_id) {
        Ok(m) => m,
        Err(e) => return cx.throw_error(format!("Model {} not found: {}", model_id, e)),
    };

    let config = model.extract_config();
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let seq_len = token_ids.len();
    let output_size = seq_len * hidden_size;

    // Load token embeddings from HOST storage
    let embedding_tensor = match model.get_tensor("token_embd.weight") {
        Some(t) => t,
        None => return cx.throw_error("token_embd.weight not found"),
    };

    // Dequantize embeddings
    let full_embeddings = match realm_core::quant::dequantize_tensor(
        &embedding_tensor.data,
        embedding_tensor.dtype,
        vocab_size * hidden_size,
    ) {
        Ok(e) => e,
        Err(e) => return cx.throw_error(format!("Failed to dequantize embeddings: {}", e)),
    };

    // Build hidden states by gathering embeddings for each token
    let mut hidden_states = vec![0.0f32; output_size];

    for (seq_idx, &token_id) in token_ids.iter().enumerate() {
        let out_start = seq_idx * hidden_size;
        let token_id_usize = token_id as usize;

        if token_id_usize >= vocab_size {
            continue; // Skip invalid tokens
        }

        // Copy embedding row for this token
        let emb_start = token_id_usize * hidden_size;
        let emb_end = emb_start + hidden_size;
        if emb_end <= full_embeddings.len() {
            hidden_states[out_start..out_start + hidden_size]
                .copy_from_slice(&full_embeddings[emb_start..emb_end]);
        }
    }

    // Convert to bytes
    let f32_bytes: Vec<u8> = hidden_states
        .iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    let mut buffer = cx.array_buffer(f32_bytes.len())?;
    buffer.as_mut_slice(&mut cx).copy_from_slice(&f32_bytes);
    Ok(buffer)
}

/// Forward through one transformer layer using HOST-side computation
///
/// # Arguments
/// * `model_id` - Model ID (number)
/// * `layer_idx` - Layer index (number)
/// * `hidden_states` - Hidden states (Float32Array/Buffer)
/// * `position` - Position in sequence (number)
///
/// # Returns
/// ArrayBuffer containing output hidden states (f32 array: same size as input)
fn forward_layer(mut cx: FunctionContext) -> JsResult<JsArrayBuffer> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;
    let layer_idx = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let hidden_states_buffer = cx.argument::<JsBuffer>(2)?;
    let _position = cx.argument::<JsNumber>(3)?.value(&mut cx) as usize;

    // Read hidden states from buffer
    let hidden_bytes = hidden_states_buffer.as_slice(&cx);
    let hidden_count = hidden_bytes.len() / 4; // f32 = 4 bytes
    let mut hidden_states = Vec::with_capacity(hidden_count);

    for chunk in hidden_bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        hidden_states.push(val);
    }

    // Get model and config
    let model = match GLOBAL_MODEL_STORAGE.get_model(model_id) {
        Ok(m) => m,
        Err(e) => return cx.throw_error(format!("Model {} not found: {}", model_id, e)),
    };

    let config = model.extract_config();
    let hidden_size = config.hidden_size;
    let seq_len = hidden_states.len() / hidden_size;

    if hidden_states.len() % hidden_size != 0 {
        return cx.throw_error(format!(
            "Invalid hidden_states length: {} not divisible by hidden_size {}",
            hidden_states.len(),
            hidden_size
        ));
    }

    // Initialize backends
    use realm_compute_cpu::{CandleCpuBackend, CandleNeuralOpsBackend};
    let candle_backend = CandleNeuralOpsBackend::new();
    let _cpu_backend = match CandleCpuBackend::new() {
        Ok(b) => b,
        Err(e) => return cx.throw_error(format!("Failed to create CPU backend: {}", e)),
    };

    // 1. Attention norm
    let attn_norm_tensor = match model.get_tensor(&format!("blk.{}.attn_norm.weight", layer_idx)) {
        Some(t) => t,
        None => {
            return cx.throw_error(format!(
                "attn_norm.weight not found for layer {}",
                layer_idx
            ))
        }
    };

    let attn_norm = match realm_core::quant::dequantize_tensor(
        &attn_norm_tensor.data,
        attn_norm_tensor.dtype,
        hidden_size,
    ) {
        Ok(d) => d,
        Err(e) => return cx.throw_error(format!("Failed to dequantize attn_norm: {}", e)),
    };

    let normed = match candle_backend.rms_norm(
        &hidden_states,
        &attn_norm,
        config.rms_norm_eps,
        seq_len,
        hidden_size,
    ) {
        Ok(n) => n,
        Err(e) => return cx.throw_error(format!("RMS norm failed: {}", e)),
    };

    // 2. Attention (simplified - identity for now, full implementation requires complex logic)
    // TODO: Implement full attention with KV cache
    let attn_output = normed.clone();

    // Residual connection
    let mut hidden = vec![0.0f32; hidden_states.len()];
    for i in 0..hidden_states.len() {
        hidden[i] = hidden_states[i] + attn_output[i];
    }

    // 3. FFN norm
    let ffn_norm_tensor = match model.get_tensor(&format!("blk.{}.ffn_norm.weight", layer_idx)) {
        Some(t) => t,
        None => {
            return cx.throw_error(format!("ffn_norm.weight not found for layer {}", layer_idx))
        }
    };

    let ffn_norm = match realm_core::quant::dequantize_tensor(
        &ffn_norm_tensor.data,
        ffn_norm_tensor.dtype,
        hidden_size,
    ) {
        Ok(d) => d,
        Err(e) => return cx.throw_error(format!("Failed to dequantize ffn_norm: {}", e)),
    };

    let ffn_normed = match candle_backend.rms_norm(
        &hidden,
        &ffn_norm,
        config.rms_norm_eps,
        seq_len,
        hidden_size,
    ) {
        Ok(n) => n,
        Err(e) => return cx.throw_error(format!("FFN RMS norm failed: {}", e)),
    };

    // 4. FFN (simplified - identity for now)
    // TODO: Implement full FFN (gate, up, down projections)
    let ffn_output = ffn_normed;

    // Final residual connection
    let mut final_hidden = vec![0.0f32; hidden.len()];
    for i in 0..hidden.len() {
        final_hidden[i] = hidden[i] + ffn_output[i];
    }

    // Convert to bytes
    let f32_bytes: Vec<u8> = final_hidden.iter().flat_map(|&f| f.to_le_bytes()).collect();

    let mut buffer = cx.array_buffer(f32_bytes.len())?;
    buffer.as_mut_slice(&mut cx).copy_from_slice(&f32_bytes);
    Ok(buffer)
}

/// Compute logits using HOST-side computation
///
/// # Arguments
/// * `model_id` - Model ID (number)
/// * `hidden_states` - Hidden states (Float32Array/Buffer)
///
/// # Returns
/// ArrayBuffer containing logits (f32 array: vocab_size)
fn compute_logits(mut cx: FunctionContext) -> JsResult<JsArrayBuffer> {
    let model_id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;
    let hidden_states_buffer = cx.argument::<JsBuffer>(1)?;

    // Read hidden states from buffer
    let hidden_bytes = hidden_states_buffer.as_slice(&cx);
    let hidden_count = hidden_bytes.len() / 4; // f32 = 4 bytes
    let mut hidden_state = Vec::with_capacity(hidden_count);

    for chunk in hidden_bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        hidden_state.push(val);
    }

    // Get model and config
    let model = match GLOBAL_MODEL_STORAGE.get_model(model_id) {
        Ok(m) => m,
        Err(e) => return cx.throw_error(format!("Model {} not found: {}", model_id, e)),
    };

    let config = model.extract_config();
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;

    if hidden_state.len() != hidden_size {
        return cx.throw_error(format!(
            "Invalid hidden_state length: {} != {}",
            hidden_state.len(),
            hidden_size
        ));
    }

    // Initialize backend
    use realm_compute_cpu::CandleNeuralOpsBackend;
    let candle_backend = CandleNeuralOpsBackend::new();

    // Load and apply output norm
    let output_norm_tensor = match model.get_tensor("output_norm.weight") {
        Some(t) => t,
        None => return cx.throw_error("output_norm.weight not found"),
    };

    let output_norm = match realm_core::quant::dequantize_tensor(
        &output_norm_tensor.data,
        output_norm_tensor.dtype,
        hidden_size,
    ) {
        Ok(n) => n,
        Err(e) => return cx.throw_error(format!("Failed to dequantize output_norm: {}", e)),
    };

    // Apply RMS norm
    let hidden_state = match candle_backend.rms_norm(
        &hidden_state,
        &output_norm,
        config.rms_norm_eps,
        1, // seq_len = 1 for single token
        hidden_size,
    ) {
        Ok(h) => h,
        Err(e) => return cx.throw_error(format!("RMS norm failed: {}", e)),
    };

    // Load LM head weights
    let lm_head_tensor = match model.get_tensor("output.weight") {
        Some(t) => t,
        None => return cx.throw_error("output.weight (lm_head) not found"),
    };

    // Dequantize LM head (vocab_size x hidden_size)
    let lm_head = match realm_core::quant::dequantize_tensor(
        &lm_head_tensor.data,
        lm_head_tensor.dtype,
        vocab_size * hidden_size,
    ) {
        Ok(l) => l,
        Err(e) => return cx.throw_error(format!("Failed to dequantize lm_head: {}", e)),
    };

    // Compute logits: hidden_state @ lm_head^T
    let mut logits = vec![0.0f32; vocab_size];
    for (i, logit) in logits.iter_mut().enumerate() {
        let weight_start = i * hidden_size;
        let mut sum = 0.0;
        for j in 0..hidden_size {
            sum += hidden_state[j] * lm_head[weight_start + j];
        }
        *logit = sum;
    }

    // Convert to bytes
    let f32_bytes: Vec<u8> = logits.iter().flat_map(|&f| f.to_le_bytes()).collect();

    let mut buffer = cx.array_buffer(f32_bytes.len())?;
    buffer.as_mut_slice(&mut cx).copy_from_slice(&f32_bytes);
    Ok(buffer)
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("storeModel", store_model)?;
    cx.export_function("getTensor", get_tensor)?;
    cx.export_function("getModelInfo", get_model_info)?;
    cx.export_function("removeModel", remove_model)?;
    cx.export_function("embedTokens", embed_tokens)?;
    cx.export_function("forwardLayer", forward_layer)?;
    cx.export_function("computeLogits", compute_logits)?;
    Ok(())
}
