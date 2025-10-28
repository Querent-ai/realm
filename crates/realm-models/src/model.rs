//! Complete transformer model implementation

use log::{debug, info, warn};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rng;
use realm_compute_cpu::{
    CandleCpuBackend, CandleNeuralOpsBackend, CpuBackendTrait, NaiveCpuBackend,
};
#[cfg(any(feature = "cuda", feature = "metal"))]
use realm_compute_gpu::CandleGpuBackend;
#[cfg(feature = "webgpu")]
use realm_compute_gpu::GpuBackend;
#[cfg(any(feature = "cuda", feature = "metal", feature = "webgpu"))]
use realm_compute_gpu::GpuBackendTrait;
use realm_core::error::Result;
use realm_core::Tokenizer;

use super::{GenerationConfig, KVCache, TransformerConfig, TransformerLayer};

/// Complete transformer model with embeddings and output head
pub struct Model {
    /// Model configuration
    pub config: TransformerConfig,
    /// Token embeddings [vocab_size, hidden_size]
    pub token_embeddings: Vec<f32>,
    /// Transformer layers
    pub layers: Vec<TransformerLayer>,
    /// Output normalization
    pub output_norm: Vec<f32>,
    /// LM head (language model head) [hidden_size, vocab_size]
    /// Often tied with token_embeddings (weight sharing)
    pub lm_head: Vec<f32>,
    /// KV caches for each layer
    pub kv_caches: Vec<KVCache>,
    /// Unified GPU backend for all GPU backends (WebGPU, CUDA, Metal)
    #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
    gpu: Option<Box<dyn GpuBackendTrait>>,
    /// Candle neural network operations backend for optimized neural network operations
    candle_backend: CandleNeuralOpsBackend,
    /// CPU backend for optimized CPU operations
    cpu_backend: Box<dyn realm_compute_cpu::CpuBackendTrait>,
    /// Memory64 model for large models (>4GB) with on-demand layer loading
    #[cfg(feature = "memory64")]
    pub memory64_model: Option<crate::memory64_layer_manager::Memory64Model>,
}

/// Transpose a matrix stored in row-major order
/// Input: [rows, cols] in row-major order
/// Output: [cols, rows] in row-major order
fn transpose_matrix(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
    transposed
}

impl Model {
    /// Create a new model with initialized (zero) weights
    pub fn new(config: TransformerConfig) -> Self {
        let mut layers = Vec::new();
        let mut kv_caches = Vec::new();
        let head_dim = config.hidden_size / config.num_heads;

        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(&config));
            kv_caches.push(KVCache::new(
                config.max_seq_len,
                config.num_kv_heads,
                head_dim,
            ));
        }

        Self {
            token_embeddings: vec![0.0; config.vocab_size * config.hidden_size],
            output_norm: vec![1.0; config.hidden_size],
            lm_head: vec![0.0; config.hidden_size * config.vocab_size],
            kv_caches,
            layers,
            config,
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu: Self::create_gpu_backend(),
            candle_backend: CandleNeuralOpsBackend::new(),
            cpu_backend: Self::create_cpu_backend(),
            #[cfg(feature = "memory64")]
            memory64_model: None,
        }
    }

    /// Create CPU backend with fallback
    fn create_cpu_backend() -> Box<dyn CpuBackendTrait> {
        #[cfg(target_arch = "wasm32")]
        {
            // In WASM environment, use Naive CPU backend directly
            // Candle CPU backend may not work in WASM
            info!("WASM: Using Naive CPU backend");
            return Box::new(NaiveCpuBackend);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // On non-WASM platforms, try Candle first
            match CandleCpuBackend::new() {
                Ok(candle_backend) => {
                    info!("Candle CPU backend initialized successfully");
                    Box::new(candle_backend)
                }
                Err(e) => {
                    warn!("Candle CPU backend initialization failed: {}, falling back to Naive CPU backend", e);
                    Box::new(NaiveCpuBackend)
                }
            }
        }
    }

    /// Create GPU backend with automatic fallback based on available features
    /// Tries backends in priority order: CUDA/Metal → WebGPU
    #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
    fn create_gpu_backend() -> Option<Box<dyn GpuBackendTrait>> {
        // Priority 1: Try Candle GPU backend (CUDA/Metal - synchronous, best performance)
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            match CandleGpuBackend::new() {
                Ok(candle_gpu) => {
                    info!("Candle GPU backend (CUDA/Metal) initialized successfully");
                    return Some(Box::new(candle_gpu));
                }
                Err(e) => {
                    warn!("Candle GPU backend initialization failed: {}", e);
                }
            }
        }

        // Priority 2: Try WebGPU backend (use block_on for synchronous initialization)
        #[cfg(feature = "webgpu")]
        {
            if GpuBackend::is_available() {
                match pollster::FutureExt::block_on(GpuBackend::new()) {
                    Ok(gpu) => {
                        info!("WebGPU backend initialized successfully");
                        return Some(Box::new(gpu));
                    }
                    Err(e) => {
                        warn!("WebGPU initialization failed: {}", e);
                    }
                }
            } else {
                info!("WebGPU not available");
            }
        }

        warn!("No GPU backend available, using CPU");
        None
    }

    /// Initialize GPU backend (tries all available backends automatically)
    #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
    pub fn init_gpu(&mut self) -> Result<()> {
        self.gpu = Self::create_gpu_backend();
        Ok(())
    }

    /// Matrix multiplication with GPU/CPU fallback
    ///
    /// Tries GPU backends in this order:
    /// 1. Candle GPU (CUDA/Metal) - Native GPU, always available
    /// 2. WebGPU - Browser GPU, only with `webgpu` feature
    /// 3. CPU (Candle-optimized) - Fallback
    ///
    /// # Arguments
    /// * `transposed_b` - If true, B is stored as [n, k] (GGUF format) and will use efficient transpose matmul
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        transposed_b: bool,
    ) -> Result<Vec<f32>> {
        // Use CPU backend for all matmuls in Model
        // GPU backends are handled by dispatch_matmul in transformer layers
        if transposed_b {
            self.cpu_backend.matmul_transposed(a, b, m, k, n)
        } else {
            self.cpu_backend.matmul(a, b, m, k, n)
        }
    }

    /// Load weights from GGUF tensors
    pub fn load_from_gguf<R: std::io::Read + std::io::Seek>(
        &mut self,
        tensor_loader: &mut realm_core::tensor_loader::TensorLoader,
        parser: &mut realm_core::formats::gguf::GGUFParser<R>,
    ) -> Result<()> {
        use realm_core::error::Error;

        // Check if we should use Memory64 for large models
        #[cfg(feature = "memory64")]
        {
            // Estimate total model size from tensor metadata
            if let Some(meta) = parser.metadata() {
                let total_size: usize = meta.tensors.iter().map(|t| t.size_bytes).sum();
                let size_gb = total_size as f64 / 1_000_000_000.0;

                info!(
                    "Model size estimate: {:.2} GB ({} bytes)",
                    size_gb, total_size
                );

                // Use Memory64 for models >3GB
                if total_size > 3_000_000_000 {
                    info!("Large model detected - using Memory64 for on-demand layer loading");

                    // Create Memory64 loader and load the model
                    let mut mem64_loader = crate::memory64_gguf::Memory64GGUFLoader::new();
                    let mem64_model = mem64_loader.load_model(parser)?;

                    // Copy embeddings and norms from Memory64Model to self
                    // (forward pass expects these in the Model struct)
                    self.token_embeddings
                        .copy_from_slice(&mem64_model.token_embeddings);
                    self.output_norm.copy_from_slice(&mem64_model.output_norm);
                    self.lm_head.copy_from_slice(&mem64_model.lm_head);

                    info!(
                        "Embeddings and norms loaded ({:.2} MB)",
                        (self.token_embeddings.len() * 4
                            + self.output_norm.len() * 4
                            + self.lm_head.len() * 4) as f64
                            / 1_000_000.0
                    );

                    // Store Memory64Model for on-demand layer access
                    self.memory64_model = Some(mem64_model);

                    info!("Memory64 model loaded - layers will be accessed on-demand");
                    info!(
                        "Memory savings: ~{:.2} GB (layers not loaded into RAM)",
                        (total_size
                            - (self.token_embeddings.len() * 4
                                + self.output_norm.len() * 4
                                + self.lm_head.len() * 4)) as f64
                            / 1_000_000_000.0
                    );

                    // Skip standard layer loading
                    return Ok(());
                }
            }
        }

        // Standard loading path for smaller models
        info!("Using standard loading (all weights in RAM)");

        // Load token embeddings - try different tensor names
        let embedding_data = if let Ok(data) =
            tensor_loader.load_tensor("token_embd.weight", parser)
        {
            debug!("Found 'token_embd.weight'");
            data
        } else if let Ok(data) = tensor_loader.load_tensor("model.embed_tokens.weight", parser) {
            data
        } else if let Ok(data) = tensor_loader.load_tensor("embed_tokens.weight", parser) {
            data
        } else {
            return Err(Error::ParseError("Missing token embeddings".to_string()));
        };

        {
            // GGUF stores token_embd.weight as [vocab_size, hidden_size]
            // Our embedding lookup correctly handles this format
            // by gathering row token_id from the matrix
            self.token_embeddings.copy_from_slice(embedding_data);
        }

        // Load output norm
        if let Ok(norm_data) = tensor_loader.load_tensor("output_norm.weight", parser) {
            // Load raw weights without arbitrary scaling
            self.output_norm.copy_from_slice(norm_data);
        } else {
            eprintln!("WARN: Failed to load output_norm.weight");
        }

        // Load LM head - try different tensor names
        if let Ok(lm_head_data) = tensor_loader.load_tensor("output.weight", parser) {
            // GGUF stores output.weight as [vocab_size, hidden_size]
            // Store as-is for matmul_transposed
            self.lm_head.copy_from_slice(lm_head_data);
        } else if let Ok(lm_head_data) = tensor_loader.load_tensor("lm_head.weight", parser) {
            // GGUF stores lm_head.weight as [vocab_size, hidden_size]
            // Store as-is for matmul_transposed
            self.lm_head.copy_from_slice(lm_head_data);
        } else if let Ok(lm_head_data) = tensor_loader.load_tensor("model.lm_head.weight", parser) {
            // GGUF stores model.lm_head.weight as [vocab_size, hidden_size]
            // Store as-is for matmul_transposed
            self.lm_head.copy_from_slice(lm_head_data);
        } else {
            // Weight tying: LM head shares weights with token embeddings
            let lm_head_transposed = transpose_matrix(
                &self.token_embeddings,
                self.config.hidden_size,
                self.config.vocab_size,
            );
            self.lm_head.copy_from_slice(&lm_head_transposed);
        }

        // Load each layer's weights
        for layer_idx in 0..self.config.num_layers {
            let layer = &mut self.layers[layer_idx];

            // Attention weights
            let wq_name = format!("blk.{}.attn_q.weight", layer_idx);
            let wk_name = format!("blk.{}.attn_k.weight", layer_idx);
            let wv_name = format!("blk.{}.attn_v.weight", layer_idx);
            let wo_name = format!("blk.{}.attn_output.weight", layer_idx);

            // Try without .weight suffix if not found
            let wq_name = if tensor_loader.get_metadata(&wq_name).is_some() {
                wq_name
            } else {
                format!("blk.{}.attn_q", layer_idx)
            };
            let wk_name = if tensor_loader.get_metadata(&wk_name).is_some() {
                wk_name
            } else {
                format!("blk.{}.attn_k", layer_idx)
            };
            let wv_name = if tensor_loader.get_metadata(&wv_name).is_some() {
                wv_name
            } else {
                format!("blk.{}.attn_v", layer_idx)
            };
            let wo_name = if tensor_loader.get_metadata(&wo_name).is_some() {
                wo_name
            } else {
                format!("blk.{}.attn_output", layer_idx)
            };

            if let Ok(wq) = tensor_loader.load_tensor(&wq_name, parser) {
                // Use GGUF weights directly - no transpose needed
                // matmul_transposed will handle the orientation efficiently
                layer.attention_weights.wq = crate::weight_format::WeightFormat::F32(wq.to_vec());
            } else if layer_idx == 0 {
                warn!("Failed to load {}", wq_name);
            }
            if let Ok(wk) = tensor_loader.load_tensor(&wk_name, parser) {
                // Use GGUF weights directly - no transpose needed
                // matmul_transposed will handle the orientation efficiently
                layer.attention_weights.wk = crate::weight_format::WeightFormat::F32(wk.to_vec());
            } else if layer_idx == 0 {
                warn!("Failed to load {}", wk_name);
            }
            if let Ok(wv) = tensor_loader.load_tensor(&wv_name, parser) {
                // Use GGUF weights directly - no transpose needed
                // matmul_transposed will handle the orientation efficiently
                layer.attention_weights.wv = crate::weight_format::WeightFormat::F32(wv.to_vec());
            } else if layer_idx == 0 {
                warn!("Failed to load {}", wv_name);
            }
            if let Ok(wo) = tensor_loader.load_tensor(&wo_name, parser) {
                layer.attention_weights.wo = crate::weight_format::WeightFormat::F32(wo.to_vec());
            } else if layer_idx == 0 {
                warn!("Failed to load {}", wo_name);
            }
            // Attention norm

            // Attention norm
            let attn_norm_name = format!("blk.{}.attn_norm.weight", layer_idx);
            if let Ok(norm) = tensor_loader.load_tensor(&attn_norm_name, parser) {
                layer.attention_norm.copy_from_slice(norm);
            }

            // FFN weights
            let ffn_gate_name = format!("blk.{}.ffn_gate.weight", layer_idx);
            let ffn_up_name = format!("blk.{}.ffn_up.weight", layer_idx);
            let ffn_down_name = format!("blk.{}.ffn_down.weight", layer_idx);

            if let Ok(gate) = tensor_loader.load_tensor(&ffn_gate_name, parser) {
                layer.ffn_weights.w_gate.copy_from_slice(gate);
            }
            if let Ok(up) = tensor_loader.load_tensor(&ffn_up_name, parser) {
                layer.ffn_weights.w_up.copy_from_slice(up);
            }
            if let Ok(down) = tensor_loader.load_tensor(&ffn_down_name, parser) {
                layer.ffn_weights.w_down.copy_from_slice(down);
            }

            // FFN norm
            let ffn_norm_name = format!("blk.{}.ffn_norm.weight", layer_idx);
            if let Ok(norm) = tensor_loader.load_tensor(&ffn_norm_name, parser) {
                layer.ffn_norm.copy_from_slice(norm);
            }
        }

        Ok(())
    }

    /// Forward pass through a single layer
    ///
    /// Supports both standard (Vec) and Memory64 (on-demand) layer access
    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden_states: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        // Check if we should use Memory64 path
        #[cfg(feature = "memory64")]
        let use_memory64 = self.memory64_model.is_some();
        #[cfg(not(feature = "memory64"))]
        let use_memory64 = false;

        if use_memory64 {
            #[cfg(feature = "memory64")]
            {
                // Memory64 path: Get layer from on-demand storage
                let layer = self
                    .memory64_model
                    .as_mut()
                    .unwrap()
                    .get_layer(layer_idx as u32)
                    .map_err(|e| {
                        realm_core::error::Error::Runtime(format!(
                            "Memory64 layer {} access failed: {}",
                            layer_idx, e
                        ))
                    })?;

                let kv_cache = &mut self.kv_caches[layer_idx];
                #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
                let gpu = self
                    .gpu
                    .as_ref()
                    .map(|g| g.as_ref() as &dyn GpuBackendTrait);

                return layer.forward(
                    hidden_states,
                    kv_cache,
                    position,
                    &self.candle_backend,
                    Some(self.cpu_backend.as_ref()),
                    #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
                    gpu,
                );
            }
        }

        // Standard path: Direct Vec access
        let kv_cache = &mut self.kv_caches[layer_idx];
        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
        let gpu = self
            .gpu
            .as_ref()
            .map(|g| g.as_ref() as &dyn GpuBackendTrait);

        self.layers[layer_idx].forward(
            hidden_states,
            kv_cache,
            position,
            &self.candle_backend,
            Some(self.cpu_backend.as_ref()),
            #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
            gpu,
        )
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs \[seq_len\]
    /// * `position` - Current position in sequence (for KV cache)
    ///
    /// # Returns
    /// Logits \[seq_len, vocab_size\]
    pub fn forward(&mut self, token_ids: &[u32], position: usize) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;

        // 1. Embed tokens
        // GGUF stores embeddings as [vocab_size, hidden_size]
        // So for token_id, we need to gather row token_id from the matrix
        let mut hidden_states = vec![0.0; seq_len * hidden_size];

        for (seq_idx, &token_id) in token_ids.iter().enumerate() {
            let out_start = seq_idx * hidden_size;

            // Gather embedding for this token from the [vocab_size, hidden_size] matrix
            // Element [row, col] is at index row * hidden_size + col
            // So embedding dimension i for token_id is at: token_id * hidden_size + i
            for dim_idx in 0..hidden_size {
                let emb_idx = (token_id as usize) * hidden_size + dim_idx;
                if emb_idx < self.token_embeddings.len() {
                    hidden_states[out_start + dim_idx] = self.token_embeddings[emb_idx];
                } else {
                    eprintln!("WARN: Token {} dim {} out of bounds", token_id, dim_idx);
                    break;
                }
            }
        }

        // 2. Pass through transformer layers
        for layer_idx in 0..self.config.num_layers {
            hidden_states = self.forward_layer(layer_idx, &hidden_states, position)?;
        }

        // 3. Final normalization using Candle
        hidden_states = self.candle_backend.rms_norm(
            &hidden_states,
            &self.output_norm,
            self.config.rms_norm_eps,
            seq_len,
            self.config.hidden_size,
        )?;

        // 4. Project to vocabulary (LM head)
        let vocab_size = self.config.vocab_size;

        // OPTIMIZATION: Extract only last token's hidden state before LM head
        // This avoids computing logits for all prompt tokens during prefill
        let last_hidden_start = (seq_len - 1) * hidden_size;
        let last_hidden = &hidden_states[last_hidden_start..last_hidden_start + hidden_size];

        // Compute logits only for the last token (seq_len = 1)
        let logits = self.matmul(last_hidden, &self.lm_head, 1, hidden_size, vocab_size, true)?;

        // Logits are already just for the last token (we computed them that way above)
        Ok(logits)
    }

    /// Sample next token from logits
    ///
    /// # Arguments
    /// * `logits` - Logits for last position \[vocab_size\]
    /// * `temperature` - Sampling temperature (default 1.0, 0.0 = greedy)
    /// * `top_p` - Nucleus sampling threshold (0.0-1.0, 1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    ///
    /// # Returns
    /// Sampled token ID
    pub fn sample(&self, logits: &[f32], temperature: f32, top_p: f32, top_k: u32) -> Result<u32> {
        let vocab_size = logits.len();

        // Greedy sampling (deterministic)
        if temperature <= 0.0 {
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0));
        }

        // Apply temperature scaling
        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Find max for numerical stability
        let max_logit = scaled_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Softmax
        let exp_logits: Vec<f32> = scaled_logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let mut probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // Create index-probability pairs
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();

        // Sort by probability (descending)
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k filtering
        if top_k > 0 && (top_k as usize) < vocab_size {
            // Zero out probabilities beyond top-k
            for (idx, _) in indexed_probs.iter().skip(top_k as usize) {
                probs[*idx] = 0.0;
            }
        }

        // Apply top-p (nucleus) filtering
        if top_p < 1.0 && top_p > 0.0 {
            let mut cumulative_prob = 0.0;
            let mut cutoff_idx = vocab_size;

            for (i, (_idx, prob)) in indexed_probs.iter().enumerate() {
                cumulative_prob += prob;
                if cumulative_prob >= top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Zero out probabilities beyond nucleus
            for (idx, _) in indexed_probs.iter().skip(cutoff_idx) {
                probs[*idx] = 0.0;
            }
        }

        // Renormalize probabilities
        let sum_filtered: f32 = probs.iter().sum();
        if sum_filtered > 0.0 {
            for p in &mut probs {
                *p /= sum_filtered;
            }
        } else {
            // Fallback: uniform over top-1
            return Ok(indexed_probs[0].0 as u32);
        }

        // Sample from filtered distribution using true randomness
        let mut rng = rng();

        // Convert probabilities to weights for weighted sampling
        // Filter out zero probabilities for efficiency
        let non_zero_probs: Vec<(usize, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(idx, &p)| (idx, p as f64))
            .collect();

        if non_zero_probs.is_empty() {
            // Fallback to greedy if all probs are zero
            return Ok(indexed_probs[0].0 as u32);
        }

        let weights: Vec<f64> = non_zero_probs.iter().map(|(_, w)| *w).collect();
        let indices: Vec<usize> = non_zero_probs.iter().map(|(i, _)| *i).collect();

        let dist = WeightedIndex::new(&weights).map_err(|e| {
            realm_core::error::Error::Runtime(format!("Weighted sampling failed: {}", e))
        })?;
        let sampled_idx = dist.sample(&mut rng);
        let token_id = indices[sampled_idx] as u32;

        Ok(token_id)
    }

    /// Generate text from a prompt
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `tokenizer` - Tokenizer for encoding/decoding
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated text
    pub fn generate(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        config: &GenerationConfig,
    ) -> Result<String> {
        let max_tokens = config.max_tokens;
        let temperature = config.temperature;
        let top_p = config.top_p;
        let top_k = config.top_k;
        let repetition_penalty = config.repetition_penalty;
        use realm_core::error::Error;

        // Create logits processor with configured sampling strategy
        let mut logits_processor = crate::LogitsProcessor::with_params(
            42, // seed - could be made configurable
            temperature as f64,
            top_p as f64,
            top_k,
            repetition_penalty,
        );

        // Clear KV cache for new generation
        self.clear_kv_cache();

        // Encode prompt (with BOS token)
        let mut tokens = tokenizer.encode(prompt, true)?;

        if tokens.is_empty() {
            return Err(Error::ParseError("Empty token sequence".to_string()));
        }

        let num_prompt_tokens = tokens.len();

        // PREFILL PHASE: Process prompt tokens in chunks to avoid memory issues
        let chunk_size = 8; // Process 8 tokens at a time for better performance
        let mut prefill_logits = Vec::new();

        for chunk_start in (0..num_prompt_tokens).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(num_prompt_tokens);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_logits = self.forward(chunk, chunk_start)?;
            prefill_logits.extend(chunk_logits);
        }

        // Get logits for the last prompt token
        let logits = prefill_logits[(prefill_logits.len() - self.config.vocab_size)..].to_vec();

        // Sample first generated token
        let mut last_logits = logits.clone();
        let next = logits_processor
            .sample(&mut last_logits)
            .map_err(Error::ParseError)?;

        tokens.push(next);

        // DECODE PHASE: Generate tokens one at a time
        let mut generated = 1;
        while generated < max_tokens {
            // Forward pass with just the last generated token
            let last_token = tokens[tokens.len() - 1];

            // Use KV cache position, not absolute position
            let cache_position = if !self.kv_caches.is_empty() {
                self.kv_caches[0].current_seq_len
            } else {
                tokens.len() - 1
            };

            last_logits = self.forward(&[last_token], cache_position)?;

            // Sample next token
            let next = logits_processor
                .sample(&mut last_logits)
                .map_err(Error::ParseError)?;

            // Check for EOS token
            if next == tokenizer.special_tokens().eos_token_id {
                break;
            }

            // Add generated token to our sequence
            tokens.push(next);
            generated += 1;
        }

        // Decode all tokens to text (skip special tokens)
        let generated_text = tokenizer.decode(&tokens, true)?;

        Ok(generated_text)
    }

    /// Generate text with streaming callback
    ///
    /// Calls the callback for each generated token in real-time.
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `tokenizer` - Tokenizer for encoding/decoding
    /// * `config` - Generation configuration
    /// * `callback` - Function called for each token: (token_id, decoded_text) -> bool (continue?)
    ///
    /// # Returns
    /// Full generated text
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        config: &GenerationConfig,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let max_tokens = config.max_tokens;
        let temperature = config.temperature;
        let top_p = config.top_p;
        let top_k = config.top_k;
        let repetition_penalty = config.repetition_penalty;
        use realm_core::error::Error;

        // Create logits processor with configured sampling strategy
        let mut logits_processor = crate::LogitsProcessor::with_params(
            42, // seed - could be made configurable
            temperature as f64,
            top_p as f64,
            top_k,
            repetition_penalty,
        );

        // Clear KV cache for new generation
        self.clear_kv_cache();

        // Encode prompt (with BOS token)
        let mut tokens = tokenizer.encode(prompt, true)?;

        if tokens.is_empty() {
            return Err(Error::ParseError("Empty token sequence".to_string()));
        }

        let num_prompt_tokens = tokens.len();

        // PHASE 1: PREFILL - Process prompt tokens in chunks to avoid memory issues
        eprintln!(
            "🔥 PREFILL: Processing {} prompt tokens in chunks",
            num_prompt_tokens
        );

        // Process prompt tokens in chunks of 8 to avoid memory issues
        let chunk_size = 8;
        let mut prefill_logits = Vec::new();

        for chunk_start in (0..num_prompt_tokens).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(num_prompt_tokens);
            let chunk = &tokens[chunk_start..chunk_end];
            let chunk_logits = self.forward(chunk, chunk_start)?;
            prefill_logits.extend(chunk_logits);
        }

        // Get the last token's logits for the first generation step
        let mut last_logits =
            prefill_logits[(prefill_logits.len() - self.config.vocab_size)..].to_vec();

        // PHASE 2: DECODE - Process ONE new token at a time
        let mut pos = num_prompt_tokens;

        while pos < num_prompt_tokens + max_tokens - 1 {
            // Generate next token
            let next = logits_processor
                .sample(&mut last_logits)
                .map_err(Error::ParseError)?;

            // Check for EOS
            if next == tokenizer.special_tokens().eos_token_id {
                break;
            }

            // Add to output
            tokens.push(next);

            // Process the new token at the correct position
            let cache_position = if !self.kv_caches.is_empty() {
                self.kv_caches[0].current_seq_len
            } else {
                tokens.len() - 1
            };

            let logits = self.forward(&[next], cache_position)?;

            // Get logits for next iteration
            last_logits = logits[(logits.len() - self.config.vocab_size)..].to_vec();

            // Decode just this token and call callback
            let token_text = tokenizer.decode(&[next], true)?;

            // Call callback - if it returns false, stop generation
            if !callback(next, &token_text) {
                break;
            }

            pos += 1;
        }

        // Return full generated text
        let generated_text = tokenizer.decode(&tokens, true)?;
        Ok(generated_text)
    }

    /// Clear all KV caches (for new sequence)
    pub fn clear_kv_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
    }

    pub fn rms_norm(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        let hidden_size = weight.len();
        let seq_len = input.len() / hidden_size;
        let mut output = vec![0.0; input.len()];

        for seq in 0..seq_len {
            let offset = seq * hidden_size;
            let slice = &input[offset..offset + hidden_size];

            // Compute RMS - FIXED: epsilon goes INSIDE sqrt like llama.cpp
            let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
            let mean = sum_sq / hidden_size as f32;
            let rms = (mean + self.config.rms_norm_eps).sqrt();

            // Normalize and scale
            for i in 0..hidden_size {
                output[offset + i] = (slice[i] / rms) * weight[i];
            }
        }

        Ok(output)
    }
}
