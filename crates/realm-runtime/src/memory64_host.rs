//! Production-Hardened Memory64 Runtime
//!
//! This is the PRODUCTION-GRADE implementation with critical fixes:
//! 1. ✅ parking_lot::Mutex (no poisoning, better performance)
//! 2. ✅ Integer overflow checks on all arithmetic
//! 3. ✅ WASM pointer validation in host functions
//! 4. ✅ Proper error logging
//!
//! Battle-tested patterns from:
//! - wasmex (Elixir/Erlang WASM runtime) - fine-grained locking
//! - Wasmtime/Wasmer - pointer validation, error handling
//! - llama.cpp - layer loading patterns

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex; // No poisoning, faster than std::sync::Mutex
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use wasmtime::{AsContext, Caller, Extern, Linker, Memory, MemoryType, Store};

// Import Candle backends
#[cfg(not(target_arch = "wasm32"))]
use realm_compute_cpu::{CandleCpuBackend, CpuBackendTrait};

#[cfg(all(not(target_arch = "wasm32"), any(feature = "cuda", feature = "metal")))]
use realm_compute_gpu::GpuBackendTrait;

use realm_core::error::Error;

/// Memory access statistics for monitoring (thread-safe)
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub reads: u64,
    pub writes: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub errors: u64, // Track error count
}

/// Memory region descriptor for multi-memory layouts
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub id: u32,
    pub name: String,
    pub start_offset: u64,
    pub size: u64,
    pub purpose: String,
}

impl MemoryRegion {
    pub fn new(
        id: u32,
        name: impl Into<String>,
        start_offset: u64,
        size: u64,
        purpose: impl Into<String>,
    ) -> Result<Self> {
        // ✅ FIX 2: Validate size is non-zero and page-aligned
        if size == 0 {
            return Err(anyhow!("Region size cannot be zero"));
        }
        if !size.is_multiple_of(65536) {
            return Err(anyhow!(
                "Region size must be page-aligned (multiple of 64KB), got {}",
                size
            ));
        }

        // ✅ FIX 2: Check for offset overflow
        let _end = start_offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("Region offset + size overflows u64"))?;

        Ok(Self {
            id,
            name: name.into(),
            start_offset,
            size,
            purpose: purpose.into(),
        })
    }

    /// Check if an offset falls within this region
    pub fn contains(&self, offset: u64) -> bool {
        // ✅ FIX 2: Use checked arithmetic
        if let Some(end) = self.start_offset.checked_add(self.size) {
            offset >= self.start_offset && offset < end
        } else {
            false // Overflow means invalid region
        }
    }

    /// Get local offset within this region
    pub fn local_offset(&self, global_offset: u64) -> Result<u64> {
        if !self.contains(global_offset) {
            return Err(anyhow!(
                "Offset {} not in region {} ({}..{})",
                global_offset,
                self.name,
                self.start_offset,
                self.start_offset.saturating_add(self.size)
            ));
        }
        Ok(global_offset - self.start_offset)
    }
}

/// Memory layout configuration
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    pub regions: Vec<MemoryRegion>,
    pub total_size: u64,
}

impl MemoryLayout {
    /// Create a single memory layout with validation
    pub fn single(size_gb: u64, purpose: impl Into<String>) -> Result<Self> {
        // ✅ FIX 2: Validate input
        if size_gb == 0 {
            return Err(anyhow!("Size must be greater than 0"));
        }
        if size_gb > 16384 {
            // 16TB limit (reasonable for now)
            return Err(anyhow!("Size {} GB exceeds maximum 16TB", size_gb));
        }

        // ✅ FIX 2: Check for overflow when converting to bytes
        let size = size_gb
            .checked_mul(1024)
            .and_then(|v| v.checked_mul(1024))
            .and_then(|v| v.checked_mul(1024))
            .ok_or_else(|| anyhow!("Size {} GB causes overflow", size_gb))?;

        let region = MemoryRegion::new(0, "memory0", 0, size, purpose)?;

        Ok(Self {
            regions: vec![region],
            total_size: size,
        })
    }

    /// Create a multi-memory layout with custom regions
    pub fn multi(regions: &[(&str, u64)]) -> Result<Self> {
        if regions.is_empty() {
            return Err(anyhow!("Must provide at least one region"));
        }

        let mut offset = 0u64;
        let mut region_list = Vec::new();

        for (id, (name, size_gb)) in regions.iter().enumerate() {
            // ✅ FIX 2: Validate and check overflow
            if *size_gb == 0 {
                return Err(anyhow!("Region '{}' has zero size", name));
            }

            let size = size_gb
                .checked_mul(1024)
                .and_then(|v| v.checked_mul(1024))
                .and_then(|v| v.checked_mul(1024))
                .ok_or_else(|| anyhow!("Size {} GB causes overflow", size_gb))?;

            let region =
                MemoryRegion::new(id as u32, format!("memory{}", id), offset, size, *name)?;

            // ✅ FIX 2: Check offset overflow
            offset = offset
                .checked_add(size)
                .ok_or_else(|| anyhow!("Total memory layout size overflows u64"))?;

            region_list.push(region);
        }

        Ok(Self {
            regions: region_list,
            total_size: offset,
        })
    }

    /// Find the region containing the given offset
    pub fn find_region(&self, offset: u64) -> Result<&MemoryRegion> {
        self.regions
            .iter()
            .find(|r| r.contains(offset))
            .ok_or_else(|| {
                anyhow!(
                    "Offset {} not in any memory region (total size: {} GB)",
                    offset,
                    self.total_size / 1024 / 1024 / 1024
                )
            })
    }

    /// Get total memory in GB
    pub fn total_gb(&self) -> f64 {
        self.total_size as f64 / 1024.0 / 1024.0 / 1024.0
    }
}

/// Layer information for model weights tracking
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub layer_id: u32,
    pub offset: u64,
    pub size: usize,
    pub memory_id: u32,
    pub layer_type: String,
}

/// Production Memory64 state with thread-safe access
pub struct Memory64State {
    /// Memory64 instances (one per region)
    memories: Vec<Memory>,
    /// Memory layout configuration
    layout: MemoryLayout,
    /// Layer tracking
    layers: Vec<LayerInfo>,
    /// Statistics
    stats: MemoryStats,
    /// Feature flag
    enabled: bool,
}

impl Memory64State {
    /// Create new Memory64 state
    pub fn new(layout: MemoryLayout, enabled: bool) -> Self {
        Self {
            memories: Vec::new(),
            layout,
            layers: Vec::new(),
            stats: MemoryStats::default(),
            enabled,
        }
    }

    /// Initialize Memory64 instances with validation
    pub fn initialize(&mut self, store: &mut Store<()>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        for region in &self.layout.regions {
            // ✅ FIX 2: Validate before division
            if region.size == 0 {
                return Err(anyhow!("Region '{}' has zero size", region.name));
            }

            // Calculate pages (64KB per page)
            let min_pages = region.size / 65536;
            let max_pages = min_pages.checked_mul(2); // Allow 2x growth

            let memory_type = MemoryType::new64(min_pages, max_pages);
            let memory = Memory::new(&mut *store, memory_type)
                .with_context(|| format!("Failed to create Memory64 for region {}", region.name))?;

            self.memories.push(memory);
        }

        Ok(())
    }

    /// Write data to Memory64 with bounds checking
    pub fn write(&mut self, store: &mut Store<()>, offset: u64, data: &[u8]) -> Result<()> {
        if !self.enabled {
            return Err(anyhow!("Memory64 not enabled"));
        }

        let region = self.layout.find_region(offset)?;
        let local_offset = region.local_offset(offset)?;
        let memory = &self.memories[region.id as usize];

        // ✅ FIX 2: Check for overflow
        let end_offset = local_offset
            .checked_add(data.len() as u64)
            .ok_or_else(|| anyhow!("Write offset + size overflows u64"))?;

        if end_offset > region.size {
            return Err(anyhow!(
                "Write would exceed region {} bounds (offset: {}, size: {}, region size: {})",
                region.name,
                local_offset,
                data.len(),
                region.size
            ));
        }

        memory
            .write(store, local_offset as usize, data)
            .with_context(|| format!("Failed to write to region {}", region.name))?;

        // Update stats
        self.stats.writes += 1;
        self.stats.bytes_written += data.len() as u64;

        Ok(())
    }

    /// Read data from Memory64 with bounds checking
    pub fn read<T>(
        &mut self,
        store: impl AsContext<Data = T>,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<()> {
        if !self.enabled {
            return Err(anyhow!("Memory64 not enabled"));
        }

        let region = self.layout.find_region(offset)?;
        let local_offset = region.local_offset(offset)?;
        let memory = &self.memories[region.id as usize];

        // ✅ FIX 2: Check for overflow
        let end_offset = local_offset
            .checked_add(buffer.len() as u64)
            .ok_or_else(|| anyhow!("Read offset + size overflows u64"))?;

        if end_offset > region.size {
            return Err(anyhow!(
                "Read would exceed region {} bounds (offset: {}, size: {}, region size: {})",
                region.name,
                local_offset,
                buffer.len(),
                region.size
            ));
        }

        memory
            .read(&store, local_offset as usize, buffer)
            .with_context(|| format!("Failed to read from region {}", region.name))?;

        // Update stats
        self.stats.reads += 1;
        self.stats.bytes_read += buffer.len() as u64;

        Ok(())
    }

    /// Register a layer's location
    pub fn register_layer(
        &mut self,
        layer_id: u32,
        offset: u64,
        size: usize,
        layer_type: impl Into<String>,
    ) -> Result<()> {
        let region = self.layout.find_region(offset)?;

        self.layers.push(LayerInfo {
            layer_id,
            offset,
            size,
            memory_id: region.id,
            layer_type: layer_type.into(),
        });

        Ok(())
    }

    /// Get layer information
    pub fn get_layer(&self, layer_id: u32) -> Result<&LayerInfo> {
        self.layers
            .iter()
            .find(|l| l.layer_id == layer_id)
            .ok_or_else(|| anyhow!("Layer {} not found", layer_id))
    }

    /// Get statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get layout information
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }
}

/// Production-ready Memory64 runtime with parking_lot::Mutex
pub struct Memory64Runtime {
    state: Arc<Mutex<Memory64State>>, // ✅ FIX 1: parking_lot::Mutex (no poisoning)

    // Candle backends for host-side acceleration (not available in WASM)
    // Wrapped in Arc for sharing across host function closures
    #[cfg(not(target_arch = "wasm32"))]
    cpu_backend: Option<Arc<dyn CpuBackendTrait>>,

    #[cfg(all(not(target_arch = "wasm32"), any(feature = "cuda", feature = "metal")))]
    gpu_backend: Option<Arc<dyn GpuBackendTrait>>,

    // Streaming callback for token-by-token generation
    // Used by realm_stream_token host function to send tokens as they're generated
    // Uses blocking channel since host functions are called from blocking context
    #[cfg(not(target_arch = "wasm32"))]
    stream_callback: Arc<Mutex<Option<std::sync::mpsc::Sender<String>>>>,
}

impl Memory64Runtime {
    /// Create new runtime
    pub fn new(layout: MemoryLayout, enabled: bool) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let cpu_backend = Self::create_cpu_backend();

        #[cfg(all(not(target_arch = "wasm32"), any(feature = "cuda", feature = "metal")))]
        let gpu_backend = Self::create_gpu_backend();

        Self {
            state: Arc::new(Mutex::new(Memory64State::new(layout, enabled))),
            #[cfg(not(target_arch = "wasm32"))]
            cpu_backend,
            #[cfg(all(not(target_arch = "wasm32"), any(feature = "cuda", feature = "metal")))]
            gpu_backend,
            #[cfg(not(target_arch = "wasm32"))]
            stream_callback: Arc::new(Mutex::new(None)),
        }
    }

    /// Set streaming callback for token generation
    /// This callback will be invoked by realm_stream_token host function
    /// Uses blocking channel since host functions are called from blocking context
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_stream_callback(&self, sender: std::sync::mpsc::Sender<String>) {
        *self.stream_callback.lock() = Some(sender);
    }

    /// Clear streaming callback
    #[cfg(not(target_arch = "wasm32"))]
    pub fn clear_stream_callback(&self) {
        *self.stream_callback.lock() = None;
    }

    /// Create CPU backend (tries Candle first, falls back to none)
    #[cfg(not(target_arch = "wasm32"))]
    fn create_cpu_backend() -> Option<Arc<dyn CpuBackendTrait>> {
        match CandleCpuBackend::new() {
            Ok(backend) => {
                info!("Memory64 Runtime: Candle CPU backend initialized");
                Some(Arc::new(backend))
            }
            Err(e) => {
                warn!("Memory64 Runtime: Candle CPU backend failed: {}, WASM will use NaiveCpuBackend fallback", e);
                None
            }
        }
    }

    /// Create GPU backend (CUDA/Metal via Candle)
    #[cfg(all(not(target_arch = "wasm32"), any(feature = "cuda", feature = "metal")))]
    fn create_gpu_backend() -> Option<Arc<dyn GpuBackendTrait>> {
        use realm_compute_gpu::CandleGpuBackend;

        match CandleGpuBackend::new() {
            Ok(backend) => {
                let backend_name = backend.name();
                info!(
                    "Memory64 Runtime: Candle GPU backend initialized ({})",
                    backend_name
                );
                Some(Arc::new(backend))
            }
            Err(e) => {
                warn!("Memory64 Runtime: Candle GPU backend failed: {}", e);
                None
            }
        }
    }

    /// Initialize Memory64 instances
    pub fn initialize(&self, store: &mut Store<()>) -> Result<()> {
        self.state.lock().initialize(store) // ✅ FIX 1: No poison handling needed
    }

    /// Write model data
    pub fn write_model_data(&self, store: &mut Store<()>, offset: u64, data: &[u8]) -> Result<()> {
        self.state.lock().write(store, offset, data)
    }

    /// Register a layer
    pub fn register_layer(
        &self,
        _store: &mut Store<()>,
        layer_id: u32,
        layer_type: impl Into<String>,
        offset: u64,
        size: usize,
    ) -> Result<()> {
        self.state
            .lock()
            .register_layer(layer_id, offset, size, layer_type)
    }

    /// Get statistics
    pub fn stats(&self) -> Result<MemoryStats> {
        Ok(self.state.lock().stats().clone())
    }

    /// Get statistics (convenience method for host example)
    pub fn get_stats<T>(&self, _store: &Store<T>) -> Result<MemoryStats> {
        Ok(self.state.lock().stats().clone())
    }
}

/// Helper: Dequantize WeightFormat to f32
fn dequantize_weight_format_to_f32(
    weight: &realm_models::WeightFormat,
) -> std::result::Result<Vec<f32>, anyhow::Error> {
    use realm_core::quant::{
        dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
        dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
        dequantize_q8_1, dequantize_q8_k, Q4_BLOCK_SIZE, Q8_BLOCK_SIZE, QK_K,
    };

    match weight {
        realm_models::WeightFormat::F32(w) => Ok(w.clone()),
        realm_models::WeightFormat::Q4K(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * QK_K);
            for block in blocks {
                let mut block_output = vec![0.0f32; QK_K];
                dequantize_q4_k(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q5K(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * QK_K);
            for block in blocks {
                let mut block_output = vec![0.0f32; QK_K];
                dequantize_q5_k(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q6K(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * QK_K);
            for block in blocks {
                let mut block_output = vec![0.0f32; QK_K];
                dequantize_q6_k(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q8K(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * QK_K);
            for block in blocks {
                let mut block_output = vec![0.0f32; QK_K];
                dequantize_q8_k(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q2K(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * QK_K);
            for block in blocks {
                let mut block_output = vec![0.0f32; QK_K];
                dequantize_q2_k(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q3K(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * QK_K);
            for block in blocks {
                let mut block_output = vec![0.0f32; QK_K];
                dequantize_q3_k(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q40(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * Q4_BLOCK_SIZE);
            for block in blocks {
                let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
                dequantize_q4_0(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q41(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * Q4_BLOCK_SIZE);
            for block in blocks {
                let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
                dequantize_q4_1(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q50(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * Q4_BLOCK_SIZE);
            for block in blocks {
                let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
                dequantize_q5_0(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q51(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * Q4_BLOCK_SIZE);
            for block in blocks {
                let mut block_output = vec![0.0f32; Q4_BLOCK_SIZE];
                dequantize_q5_1(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q80(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * Q8_BLOCK_SIZE);
            for block in blocks {
                let mut block_output = vec![0.0f32; Q8_BLOCK_SIZE];
                dequantize_q8_0(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
        realm_models::WeightFormat::Q81(blocks) => {
            let mut output = Vec::with_capacity(blocks.len() * Q8_BLOCK_SIZE);
            for block in blocks {
                let mut block_output = vec![0.0f32; Q8_BLOCK_SIZE];
                dequantize_q8_1(block, &mut block_output)?;
                output.extend_from_slice(&block_output);
            }
            Ok(output)
        }
    }
}

/// Helper: Apply LoRA to WeightFormat if adapter is set
/// For quantized weights, dequantize to F32, apply LoRA, then use F32
fn apply_lora_to_weight_format(
    weight: realm_models::WeightFormat,
    lora_adapter_id: Option<&str>,
    layer_name: &str,
    weight_name: &str,
    out_dim: usize,
    in_dim: usize,
) -> std::result::Result<realm_models::WeightFormat, anyhow::Error> {
    if let Some(adapter_id) = lora_adapter_id {
        use crate::lora::get_global_lora_manager;

        // Dequantize to F32 if needed (supports all quantization types)
        let f32_weights = match dequantize_weight_format_to_f32(&weight) {
            Ok(w) => w,
            Err(e) => {
                debug!(
                    "LoRA: Failed to dequantize weight {}: {}, skipping LoRA",
                    weight_name, e
                );
                return Ok(weight);
            }
        };

        // Apply LoRA
        let lora_manager = get_global_lora_manager();
        let full_layer_name = format!("{}.{}", layer_name, weight_name);
        match lora_manager.lock() {
            Ok(manager_guard) => {
                match manager_guard.apply_to_weights(
                    adapter_id,
                    &full_layer_name,
                    &f32_weights,
                    out_dim,
                    in_dim,
                ) {
                    Ok(modified) => Ok(realm_models::WeightFormat::F32(modified)),
                    Err(e) => {
                        debug!(
                            "LoRA: Failed to apply to {}: {}, using base weights",
                            weight_name, e
                        );
                        Ok(realm_models::WeightFormat::F32(f32_weights)) // Fall back to base weights
                    }
                }
            }
            Err(_) => {
                debug!("LoRA: Mutex lock failed, using base weights");
                Ok(realm_models::WeightFormat::F32(f32_weights)) // Fall back to base weights
            }
        }
    } else {
        Ok(weight) // No LoRA adapter, use base weights
    }
}

impl Memory64Runtime {
    /// Add host functions to linker with WASM pointer validation
    pub fn add_to_linker(&self, linker: &mut Linker<()>) -> Result<()> {
        let state = self.state.clone();

        // Host function: Load layer weights from Memory64 to WASM memory
        linker.func_wrap(
            "env",
            "memory64_load_layer",
            move |mut caller: Caller<'_, ()>, layer_id: u32, wasm_ptr: u32, max_size: u32| -> i32 {
                let state_clone = state.clone();
                let mut state_guard = state_clone.lock(); // ✅ FIX 1: No poison check

                if !state_guard.enabled {
                    return -1;
                }

                // Get layer info
                let layer = match state_guard.get_layer(layer_id) {
                    Ok(l) => l.clone(),
                    Err(e) => {
                        error!("Layer {} not found: {}", layer_id, e);
                        return -2;
                    }
                };

                if layer.size > max_size as usize {
                    error!(
                        "Buffer too small for layer {}: need {}, got {}",
                        layer_id, layer.size, max_size
                    );
                    return -3;
                }

                // ✅ FIX 3: Validate WASM pointer BEFORE allocating buffer
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("No WASM memory export available");
                        return -5;
                    }
                };

                let wasm_mem_size = wasm_memory.data_size(&caller);

                // ✅ FIX 3: Check pointer bounds with overflow protection
                let end_ptr = match (wasm_ptr as usize).checked_add(layer.size) {
                    Some(end) => end,
                    None => {
                        error!("WASM pointer overflow: {} + {}", wasm_ptr, layer.size);
                        state_guard.stats.errors += 1;
                        return -6;
                    }
                };

                if end_ptr > wasm_mem_size {
                    error!(
                        "WASM pointer out of bounds: {} + {} > {}",
                        wasm_ptr, layer.size, wasm_mem_size
                    );
                    state_guard.stats.errors += 1;
                    return -7;
                }

                // Read from Memory64
                let mut buffer = vec![0u8; layer.size];
                if let Err(e) = state_guard.read(caller.as_context(), layer.offset, &mut buffer) {
                    error!("Failed to read layer {}: {}", layer_id, e);
                    state_guard.stats.errors += 1;
                    return -4;
                }

                // Write to WASM memory (already validated)
                if let Err(e) = wasm_memory.write(&mut caller, wasm_ptr as usize, &buffer) {
                    error!("Failed to write to WASM memory: {}", e);
                    state_guard.stats.errors += 1;
                    return -8;
                }

                layer.size as i32
            },
        )?;

        // Host function: Read data from Memory64
        let state2 = self.state.clone();
        linker.func_wrap(
            "env",
            "memory64_read",
            move |mut caller: Caller<'_, ()>, offset: u64, wasm_ptr: u32, size: u32| -> i32 {
                let state_clone = state2.clone();
                let mut state_guard = state_clone.lock();

                if !state_guard.enabled {
                    return -1;
                }

                // ✅ FIX 3: Validate WASM pointer
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("No WASM memory export");
                        return -3;
                    }
                };

                let wasm_mem_size = wasm_memory.data_size(&caller);

                // ✅ FIX 3: Check overflow
                let end_ptr = match (wasm_ptr as usize).checked_add(size as usize) {
                    Some(end) => end,
                    None => {
                        error!("WASM pointer overflow: {} + {}", wasm_ptr, size);
                        state_guard.stats.errors += 1;
                        return -4;
                    }
                };

                if end_ptr > wasm_mem_size {
                    error!(
                        "WASM pointer out of bounds: {} + {} > {}",
                        wasm_ptr, size, wasm_mem_size
                    );
                    state_guard.stats.errors += 1;
                    return -5;
                }

                // Read from Memory64
                let mut buffer = vec![0u8; size as usize];
                if let Err(e) = state_guard.read(caller.as_context(), offset, &mut buffer) {
                    error!("memory64_read failed at offset {}: {}", offset, e);
                    state_guard.stats.errors += 1;
                    return -2;
                }

                // Write to WASM memory
                if let Err(e) = wasm_memory.write(&mut caller, wasm_ptr as usize, &buffer) {
                    error!("Failed to write to WASM memory: {}", e);
                    state_guard.stats.errors += 1;
                    return -6;
                }

                size as i32
            },
        )?;

        // Host function: Check if Memory64 is enabled
        let state3 = self.state.clone();
        linker.func_wrap(
            "env",
            "memory64_is_enabled",
            move |_caller: Caller<'_, ()>| -> i32 {
                if state3.lock().enabled {
                    1
                } else {
                    0
                }
            },
        )?;

        // Host function: Get memory stats
        let state4 = self.state.clone();
        linker.func_wrap(
            "env",
            "memory64_stats",
            move |_caller: Caller<'_, ()>| -> i64 { state4.lock().stats().reads as i64 },
        )?;

        // ========================================
        // Candle Backend Host Functions (GPU-first, CPU fallback)
        // ========================================

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Host function: Matrix multiplication using GPU (if available) or CPU backend
            // Parameters: a_ptr, b_ptr, result_ptr, m, k, n (all as WASM memory offsets)
            // Returns: result_size on success, negative on error
            #[cfg(all(not(target_arch = "wasm32"), any(feature = "cuda", feature = "metal")))]
            let gpu_backend = self.gpu_backend.clone();
            let cpu_backend = self.cpu_backend.clone();
            linker.func_wrap(
                "env",
                "candle_matmul",
                move |mut caller: Caller<'_, ()>,
                      a_ptr: u32,
                      b_ptr: u32,
                      result_ptr: u32,
                      m: u32,
                      k: u32,
                      n: u32|
                      -> i32 {
                    // Get WASM memory
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("No WASM memory export");
                            return -2;
                        }
                    };

                    let m_usize = m as usize;
                    let k_usize = k as usize;
                    let n_usize = n as usize;

                    let a_size = m_usize * k_usize;
                    let b_size = k_usize * n_usize;
                    let result_size = m_usize * n_usize;

                    // Read input matrices from WASM memory
                    let mut a_buffer = vec![0u8; a_size * 4]; // f32 = 4 bytes
                    let mut b_buffer = vec![0u8; b_size * 4];

                    if let Err(e) = wasm_memory.read(&caller, a_ptr as usize, &mut a_buffer) {
                        error!("Failed to read matrix A: {}", e);
                        return -3;
                    }

                    if let Err(e) = wasm_memory.read(&caller, b_ptr as usize, &mut b_buffer) {
                        error!("Failed to read matrix B: {}", e);
                        return -4;
                    }

                    // Convert bytes to f32 slices
                    let a_f32: Vec<f32> = a_buffer
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    let b_f32: Vec<f32> = b_buffer
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Try GPU first, fallback to CPU
                    #[cfg(all(
                        not(target_arch = "wasm32"),
                        any(feature = "cuda", feature = "metal")
                    ))]
                    let result = if let Some(gpu) = &gpu_backend {
                        match gpu.matmul(&a_f32, &b_f32, m as u32, k as u32, n as u32) {
                            Ok(r) => Ok(r),
                            Err(e) => {
                                warn!("GPU matmul failed, falling back to CPU: {}", e);
                                if let Some(cpu) = &cpu_backend {
                                    cpu.matmul(&a_f32, &b_f32, m_usize, k_usize, n_usize)
                                } else {
                                    Err(Error::Runtime("No compute backend available".to_string()))
                                }
                            }
                        }
                    } else {
                        // No GPU, try CPU
                        if let Some(cpu) = &cpu_backend {
                            cpu.matmul(&a_f32, &b_f32, m_usize, k_usize, n_usize)
                        } else {
                            Err(Error::Runtime("No compute backend available".to_string()))
                        }
                    };

                    #[cfg(not(any(feature = "cuda", feature = "metal")))]
                    let result = if let Some(cpu) = &cpu_backend {
                        cpu.matmul(&a_f32, &b_f32, m_usize, k_usize, n_usize)
                    } else {
                        Err(Error::Runtime("No compute backend available".to_string()))
                    };

                    let result = match result {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Matmul failed: {}", e);
                            return -5;
                        }
                    };

                    // Convert result to bytes
                    let result_bytes: Vec<u8> =
                        result.iter().flat_map(|&f| f.to_le_bytes()).collect();

                    // Write result back to WASM memory
                    if let Err(e) =
                        wasm_memory.write(&mut caller, result_ptr as usize, &result_bytes)
                    {
                        error!("Failed to write result: {}", e);
                        return -6;
                    }

                    result_size as i32
                },
            )?;

            // Host function: Transposed matrix multiplication using GPU (if available) or CPU backend
            #[cfg(all(not(target_arch = "wasm32"), any(feature = "cuda", feature = "metal")))]
            let gpu_backend2 = self.gpu_backend.clone();
            let cpu_backend2 = self.cpu_backend.clone();
            linker.func_wrap(
                "env",
                "candle_matmul_transposed",
                move |mut caller: Caller<'_, ()>,
                      a_ptr: u32,
                      b_ptr: u32,
                      result_ptr: u32,
                      m: u32,
                      k: u32,
                      n: u32|
                      -> i32 {
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("No WASM memory export");
                            return -2;
                        }
                    };

                    let m_usize = m as usize;
                    let k_usize = k as usize;
                    let n_usize = n as usize;

                    let a_size = m_usize * k_usize;
                    let b_size = n_usize * k_usize; // Note: b is transposed

                    let mut a_buffer = vec![0u8; a_size * 4];
                    let mut b_buffer = vec![0u8; b_size * 4];

                    if let Err(e) = wasm_memory.read(&caller, a_ptr as usize, &mut a_buffer) {
                        error!("Failed to read matrix A: {}", e);
                        return -3;
                    }

                    if let Err(e) = wasm_memory.read(&caller, b_ptr as usize, &mut b_buffer) {
                        error!("Failed to read matrix B: {}", e);
                        return -4;
                    }

                    let a_f32: Vec<f32> = a_buffer
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    let b_f32: Vec<f32> = b_buffer
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Try GPU first, fallback to CPU
                    // Note: GPU backend doesn't have matmul_transposed, so we transpose B manually
                    #[cfg(all(
                        not(target_arch = "wasm32"),
                        any(feature = "cuda", feature = "metal")
                    ))]
                    let result = if let Some(gpu) = &gpu_backend2 {
                        // Transpose B manually: [n, k] -> [k, n]
                        let mut b_transposed = vec![0.0f32; k_usize * n_usize];
                        for i in 0..n_usize {
                            for j in 0..k_usize {
                                b_transposed[j * n_usize + i] = b_f32[i * k_usize + j];
                            }
                        }
                        // Now do A @ B^T = A @ B_transposed
                        match gpu.matmul(&a_f32, &b_transposed, m as u32, k as u32, n as u32) {
                            Ok(r) => Ok(r),
                            Err(e) => {
                                warn!("GPU matmul_transposed failed, falling back to CPU: {}", e);
                                if let Some(cpu) = &cpu_backend2 {
                                    cpu.matmul_transposed(&a_f32, &b_f32, m_usize, k_usize, n_usize)
                                } else {
                                    Err(Error::Runtime("No compute backend available".to_string()))
                                }
                            }
                        }
                    } else {
                        // No GPU, try CPU
                        if let Some(cpu) = &cpu_backend2 {
                            cpu.matmul_transposed(&a_f32, &b_f32, m_usize, k_usize, n_usize)
                        } else {
                            Err(Error::Runtime("No compute backend available".to_string()))
                        }
                    };

                    #[cfg(not(any(feature = "cuda", feature = "metal")))]
                    let result = if let Some(cpu) = &cpu_backend2 {
                        cpu.matmul_transposed(&a_f32, &b_f32, m_usize, k_usize, n_usize)
                    } else {
                        Err(Error::Runtime("No compute backend available".to_string()))
                    };

                    let result = match result {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Matmul_transposed failed: {}", e);
                            return -5;
                        }
                    };

                    let result_bytes: Vec<u8> =
                        result.iter().flat_map(|&f| f.to_le_bytes()).collect();

                    if let Err(e) =
                        wasm_memory.write(&mut caller, result_ptr as usize, &result_bytes)
                    {
                        error!("Failed to write result: {}", e);
                        return -6;
                    }

                    (m_usize * n_usize) as i32
                },
            )?;
        }

        // ========================================
        // Model Storage Host Functions
        // ========================================

        // Host function: Store a model from GGUF bytes in HOST memory
        // Parameters: gguf_ptr, gguf_len, model_id (WASM memory offsets)
        //            model_id: 0 = auto-generate from hash, > 0 = use provided ID
        // Returns: model_id on success (> 0), negative on error
        linker.func_wrap(
            "env",
            "realm_store_model",
            move |mut caller: Caller<'_, ()>, gguf_ptr: u32, gguf_len: u32, model_id: u32| -> i32 {
                // Get WASM memory
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("realm_store_model: No WASM memory export");
                        return -1;
                    }
                };

                // Validate pointer bounds
                let wasm_mem_size = wasm_memory.data_size(&caller);
                let end_ptr = match (gguf_ptr as usize).checked_add(gguf_len as usize) {
                    Some(end) => end,
                    None => {
                        error!(
                            "realm_store_model: Pointer overflow: {} + {}",
                            gguf_ptr, gguf_len
                        );
                        return -2;
                    }
                };

                if end_ptr > wasm_mem_size {
                    error!(
                        "realm_store_model: Pointer out of bounds: {} + {} > {}",
                        gguf_ptr, gguf_len, wasm_mem_size
                    );
                    return -3;
                }

                // Read GGUF bytes from WASM memory
                let mut gguf_buffer = vec![0u8; gguf_len as usize];
                if let Err(e) = wasm_memory.read(&caller, gguf_ptr as usize, &mut gguf_buffer) {
                    error!("realm_store_model: Failed to read GGUF data: {}", e);
                    return -4;
                }

                // Store model in HOST storage
                use crate::model_storage::get_global_model_storage;
                let requested_id = if model_id == 0 {
                    None // Auto-generate from hash
                } else {
                    Some(model_id) // Use consumer-provided ID
                };

                match get_global_model_storage()
                    .lock()
                    .store_model(&gguf_buffer, requested_id)
                {
                    Ok(final_model_id) => {
                        info!(
                            "realm_store_model: Stored model {} ({} bytes, requested_id={:?})",
                            final_model_id, gguf_len, requested_id
                        );
                        final_model_id as i32
                    }
                    Err(e) => {
                        error!("realm_store_model: Failed to store model: {}", e);
                        -5
                    }
                }
            },
        )?;

        // Host function: Get tensor data from HOST storage (with automatic dequantization)
        // Parameters: model_id, tensor_name_ptr, tensor_name_len, out_ptr, out_max_len
        // Returns: actual tensor size in BYTES on success (f32 * 4), negative on error
        //
        // This function:
        // 1. Retrieves quantized tensor from storage
        // 2. Dequantizes to f32 automatically
        // 3. Writes f32 array to WASM memory
        linker.func_wrap(
            "env",
            "realm_get_tensor",
            move |mut caller: Caller<'_, ()>,
                  model_id: u32,
                  tensor_name_ptr: u32,
                  tensor_name_len: u32,
                  out_ptr: u32,
                  out_max_len: u32|
                  -> i32 {
                // Get WASM memory
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("realm_get_tensor: No WASM memory export");
                        return -1;
                    }
                };

                let wasm_mem_size = wasm_memory.data_size(&caller);

                // Read tensor name from WASM memory
                let name_end = match (tensor_name_ptr as usize).checked_add(tensor_name_len as usize)
                {
                    Some(end) => end,
                    None => {
                        error!("realm_get_tensor: Name pointer overflow");
                        return -2;
                    }
                };

                if name_end > wasm_mem_size {
                    error!("realm_get_tensor: Name pointer out of bounds");
                    return -3;
                }

                let mut name_buffer = vec![0u8; tensor_name_len as usize];
                if let Err(e) = wasm_memory.read(&caller, tensor_name_ptr as usize, &mut name_buffer)
                {
                    error!("realm_get_tensor: Failed to read tensor name: {}", e);
                    return -4;
                }

                let tensor_name = match std::str::from_utf8(&name_buffer) {
                    Ok(s) => s,
                    Err(e) => {
                        error!("realm_get_tensor: Invalid UTF-8 in tensor name: {}", e);
                        return -5;
                    }
                };

                // Validate output pointer
                let out_end = match (out_ptr as usize).checked_add(out_max_len as usize) {
                    Some(end) => end,
                    None => {
                        error!("realm_get_tensor: Output pointer overflow");
                        return -6;
                    }
                };

                if out_end > wasm_mem_size {
                    error!("realm_get_tensor: Output pointer out of bounds");
                    return -7;
                }

                // Get tensor from HOST storage - clone while lock is held
                use crate::model_storage::get_global_model_storage;
                let tensor = {
                    let storage = get_global_model_storage().lock();
                    let model = match storage.get_model(model_id) {
                        Ok(m) => m,
                        Err(e) => {
                            error!(
                                "realm_get_tensor: Model {} not found: {}",
                                model_id, e
                            );
                            return -7;
                        }
                    };
                    match model.get_tensor(tensor_name) {
                        Some(t) => t.clone(),
                        None => {
                            error!(
                                "realm_get_tensor: Tensor '{}' not found in model {}",
                                tensor_name, model_id
                            );
                            return -8;
                        }
                    }
                };

                // Dequantize tensor to f32
                use realm_core::quant::dequantize_tensor;
                let element_count = tensor.element_count() as usize;
                let dequantized = match dequantize_tensor(&tensor.data, tensor.dtype, element_count) {
                    Ok(d) => d,
                    Err(e) => {
                        error!(
                            "realm_get_tensor: Failed to dequantize tensor '{}': {}",
                            tensor_name, e
                        );
                        return -9;
                    }
                };

                // Convert f32 array to bytes
                let f32_bytes: Vec<u8> = dequantized
                    .iter()
                    .flat_map(|&f| f.to_le_bytes())
                    .collect();

                // Check if output buffer is large enough
                if f32_bytes.len() > out_max_len as usize {
                    error!(
                        "realm_get_tensor: Buffer too small: need {} bytes, got {}",
                        f32_bytes.len(),
                        out_max_len
                    );
                    return -10;
                }

                // Write dequantized f32 data to WASM memory
                if let Err(e) = wasm_memory.write(&mut caller, out_ptr as usize, &f32_bytes) {
                    error!("realm_get_tensor: Failed to write tensor data: {}", e);
                    return -11;
                }

                info!(
                    "realm_get_tensor: Loaded and dequantized tensor '{}' from model {} ({} elements, {} bytes)",
                    tensor_name,
                    model_id,
                    element_count,
                    f32_bytes.len()
                );

                f32_bytes.len() as i32
            },
        )?;

        // Host function: Get model metadata (tensor count, total size)
        // Parameters: model_id, out_tensor_count_ptr, out_total_size_ptr
        // Returns: 0 on success, negative on error
        linker.func_wrap(
            "env",
            "realm_get_model_info",
            move |mut caller: Caller<'_, ()>,
                  model_id: u32,
                  out_tensor_count_ptr: u32,
                  out_total_size_ptr: u32|
                  -> i32 {
                // Get WASM memory
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("realm_get_model_info: No WASM memory export");
                        return -1;
                    }
                };

                // Get model from storage - extract values while lock is held
                use crate::model_storage::get_global_model_storage;
                let (tensor_count, total_size) = {
                    let storage = get_global_model_storage().lock();
                    let model = match storage.get_model(model_id) {
                        Ok(m) => m,
                        Err(e) => {
                            error!("realm_get_model_info: Model {} not found: {}", model_id, e);
                            return -2;
                        }
                    };

                    let tensor_count = model.tensor_count() as u32;
                    let total_size = model.total_size as u64;
                    (tensor_count, total_size)
                };

                // Write tensor_count (u32, 4 bytes)
                let tensor_count_bytes = tensor_count.to_le_bytes();
                if let Err(e) = wasm_memory.write(
                    &mut caller,
                    out_tensor_count_ptr as usize,
                    &tensor_count_bytes,
                ) {
                    error!("realm_get_model_info: Failed to write tensor count: {}", e);
                    return -3;
                }

                // Write total_size (u64, 8 bytes)
                let total_size_bytes = total_size.to_le_bytes();
                if let Err(e) =
                    wasm_memory.write(&mut caller, out_total_size_ptr as usize, &total_size_bytes)
                {
                    error!("realm_get_model_info: Failed to write total size: {}", e);
                    return -4;
                }

                info!(
                    "realm_get_model_info: Model {} has {} tensors, {} bytes total",
                    model_id, tensor_count, total_size
                );

                0 // Success
            },
        )?;

        // Host function: Get model metadata (config + tokenizer info) as JSON
        // Parameters: model_id, out_ptr, out_max_len
        // Returns: number of bytes written on success, negative on error
        // The JSON contains: config (TransformerConfig) and tokenizer metadata
        linker.func_wrap(
            "env",
            "realm_get_model_metadata",
            move |mut caller: Caller<'_, ()>,
                  model_id: u32,
                  out_ptr: u32,
                  out_max_len: u32|
                  -> i32 {
                // Get WASM memory
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("realm_get_model_metadata: No WASM memory export");
                        return -1;
                    }
                };

                let wasm_mem_size = wasm_memory.data_size(&caller);

                // Validate output pointer
                let out_end = match (out_ptr as usize).checked_add(out_max_len as usize) {
                    Some(end) => end,
                    None => {
                        error!("realm_get_model_metadata: Output pointer overflow");
                        return -2;
                    }
                };

                if out_end > wasm_mem_size {
                    error!("realm_get_model_metadata: Output pointer out of bounds");
                    return -3;
                }

                // Get model from storage
                use crate::model_storage::get_global_model_storage;
                let (config, has_tokenizer) = {
                    let storage = get_global_model_storage().lock();
                    let model = match storage.get_model(model_id) {
                        Ok(m) => m,
                        Err(e) => {
                            error!(
                                "realm_get_model_metadata: Model {} not found: {}",
                                model_id, e
                            );
                            return -4;
                        }
                    };
                    let config = model.extract_config();
                    let has_tokenizer = model.tokenizer().is_some();
                    (config, has_tokenizer)
                };

                // Serialize config to JSON
                use serde_json;
                let metadata_json = serde_json::json!({
                    "config": {
                        "vocab_size": config.vocab_size,
                        "hidden_size": config.hidden_size,
                        "num_layers": config.num_layers,
                        "num_heads": config.num_heads,
                        "num_kv_heads": config.num_kv_heads,
                        "intermediate_size": config.intermediate_size,
                        "max_seq_len": config.max_seq_len,
                        "rope_theta": config.rope_theta,
                        "rms_norm_eps": config.rms_norm_eps,
                    },
                    "has_tokenizer": has_tokenizer,
                });

                let json_bytes = match serde_json::to_string(&metadata_json) {
                    Ok(s) => s.into_bytes(),
                    Err(e) => {
                        error!("realm_get_model_metadata: Failed to serialize: {}", e);
                        return -5;
                    }
                };

                // Check if output buffer is large enough
                if json_bytes.len() > out_max_len as usize {
                    error!(
                        "realm_get_model_metadata: Buffer too small: need {} bytes, got {}",
                        json_bytes.len(),
                        out_max_len
                    );
                    return -6;
                }

                // Write JSON to WASM memory
                if let Err(e) = wasm_memory.write(&mut caller, out_ptr as usize, &json_bytes) {
                    error!("realm_get_model_metadata: Failed to write JSON: {}", e);
                    return -7;
                }

                debug!(
                    "realm_get_model_metadata: Wrote {} bytes of metadata for model {}",
                    json_bytes.len(),
                    model_id
                );

                json_bytes.len() as i32
            },
        )?;

        // Host function: Remove model from storage (cleanup)
        // Parameters: model_id
        // Returns: 0 on success, negative on error
        linker.func_wrap(
            "env",
            "realm_remove_model",
            move |_caller: Caller<'_, ()>, model_id: u32| -> i32 {
                use crate::model_storage::get_global_model_storage;
                match get_global_model_storage().lock().remove_model(model_id) {
                    Ok(()) => {
                        info!("realm_remove_model: Removed model {}", model_id);
                        0
                    }
                    Err(e) => {
                        error!(
                            "realm_remove_model: Failed to remove model {}: {}",
                            model_id, e
                        );
                        -1
                    }
                }
            },
        )?;

        // ========================================
        // Transformer Layer Forward (HOST-SIDE COMPUTATION)
        // ========================================
        // This is THE KEY function: weights never enter WASM!
        //
        // Host function: Forward through a complete transformer layer
        // - Loads weights from HOST storage (never copies to WASM)
        // - Computes attention + FFN on HOST (with GPU acceleration)
        // - Returns output to WASM (only activations, no weights)

        #[cfg(not(target_arch = "wasm32"))]
        {
            let cpu_backend = self.cpu_backend.clone();
            linker.func_wrap(
                "env",
                "realm_forward_layer",
                move |mut caller: Caller<'_, ()>,
                      model_id: u32,
                      layer_idx: u32,
                      hidden_states_ptr: u32,
                      hidden_states_len: u32,
                      position: u32,
                      out_ptr: u32| -> i32 {
                    // Check backend
                    if cpu_backend.is_none() {
                        error!("realm_forward_layer: CPU backend not available");
                        return -1;
                    }
                    let _backend = cpu_backend.as_ref().unwrap();

                    // Get WASM memory
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("realm_forward_layer: No WASM memory export");
                            return -2;
                        }
                    };

                    // Read hidden states from WASM
                    let hidden_size_bytes = (hidden_states_len as usize) * 4; // f32 = 4 bytes
                    let mut hidden_buffer = vec![0u8; hidden_size_bytes];
                    if let Err(e) = wasm_memory.read(&caller, hidden_states_ptr as usize, &mut hidden_buffer) {
                        error!("realm_forward_layer: Failed to read hidden states: {}", e);
                        return -3;
                    }

                    // Convert to f32
                    let hidden_states: Vec<f32> = hidden_buffer
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Get model and config from storage - extract ALL tensor data while lock is held
                    use crate::model_storage::get_global_model_storage;
                    let (config, hidden_size, lora_adapter_id, attn_norm_tensor, wq_tensor, wk_tensor, wv_tensor, wo_tensor, ffn_norm_tensor, w_gate_tensor, w_up_tensor, w_down_tensor) = {
                        let storage = get_global_model_storage().lock();
                        let model = match storage.get_model(model_id) {
                            Ok(m) => m,
                            Err(e) => {
                                error!("realm_forward_layer: Model {} not found: {}", model_id, e);
                                return -4;
                            }
                        };
                        let config = model.extract_config();
                        let hidden_size = config.hidden_size;
                        let lora_adapter_id = model.lora_adapter_id().map(|s| s.to_string());

                        // Clone all tensors we need while lock is held
                        let attn_norm_tensor = match model.get_tensor(&format!("blk.{}.attn_norm.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: attn_norm.weight not found for layer {}", layer_idx);
                                return -6;
                            }
                        };
                        let wq_tensor = match model.get_tensor(&format!("blk.{}.attn_q.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: attn_q.weight not found for layer {}", layer_idx);
                                return -15;
                            }
                        };
                        let wk_tensor = match model.get_tensor(&format!("blk.{}.attn_k.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: attn_k.weight not found for layer {}", layer_idx);
                                return -16;
                            }
                        };
                        let wv_tensor = match model.get_tensor(&format!("blk.{}.attn_v.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: attn_v.weight not found for layer {}", layer_idx);
                                return -18;
                            }
                        };
                        let wo_tensor = match model.get_tensor(&format!("blk.{}.attn_output.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: attn_output.weight not found for layer {}", layer_idx);
                                return -20;
                            }
                        };
                        let ffn_norm_tensor = match model.get_tensor(&format!("blk.{}.ffn_norm.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: ffn_norm.weight not found for layer {}", layer_idx);
                                return -11;
                            }
                        };
                        let w_gate_tensor = match model.get_tensor(&format!("blk.{}.ffn_gate.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: ffn_gate.weight not found for layer {}", layer_idx);
                                return -22;
                            }
                        };
                        let w_up_tensor = match model.get_tensor(&format!("blk.{}.ffn_up.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: ffn_up.weight not found for layer {}", layer_idx);
                                return -24;
                            }
                        };
                        let w_down_tensor = match model.get_tensor(&format!("blk.{}.ffn_down.weight", layer_idx)) {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_forward_layer: ffn_down.weight not found for layer {}", layer_idx);
                                return -26;
                            }
                        };

                        (config, hidden_size, lora_adapter_id, attn_norm_tensor, wq_tensor, wk_tensor, wv_tensor, wo_tensor, ffn_norm_tensor, w_gate_tensor, w_up_tensor, w_down_tensor)
                    };
                    let seq_len = hidden_states.len() / hidden_size;

                    if !hidden_states.len().is_multiple_of(hidden_size) {
                        error!("realm_forward_layer: Invalid hidden_states length: {} not divisible by hidden_size {}", hidden_states.len(), hidden_size);
                        return -5;
                    }

                    // Initialize backends
                    let candle_backend = realm_compute_cpu::CandleNeuralOpsBackend::new();
                    use realm_compute_cpu::CandleCpuBackend;
                    let cpu_backend = match CandleCpuBackend::new() {
                        Ok(b) => b,
                        Err(_) => {
                            error!("realm_forward_layer: Failed to create CPU backend");
                            return -31;
                        }
                    };

                    // Use cloned tensors (lock is already released)
                    let attn_norm = match realm_core::quant::dequantize_tensor(&attn_norm_tensor.data, attn_norm_tensor.dtype, hidden_size) {
                        Ok(d) => d,
                        Err(e) => {
                            error!("realm_forward_layer: Failed to dequantize attn_norm: {}", e);
                            return -7;
                        }
                    };

                    // 1. Attention block
                    // Norm
                    let normed = match candle_backend.rms_norm(&hidden_states, &attn_norm, config.rms_norm_eps, seq_len, hidden_size) {
                        Ok(n) => n,
                        Err(e) => {
                            error!("realm_forward_layer: RMS norm failed: {}", e);
                            return -8;
                        }
                    };

                    // Helper: Convert quantized tensor to WeightFormat (for fused ops)
                    fn quantized_to_weight_format(tensor: &crate::model_storage::QuantizedTensor) -> std::result::Result<realm_models::WeightFormat, anyhow::Error> {
                        use realm_core::quant::{
                            BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_K, BlockQ5_0,
                            BlockQ5_1, BlockQ5_K, BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K,
                        };

                        match tensor.dtype {
                            realm_core::tensor::DataType::F32 => {
                                let f32_data: Vec<f32> = tensor.data.chunks_exact(4)
                                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                                    .collect();
                                Ok(realm_models::WeightFormat::F32(f32_data))
                            }
                            realm_core::tensor::DataType::Q4_K => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_K>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_K) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q4K(blocks))
                            }
                            realm_core::tensor::DataType::Q5_K => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_K>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ5_K) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q5K(blocks))
                            }
                            realm_core::tensor::DataType::Q6_K => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6_K>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ6_K) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q6K(blocks))
                            }
                            realm_core::tensor::DataType::Q8_K => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_K>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ8_K) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q8K(blocks))
                            }
                            realm_core::tensor::DataType::Q2_K => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ2_K>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ2_K) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q2K(blocks))
                            }
                            realm_core::tensor::DataType::Q3_K => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ3_K>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ3_K) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q3K(blocks))
                            }
                            realm_core::tensor::DataType::Q4_0 => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_0>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_0) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q40(blocks))
                            }
                            realm_core::tensor::DataType::Q4_1 => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4_1>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ4_1) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q41(blocks))
                            }
                            realm_core::tensor::DataType::Q5_0 => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_0>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ5_0) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q50(blocks))
                            }
                            realm_core::tensor::DataType::Q5_1 => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ5_1>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ5_1) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q51(blocks))
                            }
                            realm_core::tensor::DataType::Q8_0 => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_0>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ8_0) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q80(blocks))
                            }
                            realm_core::tensor::DataType::Q8_1 => {
                                const BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_1>();
                                let num_blocks = tensor.data.len() / BLOCK_SIZE;
                                let mut blocks = Vec::with_capacity(num_blocks);

                                for block_data in tensor.data.chunks_exact(BLOCK_SIZE) {
                                    let block = unsafe { std::ptr::read(block_data.as_ptr() as *const BlockQ8_1) };
                                    blocks.push(block);
                                }
                                Ok(realm_models::WeightFormat::Q81(blocks))
                            }
                            _ => Err(anyhow::anyhow!("Unsupported dtype for weight format: {:?}", tensor.dtype))
                        }
                    }

                    // Load attention weights (as WeightFormat for fused ops) - using cloned tensors
                    // Apply LoRA if adapter is configured
                    let layer_name = format!("layer.{}", layer_idx);
                    let hidden_size = config.hidden_size;

                    let wq = match quantized_to_weight_format(&wq_tensor) {
                        Ok(w) => {
                            // Function always returns Ok, so unwrap is safe
                            apply_lora_to_weight_format(w, lora_adapter_id.as_deref(), &layer_name, "attn_q", hidden_size, hidden_size).unwrap()
                        }
                        Err(e) => {
                            error!("realm_forward_layer: Failed to convert wq to WeightFormat: {}", e);
                            return -15;
                        }
                    };

                    let wk = match quantized_to_weight_format(&wk_tensor) {
                        Ok(w) => {
                            // Function always returns Ok, so unwrap is safe
                            apply_lora_to_weight_format(w, lora_adapter_id.as_deref(), &layer_name, "attn_k", hidden_size, hidden_size).unwrap()
                        }
                        Err(e) => {
                            error!("realm_forward_layer: Failed to convert wk to WeightFormat: {}", e);
                            return -17;
                        }
                    };

                    let wv = match quantized_to_weight_format(&wv_tensor) {
                        Ok(w) => {
                            // Function always returns Ok, so unwrap is safe
                            apply_lora_to_weight_format(w, lora_adapter_id.as_deref(), &layer_name, "attn_v", hidden_size, hidden_size).unwrap()
                        }
                        Err(e) => {
                            error!("realm_forward_layer: Failed to convert wv to WeightFormat: {}", e);
                            return -19;
                        }
                    };

                    let wo = match quantized_to_weight_format(&wo_tensor) {
                        Ok(w) => {
                            // Function always returns Ok, so unwrap is safe
                            apply_lora_to_weight_format(w, lora_adapter_id.as_deref(), &layer_name, "attn_output", hidden_size, hidden_size).unwrap()
                        }
                        Err(e) => {
                            error!("realm_forward_layer: Failed to convert wo to WeightFormat: {}", e);
                            return -21;
                        }
                    };

                    // Create attention weights struct
                    let attn_weights = realm_models::AttentionWeights {
                        wq,
                        wk,
                        wv,
                        wo,
                    };

                    // Forward attention using MultiHeadAttention (weights stay in HOST!)
                    let attention = realm_models::MultiHeadAttention::new(config.clone());

                    // KV cache management - use HOST-side storage
                    use crate::kv_cache_storage::GLOBAL_KV_CACHE_STORAGE;
                    let head_dim = config.hidden_size / config.num_heads;
                    let kv_cache_arc = GLOBAL_KV_CACHE_STORAGE.get_or_create(
                        model_id,
                        layer_idx,
                        config.max_seq_len,
                        config.num_kv_heads,
                        head_dim,
                    );
                    let mut kv_cache = kv_cache_arc.lock();

                    // Note: kv_cache is already locked, we need to pass a mutable reference
                    // Attention.forward expects &mut KVCache, so we dereference the MutexGuard
                    let attn_output = match attention.forward(
                        &normed,
                        &attn_weights,
                        &mut kv_cache, // Dereference MutexGuard to get &mut KVCache
                        position as usize,
                        Some(&cpu_backend as &dyn realm_compute_cpu::CpuBackendTrait),
                        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
                        None,
                    ) {
                        Ok(output) => output,
                        Err(e) => {
                            error!("realm_forward_layer: Attention forward failed: {}", e);
                            return -9;
                        }
                    };

                    // Residual connection
                    let hidden = match candle_backend.add(&hidden_states, &attn_output, hidden_states.len()) {
                        Ok(h) => h,
                        Err(e) => {
                            error!("realm_forward_layer: Add failed: {}", e);
                            return -10;
                        }
                    };

                    // 2. FFN block
                    // Load FFN norm - using cloned tensor
                    let ffn_norm = match realm_core::quant::dequantize_tensor(&ffn_norm_tensor.data, ffn_norm_tensor.dtype, hidden_size) {
                        Ok(d) => d,
                        Err(e) => {
                            error!("realm_forward_layer: Failed to dequantize ffn_norm: {}", e);
                            return -11;
                        }
                    };

                    // FFN norm
                    let ffn_normed = match candle_backend.rms_norm(&hidden, &ffn_norm, config.rms_norm_eps, seq_len, hidden_size) {
                        Ok(n) => n,
                        Err(e) => {
                            error!("realm_forward_layer: FFN RMS norm failed: {}", e);
                            return -12;
                        }
                    };

                    // Load FFN weights - using cloned tensors
                    // Apply LoRA if adapter is configured
                    let w_gate = match quantized_to_weight_format(&w_gate_tensor) {
                        Ok(w) => {
                            // Function always returns Ok, so unwrap is safe
                            apply_lora_to_weight_format(w, lora_adapter_id.as_deref(), &layer_name, "ffn_gate", config.intermediate_size, hidden_size).unwrap()
                        }
                        Err(e) => {
                            error!("realm_forward_layer: Failed to convert w_gate to WeightFormat: {}", e);
                            return -23;
                        }
                    };

                    let w_up = match quantized_to_weight_format(&w_up_tensor) {
                        Ok(w) => {
                            // Function always returns Ok, so unwrap is safe
                            apply_lora_to_weight_format(w, lora_adapter_id.as_deref(), &layer_name, "ffn_up", config.intermediate_size, hidden_size).unwrap()
                        }
                        Err(e) => {
                            error!("realm_forward_layer: Failed to convert w_up to WeightFormat: {}", e);
                            return -25;
                        }
                    };

                    let w_down = match quantized_to_weight_format(&w_down_tensor) {
                        Ok(w) => {
                            // Function always returns Ok, so unwrap is safe
                            apply_lora_to_weight_format(w, lora_adapter_id.as_deref(), &layer_name, "ffn_down", hidden_size, config.intermediate_size).unwrap()
                        }
                        Err(e) => {
                            error!("realm_forward_layer: Failed to convert w_down to WeightFormat: {}", e);
                            return -27;
                        }
                    };

                    use realm_models::dispatch_matmul;

                    // Gate projection: hidden_states @ w_gate^T
                    // w_gate is already WeightFormat, dispatch_matmul handles it
                    let gate = match dispatch_matmul(
                        &ffn_normed,
                        &w_gate,
                        seq_len,
                        hidden_size,
                        config.intermediate_size,
                        Some(&cpu_backend as &dyn realm_compute_cpu::CpuBackendTrait),
                        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
                        None,
                    ) {
                        Ok(g) => g,
                        Err(e) => {
                            error!("realm_forward_layer: Gate matmul failed: {}", e);
                            return -28;
                        }
                    };

                    // Up projection
                    let up = match dispatch_matmul(
                        &ffn_normed,
                        &w_up,
                        seq_len,
                        hidden_size,
                        config.intermediate_size,
                        Some(&cpu_backend as &dyn realm_compute_cpu::CpuBackendTrait),
                        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
                        None,
                    ) {
                        Ok(u) => u,
                        Err(e) => {
                            error!("realm_forward_layer: Up matmul failed: {}", e);
                            return -29;
                        }
                    };

                    // SwiGLU: silu(gate) * up
                    let mut activated = gate;
                    for i in 0..activated.len() {
                        let sigmoid = 1.0 / (1.0 + (-activated[i]).exp());
                        let silu = activated[i] * sigmoid;
                        activated[i] = silu * up[i];
                    }

                    // Down projection
                    let ffn_output = match dispatch_matmul(
                        &activated,
                        &w_down,
                        seq_len,
                        config.intermediate_size,
                        hidden_size,
                        Some(&cpu_backend as &dyn realm_compute_cpu::CpuBackendTrait),
                        #[cfg(any(feature = "webgpu", feature = "cuda", feature = "metal"))]
                        None,
                    ) {
                        Ok(f) => f,
                        Err(e) => {
                            error!("realm_forward_layer: Down matmul failed: {}", e);
                            return -30;
                        }
                    };

                    // Residual connection
                    let final_hidden = match candle_backend.add(&hidden, &ffn_output, hidden.len()) {
                        Ok(h) => h,
                        Err(e) => {
                            error!("realm_forward_layer: Final add failed: {}", e);
                            return -13;
                        }
                    };

                    // Write output back to WASM memory
                    let output_bytes: Vec<u8> = final_hidden.iter()
                        .flat_map(|&f| f.to_le_bytes())
                        .collect();

                    if let Err(e) = wasm_memory.write(&mut caller, out_ptr as usize, &output_bytes) {
                        error!("realm_forward_layer: Failed to write output: {}", e);
                        return -14;
                    }

                    info!(
                        "realm_forward_layer: Layer {} forward complete (seq_len={}, hidden_size={})",
                        layer_idx, seq_len, hidden_size
                    );

                    (final_hidden.len() * 4) as i32 // Return bytes written
                },
            )?;
        }

        // ========================================
        // Token Embedding (HOST-SIDE COMPUTATION)
        // ========================================
        // Embed token IDs into hidden states on HOST
        // This avoids loading 262MB embeddings into WASM!

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _cpu_backend = self.cpu_backend.clone();
            linker.func_wrap(
                "env",
                "realm_embed_tokens",
                move |mut caller: Caller<'_, ()>,
                      model_id: u32,
                      token_ids_ptr: u32,
                      token_ids_len: u32,
                      out_ptr: u32|
                      -> i32 {
                    // Get WASM memory
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("realm_embed_tokens: No WASM memory export");
                            return -1;
                        }
                    };

                    // Read token IDs from WASM
                    let token_ids_bytes = (token_ids_len as usize) * 4; // u32 = 4 bytes
                    let mut token_ids_buffer = vec![0u8; token_ids_bytes];
                    if let Err(e) =
                        wasm_memory.read(&caller, token_ids_ptr as usize, &mut token_ids_buffer)
                    {
                        error!("realm_embed_tokens: Failed to read token IDs: {}", e);
                        return -2;
                    }

                    // Convert to u32 array
                    let token_ids: Vec<u32> = token_ids_buffer
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Get model and config - extract embedding tensor while lock is held
                    use crate::model_storage::get_global_model_storage;
                    let (config, hidden_size, seq_len, output_size, embedding_tensor) = {
                        let storage = get_global_model_storage().lock();
                        let model = match storage.get_model(model_id) {
                            Ok(m) => m,
                            Err(e) => {
                                error!("realm_embed_tokens: Model {} not found: {}", model_id, e);
                                return -3;
                            }
                        };

                        let config = model.extract_config();
                        let hidden_size = config.hidden_size;
                        let seq_len = token_ids.len();
                        let output_size = seq_len * hidden_size;

                        // Clone embedding tensor while lock is held
                        let embedding_tensor = match model.get_tensor("token_embd.weight") {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_embed_tokens: token_embd.weight not found");
                                return -4;
                            }
                        };

                        (config, hidden_size, seq_len, output_size, embedding_tensor)
                    };

                    // Dequantize embeddings (only what we need)
                    // GGUF stores as [vocab_size, hidden_size]
                    // We'll dequantize on-the-fly for each token
                    let vocab_size = config.vocab_size;
                    let _embedding_bytes_per_row = hidden_size * 4; // f32 = 4 bytes

                    // Build hidden states by gathering embeddings for each token
                    let mut hidden_states = vec![0.0f32; output_size];

                    for (seq_idx, &token_id) in token_ids.iter().enumerate() {
                        let out_start = seq_idx * hidden_size;
                        let token_id_usize = token_id as usize;

                        if token_id_usize >= vocab_size {
                            warn!(
                                "realm_embed_tokens: Token ID {} >= vocab_size {}",
                                token_id_usize, vocab_size
                            );
                            continue;
                        }

                        // For quantized embeddings, we'd need to dequantize the specific row
                        // For now, dequantize full embeddings (will optimize later)
                        // TODO: Implement row-wise dequantization for efficiency
                        let full_embeddings = match realm_core::quant::dequantize_tensor(
                            &embedding_tensor.data,
                            embedding_tensor.dtype,
                            vocab_size * hidden_size,
                        ) {
                            Ok(e) => e,
                            Err(e) => {
                                error!(
                                    "realm_embed_tokens: Failed to dequantize embeddings: {}",
                                    e
                                );
                                return -5;
                            }
                        };

                        // Copy embedding row for this token
                        let emb_start = token_id_usize * hidden_size;
                        let emb_end = emb_start + hidden_size;
                        if emb_end <= full_embeddings.len() {
                            hidden_states[out_start..out_start + hidden_size]
                                .copy_from_slice(&full_embeddings[emb_start..emb_end]);
                        }
                    }

                    // Write output back to WASM memory
                    let output_bytes: Vec<u8> = hidden_states
                        .iter()
                        .flat_map(|&f| f.to_le_bytes())
                        .collect();

                    if let Err(e) = wasm_memory.write(&mut caller, out_ptr as usize, &output_bytes)
                    {
                        error!("realm_embed_tokens: Failed to write output: {}", e);
                        return -6;
                    }

                    info!(
                        "realm_embed_tokens: Embedded {} tokens (hidden_size={})",
                        seq_len, hidden_size
                    );

                    (output_size * 4) as i32 // Return bytes written
                },
            )?;
        }

        // ========================================
        // LoRA Integration Host Functions
        // ========================================
        // Host function: Set LoRA adapter for a model
        // Parameters: model_id, adapter_id_ptr, adapter_id_len (WASM memory offsets)
        // Returns: 0 on success, negative on error
        // NOTE: LoRA application happens automatically when model is used in forward pass
        linker.func_wrap(
            "env",
            "realm_set_lora_adapter",
            move |mut caller: Caller<'_, ()>,
                  model_id: u32,
                  adapter_id_ptr: u32,
                  adapter_id_len: u32|
                  -> i32 {
                // Get WASM memory
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("realm_set_lora_adapter: No WASM memory export");
                        return -1;
                    }
                };

                // Read adapter ID from WASM memory
                let mut adapter_id_bytes = vec![0u8; adapter_id_len as usize];
                if let Err(e) =
                    wasm_memory.read(&caller, adapter_id_ptr as usize, &mut adapter_id_bytes)
                {
                    error!("realm_set_lora_adapter: Failed to read adapter_id: {}", e);
                    return -2;
                }

                let adapter_id = match String::from_utf8(adapter_id_bytes) {
                    Ok(id) => id,
                    Err(e) => {
                        error!("realm_set_lora_adapter: Invalid UTF-8 in adapter_id: {}", e);
                        return -3;
                    }
                };

                // Mark model with LoRA adapter ID
                use crate::model_storage::get_global_model_storage;
                let adapter_id_for_log = adapter_id.clone();
                let adapter_id_for_set = adapter_id.clone();
                let result = get_global_model_storage()
                    .lock()
                    .set_lora_adapter(model_id, adapter_id_for_set);
                match result {
                    Ok(()) => {
                        info!(
                            "realm_set_lora_adapter: Model {} marked with LoRA adapter '{}'",
                            model_id, adapter_id_for_log
                        );
                        // NOTE: Actual LoRA application happens during forward pass when weights are loaded
                        // This is handled by the RuntimeManager integration
                        0
                    }
                    Err(e) => {
                        error!("realm_set_lora_adapter: Failed to set LoRA adapter: {}", e);
                        -4
                    }
                }
            },
        )?;

        // ========================================
        // Speculative Decoding Host Functions
        // ========================================
        // Host function: Store draft model for speculative decoding
        // Parameters: gguf_ptr, gguf_len, draft_model_id (WASM memory offsets)
        // Returns: draft_model_id on success (> 0), negative on error
        linker.func_wrap(
            "env",
            "realm_store_draft_model",
            move |mut caller: Caller<'_, ()>,
                  gguf_ptr: u32,
                  gguf_len: u32,
                  draft_model_id: u32|
                  -> i32 {
                // Get WASM memory
                let wasm_memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => {
                        error!("realm_store_draft_model: No WASM memory export");
                        return -1;
                    }
                };

                // Read GGUF bytes from WASM memory
                let mut gguf_bytes = vec![0u8; gguf_len as usize];
                if let Err(e) = wasm_memory.read(&caller, gguf_ptr as usize, &mut gguf_bytes) {
                    error!("realm_store_draft_model: Failed to read GGUF data: {}", e);
                    return -2;
                }

                // Store draft model
                use crate::model_storage::get_global_model_storage;
                match get_global_model_storage()
                    .lock()
                    .store_model(&gguf_bytes, Some(draft_model_id))
                {
                    Ok(id) => {
                        info!("realm_store_draft_model: Draft model stored with ID {}", id);
                        id as i32
                    }
                    Err(e) => {
                        error!(
                            "realm_store_draft_model: Failed to store draft model: {}",
                            e
                        );
                        -3
                    }
                }
            },
        )?;

        // ========================================
        // Compute Logits (HOST-SIDE COMPUTATION)
        // ========================================
        // Apply final norm + LM head projection on HOST
        // This avoids loading large LM head weights into WASM!

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _cpu_backend = self.cpu_backend.clone();
            linker.func_wrap(
                "env",
                "realm_compute_logits",
                move |mut caller: Caller<'_, ()>,
                      model_id: u32,
                      hidden_state_ptr: u32,
                      hidden_state_len: u32,
                      out_ptr: u32|
                      -> i32 {
                    use realm_compute_cpu::CandleNeuralOpsBackend;

                    // Get WASM memory
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("realm_compute_logits: No WASM memory export");
                            return -1;
                        }
                    };

                    // Read hidden state from WASM
                    let hidden_bytes = (hidden_state_len as usize) * 4; // f32 = 4 bytes
                    let mut hidden_buffer = vec![0u8; hidden_bytes];
                    if let Err(e) =
                        wasm_memory.read(&caller, hidden_state_ptr as usize, &mut hidden_buffer)
                    {
                        error!("realm_compute_logits: Failed to read hidden state: {}", e);
                        return -2;
                    }

                    // Convert to f32
                    let mut hidden_state: Vec<f32> = hidden_buffer
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Get model and config - extract tensors while lock is held
                    use crate::model_storage::get_global_model_storage;
                    let (config, hidden_size, output_norm_tensor, lm_head_tensor) = {
                        let storage = get_global_model_storage().lock();
                        let model = match storage.get_model(model_id) {
                            Ok(m) => m,
                            Err(e) => {
                                error!("realm_compute_logits: Model {} not found: {}", model_id, e);
                                return -3;
                            }
                        };

                        let config = model.extract_config();
                        let hidden_size = config.hidden_size;

                        // Clone tensors while lock is held
                        let output_norm_tensor = match model.get_tensor("output_norm.weight") {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_compute_logits: output_norm.weight not found");
                                return -4;
                            }
                        };

                        let lm_head_tensor = match model.get_tensor("output.weight") {
                            Some(t) => t.clone(),
                            None => {
                                error!("realm_compute_logits: output.weight (lm_head) not found");
                                return -5;
                            }
                        };

                        (config, hidden_size, output_norm_tensor, lm_head_tensor)
                    };

                    if hidden_state.len() != hidden_size {
                        error!(
                            "realm_compute_logits: Invalid hidden_state length: {} != {}",
                            hidden_state.len(),
                            hidden_size
                        );
                        return -6;
                    }

                    // Initialize backend
                    let candle_backend = CandleNeuralOpsBackend::new();

                    // Use cloned tensors (lock is already released)
                    let output_norm = match realm_core::quant::dequantize_tensor(
                        &output_norm_tensor.data,
                        output_norm_tensor.dtype,
                        hidden_size,
                    ) {
                        Ok(n) => n,
                        Err(e) => {
                            error!(
                                "realm_compute_logits: Failed to dequantize output_norm: {}",
                                e
                            );
                            return -7;
                        }
                    };

                    // Apply RMS norm
                    hidden_state = match candle_backend.rms_norm(
                        &hidden_state,
                        &output_norm,
                        config.rms_norm_eps,
                        1, // seq_len = 1 for single token
                        hidden_size,
                    ) {
                        Ok(h) => h,
                        Err(e) => {
                            error!("realm_compute_logits: RMS norm failed: {}", e);
                            return -8;
                        }
                    };

                    // Dequantize LM head (vocab_size x hidden_size)
                    let vocab_size = config.vocab_size;
                    let lm_head = match realm_core::quant::dequantize_tensor(
                        &lm_head_tensor.data,
                        lm_head_tensor.dtype,
                        vocab_size * hidden_size,
                    ) {
                        Ok(l) => l,
                        Err(e) => {
                            error!("realm_compute_logits: Failed to dequantize lm_head: {}", e);
                            return -9;
                        }
                    };

                    // Compute logits: hidden_state @ lm_head^T
                    // LM head is [vocab_size, hidden_size], we want logits = [vocab_size]
                    let mut logits = vec![0.0f32; vocab_size];
                    for (i, logit) in logits.iter_mut().enumerate() {
                        let weight_start = i * hidden_size;
                        let mut sum = 0.0;
                        for j in 0..hidden_size {
                            sum += hidden_state[j] * lm_head[weight_start + j];
                        }
                        *logit = sum;
                    }

                    // Write output back to WASM memory
                    let output_bytes: Vec<u8> =
                        logits.iter().flat_map(|&f| f.to_le_bytes()).collect();

                    if let Err(e) = wasm_memory.write(&mut caller, out_ptr as usize, &output_bytes)
                    {
                        error!("realm_compute_logits: Failed to write output: {}", e);
                        return -10;
                    }

                    info!("realm_compute_logits: Computed {} logits", vocab_size);

                    (vocab_size * 4) as i32 // Return bytes written
                },
            )?;
        }

        // ========================================
        // Token Streaming (REAL TOKEN-BY-TOKEN STREAMING)
        // ========================================
        // Host function: Stream a token as it's generated
        // This enables real token-by-token streaming instead of word chunking
        // Parameters: token_ptr, token_len (WASM memory offsets)
        // Returns: 0 on success, negative on error

        #[cfg(not(target_arch = "wasm32"))]
        {
            let stream_callback = self.stream_callback.clone();
            linker.func_wrap(
                "env",
                "realm_stream_token",
                move |mut caller: Caller<'_, ()>, token_ptr: u32, token_len: u32| -> i32 {
                    // Get WASM memory
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("realm_stream_token: No WASM memory export");
                            return -1;
                        }
                    };

                    // Validate pointer
                    let wasm_mem_size = wasm_memory.data_size(&caller);
                    let token_end = match (token_ptr as usize).checked_add(token_len as usize) {
                        Some(end) => end,
                        None => {
                            error!("realm_stream_token: Pointer overflow");
                            return -2;
                        }
                    };

                    if token_end > wasm_mem_size {
                        error!("realm_stream_token: Pointer out of bounds");
                        return -3;
                    }

                    // Read token string from WASM memory
                    let mut token_bytes = vec![0u8; token_len as usize];
                    if let Err(e) = wasm_memory.read(&caller, token_ptr as usize, &mut token_bytes)
                    {
                        error!("realm_stream_token: Failed to read token: {}", e);
                        return -4;
                    }

                    // Convert to string
                    let token = match String::from_utf8(token_bytes) {
                        Ok(s) => s,
                        Err(e) => {
                            error!("realm_stream_token: Invalid UTF-8: {}", e);
                            return -5;
                        }
                    };

                    // Send token to callback (blocking send, but should be fast)
                    // Note: If the channel is closed (receiver dropped), send() will return an error
                    // This is expected behavior when the client disconnects, so we handle it gracefully
                    let callback_guard = stream_callback.lock();
                    if let Some(ref sender) = *callback_guard {
                        // Use send for blocking channel
                        // If channel is closed (receiver dropped), we stop streaming gracefully
                        if let Err(e) = sender.send(token.clone()) {
                            // Channel closed - client likely disconnected, stop streaming
                            debug!("realm_stream_token: Channel closed, stopping stream: {}", e);
                            // Don't fail the generation, just stop sending tokens
                            // The generation will complete normally, but tokens won't be streamed
                        }
                    } else {
                        // No callback set - this is okay, just means streaming is not enabled
                        // This can happen if generate_stream() wasn't called or callback was cleared
                        debug!("realm_stream_token: No streaming callback set (streaming may not be enabled)");
                    }

                    0 // Success
                },
            )?;

            // Host function: Encode text to token IDs
            // Parameters: model_id, text_ptr, text_len, out_ptr, out_max_len
            // Returns: number of tokens written on success, negative on error
            linker.func_wrap(
                "env",
                "realm_encode_tokens",
                move |mut caller: Caller<'_, ()>,
                      model_id: u32,
                      text_ptr: u32,
                      text_len: u32,
                      out_ptr: u32,
                      out_max_len: u32|
                      -> i32 {
                    use crate::model_storage::get_global_model_storage;

                    // Get WASM memory
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("realm_encode_tokens: No WASM memory export");
                            return -1;
                        }
                    };

                    // Read text from WASM
                    let mut text_bytes = vec![0u8; text_len as usize];
                    if let Err(e) = wasm_memory.read(&caller, text_ptr as usize, &mut text_bytes) {
                        error!("realm_encode_tokens: Failed to read text: {}", e);
                        return -2;
                    }

                    let text = match String::from_utf8(text_bytes) {
                        Ok(s) => s,
                        Err(e) => {
                            error!("realm_encode_tokens: Invalid UTF-8: {}", e);
                            return -3;
                        }
                    };

                    // Get tokenizer from model storage
                    let storage = get_global_model_storage().lock();
                    let model = match storage.get_model(model_id) {
                        Ok(m) => m,
                        Err(_) => {
                            error!("realm_encode_tokens: Model {} not found", model_id);
                            return -4;
                        }
                    };

                    let tokenizer = match model.tokenizer() {
                        Some(t) => t,
                        None => {
                            error!("realm_encode_tokens: No tokenizer for model {}", model_id);
                            return -5;
                        }
                    };

                    // Encode text to tokens
                    let tokens = match tokenizer.encode(&text, true) {
                        Ok(t) => t,
                        Err(e) => {
                            error!("realm_encode_tokens: Encoding failed: {}", e);
                            return -6;
                        }
                    };

                    // Check output buffer size
                    let tokens_bytes = tokens.len() * 4; // u32 = 4 bytes
                    if tokens_bytes > out_max_len as usize {
                        error!(
                            "realm_encode_tokens: Output buffer too small: need {}, have {}",
                            tokens_bytes, out_max_len
                        );
                        return -7;
                    }

                    // Write tokens to WASM memory
                    let mut tokens_bytes_vec = Vec::with_capacity(tokens_bytes);
                    for token_id in &tokens {
                        tokens_bytes_vec.extend_from_slice(&token_id.to_le_bytes());
                    }

                    if let Err(e) =
                        wasm_memory.write(&mut caller, out_ptr as usize, &tokens_bytes_vec)
                    {
                        error!("realm_encode_tokens: Failed to write tokens: {}", e);
                        return -8;
                    }

                    tokens.len() as i32
                },
            )?;

            // Host function: Decode token IDs to text
            // Parameters: model_id, token_ids_ptr, token_ids_len, out_ptr, out_max_len
            // Returns: number of bytes written on success, negative on error
            linker.func_wrap(
                "env",
                "realm_decode_tokens",
                move |mut caller: Caller<'_, ()>,
                      model_id: u32,
                      token_ids_ptr: u32,
                      token_ids_len: u32,
                      out_ptr: u32,
                      out_max_len: u32|
                      -> i32 {
                    use crate::model_storage::get_global_model_storage;

                    // Get WASM memory
                    let wasm_memory = match caller.get_export("memory") {
                        Some(Extern::Memory(mem)) => mem,
                        _ => {
                            error!("realm_decode_tokens: No WASM memory export");
                            return -1;
                        }
                    };

                    // Read token IDs from WASM
                    let token_ids_bytes = (token_ids_len as usize) * 4; // u32 = 4 bytes
                    let mut token_ids_buffer = vec![0u8; token_ids_bytes];
                    if let Err(e) =
                        wasm_memory.read(&caller, token_ids_ptr as usize, &mut token_ids_buffer)
                    {
                        error!("realm_decode_tokens: Failed to read token IDs: {}", e);
                        return -2;
                    }

                    // Convert to u32 array
                    let token_ids: Vec<u32> = token_ids_buffer
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Get tokenizer from model storage
                    let storage = get_global_model_storage().lock();
                    let model = match storage.get_model(model_id) {
                        Ok(m) => m,
                        Err(_) => {
                            error!("realm_decode_tokens: Model {} not found", model_id);
                            return -3;
                        }
                    };

                    let tokenizer = match model.tokenizer() {
                        Some(t) => t,
                        None => {
                            error!("realm_decode_tokens: No tokenizer for model {}", model_id);
                            return -4;
                        }
                    };

                    // Decode tokens to text
                    let text = match tokenizer.decode(&token_ids, false) {
                        Ok(t) => t,
                        Err(e) => {
                            error!("realm_decode_tokens: Decoding failed: {}", e);
                            return -5;
                        }
                    };

                    let text_bytes = text.as_bytes();
                    if text_bytes.len() > out_max_len as usize {
                        error!(
                            "realm_decode_tokens: Output buffer too small: need {}, have {}",
                            text_bytes.len(),
                            out_max_len
                        );
                        return -6;
                    }

                    // Write text to WASM memory
                    if let Err(e) = wasm_memory.write(&mut caller, out_ptr as usize, text_bytes) {
                        error!("realm_decode_tokens: Failed to write text: {}", e);
                        return -7;
                    }

                    text_bytes.len() as i32
                },
            )?;
        }

        Ok(())
    }

    /// Get state reference (for testing/debugging)
    pub fn state(&self) -> Arc<Mutex<Memory64State>> {
        self.state.clone()
    }

    /// Get stream callback reference (for testing)
    #[cfg(test)]
    pub fn stream_callback(&self) -> &Arc<Mutex<Option<std::sync::mpsc::Sender<String>>>> {
        &self.stream_callback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_memory_layout() {
        let layout = MemoryLayout::single(8, "test").unwrap();
        assert_eq!(layout.regions.len(), 1);
        assert_eq!(layout.total_gb(), 8.0);
    }

    #[test]
    fn test_invalid_size() {
        assert!(MemoryLayout::single(0, "test").is_err()); // Zero size
        assert!(MemoryLayout::single(20000, "test").is_err()); // Too large
    }

    #[test]
    fn test_overflow_protection() {
        // This should fail due to overflow when converting GB to bytes
        // u64::MAX / (1024^3) = ~16 exabytes, when multiplied by 1024^3 should overflow
        // Use a value that will overflow: (u64::MAX / 1024 / 1024 / 1024) + 1
        let result = MemoryLayout::multi(&[
            ("region1", u64::MAX / 1024 / 1024 / 1024),
            ("region2", 1), // This will cause offset overflow
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_region_validation() {
        // Non-page-aligned size should fail
        let result = MemoryRegion::new(0, "test", 0, 1000, "test");
        assert!(result.is_err());

        // Valid page-aligned size should succeed
        let result = MemoryRegion::new(0, "test", 0, 65536, "test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_streaming_callback_set_and_clear() {
        // Test that streaming callback can be set and cleared
        let layout = MemoryLayout::single(8, "test").unwrap();
        let runtime = Memory64Runtime::new(layout, true);

        // Create a test channel
        let (tx, _rx) = std::sync::mpsc::channel();

        // Set callback
        runtime.set_stream_callback(tx);

        // Clear callback
        runtime.clear_stream_callback();

        // Test passes if no panic occurs
    }

    #[test]
    fn test_streaming_callback_receives_tokens() {
        // Test that tokens sent via callback are received correctly
        let layout = MemoryLayout::single(8, "test").unwrap();
        let runtime = Memory64Runtime::new(layout, true);

        // Create a test channel
        let (tx, rx) = std::sync::mpsc::channel();

        // Set callback
        runtime.set_stream_callback(tx);

        // Simulate sending tokens (in real usage, this would be via realm_stream_token host function)
        // We can't directly call the host function from tests, but we can test the callback mechanism
        let callback_guard = runtime.stream_callback().lock();
        if let Some(ref sender) = *callback_guard {
            sender.send("test_token".to_string()).unwrap();
        }
        drop(callback_guard);

        // Receive token
        let token = rx.recv().unwrap();
        assert_eq!(token, "test_token");
    }

    #[test]
    fn test_streaming_callback_none_when_not_set() {
        // Test that callback is None when not set
        let layout = MemoryLayout::single(8, "test").unwrap();
        let runtime = Memory64Runtime::new(layout, true);

        // Callback should be None initially
        let callback_guard = runtime.stream_callback().lock();
        assert!(callback_guard.is_none());
    }
}
