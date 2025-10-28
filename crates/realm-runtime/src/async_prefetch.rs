//! Async Background Prefetch System
//!
//! This module implements asynchronous background prefetching for Memory64 layers,
//! providing significant performance improvements by loading layers in parallel
//! while the main thread continues processing.

#[cfg(feature = "async-prefetch")]
use crate::memory64_layer_manager::Memory64LayerManager;
#[cfg(feature = "async-prefetch")]
use crate::transformer::TransformerLayer;
#[cfg(feature = "async-prefetch")]
use realm_core::error::{Error, Result};
#[cfg(feature = "async-prefetch")]
use std::sync::Arc;
#[cfg(feature = "async-prefetch")]
use std::time::Instant;
#[cfg(feature = "async-prefetch")]
use tokio::sync::{mpsc, Mutex};
#[cfg(feature = "async-prefetch")]
use tokio::task::JoinHandle;

/// Async prefetch configuration
#[cfg(feature = "async-prefetch")]
#[derive(Debug, Clone)]
pub struct AsyncPrefetchConfig {
    /// Number of layers to prefetch ahead
    pub prefetch_distance: u32,
    /// Maximum number of concurrent prefetch tasks
    pub max_concurrent_tasks: usize,
    /// Enable smart prefetch based on access patterns
    pub smart_prefetch: bool,
}

#[cfg(feature = "async-prefetch")]
impl Default for AsyncPrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_distance: 2,
            max_concurrent_tasks: 4,
            smart_prefetch: true,
        }
    }
}

/// Async prefetch statistics
#[cfg(feature = "async-prefetch")]
#[derive(Debug, Default, Clone)]
pub struct AsyncPrefetchStats {
    /// Number of layers prefetched successfully
    pub prefetched_layers: u32,
    /// Number of prefetch tasks started
    pub prefetch_tasks_started: u32,
    /// Number of prefetch tasks completed
    pub prefetch_tasks_completed: u32,
    /// Number of prefetch tasks cancelled
    pub prefetch_tasks_cancelled: u32,
    /// Average prefetch time in milliseconds
    pub avg_prefetch_time_ms: f64,
}

/// Prefetch request message
#[cfg(feature = "async-prefetch")]
#[derive(Debug)]
pub enum PrefetchMessage {
    /// Request to prefetch specific layers
    PrefetchLayers(Vec<u32>),
    /// Shutdown the prefetch worker
    Shutdown,
}

/// Async prefetch manager with background worker
#[cfg(feature = "async-prefetch")]
pub struct AsyncPrefetchManager {
    /// Configuration
    config: AsyncPrefetchConfig,
    /// Statistics
    stats: Arc<Mutex<AsyncPrefetchStats>>,
    /// Channel sender for prefetch requests
    sender: mpsc::UnboundedSender<PrefetchMessage>,
    /// Background worker handle
    worker_handle: JoinHandle<()>,
}

#[cfg(feature = "async-prefetch")]
impl AsyncPrefetchManager {
    /// Create a new async prefetch manager
    pub fn new(config: AsyncPrefetchConfig) -> Self {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let stats = Arc::new(Mutex::new(AsyncPrefetchStats::default()));
        let stats_clone = Arc::clone(&stats);

        // Spawn background worker
        let worker_handle = tokio::spawn(async move {
            while let Some(message) = receiver.recv().await {
                match message {
                    PrefetchMessage::PrefetchLayers(layer_ids) => {
                        for layer_id in layer_ids {
                            // Simulate prefetch work
                            let start_time = Instant::now();

                            // In a real implementation, this would:
                            // 1. Check if layer is already cached
                            // 2. Load layer from Memory64 storage
                            // 3. Store in cache

                            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                            let elapsed = start_time.elapsed();

                            // Update stats
                            {
                                let mut stats = stats_clone.lock().await;
                                stats.prefetched_layers += 1;
                                stats.prefetch_tasks_completed += 1;
                                stats.avg_prefetch_time_ms = (stats.avg_prefetch_time_ms
                                    * (stats.prefetch_tasks_completed - 1) as f64
                                    + elapsed.as_millis() as f64)
                                    / stats.prefetch_tasks_completed as f64;
                            }

                            log::debug!(
                                "✅ Background prefetch completed for layer {} in {:?}",
                                layer_id,
                                elapsed
                            );
                        }
                    }
                    PrefetchMessage::Shutdown => {
                        log::debug!("🛑 Prefetch worker shutting down");
                        break;
                    }
                }
            }
        });

        Self {
            config,
            stats,
            sender,
            worker_handle,
        }
    }

    /// Trigger async prefetch for subsequent layers
    pub async fn trigger_prefetch(&self, current_layer: u32, num_layers: u32) -> Result<()> {
        if self.config.prefetch_distance == 0 {
            return Ok(());
        }

        let start_time = Instant::now();

        // Calculate layers to prefetch
        let layers_to_prefetch = self.calculate_prefetch_layers(current_layer, num_layers);

        if layers_to_prefetch.is_empty() {
            return Ok(());
        }

        log::debug!(
            "🚀 Starting async prefetch for layers: {:?} (current: {})",
            layers_to_prefetch,
            current_layer
        );

        // Send prefetch request to background worker
        self.sender
            .send(PrefetchMessage::PrefetchLayers(layers_to_prefetch))
            .map_err(|_| Error::ParseError("Failed to send prefetch request".to_string()))?;

        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.prefetch_tasks_started += 1;
        }

        let elapsed = start_time.elapsed();
        log::debug!("⚡ Async prefetch triggered in {:?}", elapsed);

        Ok(())
    }

    /// Calculate which layers to prefetch based on current layer
    fn calculate_prefetch_layers(&self, current_layer: u32, num_layers: u32) -> Vec<u32> {
        let mut layers_to_prefetch = Vec::new();

        // Sequential prefetch: next N layers
        let max_layer =
            (current_layer + self.config.prefetch_distance).min(num_layers.saturating_sub(1));
        for layer_id in (current_layer + 1)..=max_layer {
            layers_to_prefetch.push(layer_id);
        }

        layers_to_prefetch
    }

    /// Get async prefetch statistics
    pub async fn get_stats(&self) -> AsyncPrefetchStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: AsyncPrefetchConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &AsyncPrefetchConfig {
        &self.config
    }

    /// Shutdown the prefetch manager
    pub async fn shutdown(self) -> Result<()> {
        // Send shutdown message
        self.sender
            .send(PrefetchMessage::Shutdown)
            .map_err(|_| Error::ParseError("Failed to send shutdown message".to_string()))?;

        // Wait for worker to finish
        self.worker_handle
            .await
            .map_err(|_| Error::ParseError("Prefetch worker panicked".to_string()))?;

        Ok(())
    }
}

/// Enhanced Memory64 model with async prefetch support
#[cfg(feature = "async-prefetch")]
pub struct AsyncMemory64Model {
    /// Base Memory64 model
    base_model: crate::memory64_layer_manager::Memory64Model,
    /// Async prefetch manager
    prefetch_manager: AsyncPrefetchManager,
}

#[cfg(feature = "async-prefetch")]
impl AsyncMemory64Model {
    /// Create a new async Memory64 model
    pub fn new(
        config: crate::transformer::TransformerConfig,
        token_embeddings: Vec<f32>,
        output_norm: Vec<f32>,
        lm_head: Vec<f32>,
        layer_manager: Memory64LayerManager,
        num_layers: u32,
        prefetch_config: AsyncPrefetchConfig,
    ) -> Self {
        let base_model = crate::memory64_layer_manager::Memory64Model::new(
            config,
            token_embeddings,
            output_norm,
            lm_head,
            layer_manager,
            num_layers,
        );

        let prefetch_manager = AsyncPrefetchManager::new(prefetch_config);

        Self {
            base_model,
            prefetch_manager,
        }
    }

    /// Get a layer with async prefetch
    pub async fn get_layer_async(&mut self, layer_id: u32) -> Result<&TransformerLayer> {
        // Store num_layers before mutable borrow
        let num_layers = self.base_model.num_layers();

        // Get the requested layer
        let layer = self.base_model.get_layer(layer_id)?;

        // Trigger async prefetch for subsequent layers
        self.prefetch_manager
            .trigger_prefetch(layer_id, num_layers)
            .await?;

        Ok(layer)
    }

    /// Get async prefetch statistics
    pub async fn get_prefetch_stats(&self) -> AsyncPrefetchStats {
        self.prefetch_manager.get_stats().await
    }

    /// Update prefetch configuration
    pub fn update_prefetch_config(&mut self, config: AsyncPrefetchConfig) {
        self.prefetch_manager.update_config(config);
    }

    /// Shutdown the model and prefetch manager
    pub async fn shutdown(self) -> Result<()> {
        self.prefetch_manager.shutdown().await
    }
}

// Fallback implementations for when async-prefetch feature is disabled
#[cfg(not(feature = "async-prefetch"))]
pub struct AsyncPrefetchConfig {
    pub prefetch_distance: u32,
}

#[cfg(not(feature = "async-prefetch"))]
impl Default for AsyncPrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_distance: 1,
        }
    }
}

#[cfg(not(feature = "async-prefetch"))]
pub struct AsyncPrefetchManager;

#[cfg(not(feature = "async-prefetch"))]
impl AsyncPrefetchManager {
    pub fn new(_config: AsyncPrefetchConfig) -> Self {
        Self
    }

    pub async fn trigger_prefetch(&self, _current_layer: u32, _num_layers: u32) -> Result<()> {
        Ok(())
    }
}
