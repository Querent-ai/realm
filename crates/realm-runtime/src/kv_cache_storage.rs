//! KV Cache Storage for HOST-side caching
//!
//! Stores KV caches per (model_id, layer_idx) to enable proper
//! state management across multiple forward passes.

use parking_lot::Mutex;
use realm_models::KVCache;
use std::collections::HashMap;
use std::sync::Arc;

/// Global KV cache storage
pub struct KVCacheStorage {
    /// Caches indexed by (model_id, layer_idx)
    caches: Arc<Mutex<HashMap<(u32, u32), KVCache>>>,
}

impl KVCacheStorage {
    /// Create new KV cache storage
    pub fn new() -> Self {
        Self {
            caches: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get or create KV cache for a model layer
    pub fn get_or_create(
        &self,
        model_id: u32,
        layer_idx: u32,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Arc<Mutex<KVCache>> {
        let mut caches = self.caches.lock();
        let key = (model_id, layer_idx);

        if let Some(cache) = caches.get(&key) {
            // Return existing cache
            Arc::new(Mutex::new(cache.clone()))
        } else {
            // Create new cache
            let cache = KVCache::new(max_seq_len, num_kv_heads, head_dim);
            let cache_arc = Arc::new(Mutex::new(cache.clone()));
            caches.insert(key, cache);
            cache_arc
        }
    }

    /// Get existing KV cache (returns None if not found)
    pub fn get(&self, model_id: u32, layer_idx: u32) -> Option<Arc<Mutex<KVCache>>> {
        let caches = self.caches.lock();
        let key = (model_id, layer_idx);
        caches
            .get(&key)
            .map(|cache| Arc::new(Mutex::new(cache.clone())))
    }

    /// Clear KV cache for a specific model layer
    pub fn clear(&self, model_id: u32, layer_idx: u32) {
        let mut caches = self.caches.lock();
        let key = (model_id, layer_idx);
        if let Some(cache) = caches.get_mut(&key) {
            cache.clear();
        }
    }

    /// Clear all KV caches for a model
    pub fn clear_model(&self, model_id: u32) {
        let mut caches = self.caches.lock();
        caches.retain(|(mid, _), _| *mid != model_id);
    }

    /// Get cache count (for debugging)
    pub fn cache_count(&self) -> usize {
        self.caches.lock().len()
    }
}

impl Default for KVCacheStorage {
    fn default() -> Self {
        Self::new()
    }
}

// Global singleton
lazy_static::lazy_static! {
    pub static ref GLOBAL_KV_CACHE_STORAGE: KVCacheStorage = KVCacheStorage::new();
}
