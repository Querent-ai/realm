//! Layer sharding for large models
//!
//! This module provides layer sharding capabilities to split large models
//! across multiple memory regions or compute devices for models that exceed
//! single-region memory limits.

use realm_core::error::{Error, Result};

/// Sharding strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// No sharding - all layers in one region
    None,
    /// Sequential sharding - divide layers into N equal groups
    Sequential,
    /// Custom sharding - user-defined layer boundaries
    Custom,
}

/// Shard configuration for a group of layers
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Starting layer index (inclusive)
    pub start_layer: usize,
    /// Ending layer index (exclusive)
    pub end_layer: usize,
    /// Memory region or device ID
    pub region_id: usize,
    /// Estimated memory requirement in bytes
    pub memory_bytes: usize,
}

impl ShardConfig {
    pub fn new(
        start_layer: usize,
        end_layer: usize,
        region_id: usize,
        memory_bytes: usize,
    ) -> Self {
        Self {
            start_layer,
            end_layer,
            region_id,
            memory_bytes,
        }
    }

    /// Number of layers in this shard
    pub fn layer_count(&self) -> usize {
        self.end_layer.saturating_sub(self.start_layer)
    }

    /// Check if a layer belongs to this shard
    pub fn contains_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.start_layer && layer_idx < self.end_layer
    }
}

/// Layer sharding manager
pub struct ShardingManager {
    strategy: ShardingStrategy,
    shards: Vec<ShardConfig>,
    total_layers: usize,
}

impl ShardingManager {
    /// Create a new sharding manager without sharding
    pub fn new(total_layers: usize) -> Self {
        Self {
            strategy: ShardingStrategy::None,
            shards: vec![ShardConfig::new(0, total_layers, 0, 0)],
            total_layers,
        }
    }

    /// Create with sequential sharding across N regions
    pub fn with_sequential_sharding(
        total_layers: usize,
        num_regions: usize,
        memory_per_layer: usize,
    ) -> Result<Self> {
        if num_regions == 0 {
            return Err(Error::Runtime(
                "Number of regions must be at least 1".to_string(),
            ));
        }

        let layers_per_shard = total_layers.div_ceil(num_regions);
        let mut shards = Vec::with_capacity(num_regions);

        for region_id in 0..num_regions {
            let start_layer = region_id * layers_per_shard;
            let end_layer = ((region_id + 1) * layers_per_shard).min(total_layers);

            if start_layer < total_layers {
                let layer_count = end_layer - start_layer;
                shards.push(ShardConfig::new(
                    start_layer,
                    end_layer,
                    region_id,
                    layer_count * memory_per_layer,
                ));
            }
        }

        Ok(Self {
            strategy: ShardingStrategy::Sequential,
            shards,
            total_layers,
        })
    }

    /// Create with custom shard boundaries
    pub fn with_custom_sharding(total_layers: usize, shards: Vec<ShardConfig>) -> Result<Self> {
        // Validate shards
        if shards.is_empty() {
            return Err(Error::Runtime(
                "At least one shard must be specified".to_string(),
            ));
        }

        // Check all layers are covered
        let mut covered = vec![false; total_layers];
        for shard in &shards {
            if shard.start_layer >= total_layers {
                return Err(Error::Runtime(format!(
                    "Shard start layer {} exceeds total layers {}",
                    shard.start_layer, total_layers
                )));
            }
            if shard.end_layer > total_layers {
                return Err(Error::Runtime(format!(
                    "Shard end layer {} exceeds total layers {}",
                    shard.end_layer, total_layers
                )));
            }
            if shard.start_layer >= shard.end_layer {
                return Err(Error::Runtime(format!(
                    "Invalid shard range: {} to {}",
                    shard.start_layer, shard.end_layer
                )));
            }

            // Mark layers as covered
            for (layer_idx, is_covered) in covered
                .iter_mut()
                .enumerate()
                .take(shard.end_layer)
                .skip(shard.start_layer)
            {
                if *is_covered {
                    return Err(Error::Runtime(format!(
                        "Layer {} is covered by multiple shards",
                        layer_idx
                    )));
                }
                *is_covered = true;
            }
        }

        // Ensure all layers are covered
        if !covered.iter().all(|&c| c) {
            return Err(Error::Runtime(
                "Not all layers are covered by shards".to_string(),
            ));
        }

        Ok(Self {
            strategy: ShardingStrategy::Custom,
            shards,
            total_layers,
        })
    }

    /// Get the shard containing a specific layer
    pub fn shard_for_layer(&self, layer_idx: usize) -> Option<&ShardConfig> {
        if layer_idx >= self.total_layers {
            return None;
        }
        self.shards
            .iter()
            .find(|shard| shard.contains_layer(layer_idx))
    }

    /// Get all shards
    pub fn shards(&self) -> &[ShardConfig] {
        &self.shards
    }

    /// Get sharding strategy
    pub fn strategy(&self) -> ShardingStrategy {
        self.strategy
    }

    /// Total number of layers
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Total number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Total memory across all shards
    pub fn total_memory(&self) -> usize {
        self.shards.iter().map(|s| s.memory_bytes).sum()
    }

    /// Get memory statistics per region
    pub fn memory_stats(&self) -> Vec<(usize, usize, usize)> {
        // Group by region_id: (region_id, layer_count, memory_bytes)
        let mut stats: std::collections::HashMap<usize, (usize, usize)> =
            std::collections::HashMap::new();

        for shard in &self.shards {
            let entry = stats.entry(shard.region_id).or_insert((0, 0));
            entry.0 += shard.layer_count();
            entry.1 += shard.memory_bytes;
        }

        let mut result: Vec<_> = stats
            .into_iter()
            .map(|(region_id, (layers, memory))| (region_id, layers, memory))
            .collect();
        result.sort_by_key(|(region_id, _, _)| *region_id);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_sharding() {
        let manager = ShardingManager::new(32);
        assert_eq!(manager.strategy(), ShardingStrategy::None);
        assert_eq!(manager.num_shards(), 1);
        assert_eq!(manager.total_layers(), 32);

        // All layers should be in the same shard
        for layer in 0..32 {
            let shard = manager.shard_for_layer(layer);
            assert!(shard.is_some());
            assert_eq!(shard.unwrap().region_id, 0);
        }
    }

    #[test]
    fn test_sequential_sharding() -> Result<()> {
        // 32 layers across 4 regions = 8 layers per region
        let manager = ShardingManager::with_sequential_sharding(32, 4, 1024 * 1024)?;

        assert_eq!(manager.strategy(), ShardingStrategy::Sequential);
        assert_eq!(manager.num_shards(), 4);

        // Check each shard
        let shards = manager.shards();
        assert_eq!(shards[0].start_layer, 0);
        assert_eq!(shards[0].end_layer, 8);
        assert_eq!(shards[1].start_layer, 8);
        assert_eq!(shards[1].end_layer, 16);
        assert_eq!(shards[2].start_layer, 16);
        assert_eq!(shards[2].end_layer, 24);
        assert_eq!(shards[3].start_layer, 24);
        assert_eq!(shards[3].end_layer, 32);

        // Verify all layers are covered
        for layer in 0..32 {
            let shard = manager.shard_for_layer(layer);
            assert!(shard.is_some(), "Layer {} not covered", layer);
        }

        Ok(())
    }

    #[test]
    fn test_sequential_sharding_uneven() -> Result<()> {
        // 33 layers across 4 regions = 8, 8, 8, 9 layers
        let manager = ShardingManager::with_sequential_sharding(33, 4, 1024 * 1024)?;

        assert_eq!(manager.num_shards(), 4);

        let shards = manager.shards();
        assert_eq!(shards[0].layer_count(), 9); // ceil(33/4) = 9
        assert_eq!(shards[1].layer_count(), 9);
        assert_eq!(shards[2].layer_count(), 9);
        assert_eq!(shards[3].layer_count(), 6); // remaining

        // Verify all 33 layers are covered
        for layer in 0..33 {
            assert!(manager.shard_for_layer(layer).is_some());
        }

        Ok(())
    }

    #[test]
    fn test_custom_sharding() -> Result<()> {
        let shards = vec![
            ShardConfig::new(0, 10, 0, 1024 * 1024),
            ShardConfig::new(10, 20, 1, 1024 * 1024),
            ShardConfig::new(20, 32, 2, 1024 * 1024),
        ];

        let manager = ShardingManager::with_custom_sharding(32, shards)?;

        assert_eq!(manager.strategy(), ShardingStrategy::Custom);
        assert_eq!(manager.num_shards(), 3);

        // Check layer assignments
        assert_eq!(manager.shard_for_layer(0).unwrap().region_id, 0);
        assert_eq!(manager.shard_for_layer(9).unwrap().region_id, 0);
        assert_eq!(manager.shard_for_layer(10).unwrap().region_id, 1);
        assert_eq!(manager.shard_for_layer(19).unwrap().region_id, 1);
        assert_eq!(manager.shard_for_layer(20).unwrap().region_id, 2);
        assert_eq!(manager.shard_for_layer(31).unwrap().region_id, 2);

        Ok(())
    }

    #[test]
    fn test_custom_sharding_validation() {
        // Gap in coverage (layers 10-19 missing)
        let shards = vec![
            ShardConfig::new(0, 10, 0, 1024),
            ShardConfig::new(20, 32, 1, 1024),
        ];
        let result = ShardingManager::with_custom_sharding(32, shards);
        assert!(result.is_err());

        // Overlapping shards
        let shards = vec![
            ShardConfig::new(0, 15, 0, 1024),
            ShardConfig::new(10, 32, 1, 1024),
        ];
        let result = ShardingManager::with_custom_sharding(32, shards);
        assert!(result.is_err());

        // Invalid range (start >= end)
        let shards = vec![
            ShardConfig::new(10, 10, 0, 1024),
            ShardConfig::new(10, 32, 1, 1024),
        ];
        let result = ShardingManager::with_custom_sharding(32, shards);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_stats() -> Result<()> {
        let shards = vec![
            ShardConfig::new(0, 10, 0, 1024 * 1024),
            ShardConfig::new(10, 20, 0, 2 * 1024 * 1024), // same region
            ShardConfig::new(20, 32, 1, 3 * 1024 * 1024),
        ];

        let manager = ShardingManager::with_custom_sharding(32, shards)?;
        let stats = manager.memory_stats();

        // Region 0 should have 20 layers (10 + 10) and 3MB (1MB + 2MB)
        let region0 = stats.iter().find(|(id, _, _)| *id == 0).unwrap();
        assert_eq!(region0.1, 20); // layer count
        assert_eq!(region0.2, 3 * 1024 * 1024); // memory

        // Region 1 should have 12 layers and 3MB
        let region1 = stats.iter().find(|(id, _, _)| *id == 1).unwrap();
        assert_eq!(region1.1, 12); // layer count
        assert_eq!(region1.2, 3 * 1024 * 1024); // memory

        assert_eq!(manager.total_memory(), 6 * 1024 * 1024);

        Ok(())
    }

    #[test]
    fn test_shard_config() {
        let shard = ShardConfig::new(10, 20, 1, 1024);

        assert_eq!(shard.layer_count(), 10);
        assert!(shard.contains_layer(10));
        assert!(shard.contains_layer(15));
        assert!(shard.contains_layer(19));
        assert!(!shard.contains_layer(9));
        assert!(!shard.contains_layer(20));
    }

    #[test]
    fn test_zero_regions_error() {
        let result = ShardingManager::with_sequential_sharding(32, 0, 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_shards_error() {
        let result = ShardingManager::with_custom_sharding(32, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_bounds_layer() {
        let manager = ShardingManager::new(32);
        assert!(manager.shard_for_layer(32).is_none());
        assert!(manager.shard_for_layer(100).is_none());
    }
}
