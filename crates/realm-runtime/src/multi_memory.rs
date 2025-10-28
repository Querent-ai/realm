//! Multi-memory layout for efficient memory management
//!
//! This module provides a multi-memory system that separates different types of data
//! into separate memory regions for better cache locality and memory management.

use realm_core::error::{Error, Result};

/// Memory region types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion {
    /// Model weights (read-only)
    Weights,
    /// Activation tensors (temporary, frequently allocated/deallocated)
    Activations,
    /// KV cache (persistent, grows over time)
    KVCache,
    /// Embeddings (read-only, frequently accessed)
    Embeddings,
}

/// Memory region configuration
#[derive(Debug, Clone)]
pub struct MemoryRegionConfig {
    pub region: MemoryRegion,
    pub initial_size: usize,
    pub max_size: usize,
    pub growable: bool,
}

impl MemoryRegionConfig {
    pub fn new(region: MemoryRegion, initial_size: usize, max_size: usize, growable: bool) -> Self {
        Self {
            region,
            initial_size,
            max_size,
            growable,
        }
    }
}

/// Memory region manager
struct MemoryRegionData {
    config: MemoryRegionConfig,
    allocated: usize,
    used: usize,
}

impl MemoryRegionData {
    fn new(config: MemoryRegionConfig) -> Self {
        Self {
            config,
            allocated: 0,
            used: 0,
        }
    }

    fn can_allocate(&self, size: usize) -> bool {
        self.used + size <= self.config.max_size
    }

    fn allocate(&mut self, size: usize) -> Result<()> {
        if !self.can_allocate(size) {
            return Err(Error::AllocationFailed(format!(
                "Cannot allocate {} bytes in {:?} region (used: {}, max: {})",
                size, self.config.region, self.used, self.config.max_size
            )));
        }
        self.used += size;
        if self.used > self.allocated {
            self.allocated = self.used;
        }
        Ok(())
    }

    fn deallocate(&mut self, size: usize) {
        self.used = self.used.saturating_sub(size);
    }

    fn usage_percent(&self) -> f32 {
        if self.config.max_size == 0 {
            return 0.0;
        }
        (self.used as f32 / self.config.max_size as f32) * 100.0
    }
}

/// Multi-memory layout manager
pub struct MultiMemoryLayout {
    regions: Vec<MemoryRegionData>,
}

impl MultiMemoryLayout {
    /// Create a new multi-memory layout with default configuration
    pub fn new() -> Self {
        let regions = vec![
            MemoryRegionData::new(MemoryRegionConfig::new(
                MemoryRegion::Weights,
                2 * 1024 * 1024 * 1024,                // 2GB initial
                (8_u64 * 1024 * 1024 * 1024) as usize, // 8GB max
                false,                                 // not growable (static)
            )),
            MemoryRegionData::new(MemoryRegionConfig::new(
                MemoryRegion::Activations,
                512 * 1024 * 1024,                     // 512MB initial
                (4_u64 * 1024 * 1024 * 1024) as usize, // 4GB max
                true,                                  // growable
            )),
            MemoryRegionData::new(MemoryRegionConfig::new(
                MemoryRegion::KVCache,
                256 * 1024 * 1024,      // 256MB initial
                2 * 1024 * 1024 * 1024, // 2GB max
                true,                   // growable
            )),
            MemoryRegionData::new(MemoryRegionConfig::new(
                MemoryRegion::Embeddings,
                512 * 1024 * 1024,  // 512MB initial
                1024 * 1024 * 1024, // 1GB max
                false,              // not growable
            )),
        ];

        Self { regions }
    }

    /// Create with custom configurations
    pub fn with_configs(configs: Vec<MemoryRegionConfig>) -> Self {
        let regions = configs.into_iter().map(MemoryRegionData::new).collect();
        Self { regions }
    }

    /// Get the number of memory regions
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Get region index
    fn region_index(&self, region: MemoryRegion) -> Option<usize> {
        self.regions.iter().position(|r| r.config.region == region)
    }

    /// Allocate memory in a specific region
    pub fn allocate(&mut self, region: MemoryRegion, size: usize) -> Result<()> {
        let idx = self
            .region_index(region)
            .ok_or_else(|| Error::Runtime(format!("Region {:?} not found", region)))?;

        self.regions[idx].allocate(size)
    }

    /// Deallocate memory from a specific region
    pub fn deallocate(&mut self, region: MemoryRegion, size: usize) -> Result<()> {
        let idx = self
            .region_index(region)
            .ok_or_else(|| Error::Runtime(format!("Region {:?} not found", region)))?;

        self.regions[idx].deallocate(size);
        Ok(())
    }

    /// Check if allocation is possible in a region
    pub fn can_allocate(&self, region: MemoryRegion, size: usize) -> bool {
        self.region_index(region)
            .map(|idx| self.regions[idx].can_allocate(size))
            .unwrap_or(false)
    }

    /// Get memory usage for a region
    pub fn region_usage(&self, region: MemoryRegion) -> Option<(usize, usize, f32)> {
        self.region_index(region).map(|idx| {
            let r = &self.regions[idx];
            (r.used, r.config.max_size, r.usage_percent())
        })
    }

    /// Get total memory usage across all regions
    pub fn total_usage(&self) -> (usize, usize, f32) {
        let total_used: usize = self.regions.iter().map(|r| r.used).sum();
        let total_max: usize = self.regions.iter().map(|r| r.config.max_size).sum();
        let usage_percent = if total_max > 0 {
            (total_used as f32 / total_max as f32) * 100.0
        } else {
            0.0
        };
        (total_used, total_max, usage_percent)
    }

    /// Reset all regions (for testing)
    pub fn reset(&mut self) {
        for region in &mut self.regions {
            region.used = 0;
            region.allocated = 0;
        }
    }

    /// Get statistics for all regions
    pub fn stats(&self) -> Vec<(MemoryRegion, usize, usize, f32)> {
        self.regions
            .iter()
            .map(|r| {
                (
                    r.config.region,
                    r.used,
                    r.config.max_size,
                    r.usage_percent(),
                )
            })
            .collect()
    }
}

impl Default for MultiMemoryLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_memory_creation() {
        let layout = MultiMemoryLayout::new();
        assert_eq!(layout.regions.len(), 4);

        // Verify all regions are present
        assert!(layout.region_index(MemoryRegion::Weights).is_some());
        assert!(layout.region_index(MemoryRegion::Activations).is_some());
        assert!(layout.region_index(MemoryRegion::KVCache).is_some());
        assert!(layout.region_index(MemoryRegion::Embeddings).is_some());
    }

    #[test]
    fn test_region_allocation() -> Result<()> {
        let mut layout = MultiMemoryLayout::new();

        // Allocate in weights region
        layout.allocate(MemoryRegion::Weights, 1024 * 1024 * 1024)?; // 1GB

        let (used, _max, percent) = layout.region_usage(MemoryRegion::Weights).unwrap();
        assert_eq!(used, 1024 * 1024 * 1024);
        assert!(percent > 0.0);
        assert!(percent < 100.0);

        Ok(())
    }

    #[test]
    fn test_allocation_limits() {
        let mut layout = MultiMemoryLayout::new();

        // Try to allocate more than max in embeddings region (max 1GB)
        let result = layout.allocate(MemoryRegion::Embeddings, 2 * 1024 * 1024 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_deallocation() -> Result<()> {
        let mut layout = MultiMemoryLayout::new();

        // Allocate and deallocate
        layout.allocate(MemoryRegion::Activations, 500 * 1024 * 1024)?; // 500MB
        let (used1, _, _) = layout.region_usage(MemoryRegion::Activations).unwrap();
        assert_eq!(used1, 500 * 1024 * 1024);

        layout.deallocate(MemoryRegion::Activations, 200 * 1024 * 1024)?; // -200MB
        let (used2, _, _) = layout.region_usage(MemoryRegion::Activations).unwrap();
        assert_eq!(used2, 300 * 1024 * 1024);

        Ok(())
    }

    #[test]
    fn test_can_allocate() {
        let layout = MultiMemoryLayout::new();

        // KVCache has 2GB max
        assert!(layout.can_allocate(MemoryRegion::KVCache, 1024 * 1024 * 1024)); // 1GB - should fit
        assert!(!layout.can_allocate(MemoryRegion::KVCache, 3 * 1024 * 1024 * 1024));
        // 3GB - too large
    }

    #[test]
    fn test_total_usage() -> Result<()> {
        let mut layout = MultiMemoryLayout::new();

        // Allocate in multiple regions
        layout.allocate(MemoryRegion::Weights, 1024 * 1024 * 1024)?; // 1GB
        layout.allocate(MemoryRegion::Activations, 512 * 1024 * 1024)?; // 512MB
        layout.allocate(MemoryRegion::KVCache, 256 * 1024 * 1024)?; // 256MB

        let (total_used, total_max, _) = layout.total_usage();
        assert_eq!(
            total_used,
            1024 * 1024 * 1024 + 512 * 1024 * 1024 + 256 * 1024 * 1024
        );
        assert!(total_max > 0);

        Ok(())
    }

    #[test]
    fn test_reset() -> Result<()> {
        let mut layout = MultiMemoryLayout::new();

        // Allocate some memory
        layout.allocate(MemoryRegion::Weights, 1024 * 1024 * 1024)?;
        layout.allocate(MemoryRegion::Activations, 512 * 1024 * 1024)?;

        // Reset
        layout.reset();

        // Verify all regions are empty
        let (total_used, _, _) = layout.total_usage();
        assert_eq!(total_used, 0);

        Ok(())
    }

    #[test]
    fn test_stats() -> Result<()> {
        let mut layout = MultiMemoryLayout::new();

        layout.allocate(MemoryRegion::Weights, 1024 * 1024 * 1024)?;

        let stats = layout.stats();
        assert_eq!(stats.len(), 4);

        // Find weights stat
        let weights_stat = stats
            .iter()
            .find(|(region, _, _, _)| *region == MemoryRegion::Weights)
            .unwrap();

        assert_eq!(weights_stat.1, 1024 * 1024 * 1024); // used
        assert!(weights_stat.3 > 0.0); // percentage

        Ok(())
    }

    #[test]
    fn test_custom_config() -> Result<()> {
        let configs = vec![
            MemoryRegionConfig::new(MemoryRegion::Weights, 1024, 2048, false),
            MemoryRegionConfig::new(MemoryRegion::Activations, 512, 1024, true),
        ];

        let mut layout = MultiMemoryLayout::with_configs(configs);

        // Should have only 2 regions
        assert_eq!(layout.regions.len(), 2);

        // Test allocation with custom limits
        layout.allocate(MemoryRegion::Weights, 1500)?;
        let result = layout.allocate(MemoryRegion::Weights, 1000); // would exceed 2048
        assert!(result.is_err());

        Ok(())
    }
}
