//! Memory management primitives for wasm-chord
//!
//! Provides abstractions for multi-memory layout and allocation strategies.

use crate::error::{Error, Result};

/// Memory region identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion {
    /// Model weights (read-only)
    Weights,
    /// KV cache (attention cache, read/write)
    KvCache,
    /// Scratch space for temporary tensors
    Scratch,
    /// Host-allocated buffers
    Host,
}

/// Simple bump allocator for deterministic memory management
pub struct BumpAllocator {
    base: *mut u8,
    size: usize,
    offset: usize,
}

impl BumpAllocator {
    /// Create new allocator from raw memory region
    ///
    /// # Safety
    /// Caller must ensure base pointer is valid and size is accurate
    pub unsafe fn new(base: *mut u8, size: usize) -> Self {
        Self {
            base,
            size,
            offset: 0,
        }
    }

    /// Allocate aligned bytes
    pub fn alloc(&mut self, size: usize, align: usize) -> Result<*mut u8> {
        let align_mask = align - 1;
        let aligned_offset = (self.offset + align_mask) & !align_mask;

        if aligned_offset + size > self.size {
            return Err(Error::AllocationFailed(format!(
                "OOM: requested {} bytes, {} available",
                size,
                self.size - aligned_offset
            )));
        }

        let ptr = unsafe { self.base.add(aligned_offset) };
        self.offset = aligned_offset + size;

        Ok(ptr)
    }

    /// Reset allocator (invalidates all previous allocations)
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Get current usage in bytes
    pub fn usage(&self) -> usize {
        self.offset
    }

    /// Get available bytes
    pub fn available(&self) -> usize {
        self.size - self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_allocator() {
        let mut buffer = vec![0u8; 1024];
        let mut alloc = unsafe { BumpAllocator::new(buffer.as_mut_ptr(), buffer.len()) };

        // Allocate 16 bytes aligned to 8
        let ptr1 = alloc.alloc(16, 8).unwrap();
        assert!(!ptr1.is_null());
        assert_eq!(alloc.usage(), 16);

        // Allocate 32 bytes aligned to 16
        let ptr2 = alloc.alloc(32, 16).unwrap();
        assert!(!ptr2.is_null());
        assert_eq!(alloc.usage(), 48);

        // Reset
        alloc.reset();
        assert_eq!(alloc.usage(), 0);
    }

    #[test]
    fn test_oom() {
        let mut buffer = vec![0u8; 64];
        let mut alloc = unsafe { BumpAllocator::new(buffer.as_mut_ptr(), buffer.len()) };

        // This should fail
        let result = alloc.alloc(128, 1);
        assert!(result.is_err());
    }
}
