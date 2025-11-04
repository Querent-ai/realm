//! Continuous batching for dynamic request batching
//!
//! This module implements continuous batching which allows multiple requests to be
//! batched together dynamically, improving GPU utilization and throughput.

use realm_core::error::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// A request in the batch queue
#[derive(Debug, Clone)]
pub struct BatchedRequest {
    pub request_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub is_complete: bool,
}

impl BatchedRequest {
    pub fn new(request_id: u64, prompt_tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            request_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            is_complete: false,
        }
    }

    pub fn current_seq_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    pub fn all_tokens(&self) -> Vec<u32> {
        let mut all = self.prompt_tokens.clone();
        all.extend_from_slice(&self.generated_tokens);
        all
    }
}

/// Continuous batching manager
pub struct ContinuousBatcher {
    /// Queue of active requests
    requests: Arc<Mutex<VecDeque<BatchedRequest>>>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum sequence length per request
    max_seq_len: usize,
}

impl ContinuousBatcher {
    /// Create a new continuous batcher
    pub fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        Self {
            requests: Arc::new(Mutex::new(VecDeque::new())),
            max_batch_size,
            max_seq_len,
        }
    }

    /// Add a new request to the batch queue
    pub fn add_request(&self, request: BatchedRequest) -> Result<()> {
        let mut requests = self.requests.lock().unwrap();
        if requests.len() >= self.max_batch_size {
            return Err(realm_core::error::Error::Runtime(
                "Batch queue is full".to_string(),
            ));
        }
        requests.push_back(request);
        Ok(())
    }

    /// Get the next batch of requests to process
    ///
    /// Returns requests that are ready for inference (not complete, within max_seq_len)
    pub fn get_batch(&self) -> Vec<BatchedRequest> {
        let mut requests = self.requests.lock().unwrap();
        let mut batch = Vec::new();
        let mut remaining = VecDeque::new();

        // Filter active requests
        while let Some(mut req) = requests.pop_front() {
            if req.is_complete {
                continue; // Skip completed requests
            }
            if req.current_seq_len() >= self.max_seq_len {
                req.is_complete = true;
                continue; // Skip requests that exceeded max length
            }
            if req.generated_tokens.len() >= req.max_tokens {
                req.is_complete = true;
                continue; // Skip requests that reached max tokens
            }

            batch.push(req);
            if batch.len() >= self.max_batch_size {
                break;
            }
        }

        // Put remaining requests back
        while let Some(req) = requests.pop_front() {
            remaining.push_back(req);
        }
        *requests = remaining;

        batch
    }

    /// Update a request after generation step
    pub fn update_request(&self, request_id: u64, new_token: u32) -> Result<()> {
        let mut requests = self.requests.lock().unwrap();
        for req in requests.iter_mut() {
            if req.request_id == request_id {
                req.generated_tokens.push(new_token);
                if req.generated_tokens.len() >= req.max_tokens {
                    req.is_complete = true;
                }
                return Ok(());
            }
        }
        Err(realm_core::error::Error::Runtime(format!(
            "Request {} not found",
            request_id
        )))
    }

    /// Remove a completed request
    pub fn remove_request(&self, request_id: u64) -> Result<()> {
        let mut requests = self.requests.lock().unwrap();
        let mut found = false;
        requests.retain(|req| {
            if req.request_id == request_id {
                found = true;
                false
            } else {
                true
            }
        });
        if found {
            Ok(())
        } else {
            Err(realm_core::error::Error::Runtime(format!(
                "Request {} not found",
                request_id
            )))
        }
    }

    /// Get number of active requests
    pub fn active_count(&self) -> usize {
        let requests = self.requests.lock().unwrap();
        requests.iter().filter(|r| !r.is_complete).count()
    }

    /// Get batch statistics
    pub fn stats(&self) -> BatchStats {
        let requests = self.requests.lock().unwrap();
        let total = requests.len();
        let active = requests.iter().filter(|r| !r.is_complete).count();
        let avg_seq_len = if total > 0 {
            requests.iter().map(|r| r.current_seq_len()).sum::<usize>() / total
        } else {
            0
        };

        BatchStats {
            total_requests: total,
            active_requests: active,
            average_sequence_length: avg_seq_len,
        }
    }
}

/// Batch statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub total_requests: usize,
    pub active_requests: usize,
    pub average_sequence_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batcher_creation() {
        let batcher = ContinuousBatcher::new(32, 2048);
        assert_eq!(batcher.active_count(), 0);
    }

    #[test]
    fn test_add_and_get_batch() {
        let batcher = ContinuousBatcher::new(32, 2048);

        let req1 = BatchedRequest::new(1, vec![1, 2, 3], 100);
        let req2 = BatchedRequest::new(2, vec![4, 5, 6], 100);

        batcher.add_request(req1).unwrap();
        batcher.add_request(req2).unwrap();

        let batch = batcher.get_batch();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].request_id, 1);
        assert_eq!(batch[1].request_id, 2);
    }

    #[test]
    fn test_update_request() {
        let batcher = ContinuousBatcher::new(32, 2048);

        let req = BatchedRequest::new(1, vec![1, 2, 3], 100);
        batcher.add_request(req).unwrap();

        batcher.update_request(1, 42).unwrap();

        let batch = batcher.get_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].generated_tokens, vec![42]);
    }

    #[test]
    fn test_complete_request() {
        let batcher = ContinuousBatcher::new(32, 2048);

        let req = BatchedRequest::new(1, vec![1, 2, 3], 1); // max_tokens = 1
        batcher.add_request(req).unwrap();

        batcher.update_request(1, 42).unwrap();

        // Request should be marked complete
        let batch = batcher.get_batch();
        assert_eq!(batch.len(), 0); // Completed request should not be in batch
    }
}
