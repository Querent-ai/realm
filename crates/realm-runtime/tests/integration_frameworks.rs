//! Integration tests for LoRA, Speculative Decoding, and Continuous Batching frameworks

use realm_core::error::Result;
use realm_runtime::batching::{BatchedRequest, ContinuousBatcher};
use realm_runtime::lora::{get_global_lora_manager, LoRAManager, LoRAWeights};
use realm_runtime::speculative::{DraftModel, SpeculativeConfig, SpeculativeDecoder, TargetModel};

#[test]
fn test_lora_integration() {
    // Test LoRA adapter loading and application
    let manager = LoRAManager::new();

    // Create adapter with correct key format for apply_to_weights
    // apply_to_weights expects keys like "layer.0.attn_q.lora_a" and "layer.0.attn_q.lora_b"
    let mut adapter = LoRAWeights::new("integration_test".to_string(), 4, 8.0);
    adapter
        .lora_a
        .insert("layer.0.attn_q.lora_a".to_string(), vec![0.1; 16]);
    adapter
        .lora_b
        .insert("layer.0.attn_q.lora_b".to_string(), vec![0.2; 16]);

    manager.load_adapter(adapter).unwrap();

    // Test application
    let base_weights = vec![1.0; 16]; // 4x4 matrix
    let result =
        manager.apply_to_weights("integration_test", "layer.0.attn_q", &base_weights, 4, 4);
    assert!(result.is_ok());

    let modified = result.unwrap();
    assert_eq!(modified.len(), 16);
    assert_ne!(modified, base_weights);

    // Test global manager - load adapter there too
    let global_manager = get_global_lora_manager();
    match global_manager.lock() {
        Ok(guard) => {
            // Load adapter into global manager
            let mut global_adapter =
                LoRAWeights::new("global_integration_test".to_string(), 4, 8.0);
            global_adapter
                .lora_a
                .insert("layer.0.attn_q.lora_a".to_string(), vec![0.1; 16]);
            global_adapter
                .lora_b
                .insert("layer.0.attn_q.lora_b".to_string(), vec![0.2; 16]);
            guard.load_adapter(global_adapter).unwrap();

            let adapters = guard.list_adapters();
            // Global manager should now have at least our adapter
            assert!(adapters.contains(&"global_integration_test".to_string()));
        }
        Err(_) => panic!("Failed to lock global LoRA manager"),
    }
}

#[test]
fn test_batching_integration() {
    // Test continuous batching with multiple requests
    let batcher = ContinuousBatcher::new(32, 2048);

    // Add multiple requests with prompt text
    let req1 = BatchedRequest::with_prompt_text(
        1,
        vec![1, 2, 3],
        "What is the capital of France?".to_string(),
        50,
    );
    let req2 =
        BatchedRequest::with_prompt_text(2, vec![4, 5, 6], "Tell me about Paris.".to_string(), 50);
    let req3 =
        BatchedRequest::with_prompt_text(3, vec![7, 8, 9], "France is a country.".to_string(), 50);

    batcher.add_request(req1).unwrap();
    batcher.add_request(req2).unwrap();
    batcher.add_request(req3).unwrap();

    // Update requests while they're still in the queue (before get_batch removes them)
    batcher.update_request(1, 100).unwrap();
    batcher.update_request(2, 200).unwrap();
    batcher.update_request(3, 300).unwrap();

    // Get batch (this removes requests from queue)
    let batch = batcher.get_batch();
    assert_eq!(batch.len(), 3);

    // Verify prompt text is preserved
    assert_eq!(
        batch[0].prompt_text,
        Some("What is the capital of France?".to_string())
    );
    assert_eq!(
        batch[1].prompt_text,
        Some("Tell me about Paris.".to_string())
    );
    assert_eq!(
        batch[2].prompt_text,
        Some("France is a country.".to_string())
    );

    // Verify tokens were added (order may vary, so check by request_id)
    let req1_tokens = batch
        .iter()
        .find(|r| r.request_id == 1)
        .map(|r| &r.generated_tokens);
    let req2_tokens = batch
        .iter()
        .find(|r| r.request_id == 2)
        .map(|r| &r.generated_tokens);
    let req3_tokens = batch
        .iter()
        .find(|r| r.request_id == 3)
        .map(|r| &r.generated_tokens);

    assert_eq!(req1_tokens, Some(&vec![100]));
    assert_eq!(req2_tokens, Some(&vec![200]));
    assert_eq!(req3_tokens, Some(&vec![300]));

    // Test stats - after get_batch(), requests are removed from queue
    // So active_requests should be 0
    let stats = batcher.stats();
    // After get_batch(), requests are removed, so active should be 0
    assert_eq!(stats.active_requests, 0);
}

#[test]
fn test_speculative_decoding_integration() {
    // Test speculative decoding with mock models
    struct MockDraft;
    struct MockTarget;

    impl DraftModel for MockDraft {
        fn generate_draft(&mut self, _prompt: &[u32], k: usize) -> Result<Vec<u32>> {
            Ok((0..k).map(|i| i as u32 + 1000).collect())
        }
    }

    impl TargetModel for MockTarget {
        fn verify_draft(&mut self, _prompt: &[u32], draft_tokens: &[u32]) -> Result<Vec<u32>> {
            // Accept all tokens for simplicity
            Ok(draft_tokens.to_vec())
        }
    }

    let config = SpeculativeConfig {
        draft_k: 4,
        max_draft_tokens: 8,
    };

    let mut decoder = SpeculativeDecoder::new(MockDraft, MockTarget, config);

    let prompt = vec![1, 2, 3];
    let result = decoder.generate(&prompt, 20);

    assert!(result.is_ok());
    let generated = result.unwrap();
    assert!(!generated.is_empty());
}

#[test]
fn test_frameworks_together() {
    // Test that all three frameworks can work together conceptually
    // (In practice, they would be used in different parts of the system)

    // 1. LoRA: Load adapter
    let lora_manager = LoRAManager::new();
    let adapter = LoRAWeights::new("combined_test".to_string(), 4, 8.0);
    lora_manager.load_adapter(adapter).unwrap();

    // 2. Batching: Create batch
    let batcher = ContinuousBatcher::new(32, 2048);
    let req = BatchedRequest::with_prompt_text(1, vec![1, 2, 3], "Test prompt".to_string(), 50);
    batcher.add_request(req).unwrap();

    // 3. Speculative: Create decoder
    struct MockDraft;
    struct MockTarget;

    impl DraftModel for MockDraft {
        fn generate_draft(&mut self, _prompt: &[u32], k: usize) -> Result<Vec<u32>> {
            Ok((0..k).map(|i| i as u32).collect())
        }
    }

    impl TargetModel for MockTarget {
        fn verify_draft(&mut self, _prompt: &[u32], draft_tokens: &[u32]) -> Result<Vec<u32>> {
            Ok(draft_tokens.to_vec())
        }
    }

    let config = SpeculativeConfig::default();
    let _decoder = SpeculativeDecoder::new(MockDraft, MockTarget, config);

    // All frameworks initialized successfully
    assert_eq!(lora_manager.list_adapters().len(), 1);
    assert_eq!(batcher.active_count(), 1);
}
