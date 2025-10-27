//! Tokenizer implementation for text encoding/decoding
//!
//! Supports BPE (Byte-Pair Encoding) tokenization with vocabulary loaded from GGUF metadata.

use crate::error::{Error, Result};
use crate::formats::gguf::ModelMeta;
use std::collections::HashMap;

/// Special tokens used by the tokenizer
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token: String,
    /// End of sequence token
    pub eos_token: String,
    /// Padding token
    pub pad_token: Option<String>,
    /// Unknown token
    pub unk_token: String,

    /// Token IDs
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            pad_token: None,
            unk_token: "<unk>".to_string(),
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
            unk_token_id: 0,
        }
    }
}

/// BPE Tokenizer with vocabulary
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Vocabulary: token string -> token ID
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary: token ID -> token string
    reverse_vocab: HashMap<u32, String>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// BPE merges (pair -> merged token)
    #[allow(dead_code)] // Will be used for full BPE implementation
    merges: Vec<(String, String)>,
    /// Whether to add BOS token
    add_bos_token: bool,
    /// Whether to add EOS token
    add_eos_token: bool,
}

impl Tokenizer {
    /// Create a new tokenizer with vocabulary
    pub fn new(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
    ) -> Self {
        let reverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        Self {
            vocab,
            reverse_vocab,
            special_tokens,
            merges,
            add_bos_token: true,
            add_eos_token: false,
        }
    }

    /// Create a tokenizer from GGUF ModelMeta
    pub fn from_gguf(meta: &ModelMeta) -> Result<Self> {
        // Extract tokens array from metadata
        let tokens_array = meta
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::ParseError("No tokenizer.ggml.tokens in metadata".to_string()))?;

        let mut vocab = HashMap::new();
        for (id, token_val) in tokens_array.iter().enumerate() {
            let token_str = token_val
                .as_string()
                .ok_or_else(|| Error::ParseError(format!("Token {} is not a string", id)))?
                .to_string();
            vocab.insert(token_str, id as u32);
        }

        if vocab.is_empty() {
            return Err(Error::ParseError(
                "No vocabulary found in metadata".to_string(),
            ));
        }

        // Extract BPE merges
        let merges = if let Some(merges_array) = meta
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
        {
            merges_array
                .iter()
                .filter_map(|v| {
                    v.as_string().and_then(|s| {
                        let parts: Vec<&str> = s.split(' ').collect();
                        if parts.len() == 2 {
                            Some((parts[0].to_string(), parts[1].to_string()))
                        } else {
                            None
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        // Extract special tokens
        let special_tokens = SpecialTokens {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            unk_token: "<unk>".to_string(),
            pad_token: None,
            bos_token_id: meta
                .metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u32())
                .unwrap_or(1),
            eos_token_id: meta
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u32())
                .unwrap_or(2),
            unk_token_id: meta
                .metadata
                .get("tokenizer.ggml.unknown_token_id")
                .and_then(|v| v.as_u32())
                .unwrap_or(0),
            pad_token_id: None,
        };

        Ok(Self::new(vocab, merges, special_tokens))
    }

    /// Create a tokenizer from GGUF metadata (legacy HashMap format)
    pub fn from_gguf_metadata(metadata: &HashMap<String, String>) -> Result<Self> {
        // Extract vocabulary from metadata
        let vocab_size = metadata
            .get("vocab_size")
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| Error::ParseError("Missing vocab_size in metadata".to_string()))?;

        let mut vocab = HashMap::new();
        for i in 0..vocab_size {
            let key = format!("tokenizer.ggml.token.{}", i);
            if let Some(token) = metadata.get(&key) {
                vocab.insert(token.clone(), i as u32);
            }
        }

        if vocab.is_empty() {
            return Err(Error::ParseError(
                "No vocabulary found in metadata".to_string(),
            ));
        }

        // Extract special tokens (with fallback defaults)
        let special_tokens = SpecialTokens {
            bos_token: metadata
                .get("tokenizer.ggml.bos_token")
                .cloned()
                .unwrap_or_else(|| "<s>".to_string()),
            eos_token: metadata
                .get("tokenizer.ggml.eos_token")
                .cloned()
                .unwrap_or_else(|| "</s>".to_string()),
            unk_token: metadata
                .get("tokenizer.ggml.unk_token")
                .cloned()
                .unwrap_or_else(|| "<unk>".to_string()),
            pad_token: metadata.get("tokenizer.ggml.pad_token").cloned(),
            bos_token_id: metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
            eos_token_id: metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|s| s.parse().ok())
                .unwrap_or(2),
            unk_token_id: metadata
                .get("tokenizer.ggml.unk_token_id")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            pad_token_id: metadata
                .get("tokenizer.ggml.pad_token_id")
                .and_then(|s| s.parse().ok()),
        };

        // BPE merges (if available)
        let merges = Vec::new(); // TODO: Parse from metadata if available

        Ok(Self::new(vocab, merges, special_tokens))
    }

    /// Encode text into token IDs using BPE algorithm
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokens = Vec::new();

        if add_special_tokens && self.add_bos_token {
            tokens.push(self.special_tokens.bos_token_id);
        }

        // SentencePiece-style encoding: add ▁ for spaces
        let text_with_space_marker = format!("▁{}", text.replace(' ', "▁"));

        // Encode using BPE
        let encoded = self.encode_bpe(&text_with_space_marker)?;
        tokens.extend(encoded);

        if add_special_tokens && self.add_eos_token {
            tokens.push(self.special_tokens.eos_token_id);
        }

        Ok(tokens)
    }

    /// BPE encoding implementation (greedy longest-match)
    ///
    /// Uses a greedy approach: find the longest token in vocab that matches
    /// the current position, consume it, repeat. This is O(N*V) instead of O(N²*M).
    fn encode_bpe(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // If we have no merges or vocab is small, use fallback
        if self.merges.is_empty() || self.vocab.is_empty() {
            return self.fallback_encode(text);
        }

        // Greedy longest-match tokenization
        let mut result = Vec::new();
        let text_bytes = text.as_bytes();
        let mut i = 0;

        while i < text_bytes.len() {
            let mut matched = false;

            // Try to match the longest possible token starting at position i
            // Start from current position to end, trying progressively shorter substrings
            let mut max_len = (text_bytes.len() - i).min(256); // Limit max token length

            while max_len > 0 {
                let end = i + max_len;
                if end <= text_bytes.len() {
                    if let Ok(substring) = std::str::from_utf8(&text_bytes[i..end]) {
                        if let Some(&id) = self.vocab.get(substring) {
                            result.push(id);
                            i = end;
                            matched = true;
                            break;
                        }
                    }
                }
                max_len -= 1;
            }

            // If no token matched, try byte fallback
            if !matched {
                let byte = text_bytes[i];
                let byte_token = format!("<0x{:02X}>", byte);
                if let Some(&id) = self.vocab.get(&byte_token) {
                    result.push(id);
                } else {
                    result.push(self.special_tokens.unk_token_id);
                }
                i += 1;
            }
        }

        Ok(result)
    }

    /// Fallback encoding when no merges available
    fn fallback_encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut result = Vec::new();

        // Try to match longest substrings in vocab
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut found = false;

            // Try progressively shorter substrings
            for len in (1..=(chars.len() - i).min(20)).rev() {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.vocab.get(&substr) {
                    result.push(id);
                    i += len;
                    found = true;
                    break;
                }
            }

            if !found {
                // Byte fallback
                let ch = chars[i];
                for byte in ch.to_string().bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.vocab.get(&byte_token) {
                        result.push(id);
                    } else {
                        result.push(self.special_tokens.unk_token_id);
                    }
                }
                i += 1;
            }
        }

        Ok(result)
    }

    /// Decode token IDs into text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut pieces = Vec::new();
        let special_ids = if skip_special_tokens {
            vec![
                self.special_tokens.bos_token_id,
                self.special_tokens.eos_token_id,
                self.special_tokens.unk_token_id,
            ]
        } else {
            Vec::new()
        };

        for &token_id in token_ids {
            if skip_special_tokens && special_ids.contains(&token_id) {
                continue;
            }

            if let Some(token_str) = self.reverse_vocab.get(&token_id) {
                // Handle byte tokens
                if token_str.starts_with("<0x") && token_str.ends_with('>') {
                    // Decode byte token
                    let hex = &token_str[3..token_str.len() - 1];
                    if let Ok(byte_val) = u8::from_str_radix(hex, 16) {
                        pieces.push(vec![byte_val]);
                    }
                } else {
                    // Regular token - add as UTF-8 bytes
                    pieces.push(token_str.as_bytes().to_vec());
                }
            }
        }

        // Concatenate all bytes and convert to string
        let all_bytes: Vec<u8> = pieces.into_iter().flatten().collect();
        let mut result = String::from_utf8_lossy(&all_bytes).to_string();

        // Replace SentencePiece space markers with actual spaces
        result = result.replace('▁', " ");

        // Trim leading space (from the initial ▁)
        Ok(result.trim().to_string())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token ID for a specific token
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get token string for a specific ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.reverse_vocab.get(&id).map(|s| s.as_str())
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Set whether to add BOS token during encoding
    pub fn set_add_bos_token(&mut self, add: bool) {
        self.add_bos_token = add;
    }

    /// Set whether to add EOS token during encoding
    pub fn set_add_eos_token(&mut self, add: bool) {
        self.add_eos_token = add;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tokenizer() -> Tokenizer {
        let mut vocab = HashMap::new();
        // Add byte fallback tokens
        for byte in 0..=255 {
            vocab.insert(format!("<0x{:02X}>", byte), byte as u32 + 3);
        }
        // Add special tokens
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<s>".to_string(), 1);
        vocab.insert("</s>".to_string(), 2);
        // Add SentencePiece-style tokens with space marker
        vocab.insert("▁hello".to_string(), 259);
        vocab.insert("▁world".to_string(), 260);
        vocab.insert("▁test".to_string(), 261);

        Tokenizer::new(vocab, Vec::new(), SpecialTokens::default())
    }

    #[test]
    fn test_encode_simple() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("hello world", true).unwrap();

        // Should have BOS + tokens
        assert!(tokens.len() >= 2);
        assert_eq!(tokens[0], 1); // BOS
    }

    #[test]
    fn test_decode_simple() {
        let tokenizer = create_test_tokenizer();
        // Use the SentencePiece tokens
        let text = tokenizer.decode(&[259, 260], false).unwrap();

        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("hello", true).unwrap();

        // Should start with BOS
        assert_eq!(tokens[0], 1);
    }

    #[test]
    fn test_unknown_token() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("unknown", true).unwrap();

        // Should use byte fallback tokens for unknown words
        assert!(tokens.len() > 1);
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = create_test_tokenizer();
        // 256 byte tokens + 3 special + 3 word tokens = 262
        assert_eq!(tokenizer.vocab_size(), 262);
    }

    #[test]
    fn test_token_lookup() {
        let tokenizer = create_test_tokenizer();

        assert_eq!(tokenizer.token_to_id("▁hello"), Some(259));
        assert_eq!(tokenizer.id_to_token(259), Some("▁hello"));
        assert_eq!(tokenizer.token_to_id("nonexistent"), None);
    }
}
