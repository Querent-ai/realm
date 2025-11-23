//! WebSocket Protocol Definition
//!
//! Defines the message protocol for WebSocket-based function dispatch,
//! inspired by Polkadot's parachain runtime model.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A function call request from client to server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Unique request ID (for matching responses)
    pub id: String,

    /// Function name to call (e.g., "generate", "embed", "chat")
    pub function: String,

    /// Function parameters as JSON
    pub params: serde_json::Value,

    /// Optional tenant ID for multi-tenancy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
}

impl FunctionCall {
    /// Create a new function call with a random ID
    pub fn new(function: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            function: function.into(),
            params,
            tenant_id: None,
        }
    }

    /// Create a new function call with tenant ID
    pub fn with_tenant(
        function: impl Into<String>,
        params: serde_json::Value,
        tenant_id: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            function: function.into(),
            params,
            tenant_id: Some(tenant_id.into()),
        }
    }
}

/// Response status for function execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Function is executing, streaming results
    Streaming,
    /// Function completed successfully
    Complete,
    /// Function failed with error
    Error,
    /// Function was cancelled
    Cancelled,
}

/// A response from server to client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// Request ID this response is for
    pub id: String,

    /// Response status
    pub status: ResponseStatus,

    /// Response data (varies by function and status)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,

    /// Error information (only if status = Error)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorInfo>,
}

impl FunctionResponse {
    /// Create a streaming response
    pub fn streaming(id: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            status: ResponseStatus::Streaming,
            data: Some(data),
            error: None,
        }
    }

    /// Create a complete response
    pub fn complete(id: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            status: ResponseStatus::Complete,
            data: Some(data),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: impl Into<String>, error: ErrorInfo) -> Self {
        Self {
            id: id.into(),
            status: ResponseStatus::Error,
            data: None,
            error: Some(error),
        }
    }
}

/// Error information in responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error code (e.g., "RATE_LIMIT", "INVALID_PARAMS", "INTERNAL_ERROR")
    pub code: String,

    /// Human-readable error message
    pub message: String,

    /// Optional additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ErrorInfo {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn with_details(
        code: impl Into<String>,
        message: impl Into<String>,
        details: serde_json::Value,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: Some(details),
        }
    }
}

/// Runtime metadata (like Polkadot's runtime metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetadata {
    /// Runtime version
    pub version: String,

    /// Available functions
    pub functions: Vec<FunctionMetadata>,
}

/// Metadata for a single function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    /// Function name
    pub name: String,

    /// Function description
    pub description: String,

    /// Parameter definitions
    pub params: Vec<ParamMetadata>,

    /// Return type description
    pub returns: String,

    /// Whether this function supports streaming
    pub streaming: bool,
}

/// Metadata for a function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamMetadata {
    /// Parameter name
    pub name: String,

    /// Parameter type
    #[serde(rename = "type")]
    pub param_type: String,

    /// Parameter description
    pub description: String,

    /// Whether parameter is required
    pub required: bool,

    /// Default value (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

/// Token data for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    /// The token text
    pub text: String,

    /// Token index in sequence
    pub index: usize,

    /// Log probability (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob: Option<f32>,

    /// Whether this is the last token
    #[serde(default)]
    pub is_final: bool,
}

impl TokenData {
    pub fn new(text: impl Into<String>, index: usize) -> Self {
        Self {
            text: text.into(),
            index,
            logprob: None,
            is_final: false,
        }
    }

    pub fn final_token(text: impl Into<String>, index: usize) -> Self {
        Self {
            text: text.into(),
            index,
            logprob: None,
            is_final: true,
        }
    }
}

/// Generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,

    /// Number of tokens generated
    pub tokens_generated: usize,

    /// Number of prompt tokens (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<usize>,

    /// Cost in USD (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,

    /// Time taken in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_ms: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_call_serialization() {
        let call = FunctionCall::new(
            "generate",
            serde_json::json!({
                "prompt": "Test",
                "max_tokens": 100
            }),
        );

        let json = serde_json::to_string(&call).expect("FunctionCall should serialize to JSON");
        assert!(json.contains("generate"));
        assert!(json.contains("Test"));
    }

    #[test]
    fn test_function_response_streaming() {
        let resp = FunctionResponse::streaming(
            "req_123",
            serde_json::json!({
                "token": "Hello",
                "index": 0
            }),
        );

        assert_eq!(resp.status, ResponseStatus::Streaming);
        assert!(resp.data.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_function_response_error() {
        let resp = FunctionResponse::error(
            "req_456",
            ErrorInfo::new("RATE_LIMIT", "Rate limit exceeded"),
        );

        assert_eq!(resp.status, ResponseStatus::Error);
        assert!(resp.data.is_none());
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_token_data() {
        let token = TokenData::new("Hello", 0);
        assert_eq!(token.text, "Hello");
        assert_eq!(token.index, 0);
        assert!(!token.is_final);

        let final_token = TokenData::final_token("!", 10);
        assert!(final_token.is_final);
    }
}
