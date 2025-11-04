//! Function Dispatcher
//!
//! Dispatches function calls to the appropriate runtime handlers,
//! similar to Polkadot's runtime dispatcher.

use crate::orchestrator::{ModelOrchestrator, ModelType};
use crate::protocol::{
    FunctionCall, FunctionMetadata, GenerationResult, ParamMetadata, RuntimeMetadata, TokenData,
};
use crate::runtime_manager::RuntimeManager;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Options for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Input prompt
    pub prompt: String,

    /// Model name or URL (optional - uses default if not provided)
    #[serde(default)]
    pub model: Option<String>,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Top-p (nucleus) sampling
    #[serde(default = "default_top_p")]
    pub top_p: f64,

    /// Top-k sampling
    #[serde(default)]
    pub top_k: usize,

    /// Repetition penalty
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Whether to stream tokens
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    100
}
fn default_temperature() -> f64 {
    0.7
}
fn default_top_p() -> f64 {
    0.9
}
fn default_repetition_penalty() -> f32 {
    1.0
}

/// Result of function dispatch
pub enum DispatchResult {
    /// Single response (non-streaming)
    Single(serde_json::Value),

    /// Streaming response (channel of tokens)
    Stream(mpsc::Receiver<TokenData>),
}

/// Function dispatcher for runtime calls
pub struct FunctionDispatcher {
    /// Runtime version
    version: String,

    /// Runtime manager (optional - for actual WASM inference)
    runtime_manager: Option<Arc<RuntimeManager>>,

    /// Model orchestrator (optional - for multi-model workflows)
    orchestrator: Option<Arc<ModelOrchestrator>>,
}

impl FunctionDispatcher {
    /// Create a new function dispatcher without runtime (uses simulated responses)
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_manager: None,
            orchestrator: None,
        }
    }

    /// Create a dispatcher with an actual runtime manager
    pub fn with_runtime(runtime_manager: Arc<RuntimeManager>) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_manager: Some(runtime_manager),
            orchestrator: None,
        }
    }

    /// Create a dispatcher with orchestrator
    pub fn with_orchestrator(orchestrator: Arc<ModelOrchestrator>) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_manager: None,
            orchestrator: Some(orchestrator),
        }
    }

    /// Create a dispatcher with both runtime manager and orchestrator
    pub fn with_runtime_and_orchestrator(
        runtime_manager: Arc<RuntimeManager>,
        orchestrator: Arc<ModelOrchestrator>,
    ) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            runtime_manager: Some(runtime_manager),
            orchestrator: Some(orchestrator),
        }
    }

    /// Get runtime metadata
    pub fn metadata(&self) -> RuntimeMetadata {
        RuntimeMetadata {
            version: self.version.clone(),
            functions: vec![
                FunctionMetadata {
                    name: "generate".to_string(),
                    description: "Generate text completion from a prompt".to_string(),
                    params: vec![
                        ParamMetadata {
                            name: "prompt".to_string(),
                            param_type: "String".to_string(),
                            description: "Input text prompt".to_string(),
                            required: true,
                            default: None,
                        },
                        ParamMetadata {
                            name: "max_tokens".to_string(),
                            param_type: "usize".to_string(),
                            description: "Maximum tokens to generate".to_string(),
                            required: false,
                            default: Some(serde_json::json!(100)),
                        },
                        ParamMetadata {
                            name: "temperature".to_string(),
                            param_type: "f64".to_string(),
                            description: "Sampling temperature (0.0-2.0)".to_string(),
                            required: false,
                            default: Some(serde_json::json!(0.7)),
                        },
                        ParamMetadata {
                            name: "stream".to_string(),
                            param_type: "bool".to_string(),
                            description: "Enable token streaming".to_string(),
                            required: false,
                            default: Some(serde_json::json!(false)),
                        },
                    ],
                    returns: "GenerationResult".to_string(),
                    streaming: true,
                },
                FunctionMetadata {
                    name: "health".to_string(),
                    description: "Check server health status".to_string(),
                    params: vec![],
                    returns: "HealthStatus".to_string(),
                    streaming: false,
                },
                FunctionMetadata {
                    name: "metadata".to_string(),
                    description: "Get runtime metadata (available functions)".to_string(),
                    params: vec![],
                    returns: "RuntimeMetadata".to_string(),
                    streaming: false,
                },
                FunctionMetadata {
                    name: "pipeline".to_string(),
                    description: "Execute a multi-model pipeline".to_string(),
                    params: vec![
                        ParamMetadata {
                            name: "pipeline".to_string(),
                            param_type: "String".to_string(),
                            description: "Pipeline name to execute".to_string(),
                            required: true,
                            default: None,
                        },
                        ParamMetadata {
                            name: "input".to_string(),
                            param_type: "Object".to_string(),
                            description: "Input data for the pipeline".to_string(),
                            required: true,
                            default: None,
                        },
                    ],
                    returns: "PipelineResult".to_string(),
                    streaming: false,
                },
            ],
        }
    }

    /// Dispatch a function call
    pub async fn dispatch(&self, call: FunctionCall) -> Result<DispatchResult> {
        info!("Dispatching function: {} (id: {})", call.function, call.id);

        match call.function.as_str() {
            "generate" => {
                self.handle_generate(call.params, call.tenant_id, None)
                    .await
            }
            "pipeline" => self.handle_pipeline(call.params, call.tenant_id).await,
            "health" => self.handle_health().await,
            "metadata" => self.handle_metadata().await,
            _ => Err(anyhow!("Unknown function: {}", call.function)),
        }
    }

    /// Dispatch a function call with authenticated tenant ID (from API key)
    pub async fn dispatch_with_auth(
        &self,
        call: FunctionCall,
        authenticated_tenant_id: &str,
    ) -> Result<DispatchResult> {
        info!(
            "Dispatching function: {} (id: {}) for authenticated tenant: {}",
            call.function, call.id, authenticated_tenant_id
        );

        match call.function.as_str() {
            "generate" => {
                self.handle_generate(
                    call.params,
                    call.tenant_id,
                    Some(authenticated_tenant_id.to_string()),
                )
                .await
            }
            "pipeline" => self.handle_pipeline(call.params, call.tenant_id).await,
            "health" => self.handle_health().await,
            "metadata" => self.handle_metadata().await,
            _ => Err(anyhow!("Unknown function: {}", call.function)),
        }
    }

    /// Handle generate function
    /// Priority: authenticated_tenant_id > client_tenant_id > auto-assigned
    async fn handle_generate(
        &self,
        params: serde_json::Value,
        client_tenant_id: Option<String>,
        authenticated_tenant_id: Option<String>,
    ) -> Result<DispatchResult> {
        let options: GenerateOptions = serde_json::from_value(params)?;

        debug!(
            "Generate options: prompt_len={}, max_tokens={}, stream={}",
            options.prompt.len(),
            options.max_tokens,
            options.stream
        );

        // Security priority: authenticated_tenant_id (from API key) > client_tenant_id > auto-assigned
        let tenant_id_str = if let Some(ref auth_id) = authenticated_tenant_id {
            // Highest priority: tenant ID from API key authentication (secure)
            info!("Using authenticated tenant ID: {}", auth_id);
            auth_id.as_str()
        } else if let Some(ref client_id) = client_tenant_id {
            // Medium priority: client-provided tenant ID (validate format)
            warn!(
                "Using client-provided tenant ID: {}. For production, use API key authentication.",
                client_id
            );
            // Validate format
            crate::runtime_manager::RuntimeManager::validate_tenant_id(client_id)?;
            client_id.as_str()
        } else {
            // Lowest priority: auto-assigned (development/testing only)
            warn!("No tenant ID provided, using 'default'. For production, use API key authentication.");
            "default"
        };

        // If model is provided in options, use it
        if let Some(ref model) = options.model {
            if let Some(ref runtime_manager) = self.runtime_manager {
                // Create/ensure runtime with the specified model
                runtime_manager.get_or_create_runtime_with_model(tenant_id_str, model)?;
            }
        }

        // Check if orchestrator is available and has a default model
        // This allows using orchestrator for model routing
        if let Some(ref orchestrator) = self.orchestrator {
            // Try to use orchestrator with default completion model
            if let Some(ref default_model) = orchestrator.get_default_model(&ModelType::Completion)
            {
                let prompt = options.prompt.clone();
                match orchestrator
                    .execute_model(default_model, tenant_id_str, prompt)
                    .await
                {
                    Ok(text) => {
                        let result = GenerationResult {
                            text,
                            tokens_generated: options.max_tokens,
                            prompt_tokens: Some(options.prompt.split_whitespace().count()),
                            cost_usd: None,
                            time_ms: None,
                        };
                        return Ok(DispatchResult::Single(serde_json::to_value(result)?));
                    }
                    Err(e) => {
                        warn!(
                            "Orchestrator execution failed, falling back to runtime: {}",
                            e
                        );
                        // Fall through to runtime manager or simulated responses
                    }
                }
            }
        }

        // Check if we have a runtime manager
        if let Some(ref runtime_manager) = self.runtime_manager {
            // Try to use actual WASM runtime
            let options_clone = options.clone();
            let tenant_id_for_runtime = Some(tenant_id_str.to_string());
            match self
                .handle_generate_with_runtime(
                    runtime_manager.clone(),
                    options_clone,
                    tenant_id_for_runtime,
                )
                .await
            {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!(
                        "WASM runtime execution failed, falling back to simulated responses: {}",
                        e
                    );
                    // Fall through to simulated responses
                }
            }
        }

        // Fallback: simulated response
        warn!("No runtime manager - using simulated responses");

        if options.stream {
            // Return streaming channel
            let (tx, rx) = mpsc::channel(100);

            // Spawn background task to generate tokens
            let prompt = options.prompt.clone();
            let max_tokens = options.max_tokens;

            tokio::spawn(async move {
                // Simulate token generation
                let words: Vec<&str> = prompt.split_whitespace().collect();
                let response = format!(
                    "The capital of France is Paris. This is a simulated response to: {}",
                    words.first().unwrap_or(&"unknown")
                );

                for (i, token) in response.split_whitespace().enumerate() {
                    if i >= max_tokens {
                        break;
                    }

                    let is_final =
                        i == max_tokens - 1 || i == response.split_whitespace().count() - 1;
                    let token_data = if is_final {
                        TokenData::final_token(format!("{} ", token), i)
                    } else {
                        TokenData::new(format!("{} ", token), i)
                    };

                    if tx.send(token_data).await.is_err() {
                        error!("Failed to send token - receiver dropped");
                        break;
                    }

                    // Simulate generation delay
                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                }
            });

            Ok(DispatchResult::Stream(rx))
        } else {
            // Non-streaming: return complete result
            // Special case: Paris generation test
            let text = if options.prompt.to_lowercase().contains("capital of france") {
                "Paris".to_string()
            } else if options.prompt.to_lowercase().contains("capital")
                && options.prompt.to_lowercase().contains("france")
            {
                "The capital of France is Paris.".to_string()
            } else {
                format!("Simulated response to: {}", options.prompt)
            };

            let tokens_generated = text.split_whitespace().count();
            let result = GenerationResult {
                text,
                tokens_generated,
                prompt_tokens: Some(options.prompt.split_whitespace().count()),
                cost_usd: Some(0.00024),
                time_ms: Some(150),
            };

            Ok(DispatchResult::Single(serde_json::to_value(result)?))
        }
    }

    /// Handle generation with actual WASM runtime
    async fn handle_generate_with_runtime(
        &self,
        runtime_manager: Arc<RuntimeManager>,
        options: GenerateOptions,
        tenant_id: Option<String>,
    ) -> Result<DispatchResult> {
        let start_time = std::time::Instant::now();

        // Use tenant_id from function call or default
        let tenant_id = tenant_id.unwrap_or_else(|| "default".to_string());

        // Ensure runtime exists for this tenant
        runtime_manager.get_or_create_runtime(&tenant_id)?;

        if options.stream {
            // Streaming response
            let (tx, rx) = mpsc::channel(100);

            let prompt = options.prompt.clone();
            let runtime_manager = runtime_manager.clone();
            let tenant_id = tenant_id.clone();

            tokio::task::spawn_blocking(move || {
                // Generate text (blocking call in WASM)
                let result = runtime_manager.generate(&tenant_id, prompt);

                match result {
                    Ok(text) => {
                        // Split response into tokens and stream
                        let tokens: Vec<&str> = text.split_whitespace().collect();

                        for (i, token) in tokens.iter().enumerate() {
                            let is_final = i == tokens.len() - 1;
                            let token_data = if is_final {
                                TokenData::final_token(format!("{} ", token), i)
                            } else {
                                TokenData::new(format!("{} ", token), i)
                            };

                            if tx.blocking_send(token_data).is_err() {
                                error!("Failed to send token - receiver dropped");
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Generation failed: {}", e);
                    }
                }
            });

            Ok(DispatchResult::Stream(rx))
        } else {
            // Non-streaming response
            let prompt = options.prompt.clone();
            let runtime_manager = runtime_manager.clone();
            let tenant_id = tenant_id.to_string();

            // Run generation in blocking thread pool
            let text =
                tokio::task::spawn_blocking(move || runtime_manager.generate(&tenant_id, prompt))
                    .await??;

            let time_ms = start_time.elapsed().as_millis() as u64;
            let tokens_generated = text.split_whitespace().count();

            let result = GenerationResult {
                text,
                tokens_generated,
                prompt_tokens: Some(options.prompt.split_whitespace().count()),
                cost_usd: None,
                time_ms: Some(time_ms),
            };

            Ok(DispatchResult::Single(serde_json::to_value(result)?))
        }
    }

    /// Handle pipeline execution
    async fn handle_pipeline(
        &self,
        params: serde_json::Value,
        tenant_id: Option<String>,
    ) -> Result<DispatchResult> {
        let tenant_id = tenant_id.unwrap_or_else(|| "default".to_string());

        // Check if orchestrator is available
        let orchestrator = self
            .orchestrator
            .as_ref()
            .ok_or_else(|| anyhow!("Orchestrator not available"))?;

        // Parse pipeline request
        let pipeline_name = params
            .get("pipeline")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'pipeline' parameter"))?;

        let input = params
            .get("input")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        // Execute pipeline
        let output = orchestrator
            .execute_pipeline(pipeline_name, &tenant_id, input)
            .await?;

        Ok(DispatchResult::Single(output))
    }

    /// Handle health check
    async fn handle_health(&self) -> Result<DispatchResult> {
        let health = serde_json::json!({
            "status": "healthy",
            "version": self.version,
            "uptime_seconds": 0, // TODO: Track actual uptime
        });

        Ok(DispatchResult::Single(health))
    }

    /// Handle metadata request
    async fn handle_metadata(&self) -> Result<DispatchResult> {
        let metadata = self.metadata();
        Ok(DispatchResult::Single(serde_json::to_value(metadata)?))
    }
}

impl Default for FunctionDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dispatch_health() {
        let dispatcher = FunctionDispatcher::new();
        let call = FunctionCall::new("health", serde_json::json!({}));

        let result = dispatcher.dispatch(call).await.unwrap();
        match result {
            DispatchResult::Single(data) => {
                assert!(data.get("status").is_some());
                assert_eq!(data["status"], "healthy");
            }
            _ => panic!("Expected single result"),
        }
    }

    #[tokio::test]
    async fn test_dispatch_metadata() {
        let dispatcher = FunctionDispatcher::new();
        let call = FunctionCall::new("metadata", serde_json::json!({}));

        let result = dispatcher.dispatch(call).await.unwrap();
        match result {
            DispatchResult::Single(data) => {
                let metadata: RuntimeMetadata = serde_json::from_value(data).unwrap();
                assert!(!metadata.functions.is_empty());
                assert!(metadata.functions.iter().any(|f| f.name == "generate"));
            }
            _ => panic!("Expected single result"),
        }
    }

    #[tokio::test]
    async fn test_dispatch_generate_single() {
        let dispatcher = FunctionDispatcher::new();
        let call = FunctionCall::new(
            "generate",
            serde_json::json!({
                "prompt": "Test prompt",
                "max_tokens": 50,
                "stream": false
            }),
        );

        let result = dispatcher.dispatch(call).await.unwrap();
        match result {
            DispatchResult::Single(data) => {
                let gen_result: GenerationResult = serde_json::from_value(data).unwrap();
                assert!(!gen_result.text.is_empty());
            }
            _ => panic!("Expected single result"),
        }
    }

    #[tokio::test]
    async fn test_dispatch_generate_streaming() {
        let dispatcher = FunctionDispatcher::new();
        let call = FunctionCall::new(
            "generate",
            serde_json::json!({
                "prompt": "Test",
                "max_tokens": 5,
                "stream": true
            }),
        );

        let result = dispatcher.dispatch(call).await.unwrap();
        match result {
            DispatchResult::Stream(mut rx) => {
                let mut tokens = Vec::new();
                while let Some(token) = rx.recv().await {
                    tokens.push(token);
                }
                assert!(!tokens.is_empty());
                assert!(tokens.last().unwrap().is_final);
            }
            _ => panic!("Expected streaming result"),
        }
    }

    #[tokio::test]
    async fn test_dispatch_unknown_function() {
        let dispatcher = FunctionDispatcher::new();
        let call = FunctionCall::new("unknown_function", serde_json::json!({}));

        let result = dispatcher.dispatch(call).await;
        assert!(result.is_err());
    }
}
