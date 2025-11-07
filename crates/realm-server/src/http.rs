//! HTTP/SSE Server for OpenAI-compatible API
//!
//! Provides REST endpoints with Server-Sent Events (SSE) streaming support

use crate::auth::ApiKeyStore;
use crate::runtime_manager::RuntimeManager;
use anyhow::{anyhow, Result};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::{self, Stream, StreamExt};
use realm_metrics::{LatencyMetrics, MetricsCollector};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc as StdArc;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

/// Count tokens in text using a simple approximation
///
/// Note: This is an approximation. For accurate token counting, use the tokenizer
/// from RuntimeManager when available.
fn count_tokens_approx(text: &str) -> usize {
    // Simple approximation: count words and add some overhead for special tokens
    // This is a rough estimate - actual tokenization would be more accurate
    let word_count = text.split_whitespace().count();
    // Add 20% overhead for subword tokenization and special tokens
    ((word_count as f64) * 1.2) as usize
}

/// OpenAI-compatible chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

fn default_max_tokens() -> usize {
    100
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

/// Chat message structure
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat completion response (non-streaming)
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Server state shared across requests
#[derive(Clone)]
pub struct ServerState {
    pub runtime_manager: Arc<Mutex<RuntimeManager>>,
    pub api_key_store: Option<Arc<ApiKeyStore>>,
    pub metrics: Option<StdArc<StdMutex<MetricsCollector>>>,
}

/// Create HTTP router with OpenAI-compatible endpoints
pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_endpoint))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Metrics endpoint (Prometheus format)
async fn metrics_endpoint(State(state): State<ServerState>) -> impl IntoResponse {
    let metrics_text = if let Some(ref metrics) = state.metrics {
        let collector = metrics.lock().unwrap();
        collector.export_prometheus()
    } else {
        "# Metrics not enabled\n".to_string()
    };

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        metrics_text,
    )
}

/// Extract tenant ID from Authorization header or use default
fn extract_tenant_id(headers: &HeaderMap) -> Result<String> {
    if let Some(auth_header) = headers.get("authorization") {
        let auth_str = auth_header
            .to_str()
            .map_err(|_| anyhow!("Invalid authorization header"))?;

        if let Some(api_key) = auth_str.strip_prefix("Bearer ") {
            // For now, use API key as tenant ID (can be enhanced with proper auth)
            return Ok(api_key.to_string());
        }
    }

    // Default tenant if no auth provided
    Ok("default".to_string())
}

/// Chat completions endpoint (OpenAI-compatible)
async fn chat_completions(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    info!(
        "Chat completion request: model={}, stream={}, messages={}",
        request.model,
        request.stream,
        request.messages.len()
    );

    // Extract tenant ID from auth
    let tenant_id = extract_tenant_id(&headers).map_err(|e| {
        error!("Failed to extract tenant ID: {}", e);
        StatusCode::UNAUTHORIZED
    })?;

    // Validate API key if auth is enabled
    if let Some(ref api_key_store) = state.api_key_store {
        if let Some(auth_header) = headers.get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if let Some(api_key) = auth_str.strip_prefix("Bearer ") {
                    if api_key_store.validate(api_key).is_err() {
                        return Err(StatusCode::UNAUTHORIZED);
                    }
                }
            }
        } else {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    // Get or create runtime for tenant
    let runtime_manager = state.runtime_manager.clone();
    let rm = runtime_manager.lock().await;

    // Ensure runtime exists
    if !rm.is_tenant_id_taken(&tenant_id) {
        rm.get_or_create_runtime(&tenant_id).map_err(|e| {
            error!("Failed to create runtime for tenant {}: {}", tenant_id, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    }

    // Convert messages to prompt
    let prompt = request
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Record metrics if available
    let start_time = std::time::Instant::now();
    if let Some(ref metrics) = state.metrics {
        metrics.lock().unwrap().start_request();
    }

    // Generate response
    if request.stream {
        // Streaming response (SSE)
        // Clone Arc before dropping the lock
        let runtime_manager_clone = runtime_manager.clone();
        let metrics_clone = state.metrics.clone();
        drop(rm); // Release lock before async operation
        let stream = generate_stream(
            tenant_id,
            prompt,
            request.max_tokens,
            runtime_manager_clone,
            metrics_clone,
        )
        .await;
        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response
        let response = rm.generate(&tenant_id, prompt.clone()).map_err(|e| {
            error!("Generation failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        // Count tokens (approximation)
        let prompt_tokens = count_tokens_approx(&prompt);
        let completion_tokens = count_tokens_approx(&response);
        let total_tokens = prompt_tokens + completion_tokens;

        // Record latency metrics
        if let Some(ref metrics) = state.metrics {
            let duration = start_time.elapsed();
            let tokens_per_sec = if duration.as_secs_f64() > 0.0 {
                completion_tokens as f64 / duration.as_secs_f64()
            } else {
                0.0
            };
            let per_token_latency = if completion_tokens > 0 {
                duration / completion_tokens as u32
            } else {
                duration
            };
            metrics.lock().unwrap().record_latency(LatencyMetrics {
                ttft: duration, // For non-streaming, TTFT = total time
                tokens_per_sec,
                total_tokens: completion_tokens as u64,
                total_time: duration,
                per_token_latency,
            });
            metrics.lock().unwrap().finish_request();
        }

        let completion = ChatCompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: request.model,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: response,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            },
        };

        Ok(Json(completion).into_response())
    }
}

/// Generate streaming response
///
/// Uses RuntimeManager::generate_stream() to stream tokens as they're generated.
/// Currently streams word-by-word (simulates token streaming).
///
/// Note: True token-by-token streaming requires WASM module to support host function callbacks.
async fn generate_stream(
    tenant_id: String,
    prompt: String,
    max_tokens: usize,
    runtime_manager: Arc<Mutex<RuntimeManager>>,
    metrics: Option<StdArc<StdMutex<MetricsCollector>>>,
) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
    let start_time = std::time::Instant::now();
    let completion_id = uuid::Uuid::new_v4().to_string();
    let created = chrono::Utc::now().timestamp();
    let prompt_tokens = count_tokens_approx(&prompt);

    // Get stream receiver from RuntimeManager
    let rm = runtime_manager.lock().await;
    let rx = match rm.generate_stream(&tenant_id, prompt.clone()) {
        Ok(receiver) => receiver,
        Err(_) => {
            drop(rm);
            return stream::iter(vec![Ok(
                Event::default().data(r#"{"error": "Failed to start generation"}"#)
            )])
            .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }))
            .boxed();
        }
    };
    drop(rm); // Release lock

    // Track completion for metrics
    let accumulated_content = String::new();
    let token_count = 0;
    let send_done = false;

    // Stream tokens as they arrive
    stream::unfold(
        (
            rx,
            completion_id,
            created,
            prompt_tokens,
            start_time,
            metrics,
            accumulated_content,
            token_count,
            max_tokens,
            send_done,
        ),
        |state| async move {
            let (
                mut rx,
                completion_id,
                created,
                prompt_tokens,
                start_time,
                metrics,
                mut accumulated_content,
                mut token_count,
                max_tokens,
                send_done,
            ) = state;

            if send_done {
                return None;
            }

            match rx.recv().await {
                Some(chunk) => {
                    // Check if this is an error
                    if chunk.starts_with("Error:") {
                        return Some((
                            Ok(Event::default().data(format!(r#"{{"error": "{}"}}"#, chunk))),
                            (
                                rx,
                                completion_id,
                                created,
                                prompt_tokens,
                                start_time,
                                metrics,
                                accumulated_content,
                                token_count,
                                max_tokens,
                                true,
                            ),
                        ));
                    }

                    accumulated_content.push_str(&chunk);
                    token_count += 1;

                    // Check max_tokens limit
                    if token_count >= max_tokens {
                        let completion_tokens = count_tokens_approx(&accumulated_content);
                        let final_data = json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "realm-model",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "length"
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens
                            }
                        });

                        if let Some(ref m) = metrics {
                            let duration = start_time.elapsed();
                            let tokens_per_sec = if duration.as_secs_f64() > 0.0 {
                                completion_tokens as f64 / duration.as_secs_f64()
                            } else {
                                0.0
                            };
                            let per_token_latency = if completion_tokens > 0 {
                                duration / completion_tokens as u32
                            } else {
                                duration
                            };
                            m.lock().unwrap().record_latency(LatencyMetrics {
                                ttft: duration,
                                tokens_per_sec,
                                total_tokens: completion_tokens as u64,
                                total_time: duration,
                                per_token_latency,
                            });
                            m.lock().unwrap().finish_request();
                        }

                        return Some((
                            Ok(Event::default()
                                .data(serde_json::to_string(&final_data).unwrap_or_default())),
                            (
                                rx,
                                completion_id,
                                created,
                                prompt_tokens,
                                start_time,
                                metrics,
                                accumulated_content,
                                token_count,
                                max_tokens,
                                false,
                            ),
                        ));
                    }

                    // Send chunk event
                    let data = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "realm-model",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": null
                        }]
                    });

                    Some((
                        Ok(Event::default().data(serde_json::to_string(&data).unwrap_or_default())),
                        (
                            rx,
                            completion_id,
                            created,
                            prompt_tokens,
                            start_time,
                            metrics,
                            accumulated_content,
                            token_count,
                            max_tokens,
                            false,
                        ),
                    ))
                }
                None => {
                    // Stream ended, send final event
                    let completion_tokens = count_tokens_approx(&accumulated_content);
                    let final_data = json!({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "realm-model",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    });

                    if let Some(ref m) = metrics {
                        let duration = start_time.elapsed();
                        let tokens_per_sec = if duration.as_secs_f64() > 0.0 {
                            completion_tokens as f64 / duration.as_secs_f64()
                        } else {
                            0.0
                        };
                        let per_token_latency = if completion_tokens > 0 {
                            duration / completion_tokens as u32
                        } else {
                            duration
                        };
                        m.lock().unwrap().record_latency(LatencyMetrics {
                            ttft: duration,
                            tokens_per_sec,
                            total_tokens: completion_tokens as u64,
                            total_time: duration,
                            per_token_latency,
                        });
                        m.lock().unwrap().finish_request();
                    }

                    Some((
                        Ok(Event::default()
                            .data(serde_json::to_string(&final_data).unwrap_or_default())),
                        (
                            rx,
                            completion_id,
                            created,
                            prompt_tokens,
                            start_time,
                            metrics,
                            accumulated_content,
                            token_count,
                            max_tokens,
                            false,
                        ),
                    ))
                }
            }
        },
    )
    .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }))
    .boxed()
}
