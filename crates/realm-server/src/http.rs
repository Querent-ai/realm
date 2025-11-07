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
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

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
async fn metrics_endpoint() -> impl IntoResponse {
    // TODO: Integrate with realm-metrics
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        "# Metrics endpoint - TODO: integrate realm-metrics\n",
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

    // Generate response
    if request.stream {
        // Streaming response (SSE)
        // Clone Arc before dropping the lock
        let runtime_manager_clone = runtime_manager.clone();
        drop(rm); // Release lock before async operation
        let stream =
            generate_stream(tenant_id, prompt, request.max_tokens, runtime_manager_clone).await;
        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response
        let response = rm.generate(&tenant_id, prompt).map_err(|e| {
            error!("Generation failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

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
                prompt_tokens: 0,     // TODO: Count tokens
                completion_tokens: 0, // TODO: Count tokens
                total_tokens: 0,
            },
        };

        Ok(Json(completion).into_response())
    }
}

/// Generate streaming response
async fn generate_stream(
    tenant_id: String,
    prompt: String,
    _max_tokens: usize,
    runtime_manager: Arc<Mutex<RuntimeManager>>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    // TODO: Integrate with WASM streaming callback for real token-by-token streaming
    // For now, generate the full response and stream it
    let rm = runtime_manager.lock().await;
    match rm.generate(&tenant_id, prompt) {
        Ok(response) => {
            // Split response into chunks for streaming
            let chunks: Vec<String> = response
                .chars()
                .collect::<Vec<_>>()
                .chunks(10)
                .map(|chunk| chunk.iter().collect())
                .collect();

            let events: Vec<Result<Event, Infallible>> = chunks
                .into_iter()
                .map(|chunk| {
                    let data = json!({
                        "id": uuid::Uuid::new_v4().to_string(),
                        "object": "chat.completion.chunk",
                        "created": chrono::Utc::now().timestamp(),
                        "model": "realm-model",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": null
                        }]
                    });
                    Ok(Event::default().data(format!(
                        "data: {}",
                        serde_json::to_string(&data).unwrap_or_default()
                    )))
                })
                .chain(std::iter::once(Ok(Event::default().data("data: [DONE]"))))
                .collect();

            stream::iter(events)
        }
        Err(_) => {
            // Return error event
            stream::iter(vec![Ok(
                Event::default().data(r#"data: {"error": "Generation failed"}"#)
            )])
        }
    }
}
