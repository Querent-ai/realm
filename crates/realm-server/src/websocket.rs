//! WebSocket Connection Handler
//!
//! Handles individual WebSocket connections, dispatching function calls
//! and streaming responses back to clients.

use crate::auth::ApiKeyStore;
use crate::dispatcher::{DispatchResult, FunctionDispatcher};
use crate::protocol::{ErrorInfo, FunctionCall, FunctionResponse};
use crate::rate_limiter::RateLimiter;
use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use tracing::{debug, error, info, warn};

/// Authentication message from client
#[derive(Debug, Deserialize, Serialize)]
struct AuthMessage {
    /// Message type (should be "auth")
    #[serde(rename = "type")]
    msg_type: String,

    /// API key
    api_key: String,
}

/// WebSocket connection handler
pub struct WebSocketHandler {
    /// Function dispatcher
    dispatcher: Arc<FunctionDispatcher>,

    /// Optional tenant ID (from authentication)
    tenant_id: Option<String>,

    /// Rate limiter (optional)
    rate_limiter: Option<Arc<RateLimiter>>,
}

impl WebSocketHandler {
    /// Create a new WebSocket handler
    pub fn new(dispatcher: Arc<FunctionDispatcher>) -> Self {
        Self {
            dispatcher,
            tenant_id: None,
            rate_limiter: None,
        }
    }

    /// Create a new WebSocket handler with rate limiter
    pub fn with_rate_limiter(
        dispatcher: Arc<FunctionDispatcher>,
        rate_limiter: Option<Arc<RateLimiter>>,
    ) -> Self {
        Self {
            dispatcher,
            tenant_id: None,
            rate_limiter,
        }
    }

    /// Create a new WebSocket handler with tenant ID
    pub fn with_tenant(
        dispatcher: Arc<FunctionDispatcher>,
        tenant_id: impl Into<String>,
        rate_limiter: Option<Arc<RateLimiter>>,
    ) -> Self {
        Self {
            dispatcher,
            tenant_id: Some(tenant_id.into()),
            rate_limiter,
        }
    }

    /// Handle authenticated WebSocket connection
    /// Expects first message to be an authentication message with API key
    pub async fn with_auth(
        dispatcher: Arc<FunctionDispatcher>,
        api_key_store: Arc<ApiKeyStore>,
        rate_limiter: Option<Arc<RateLimiter>>,
        mut ws: WebSocketStream<TcpStream>,
    ) -> Result<()> {
        info!("Awaiting authentication...");

        // Send auth request
        let auth_request = serde_json::json!({
            "type": "auth_required",
            "message": "Please send authentication message with your API key"
        });
        ws.send(Message::Text(auth_request.to_string())).await?;

        // Wait for authentication message (with timeout)
        let auth_timeout = tokio::time::Duration::from_secs(10);
        let auth_result = tokio::time::timeout(auth_timeout, ws.next()).await;

        match auth_result {
            Ok(Some(Ok(Message::Text(text)))) => {
                // Try to parse as auth message
                match serde_json::from_str::<AuthMessage>(&text) {
                    Ok(auth_msg) if auth_msg.msg_type == "auth" => {
                        // Validate API key
                        match api_key_store.validate(&auth_msg.api_key) {
                            Ok(tenant_id) => {
                                info!("Authentication successful for tenant: {}", tenant_id);

                                // Send auth success
                                let auth_success = serde_json::json!({
                                    "type": "auth_success",
                                    "tenant_id": tenant_id,
                                });
                                ws.send(Message::Text(auth_success.to_string())).await?;

                                // Create handler with tenant ID and handle connection
                                let handler =
                                    Self::with_tenant(dispatcher, tenant_id, rate_limiter);
                                handler.handle_connection(ws).await?;

                                Ok(())
                            }
                            Err(e) => {
                                warn!("Authentication failed: {}", e);

                                // Send auth failure
                                let auth_failure = serde_json::json!({
                                    "type": "auth_failed",
                                    "error": format!("{}", e),
                                });
                                ws.send(Message::Text(auth_failure.to_string())).await?;
                                ws.close(None).await?;

                                Err(anyhow!("Authentication failed: {}", e))
                            }
                        }
                    }
                    _ => {
                        warn!("Invalid authentication message format");

                        let auth_failure = serde_json::json!({
                            "type": "auth_failed",
                            "error": "Invalid authentication message format. Expected: {\"type\":\"auth\",\"api_key\":\"your-key\"}",
                        });
                        ws.send(Message::Text(auth_failure.to_string())).await?;
                        ws.close(None).await?;

                        Err(anyhow!("Invalid authentication message"))
                    }
                }
            }
            Ok(Some(Ok(_))) => {
                warn!("Non-text authentication message received");

                let auth_failure = serde_json::json!({
                    "type": "auth_failed",
                    "error": "Authentication message must be text",
                });
                ws.send(Message::Text(auth_failure.to_string())).await?;
                ws.close(None).await?;

                Err(anyhow!("Non-text authentication message"))
            }
            Ok(Some(Err(e))) => {
                error!("WebSocket error during authentication: {}", e);
                Err(anyhow!("WebSocket error: {}", e))
            }
            Ok(None) => {
                warn!("Connection closed before authentication");
                Err(anyhow!("Connection closed"))
            }
            Err(_) => {
                warn!("Authentication timeout");

                let auth_failure = serde_json::json!({
                    "type": "auth_failed",
                    "error": "Authentication timeout (10 seconds)",
                });
                ws.send(Message::Text(auth_failure.to_string())).await?;
                ws.close(None).await?;

                Err(anyhow!("Authentication timeout"))
            }
        }
    }

    /// Handle a WebSocket connection
    pub async fn handle_connection(&self, mut ws: WebSocketStream<TcpStream>) -> Result<()> {
        info!("New WebSocket connection (tenant: {:?})", self.tenant_id);

        // Send welcome message
        let welcome = serde_json::json!({
            "status": "connected",
            "version": env!("CARGO_PKG_VERSION"),
            "tenant_id": self.tenant_id,
        });

        ws.send(Message::Text(welcome.to_string())).await?;

        // Message loop
        while let Some(msg_result) = ws.next().await {
            match msg_result {
                Ok(msg) => {
                    if let Err(e) = self.handle_message(&mut ws, msg).await {
                        error!("Error handling message: {}", e);

                        // Send error response
                        let error_response = FunctionResponse::error(
                            "unknown".to_string(),
                            ErrorInfo::new("INTERNAL_ERROR", format!("{}", e)),
                        );

                        if let Ok(json) = serde_json::to_string(&error_response) {
                            let _ = ws.send(Message::Text(json)).await;
                        }
                    }
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
            }
        }

        info!("WebSocket connection closed (tenant: {:?})", self.tenant_id);
        Ok(())
    }

    /// Handle a single WebSocket message
    async fn handle_message(
        &self,
        ws: &mut WebSocketStream<TcpStream>,
        msg: Message,
    ) -> Result<()> {
        match msg {
            Message::Text(text) => {
                debug!("Received text message: {}", text);

                // Parse function call
                let call: FunctionCall = serde_json::from_str(&text)
                    .map_err(|e| anyhow!("Invalid function call: {}", e))?;

                // Validate tenant ID if set
                if let Some(ref tenant) = self.tenant_id {
                    if call.tenant_id.as_ref() != Some(tenant) {
                        warn!(
                            "Tenant ID mismatch: expected {}, got {:?}",
                            tenant, call.tenant_id
                        );

                        let error_response = FunctionResponse::error(
                            call.id.clone(),
                            ErrorInfo::new("UNAUTHORIZED", "Tenant ID mismatch"),
                        );

                        ws.send(Message::Text(serde_json::to_string(&error_response)?))
                            .await?;

                        return Ok(());
                    }

                    // Check rate limit for tenant
                    if let Some(ref limiter) = self.rate_limiter {
                        if let Err(e) = limiter.check_rate_limit(tenant) {
                            warn!("Rate limit exceeded for tenant {}: {}", tenant, e);

                            let error_response = FunctionResponse::error(
                                call.id.clone(),
                                ErrorInfo::new("RATE_LIMIT_EXCEEDED", format!("{}", e)),
                            );

                            ws.send(Message::Text(serde_json::to_string(&error_response)?))
                                .await?;

                            return Ok(());
                        }
                    }
                }

                // Dispatch function call
                let request_id = call.id.clone();
                match self.dispatcher.dispatch(call).await {
                    Ok(result) => {
                        self.handle_dispatch_result(ws, request_id, result).await?;
                    }
                    Err(e) => {
                        error!("Dispatch error: {}", e);

                        let error_response = FunctionResponse::error(
                            request_id,
                            ErrorInfo::new("DISPATCH_ERROR", format!("{}", e)),
                        );

                        ws.send(Message::Text(serde_json::to_string(&error_response)?))
                            .await?;
                    }
                }
            }
            Message::Binary(data) => {
                warn!(
                    "Received binary message ({} bytes) - not supported",
                    data.len()
                );

                let error_response = serde_json::json!({
                    "error": "Binary messages not supported"
                });

                ws.send(Message::Text(error_response.to_string())).await?;
            }
            Message::Ping(data) => {
                debug!("Received ping");
                ws.send(Message::Pong(data)).await?;
            }
            Message::Pong(_) => {
                debug!("Received pong");
            }
            Message::Close(_) => {
                info!("Received close message");
                return Err(anyhow!("Connection closed by client"));
            }
            Message::Frame(_) => {
                // Raw frames are handled internally by tungstenite
            }
        }

        Ok(())
    }

    /// Handle dispatch result (single or streaming)
    async fn handle_dispatch_result(
        &self,
        ws: &mut WebSocketStream<TcpStream>,
        request_id: String,
        result: DispatchResult,
    ) -> Result<()> {
        match result {
            DispatchResult::Single(data) => {
                // Send single response
                let response = FunctionResponse::complete(request_id, data);
                ws.send(Message::Text(serde_json::to_string(&response)?))
                    .await?;
            }
            DispatchResult::Stream(mut rx) => {
                // Stream tokens back
                while let Some(token_data) = rx.recv().await {
                    let response = FunctionResponse::streaming(
                        request_id.clone(),
                        serde_json::to_value(&token_data)?,
                    );

                    ws.send(Message::Text(serde_json::to_string(&response)?))
                        .await?;

                    // If this is the final token, send complete message
                    if token_data.is_final {
                        let complete_response = FunctionResponse::complete(
                            request_id.clone(),
                            serde_json::json!({
                                "status": "complete",
                                "tokens_generated": token_data.index + 1
                            }),
                        );

                        ws.send(Message::Text(serde_json::to_string(&complete_response)?))
                            .await?;

                        break;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::FunctionDispatcher;

    #[test]
    fn test_handler_creation() {
        let dispatcher = Arc::new(FunctionDispatcher::new());
        let handler = WebSocketHandler::new(dispatcher.clone());
        assert!(handler.tenant_id.is_none());

        let handler_with_tenant = WebSocketHandler::with_tenant(dispatcher, "tenant1", None);
        assert_eq!(handler_with_tenant.tenant_id, Some("tenant1".to_string()));
    }
}
